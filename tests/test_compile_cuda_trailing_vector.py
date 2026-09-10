"""Public matrix/vector capture, independent of scoring corpora and case names."""
import ctypes
from dataclasses import replace
import gc
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_bytecode, _compile_trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_mul_scalar import POLICIES, call_without_python, make_program
from tests.test_cuda_add import Comparison, available, runtime, torch, upload

SEEDS = (823109, 672041)
FORMS = ("x + y", "y + x", "x.add(y)", "y.add(x)",
         "x.__add__(y)", "y.__add__(x)", "x.__radd__(y)", "y.__radd__(x)",
         "-(x * -0.5 + y.multiply(1.375)).add(x) * 2",
         "(y.mul(0.25) + x.negative()).neg() + y")
CAPTURE = None
SCALE = 0.5
MULTIPLY = native.multiply


def helper(x):
    return MULTIPLY(x, SCALE) + CAPTURE


def via_helper(x):
    return -helper(x) + CAPTURE


def addition(x, y):
    return x + y


def composed(x, y):
    return (-(x * 0.5 + y)).add(y)


def nested(x, y):
    z = x + y
    shared = [z, x]
    return [shared, (z, shared, y)]


def shapes():
    rng = np.random.default_rng(940617)
    return [(1, 1), (1, 17), (13, 1), (0, 11), (9, 0), (0, 0),
            *[tuple(map(int, rng.integers(2, 48, size=2))) for _ in range(4)]]


def inputs(module, shape, seed, offset=0, device="cuda:0"):
    rng = np.random.default_rng(seed)
    result = []
    for dimensions in (shape, (shape[1],)):
        count = int(np.prod(dimensions))
        values = rng.normal(size=count + offset).astype(np.float32)
        result.append(upload(module, values, (count + offset,), device)[offset:].reshape(dimensions))
    return result


class TrailingVectorMetadataTests(unittest.TestCase):
    def test_exact_shape_relation_and_cpu_broadcasting(self):
        def record(left, right, device):
            recorder = _compile_trace.CompileTraceRecorder()
            return recorder.input(shape=left, device=device) + recorder.input(shape=right, device=device)
        for matrix in shapes() + [(0, 2**40)]:
            for left, right in ((matrix, (matrix[1],)), ((matrix[1],), matrix)):
                out = record(left, right, "cuda:0")
                self.assertEqual(out.metadata.shape, matrix)
                self.assertEqual(out.metadata.storage_offset, 0)
        for left, right in (((3, 7), (1,)), ((3, 7), (1, 7)),
                            ((3, 7), (3, 1)), ((2, 3, 7), (7,)), ((3, 7), ())):
            for a, b in ((left, right), (right, left)):
                with self.assertRaisesRegex(NotImplementedError, "broader broadcasting"):
                    record(a, b, "cuda:0")
                self.assertEqual(record(a, b, "cpu").metadata.shape, left)


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CompileCudaTrailingVectorTests(Comparison, unittest.TestCase):
    def test_all_add_forms_and_compositions_held_out(self):
        for expression in FORMS:
            program = make_program(f"def program(x, y):\n    return {expression}\n")
            for policy in POLICIES:
                for shape in shapes():
                    with self.subTest(expression=expression, policy=policy, shape=shape):
                        compiled, cache = compile_with_cache(program, *policy)
                        torch._dynamo.reset()
                        reference = torch.compile(program, backend="eager", fullgraph=policy[0], dynamic=policy[1])
                        for seed in SEEDS:
                            args, refs = inputs(native, shape, seed, 3), inputs(torch, shape, seed, 3)
                            expected = reference(*refs)
                            self.compare(call_without_python(compiled, (program.__code__,), *args), expected)
                            for a, b in zip(args, refs):
                                self.compare(a, b)
                        self.assertEqual(len(cache.graphs), 1)

    def test_noncanonical_singleton_empty_and_offset_metadata(self):
        base = native.tensor(list(range(120)), dtype=native.float32).to("cuda:0")
        refbase = torch.arange(120, dtype=torch.float32, device="cuda:0")
        views = (lambda b: (b[5:26].reshape(3, 7), b[9:16]),
                 lambda b: (b[:7].reshape(7, 1).t(), b[10:17]),
                 lambda b: (b[:7].reshape(1, 7).t(), b[1:2]),
                 lambda b: (b.reshape(12, 10)[:, 10:], b[120:]),
                 lambda b: (b.reshape(12, 10)[12:], b[:10]),
                 lambda b: (b.reshape(12, 10).t()[:0], b[:12]))
        for view in views:
            args, refs = view(base), view(refbase)
            for reverse in (False, True):
                args2, refs2 = (args[::-1], refs[::-1]) if reverse else (args, refs)
                for program in (addition, composed):
                    torch._dynamo.reset()
                    compiled, cache = compile_with_cache(program)
                    expected = torch.compile(program, backend="eager", fullgraph=True)(*refs2)
                    self.compare(compiled(*args2), expected)
                    graph = next(iter(cache.graphs.values()))
                    output = graph.forward(*args2)
                    self.compare(output, expected)
                    self.assertEqual(graph.operations[-1].metadata.stride, output.stride())
                    self.assertEqual(graph.operations[-1].metadata.storage_offset, 0)
                    self.compare(_compile_trace._native._compile_trace_binary(*args2, "add"), refs2[0] + refs2[1])

    def test_dynamic_guards_invalidated_cache_and_forged_metadata(self):
        for dynamic in (None, False, True):
            compiled, cache = compile_with_cache(composed, dynamic=dynamic)
            for shape, offset in (((3, 7), 2), ((5, 7), 2), ((5, 7), 3), ((2, 9), 3)):
                args, refs = inputs(native, shape, SEEDS[0], offset), inputs(torch, shape, SEEDS[0], offset)
                self.compare(compiled(*args), composed(*refs))
            self.assertEqual(len(cache.graphs), 3 if dynamic else 4)
            graph = next(iter(cache.graphs.values()))
            args = inputs(native, (3, 7), SEEDS[0], 2)
            for field, value in (("shape", (21,)), ("stride", (1, 3)),
                                 ("storage_offset", 2), ("device", "cuda:1"),
                                 ("requires_grad", True)):
                node = graph.operations[-1]
                malformed = replace(graph, operations=(*graph.operations[:-1],
                    replace(node, metadata=replace(node.metadata, **{field: value}))))
                with patch.object(_compile_trace._native, "_compile_trace_binary", side_effect=AssertionError("early execution")):
                    with self.assertRaises((ValueError, NotImplementedError)):
                        malformed.forward(*args)
            before = dict(cache.graphs)
            with patch.object(_compile_trace._native, "_compile_trace_scalar", side_effect=AssertionError("early execution")):
                with self.assertRaises(NotImplementedError):
                    compiled(args[0], native.ones((1, 7)).to("cuda:0"))
            self.assertEqual(cache.graphs, before)
            native.compiler.reset()
            self.assertEqual(cache.graphs, {})
            self.compare(compiled(*args), composed(*inputs(torch, (3, 7), SEEDS[0], 2)))
            with patch.object(_compile_bytecode, "lower_compile_graph", side_effect=AssertionError("cache miss")):
                fresh = inputs(native, (3, 7), SEEDS[1], 2)
                self.compare(compiled(*fresh), composed(*inputs(torch, (3, 7), SEEDS[1], 2)))
        compiled, cache = compile_with_cache(addition, limit=1)
        compiled(*args)
        with self.assertRaisesRegex(NotImplementedError, "recompile_limit"):
            compiled(*inputs(native, (4, 7), SEEDS[0], 2))
        self.assertEqual(len(cache.graphs), 1)

    def test_globals_helpers_callable_guards_and_lifetimes(self):
        x, y = inputs(native, (3, 7), SEEDS[0], 2)
        tx, ty = inputs(torch, (3, 7), SEEDS[0], 2)
        compiled, cache = compile_with_cache(via_helper)
        for scalar, vector in ((0.5, y), (1.375, y), (1.375, y * 2)):
            with patch(__name__ + ".CAPTURE", vector), patch(__name__ + ".SCALE", scalar):
                self.compare(call_without_python(compiled, (via_helper.__code__, helper.__code__), x),
                             -(tx * scalar + (ty if vector is y else ty * 2)) + (ty if vector is y else ty * 2))
                before = dict(cache.graphs)
                with patch(__name__ + ".MULTIPLY", lambda a, b: a):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
                self.assertEqual(cache.graphs, before)
        self.assertEqual(len(cache.graphs), 3)
        nested_compiled = native.compile(nested, backend="eager", fullgraph=True)
        result = nested_compiled(x, y)
        self.assertIs(result[0], result[1][1])
        self.assertIs(result[0][0], result[1][0])
        self.assertIs(result[0][1], x)
        self.assertIs(result[1][2], y)
        keep = result[0][0]
        self.assertNotIn(keep.data_ptr(), (x.data_ptr(), y.data_ptr()))
        # Aliased matrix/vector inputs and outputs outliving sources and caches.
        self.compare(native.compile(addition, backend="eager")(x, x[0]), tx + tx[0])
        del result, x, y, compiled, cache
        native.compiler.reset()
        gc.collect()
        for _ in range(40):
            discarded = native.compile(addition, backend="eager")(*inputs(native, (3, 7), SEEDS[1]))
            self.assertNotEqual(discarded.data_ptr(), keep.data_ptr())
        self.compare(keep, tx + ty)

    def test_fail_closed_public_and_private(self):
        x, y = inputs(native, (3, 7), SEEDS[0])
        compiled, cache = compile_with_cache(addition)
        compiled(x, y)
        before = dict(cache.graphs)
        invalid = [(x, native.ones(s).to("cuda:0")) for s in ((1,), (), (1, 7), (3, 1), (3,))]
        invalid += [(native.ones((2, 3, 7)).to("cuda:0"), y), (x.t(), y),
                    (x[:, 1:], y[:6]), (x, native.ones((7, 2)).to("cuda:0").select(1, 0)), (x, y.cpu())]
        for a, b in invalid:
            for args in ((a, b), (b, a)):
                with self.assertRaises(NotImplementedError):
                    compiled(*args)
                with self.assertRaises(NotImplementedError):
                    _compile_trace._native._compile_trace_binary(*args, "add")
                self.assertEqual(cache.graphs, before)
        for expression in ("x.add(y, alpha=1)", "x.add(y, alpha=2)", "x.add(other=y)",
                           "x.add(y, out=None)", "m.add(x, y)", "x + 1", "x * y", "x - y"):
            program = make_program(f"def program(x, y):\n    return {expression}\n")
            with self.assertRaises(NotImplementedError):
                native.compile(program, backend="eager", fullgraph=True)(x, y)
        for dtype in (torch.float64, torch.int64):
            with self.assertRaises((NotImplementedError, TypeError)):
                native.ones((3, 7), dtype=dtype).to("cuda:0")
        for context in (native.enable_grad(), native.no_grad()):
            with context:
                with self.assertRaises(NotImplementedError):
                    native.ones((3, 7), requires_grad=True).to("cuda:0")
        meta = _compile_trace._metadata_from_native_tensor(x)
        vector_meta = _compile_trace._metadata_from_native_tensor(y)
        for bad in (replace(meta, requires_grad=True), replace(meta, stride=(1, 3)),
                    replace(meta, dtype=_compile_trace.CompileTraceDType("torch.float64"))):
            with self.assertRaises(NotImplementedError):
                _compile_trace._binary_output_metadata(bad, vector_meta)
        for args in ((object(), y, "add"), (x, object(), "add"), (x, y, "mul")):
            with self.assertRaises((TypeError, NotImplementedError)):
                _compile_trace._native._compile_trace_binary(*args)

    def test_no_installed_torch_python_body_or_public_eager_dispatch(self):
        script = '''
import sys
from unittest.mock import patch
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('imported installed PyTorch')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
def program(x, y):
    return -(y + x.mul(0.5)).add(y)
def reject(frame, event, arg):
    if event == 'call' and frame.f_code in blocked:
        raise AssertionError('executed original body')
x = m.tensor([[2., 4.], [6., 8.]]).to('cuda:0')
y = m.tensor([1., -1.]).to('cuda:0')
f = m.compile(program, backend='eager', fullgraph=True)
blocked = {program.__code__}
for owner, names in ((m.Tensor, ('add', '__add__', 'mul', 'neg')), (m, ('add', 'mul', 'neg'))):
    for name in names:
        code = getattr(getattr(owner, name), '__code__', None)
        if code is not None:
            blocked.add(code)
sys.setprofile(reject)
for _ in range(3):
    assert f(x, y).cpu().tolist() == [[-3., 0.], [-5., -2.]]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CompileCudaTrailingVectorDeviceTests(Comparison, unittest.TestCase):
    def test_ordinals_and_restoration(self):
        compiled, cache = compile_with_cache(composed)
        previous = torch.cuda.current_device()
        lib = runtime()
        try:
            for ordinal in (0, 1, 0, 1):
                torch.cuda.set_device(1 - ordinal)
                args = inputs(native, (5, 11), SEEDS[0], 3, f"cuda:{ordinal}")
                refs = inputs(torch, (5, 11), SEEDS[0], 3, f"cuda:{ordinal}")
                self.compare(compiled(*args), composed(*refs))
                wrong = args[1].cpu().to(f"cuda:{1 - ordinal}")
                before = dict(cache.graphs)
                with self.assertRaises(NotImplementedError):
                    compiled(args[0], wrong)
                with self.assertRaises(NotImplementedError):
                    _compile_trace._native._compile_trace_binary(args[0], wrong, "add")
                self.assertEqual(cache.graphs, before)
                current = ctypes.c_int()
                self.assertEqual(lib.cudaGetDevice(ctypes.byref(current)), 0)
                self.assertEqual(current.value, 1 - ordinal)
            self.assertEqual(len(cache.graphs), 2)
        finally:
            torch.cuda.set_device(previous)
