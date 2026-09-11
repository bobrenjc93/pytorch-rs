"""Ordinary CUDA mul/neg/add graphlets; independent of every scoring corpus."""
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
from tests.test_cuda_add import Comparison, available, runtime, torch, upload


SCALE = 1.375


def scaled(x):
    return x * SCALE


def scalar_helper(x):
    return SCALE * x


def with_helper(x):
    return -scalar_helper(x) + x


def composed(x, y):
    a = (x * -1.375).negative()
    return (0.25 * y).add(a) * 2


def nested(x, y):
    a = x.multiply(0.5)
    shared = [a, x]
    return [shared, (a, shared, y), -(a + y) * 2]


def make_program(source, module=native, **values):
    namespace = {"__name__": __name__, "m": module, "mul": module.mul, **values}
    exec(source, namespace)
    return namespace["program"]


def generated_source(rng, length):
    """Generate ordinary Python; no implementation-selected name or marker."""
    lines = ["def program(x, y):"]
    values = ["x", "y"]
    for i in range(length):
        a, b = (str(rng.choice(values)) for _ in range(2))
        scalar = repr(float(rng.choice([-2.75, -0.125, 0.0, 0.375, 1.1, 3.25])))
        forms = [f"{a} * {scalar}", f"{scalar} * {a}", f"{a}.mul({scalar})",
                 f"{a}.multiply({scalar})", f"-{a}", f"{a}.negative()",
                 f"{a} + {b}", f"{a}.add({b})"]
        lines.append(f"    v{i} = {rng.choice(forms)}")
        values.append(f"v{i}")
    lines.append(f"    return {values[-1]} * -1.375")
    return "\n".join(lines) + "\n"


def call_without_python(compiled, codes, *args):
    def reject(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            raise AssertionError("compiled call executed the original Python body")
    previous = sys.getprofile()
    try:
        sys.setprofile(reject)
        return compiled(*args)
    finally:
        sys.setprofile(previous)


POLICIES = ((True, None), (True, False), (True, True), (False, None))


class CompileScalarMetadataTests(unittest.TestCase):
    def test_recorder_scalar_nodes_reject_cpu_and_unsupported_metadata(self):
        trace = _compile_trace
        for options in ({"device": "cpu"}, {"device": "cuda:0", "requires_grad": True},
                        {"device": "cuda:0", "stride": (1, 3)}):
            recorder = trace.CompileTraceRecorder()
            x = recorder.input(shape=(3, 5), **options)
            with self.assertRaises(NotImplementedError):
                x * 2
            self.assertEqual(recorder._operations, [])
        recorder = trace.CompileTraceRecorder()
        x = recorder.input(shape=(3, 1, 5), stride=(5, 15, 1), device="cuda:0")
        y = 2 * x
        graph = recorder.finish(y)
        self.assertEqual(graph.operations[0].target, "mul_scalar")
        self.assertEqual(graph.operations[0].scalar, 2.)
        self.assertEqual(y.metadata.stride, (5, 15, 1))

    def test_scalar_conversion_and_native_fail_closed_without_gpu(self):
        normalize = _compile_trace._normalize_mul_scalar
        for value in (True, False, 2**64-1, -(2**63), -0., 1.1, 1e40):
            self.assertIs(type(normalize(value)), float)
        for value in (2**64, -(2**63)-1):
            with self.assertRaises(OverflowError):
                normalize(value)
        class Scalar(float):
            def __float__(self):
                raise AssertionError("conversion callback")
        for value in (None, 1j, Scalar(2), np.float32(2)):
            with self.assertRaises(NotImplementedError):
                normalize(value)
        hook = _compile_trace._native._compile_trace_scalar
        with self.assertRaises(TypeError):
            hook(object(), 2, "mul_scalar")
        with self.assertRaises(NotImplementedError):
            hook(native.ones(1), 2, "mul_scalar")
        with self.assertRaises(NotImplementedError):
            hook(native.ones(1), 2, "other")


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CompileCudaMulScalarTests(Comparison, unittest.TestCase):
    def test_generated_graphlets_and_operator_method_top_level_forms(self):
        rng = np.random.default_rng(481903)
        sources = [f"def program(x, y):\n    return {expr}\n" for expr in (
            "x * 1.375", "1.375 * x", "x.mul(-2)", "x.multiply(False)",
            "x.__mul__(True)", "x.__rmul__(0.1)", "m.mul(x, -1.25)",
            "m.multiply(0.375, x)", "mul(x, 2)", "-(x * 0.5) + y.mul(-2)")]
        sources += [generated_source(rng, n) for n in (1, 2, 7, 31)]
        sources += ["def program(x, y):\n    scale = -0.25\n    return x * scale + y\n"]
        shapes = [(), (0,), (2, 0, 3), (0, 2**40), (1,), (255,), (256,), (257,), (65539,)]
        shapes += [tuple(int(n) for n in rng.integers(1, 9, size=3)) for _ in range(3)]
        for source in sources:
            program, ref_program = make_program(source), make_program(source, torch)
            for fullgraph, dynamic in POLICIES:
                for shape in shapes:
                    with self.subTest(source=source, policy=(fullgraph, dynamic), shape=shape):
                        compiled, cache = compile_with_cache(program, fullgraph, dynamic)
                        torch._dynamo.reset()
                        ref = torch.compile(ref_program, backend="eager", fullgraph=fullgraph,
                                            dynamic=dynamic)
                        for _ in range(2):
                            values = [rng.normal(size=int(np.prod(shape))).astype(np.float32) for _ in range(2)]
                            args = [upload(native, v, shape) for v in values]
                            refs = [upload(torch, v, shape) for v in values]
                            expected = ref(*refs)
                            torch.testing.assert_close(expected, ref_program(*refs), rtol=0, atol=0)
                            result = call_without_python(compiled, (program.__code__,), *args)
                            self.compare(result, expected)
                            self.assertIsNot(result, args[0])
                            for x, tx in zip(args, refs):
                                self.compare(x, tx)
                                if x.numel():
                                    self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        self.assertEqual(len(cache.graphs), 1)

    def test_ieee_scalar_conversion_singleton_strides_offsets_and_empty_views(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x80800000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                         0x7fc00000, 0xffc12345], dtype=np.uint32)
        values = np.tile(bits.view(np.float32), 35)
        base, refbase = upload(native, values, (420,)), upload(torch, values, (420,))
        for view in (lambda x: x, lambda x: x[7:414].reshape(11, 37),
                     lambda x: x.select(0, 1), lambda x: x.reshape(1, 7, 60).transpose(0, 1),
                     lambda x: x.reshape(7, 60)[7:7][:, 60:60],
                     lambda x: x[420:].reshape(2, 0, 3)):
            x, tx = view(base), view(refbase)
            for scalar in (True, False, -7, 2**63, 2**64-1, -(2**63), 0., -0.,
                           0.1, 1e-40, 1e40, float('inf'), -float('inf'), float('nan')):
                program = make_program("def program(x):\n    return x * scalar\n", scalar=scalar)
                ref_program = make_program("def program(x):\n    return x * scalar\n", torch, scalar=scalar)
                for fullgraph in (True, False):
                    with self.subTest(shape=tuple(x.shape), stride=x.stride(), scalar=scalar, fullgraph=fullgraph):
                        torch._dynamo.reset()
                        reference = torch.compile(ref_program, backend="eager", fullgraph=fullgraph)
                        expected = reference(tx)
                        self.compare(native.compile(program, backend="eager", fullgraph=fullgraph)(x), expected)
                        self.compare(x, tx)

    def test_guarded_globals_helper_rebinding_and_limit(self):
        x = native.tensor([0., -0., 1., -2.]).to('cuda:0')
        tx = torch.tensor([0., -0., 1., -2.], device='cuda:0')
        for fullgraph, dynamic in POLICIES:
            for program in (scaled, with_helper):
                compiled, cache = compile_with_cache(program, fullgraph, dynamic, limit=16)
                for scalar in (0., -0., True, 1, 1., -2.75, float('nan')):
                    with patch(__name__ + '.SCALE', scalar):
                        result = call_without_python(compiled, (program.__code__, scalar_helper.__code__), x)
                        self.compare(result, program(tx))
                        before = dict(cache.graphs)
                        with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                            self.compare(compiled(x), program(tx))
                        self.assertEqual(cache.graphs, before)
                self.assertEqual(len(cache.graphs), 7)
                class MutableFloat(float):
                    pass
                for scalar in (MutableFloat(2), np.float32(2), object(), 1j, 2**64, -(2**63)-1):
                    before = dict(cache.graphs)
                    with patch(__name__ + '.SCALE', scalar), patch.object(
                        _compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
                        with self.assertRaises((NotImplementedError, OverflowError)):
                            compiled(x)
                    self.assertEqual(cache.graphs, before)
            compiled, cache = compile_with_cache(scaled, fullgraph, dynamic, limit=1)
            compiled(x)
            with patch(__name__ + '.SCALE', -9.), self.assertRaisesRegex(NotImplementedError, 'recompile_limit'):
                compiled(x)
            self.assertEqual(len(cache.graphs), 1)

    def test_dynamic_static_guards_cpu_separation_and_nested_aliases(self):
        for fullgraph, dynamic in POLICIES:
            compiled, cache = compile_with_cache(composed, fullgraph, dynamic)
            for size in (5, 11, 257):
                x = native.ones((size,)).to('cuda:0')
                self.assertEqual(compiled(x, x).cpu().tolist(), [3.25] * size)
            self.assertEqual(len(cache.graphs), 1 if dynamic else 3)
            before = dict(cache.graphs)
            with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
                for args in ((native.ones((5,)), native.ones((5,))),
                             (x, native.ones((257,))), (x, x[:2])):
                    with self.assertRaises(NotImplementedError):
                        compiled(*args)
            self.assertEqual(cache.graphs, before)
            # Same wrapper first warms a supported CPU branch, then CUDA mul.
            source = 'def program(x):\n    if x.requires_grad:\n        return -x\n    return x * 2\n'
            branch, branch_cache = compile_with_cache(make_program(source), fullgraph, dynamic)
            branch(native.ones((5,), requires_grad=True))
            branch(native.ones((5,)).to('cuda:0'))
            self.assertEqual(len(branch_cache.graphs), 2)
            cpu_graph = next(g for g in branch_cache.graphs.values() if g.inputs[0].metadata.device.type == 'cpu')
            with self.assertRaises(ValueError):
                cpu_graph.forward(native.ones((5,)).to('cuda:0'))
            offset, offset_cache = compile_with_cache(scaled, fullgraph, dynamic)
            base = native.ones((30,)).to('cuda:0')
            offset(base[1:12]); offset(base[2:13])
            self.assertEqual(len(offset_cache.graphs), 2)
            with self.assertRaisesRegex(ValueError, 'storage_offset'):
                next(iter(offset_cache.graphs.values())).forward(base[2:13])
            compiled, _ = compile_with_cache(nested, fullgraph, dynamic)
            result = call_without_python(compiled, (nested.__code__,), x, x)
            expected = torch.compile(nested, backend='eager', fullgraph=fullgraph)(torch.ones(257, device='cuda:0'), torch.ones(257, device='cuda:0'))
            self.compare(result[0][0], expected[0][0]); self.compare(result[2], expected[2])
            self.assertIs(expected[0], expected[1][1])
            self.assertIs(expected[0][0], expected[1][0])
            self.assertIs(result[0], result[1][1]); self.assertIs(result[0][0], result[1][0])
            self.assertIs(result[0][1], x); self.assertIs(result[1][2], x)

    def test_tensor_capture_and_stride_changes(self):
        source = 'def program(x):\n    return x + bias.multiply(2)\n'
        x = native.ones((1, 3, 5)).to('cuda:0')
        tx = torch.ones((1, 3, 5), device='cuda:0')
        program = make_program(source, bias=x)
        compiled, cache = compile_with_cache(program, dynamic=True)
        self.compare(compiled(x), tx + tx * 2)
        program.__globals__['bias'] = x * -1.25
        self.compare(compiled(x), tx + tx * -2.5)
        self.assertEqual(len(cache.graphs), 2)
        before = dict(cache.graphs)
        program.__globals__['bias'] = x.cpu()
        with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
            with self.assertRaises(NotImplementedError):
                compiled(x)
        self.assertEqual(cache.graphs, before)
        multiply, variants = compile_with_cache(scaled, dynamic=True)
        # Equal shapes isolate the stride guard: singleton dimensions allow
        # both layouts to be contiguous, but multiplication preserves them.
        views = (x.reshape(3, 1, 5), x.transpose(0, 1))
        references = (tx.reshape(3, 1, 5), tx.transpose(0, 1))
        self.assertEqual(views[0].shape, views[1].shape)
        self.assertNotEqual(views[0].stride(), views[1].stride())
        for view, reference in zip(views, references):
            self.assertTrue(view.is_contiguous())
            self.compare(multiply(view), reference * SCALE)
        self.assertEqual(len(variants.graphs), 2)
        with patch.object(_compile_bytecode, 'lower_compile_graph',
                          side_effect=AssertionError('unexpected stride cache miss')):
            for view, reference in zip(views, references):
                self.compare(multiply(view), reference * SCALE)
        graph = next(iter(variants.graphs.values()))
        with patch.object(_compile_trace, '_execute_operation',
                          side_effect=AssertionError('executed before stride guard')):
            with self.assertRaisesRegex(ValueError, 'stride'):
                graph.forward(views[1])

    def test_reject_before_execution_cache_and_native_hooks(self):
        x = native.ones((3, 5)).to('cuda:0')
        closed = 2.
        def closure(x):
            return x * closed
        programs = [lambda x: (x * 2) * x, lambda x: (x * 2).absolute(),
                    lambda x: (x * 2).detach(), lambda x: (x * 2).float(),
                    lambda x: x.mul(other=2), lambda x: native.mul(x, 2, out=None),
                    lambda x: x * 1j, closure, lambda x, scalar: x * scalar]
        for fullgraph, dynamic in POLICIES:
            for program in programs:
                compiled, cache = compile_with_cache(program, fullgraph, dynamic)
                args = (x, 2.) if program.__code__.co_argcount == 2 else (x,)
                with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
                    with self.assertRaises((NotImplementedError, TypeError)):
                        compiled(*args)
                self.assertEqual(cache.graphs, {})
            for tensor in (x.t(), x[:, 1:4], native.ones((3, 5))):
                compiled, cache = compile_with_cache(scaled, fullgraph, dynamic)
                with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
                    with self.assertRaises(NotImplementedError):
                        compiled(tensor)
                self.assertEqual(cache.graphs, {})
        meta = _compile_trace._metadata_from_native_tensor(x)
        for invalid in (replace(meta, requires_grad=True), replace(meta, stride=(1, 3)),
                        replace(meta, dtype=_compile_trace.CompileTraceDType('torch.float64'))):
            with self.assertRaises(NotImplementedError):
                _compile_bytecode.lower_compile_graph(scaled, (invalid,))
        for scalar in (None, object(), x, 1j, np.float32(2)):
            with self.assertRaises(NotImplementedError):
                _compile_trace._native._compile_trace_scalar(x, scalar, 'mul_scalar')
        with self.assertRaises(NotImplementedError):
            _compile_trace._native._compile_trace_scalar(x.t(), 2, 'mul_scalar')
        with self.assertRaises(NotImplementedError):
            _compile_trace._native._compile_trace_scalar(x.cpu(), 2, 'mul_scalar')

    def test_malformed_late_nodes_and_execution_failure(self):
        x = native.ones((5,)).to('cuda:0')
        program = make_program('def program(x):\n    return (-x) * 2\n')
        graph = _compile_bytecode.lower_compile_graph(program, (_compile_trace._metadata_from_native_tensor(x),))
        node = graph.operations[-1]
        for change in ({'op': 'call_function'}, {'target': 'divide'}, {'inputs': ()},
                       {'inputs': ('missing',)}, {'inputs': ('x', 'x')}, {'scalar': None},
                       {'scalar': x}, {'scalar': 1j}, {'scalar': 2**64},
                       {'metadata': replace(node.metadata, device='cpu')}):
            malformed = replace(graph, operations=(*graph.operations[:-1], replace(node, **change)))
            with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
                with self.assertRaises((NotImplementedError, OverflowError, ValueError)):
                    malformed.forward(x)
        malformed = replace(graph, operations=(replace(graph.operations[0], scalar=2), node))
        with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
            with self.assertRaises(NotImplementedError):
                malformed.forward(x)
        compiled, cache = compile_with_cache(scaled)
        with patch.object(_compile_trace._native, '_compile_trace_scalar', side_effect=RuntimeError('launch failed')):
            with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                compiled(x)
        self.assertEqual(cache.graphs, {})

    def test_binding_guards_and_code_replacement(self):
        x = native.ones((5,)).to('cuda:0')
        for expr in ('m.mul(x, 2)', 'm.multiply(2, x)', 'mul(x, 2)', 'x.mul(2)', 'x * 2'):
            program = make_program(f'def program(x):\n    return {expr}\n')
            compiled, cache = compile_with_cache(program)
            compiled(x)
            before = dict(cache.graphs)
            for owner, name in ((native, 'mul'), (native, 'multiply')) if expr.startswith('m.') else ():
                with patch.object(owner, name, side_effect=AssertionError('called patched binding')):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
            if expr.startswith('mul('):
                program.__globals__['mul'] = lambda *args: (_ for _ in ()).throw(AssertionError('called patched binding'))
                with self.assertRaises(NotImplementedError):
                    compiled(x)
            self.assertEqual(cache.graphs, before)
        for name in ('__mul__', '__rmul__', 'mul', 'multiply'):
            compiled, cache = compile_with_cache(scaled)
            compiled(x)
            missing = object()
            original = native.Tensor.__dict__.get(name, missing)
            setattr(native.Tensor, name, lambda *args: (_ for _ in ()).throw(AssertionError('patched method executed')))
            try:
                with self.assertRaisesRegex(NotImplementedError, 'patched Tensor operation bindings'):
                    compiled(x)
                self.assertEqual(len(cache.graphs), 1)
            finally:
                if original is missing:
                    delattr(native.Tensor, name)
                else:
                    setattr(native.Tensor, name, original)
        program = make_program('def program(x):\n    return x * 0.0\n')
        compiled, cache = compile_with_cache(program)
        compiled(x)
        program.__code__ = make_program('def program(x):\n    return x * -0.0\n').__code__
        self.compare(compiled(x), torch.ones(5, device='cuda:0') * -0.)
        self.assertEqual(len(cache.graphs), 2)

    def test_dynamic_scalar_node_metadata_rejects_before_execution(self):
        x = native.ones((5,)).to('cuda:0')
        compiled, cache = compile_with_cache(composed, dynamic=True)
        compiled(x, x)
        graph = next(iter(cache.graphs.values()))
        node = graph.operations[-1]
        for metadata in (
            replace(node.metadata, device='cpu'),
            replace(node.metadata, device='cuda:1'),
            replace(node.metadata, dtype=_compile_trace.CompileTraceDType('torch.float64')),
            replace(node.metadata, requires_grad=True),
            replace(node.metadata, stride=(2,)),
            replace(node.metadata, storage_offset=1),
            None,
        ):
            with self.subTest(metadata=metadata):
                malformed = replace(graph, operations=(
                    *graph.operations[:-1], replace(node, metadata=metadata)))
                with patch.object(_compile_trace, '_execute_operation',
                                  side_effect=AssertionError('executed malformed graph')):
                    with self.assertRaises((NotImplementedError, ValueError)):
                        malformed.forward(x, x)

    def test_large_complete_data_and_no_pytorch_import(self):
        values = np.random.default_rng(713924).uniform(-3, 4, 17_000_003).astype(np.float32)
        x, tx = upload(native, values, values.shape), upload(torch, values, values.shape)
        program = make_program('def program(x):\n    return -(x * -0.375) + x\n')
        compiled, _ = compile_with_cache(program)
        reference = torch.compile(program, backend='eager', fullgraph=True)
        expected = reference(tx)
        torch.testing.assert_close(expected, program(tx), rtol=0, atol=0)
        self.compare(call_without_python(compiled, (program.__code__,), x), expected)
        source = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('imported PyTorch')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
def program(x):
    return -m.multiply(x, 2) + x
x = m.tensor([1., -3.]).to('cuda:0')
assert m.compile(program, backend='eager', fullgraph=True)(x).cpu().tolist() == [-1., 3.]
'''
        result = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CompileCudaMulScalarDeviceTests(Comparison, unittest.TestCase):
    def test_ownership_restoration_and_cache_separation(self):
        lib, previous = runtime(), torch.cuda.current_device()
        driver = ctypes.CDLL('libcuda.so.1')
        driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
        driver.cuPointerGetAttribute.restype = ctypes.c_int
        try:
            for fullgraph, dynamic in POLICIES:
                compiled, cache = compile_with_cache(composed, fullgraph, dynamic)
                torch._dynamo.reset()
                reference = torch.compile(composed, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                for current, target in ((1, 0), (0, 1), (1, 0)):
                    torch.cuda.set_device(current)
                    for shape in ((), (0,), (3, 5)):
                        x = native.full(shape, -1.75).to(f'cuda:{target}')
                        result = compiled(x, x)
                        ordinal = ctypes.c_int()
                        self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                        self.assertEqual(ordinal.value, current)
                        tx = torch.full(shape, -1.75, device=f'cuda:{target}')
                        self.compare(result, reference(tx, tx))
                        if result.numel():
                            for attribute, expected in ((2, 2), (9, target), (8, 0)):
                                value = ctypes.c_int()
                                self.assertEqual(driver.cuPointerGetAttribute(ctypes.byref(value), attribute, result.data_ptr()), 0)
                                self.assertEqual(value.value, expected)
                        before = dict(cache.graphs)
                        with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed')):
                            with self.assertRaises(NotImplementedError):
                                compiled(x, native.ones(shape).to(f'cuda:{current}'))
                        self.assertEqual(cache.graphs, before)
                        del x, result
                        gc.collect()
                        self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                        self.assertEqual(ordinal.value, current)
                self.assertEqual({str(g.inputs[0].metadata.device) for g in cache.graphs.values()}, {'cuda:0', 'cuda:1'})
        finally:
            torch.cuda.set_device(previous)
