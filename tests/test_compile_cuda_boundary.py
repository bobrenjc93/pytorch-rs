"""Device-aware native eager graph capture, independent of the scoring corpus."""
import ctypes
from dataclasses import replace
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_bytecode, _compile_trace, _compiler_state
from tests.test_cuda_add import Comparison, available, runtime, torch, upload


CAPTURE = None


def add_inputs(x, y):
    return x + y


def add_capture(x):
    return x + CAPTURE


def doubled(value):
    return value.add(value)


def held_out_meridian(first, second):
    temporary = second + first
    return temporary.add(first) + second


def held_out_canopy(seed):
    branch = doubled(seed)
    return [branch + seed, (branch, seed)]


def late_add(x, y):
    a = x + x
    return a + y


def compile_with_cache(model, fullgraph=True, dynamic=None, limit=16):
    cache = _compiler_state.new_native_eager_compile_cache()
    with patch.object(_compiler_state, "new_native_eager_compile_cache", return_value=cache):
        compiled = native.compile(model, backend="eager", fullgraph=fullgraph,
                                  dynamic=dynamic, recompile_limit=limit)
    return compiled, cache


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CompileCudaBoundaryTests(Comparison, unittest.TestCase):
    def test_cpu_cuda_cache_separation_and_original_bypass(self):
        cpu = native.tensor([1., 2., 3.])
        cuda = cpu.to("cuda:0")
        for fullgraph in (True, False):
            for dynamic in ((None, False, True) if fullgraph else (None,)):
                for first, second in ((cpu, cuda), (cuda, cpu)):
                    with self.subTest(fullgraph=fullgraph, dynamic=dynamic, first=str(first.device)):
                        compiled, cache = compile_with_cache(add_inputs, fullgraph, dynamic)
                        def reject_original(frame, event, arg):
                            if event == "call" and frame.f_code is add_inputs.__code__:
                                raise AssertionError("original function executed")
                        old_profile = sys.getprofile()
                        try:
                            sys.setprofile(reject_original)
                            for value in (first, second, first):
                                self.assertEqual(compiled(value, value).cpu().tolist(), [2., 4., 6.])
                            with patch.object(_compile_bytecode, "lower_compile_graph",
                                              side_effect=AssertionError("cache miss")):
                                changed = native.tensor([9., -2., 4.]).to("cuda:0")
                                self.assertEqual(compiled(changed, changed).cpu().tolist(), [18., -4., 8.])
                        finally:
                            sys.setprofile(old_profile)
                        self.assertEqual(len(cache.graphs), 2)
                        self.assertEqual({str(g.inputs[0].metadata.device) for g in cache.graphs.values()},
                                         {"cpu", "cuda:0"})
                        before = dict(cache.graphs)
                        for args in ((cuda, cpu), (cpu, cuda)):
                            with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                                compiled(*args)
                        self.assertEqual(cache.graphs, before)
                        # A recorded CPU graph cannot execute CUDA (or conversely).
                        for graph in cache.graphs.values():
                            wrong = cuda if graph.inputs[0].metadata.device.type == "cpu" else cpu
                            with self.assertRaisesRegex(ValueError, "device"):
                                graph.forward(wrong, wrong)

    def test_seeded_shapes_structures_and_changed_data_against_reference(self):
        rng = np.random.default_rng(2675909)
        shapes = [(), (0,), (3, 0, 2), (0, 2**40), (257,), (65539,)]
        shapes += [tuple(int(n) for n in rng.integers(1, 12, size=int(rng.integers(1, 5))))
                   for _ in range(8)]
        for program, arity in ((add_inputs, 2), (doubled, 1), (held_out_meridian, 2)):
            # Each wrapper sees fresh data on its existing shape specialization.
            for shape in shapes:
                compiled, cache = compile_with_cache(program)
                torch._dynamo.reset()
                reference = torch.compile(program, backend="eager", fullgraph=True)
                for _ in range(2):
                    values = [rng.normal(size=int(np.prod(shape))).astype(np.float32) for _ in range(arity)]
                    args = [upload(native, v, shape) for v in values]
                    refs = [upload(torch, v, shape) for v in values]
                    with self.subTest(program=program.__name__, shape=shape):
                        self.compare(compiled(*args), reference(*refs))
                self.assertEqual(len(cache.graphs), 1)
        compiled, cache = compile_with_cache(held_out_canopy)
        ref = torch.compile(held_out_canopy, backend="eager", fullgraph=True)
        x = native.tensor([1., -3.]).to("cuda:0")
        actual, expected = compiled(x), ref(torch.tensor([1., -3.], device="cuda:0"))
        self.compare(actual[0], expected[0])
        self.compare(actual[1][0], expected[1][0])
        self.assertIs(actual[1][1], x)

    def test_offset_layout_metadata_and_cache_guards(self):
        base = native.tensor(list(range(30))).to("cuda:0")
        refbase = torch.arange(30, dtype=torch.float32, device="cuda:0")
        views = (lambda x: x[3:15].reshape(3, 4), lambda x: x[4:16].reshape(3, 4),
                 lambda x: x.select(0, 7), lambda x: x[30:30].reshape(2, 0, 3),
                 lambda x: x.reshape(1, 5, 6).transpose(0, 1))
        compiled, cache = compile_with_cache(add_inputs)
        reference = torch.compile(add_inputs, backend="eager", fullgraph=True)
        for view in views:
            x, tx = view(base), view(refbase)
            meta = _compile_trace._metadata_from_native_tensor(x)
            self.assertEqual((meta.shape, meta.stride, str(meta.dtype), str(meta.device), meta.storage_offset),
                             (tuple(tx.shape), tx.stride(), str(tx.dtype), str(tx.device), tx.storage_offset()))
            self.compare(compiled(x, x), reference(tx, tx))
        self.assertEqual(len(cache.graphs), len(views))
        graph = next(iter(cache.graphs.values()))
        with self.assertRaisesRegex(ValueError, "storage_offset"):
            graph.forward(views[1](base), views[1](base))

    def test_ieee_edges_and_dynamic_offset_guards(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000,
                         0x3f800000, 0xbf800000], dtype=np.uint32)
        a = np.repeat(bits.view(np.float32), len(bits))
        b = np.tile(bits.view(np.float32), len(bits))
        compiled, cache = compile_with_cache(add_inputs, dynamic=True)
        reference = torch.compile(add_inputs, backend="eager", fullgraph=True, dynamic=True)
        self.compare(compiled(upload(native, a, a.shape), upload(native, b, b.shape)),
                     reference(upload(torch, a, a.shape), upload(torch, b, b.shape)))
        x = native.tensor([0., 1., 2., 3.]).to("cuda:0")
        self.assertEqual(compiled(x[1:], x[1:]).cpu().tolist(), [2., 4., 6.])
        self.assertEqual(len(cache.graphs), 2)  # Same rank/stride; different offset.
        before = dict(cache.graphs)
        self.assertEqual(compiled(x[1:3], x[1:3]).cpu().tolist(), [2., 4.])
        self.assertEqual(cache.graphs, before)

    def test_global_capture_transitions_and_recompile_limit(self):
        cpu = native.tensor([1., 2., 3.])
        cuda = cpu.to("cuda:0")
        for fullgraph in (True, False):
            for warmed in (False, True):
                compiled, cache = compile_with_cache(add_capture, fullgraph)
                with patch(__name__ + ".CAPTURE", cpu):
                    if warmed:
                        compiled(cpu)
                    before = dict(cache.graphs)
                    with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                        compiled(cuda)
                    self.assertEqual(cache.graphs, before)
                with patch(__name__ + ".CAPTURE", cuda):
                    self.assertEqual(compiled(cuda).cpu().tolist(), [2., 4., 6.])
                    before = dict(cache.graphs)
                    with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                        compiled(cpu)
                    self.assertEqual(cache.graphs, before)
                replacement = native.tensor([8., -1., 0.]).to("cuda:0")
                with patch(__name__ + ".CAPTURE", replacement):
                    self.assertEqual(compiled(cuda).cpu().tolist(), [9., 1., 3.])
                self.assertEqual(len(cache.graphs), 2 + int(warmed))
        compiled, cache = compile_with_cache(add_inputs, limit=1)
        compiled(cpu, cpu)
        before = dict(cache.graphs)
        with self.assertRaisesRegex(NotImplementedError, "recompile_limit=1"):
            compiled(cuda, cuda)
        self.assertEqual(cache.graphs, before)
        self.assertEqual(compiled(cpu, cpu).tolist(), [2., 4., 6.])

    def test_unsupported_fresh_and_cpu_warmed_leave_cache_unchanged(self):
        cpu = native.ones((3, 4))
        cuda = cpu.to("cuda:0")
        unary = (lambda x: x.abs(), lambda x: x.relu(),
                 lambda x: x.square(), lambda x: x.detach(), lambda x: x.float())
        for program in unary:
            for warmed in (False, True):
                compiled, cache = compile_with_cache(program)
                if warmed:
                    compiled(cpu)
                before = dict(cache.graphs)
                with self.assertRaisesRegex(NotImplementedError, "CUDA unary"):
                    compiled(cuda)
                self.assertEqual(cache.graphs, before)
        for program in (lambda x: x * x, lambda x: x - x, lambda x: x.sum(),
                        lambda x: x + 1, lambda x: x.add(x, alpha=2),
                        lambda x: (x + x).relu()):
            compiled, cache = compile_with_cache(program)
            with patch.object(_compile_trace._native, "_compile_trace_binary",
                              side_effect=AssertionError("executed rejected graph")):
                with self.assertRaises(NotImplementedError):
                    compiled(cuda)
            self.assertEqual(cache.graphs, {})
        compiled, cache = compile_with_cache(add_inputs)
        compiled(cpu, cpu)
        before = dict(cache.graphs)
        for left, right in ((cuda, cuda[:1]), (cuda.t(), cuda.t()),
                            (cuda[:, 1:], cuda[:, 1:]), (cpu, cuda),
                            (native.ones((3, 4), requires_grad=True), cuda)):
            with self.assertRaises(NotImplementedError):
                compiled(left, right)
            self.assertEqual(cache.graphs, before)
        with self.assertRaises(NotImplementedError):
            native.ones((3, 4), requires_grad=True).to("cuda:0")
        with self.assertRaises((NotImplementedError, TypeError)):
            cuda.to(dtype=torch.float64)
        # The substrate cannot construct these tensors; metadata validation must
        # still reject them if future storage support exposes them to tracing.
        metadata = _compile_trace._metadata_from_native_tensor(cuda)
        for invalid in (replace(metadata, requires_grad=True),
                        replace(metadata, dtype=_compile_trace.CompileTraceDType("torch.float64"))):
            with self.assertRaises(NotImplementedError):
                _compile_trace._binary_output_metadata(invalid, invalid)

    def test_dynamic_cache_hit_preflights_late_shape_mismatch(self):
        compiled, cache = compile_with_cache(late_add, dynamic=True)
        x = native.ones((4,)).to("cuda:0")
        compiled(x, x)
        before = dict(cache.graphs)
        with patch.object(_compile_trace._native, "_compile_trace_binary",
                          side_effect=AssertionError("execution before validation")):
            with self.assertRaisesRegex(NotImplementedError, "same shape"):
                compiled(x, native.ones((7,)).to("cuda:0"))
        self.assertEqual(cache.graphs, before)
        y = native.full((7,), 3.).to("cuda:0")
        self.assertEqual(compiled(y, y).cpu().tolist(), [9.] * 7)
        self.assertEqual(cache.graphs, before)

    def test_native_failure_does_not_publish_cache(self):
        compiled, cache = compile_with_cache(add_inputs)
        x = native.ones((4,)).to("cuda:0")
        with patch.object(_compile_trace._native, "_compile_trace_binary", side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                compiled(x, x)
        self.assertEqual(cache.graphs, {})
        self.assertEqual(compiled(x, x).cpu().tolist(), [2.] * 4)
        self.assertEqual(len(cache.graphs), 1)

    def test_private_trace_entrypoints_keep_unsupported_guards(self):
        cpu = native.tensor([1., 2., 3.])
        cuda = cpu.to("cuda:0")
        backend = _compile_trace._native
        self.assertEqual(backend._compile_trace_tensor_metadata(cuda),
                         ((3,), (1,), False, "torch.float32", "cuda:0", 0))
        self.assertEqual(backend._compile_trace_binary(cuda, cuda, "add").cpu().tolist(), [2., 4., 6.])
        self.assertEqual(backend._compile_trace_unary(cuda, "neg").cpu().tolist(), [-1., -2., -3.])
        with self.assertRaisesRegex(NotImplementedError, "contiguous"):
            backend._compile_trace_unary(native.ones((2, 3)).to("cuda:0").t(), "neg")
        for target in ("float", "detach", "abs", "square", "relu"):
            with self.assertRaisesRegex(NotImplementedError, "CPU.*CUDA"):
                backend._compile_trace_unary(cuda, target)
        for left, right in ((cpu, cuda), (cuda, cpu), (cuda, cuda[:1])):
            with self.assertRaises(NotImplementedError):
                backend._compile_trace_binary(left, right, "add")
        with self.assertRaises(NotImplementedError):
            backend._compile_trace_binary(cuda, cuda, "mul")

    def test_no_pytorch_import_or_original_execution(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("production imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
def unrelated_name(a, b):
    return (b + a).add(a)
def reject(frame, event, arg):
    if event == "call" and frame.f_code is unrelated_name.__code__:
        raise AssertionError("original executed")
f = m.compile(unrelated_name, backend="eager", fullgraph=True)
sys.setprofile(reject)
for i in range(3):
    a = m.tensor([float(i), 2.]).to("cuda:0")
    b = m.tensor([1., -3.]).to("cuda:0")
    assert f(a, b).cpu().tolist() == [float(2*i+1), 1.]
def another_name(a, b):
    return -(a.neg() + b.negative())
g = m.compile(another_name, backend="eager", fullgraph=True)
def reject_both(frame, event, arg):
    if event == "call" and frame.f_code in (unrelated_name.__code__, another_name.__code__):
        raise AssertionError("original executed")
sys.setprofile(reject_both)
assert g(a, b).cpu().tolist() == [3., -1.]
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CompileCudaDeviceTests(Comparison, unittest.TestCase):
    def test_ordinals_captures_and_current_device_restoration(self):
        compiled, cache = compile_with_cache(add_inputs)
        captured, capture_cache = compile_with_cache(add_capture)
        reference = torch.compile(add_inputs, backend="eager", fullgraph=True)
        x = [native.tensor([2., -3.]).to(f"cuda:{i}") for i in (0, 1)]
        tx = [torch.tensor([2., -3.], device=f"cuda:{i}") for i in (0, 1)]
        lib = runtime()
        previous = torch.cuda.current_device()
        try:
            for target in (0, 1, 0, 1):
                current = 1 - target
                torch.cuda.set_device(current)
                self.compare(compiled(x[target], x[target]), reference(tx[target], tx[target]))
                before = dict(cache.graphs)
                with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                    compiled(x[0], x[1])
                self.assertEqual(cache.graphs, before)
                with patch(__name__ + ".CAPTURE", x[target]):
                    self.compare(captured(x[target]), reference(tx[target], tx[target]))
                    before = dict(capture_cache.graphs)
                    with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                        captured(x[current])
                    self.assertEqual(capture_cache.graphs, before)
                ordinal = ctypes.c_int()
                self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                self.assertEqual(ordinal.value, current)
                self.assertEqual(torch.cuda.current_device(), current)
            self.assertEqual(len(cache.graphs), 2)
            self.assertEqual(len(capture_cache.graphs), 2)
            graph = next(iter(cache.graphs.values()))
            with self.assertRaisesRegex(ValueError, "device"):
                graph.forward(x[1], x[1])
        finally:
            torch.cuda.set_device(previous)
