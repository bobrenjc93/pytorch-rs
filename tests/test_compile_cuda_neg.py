"""Generic CUDA neg/add capture differentials; separate from scoring corpora."""
import ctypes
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_bytecode, _compile_trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_cuda_add import Comparison, available, runtime, torch, upload


CAPTURE = None


def negate(x):
    return -x


def compose(x, y):
    a = x.neg()
    return -(a + y.negative())


def helper(x):
    return x.negative()


def captured(x):
    return helper(CAPTURE) + -x


def nested(x, y):
    a = x.neg()
    b = -(a + y)
    return [a, (b, x, a), [y.negative(), b]]


def generated_program(rng, length):
    # Ordinary straight-line functions with independently generated names and
    # operation choices, no workload markers or function-name dispatch.
    name = f"program_{int(rng.integers(1 << 30))}"
    lines = [f"def {name}(x, y):"]
    values = ["x", "y"]
    for index in range(length):
        left = str(rng.choice(values))
        forms = [f"-{left}", f"{left}.neg()", f"{left}.negative()",
                 f"{left} + {rng.choice(values)}", f"{left}.add({rng.choice(values)})"]
        lines.append(f"    v{index} = {rng.choice(forms)}")
        values.append(f"v{index}")
    lines.append(f"    return -{values[-1]}")
    namespace = {"__name__": __name__}
    exec("\n".join(lines), namespace)
    return namespace[name]


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CompileCudaNegTests(Comparison, unittest.TestCase):
    def test_generated_shapes_lengths_and_forms(self):
        rng = np.random.default_rng(903217)
        shapes = [(), (0,), (2, 0, 3), (0, 2**40), (257,), (65539,)]
        shapes += [tuple(int(n) for n in rng.integers(1, 9, size=int(rng.integers(1, 5))))
                   for _ in range(6)]
        programs = [negate, lambda x: x.neg(), lambda x: x.negative(), compose]
        programs += [generated_program(rng, n) for n in (1, 2, 7, 31)]
        for program in programs:
            arity = program.__code__.co_argcount
            for fullgraph in (True, False):
                for shape in shapes:
                    with self.subTest(program=program.__name__, fullgraph=fullgraph, shape=shape):
                        compiled, cache = compile_with_cache(program, fullgraph)
                        torch._dynamo.reset()
                        reference = torch.compile(program, backend="eager", fullgraph=fullgraph)
                        for _ in range(2):
                            values = [rng.normal(size=int(np.prod(shape))).astype(np.float32)
                                      for _ in range(arity)]
                            args = [upload(native, v, shape) for v in values]
                            refs = [upload(torch, v, shape) for v in values]
                            expected = reference(*refs)
                            torch.testing.assert_close(expected, program(*refs), rtol=0, atol=0)
                            def reject(frame, event, arg):
                                if event == "call" and frame.f_code is program.__code__:
                                    raise AssertionError("original Python function executed")
                            previous = sys.getprofile()
                            try:
                                sys.setprofile(reject)
                                result = compiled(*args)
                            finally:
                                sys.setprofile(previous)
                            self.compare(result, expected)
                            self.assertEqual(result.storage_offset(), 0)
                            for x, tx in zip(args, refs):
                                self.compare(x, tx)
                                self.assertIsNot(result, x)
                                if values[0].size:
                                    self.assertNotEqual(result.data_ptr(), x.data_ptr())
                        self.assertEqual(len(cache.graphs), 1)

    def test_cold_cpu_warmed_and_device_guards(self):
        cpu = native.tensor([1., -2., 3.])
        cuda = cpu.to("cuda:0")
        for program in (negate, compose):
            arity = program.__code__.co_argcount
            for fullgraph, dynamic in ((True, None), (True, False), (True, True), (False, None)):
                for order in ((cpu, cuda), (cuda, cpu)):
                    compiled, cache = compile_with_cache(program, fullgraph, dynamic)
                    for x in order:
                        self.assertEqual(compiled(*([x] * arity)).cpu().tolist(),
                                         [-1., 2., -3.] if arity == 1 else [2., -4., 6.])
                    with patch.object(_compile_bytecode, "lower_compile_graph",
                                      side_effect=AssertionError("unexpected cache miss")):
                        compiled(*([cuda] * arity))
                    self.assertEqual(len(cache.graphs), 2)
                    for graph in cache.graphs.values():
                        wrong = cuda if graph.inputs[0].metadata.device.type == "cpu" else cpu
                        with self.assertRaisesRegex(ValueError, "device"):
                            graph.forward(*([wrong] * arity))

    def test_offsets_ieee_metadata_dynamic_and_stride_guards(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                         0x7fc00000, 0xffc12345], dtype=np.uint32)
        values = np.tile(bits.view(np.float32), 4)
        base, refbase = upload(native, values, values.shape), upload(torch, values, values.shape)
        views = (lambda x: x[3:15].reshape(3, 4), lambda x: x[4:16].reshape(3, 4),
                 lambda x: x.select(0, 1), lambda x: x[48:].reshape(2, 0, 3),
                 lambda x: x.reshape(1, 6, 8).transpose(0, 1))
        compiled, cache = compile_with_cache(negate)
        reference = torch.compile(negate, backend="eager", fullgraph=True)
        for view in views:
            x, tx = view(base), view(refbase)
            self.compare(compiled(x), reference(tx))
            self.compare(x, tx)
        self.assertEqual(len(cache.graphs), len(views))
        graph = next(iter(cache.graphs.values()))
        for x, field in ((views[1](base), "storage_offset"), (base, "shape"),
                         (views[0](base).t(), "contiguous")):
            with self.assertRaisesRegex((ValueError, NotImplementedError), field):
                graph.forward(x)
        for graph in cache.graphs.values():
            self.assertEqual(graph.operations[0].metadata.storage_offset, 0)
        # Different contiguous singleton strides must not share static guards.
        a = base.reshape(1, 6, 8).transpose(0, 1)
        b = base.reshape(6, 1, 8)
        self.assertNotEqual(a.stride(), b.stride())
        with self.assertRaisesRegex(ValueError, "stride"):
            list(cache.graphs.values())[-1].forward(b)
        compiled(b)
        self.assertEqual(len(cache.graphs), len(views) + 1)
        dynamic, cache = compile_with_cache(negate, dynamic=True)
        dynamic(base[1:])
        before = dict(cache.graphs)
        self.compare(dynamic(base[1:4]), -refbase[1:4])
        self.assertEqual(cache.graphs, before)
        dynamic(base[2:])
        self.assertEqual(len(cache.graphs), 2)

    def test_captures_nested_outputs_and_repeated_inputs(self):
        cpu = native.tensor([1., -3.])
        x, tx = cpu.to("cuda:0"), torch.tensor([1., -3.], device="cuda:0")
        compiled, cache = compile_with_cache(captured)
        with patch(__name__ + ".CAPTURE", cpu):
            compiled(cpu)
        for value in (x, native.tensor([4., -2.]).to("cuda:0")):
            with patch(__name__ + ".CAPTURE", value):
                actual = compiled(x)
                before = dict(cache.graphs)
                with patch.object(_compile_trace, "_execute_operation", side_effect=AssertionError("executed")):
                    with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                        compiled(cpu)
                self.assertEqual(cache.graphs, before)
            with patch(__name__ + ".CAPTURE", torch.tensor(value.cpu().tolist(), device="cuda:0")):
                reference = torch.compile(captured, backend="eager", fullgraph=True)
                self.compare(actual, reference(tx))
        self.assertEqual(len(cache.graphs), 3)
        compiled, cache = compile_with_cache(nested)
        actual = compiled(x, x)
        reference = torch.compile(nested, backend="eager", fullgraph=True)(tx, tx)
        self.compare(actual[0], reference[0])
        self.compare(actual[1][0], reference[1][0])
        self.compare(actual[2][0], reference[2][0])
        self.assertIs(actual[1][1], x)
        self.assertIs(actual[0], actual[1][2])
        self.assertIs(actual[1][0], actual[2][1])
        self.compare(x, tx)

    def test_global_offset_shape_and_layout_guards(self):
        base = native.tensor(list(range(20))).to("cuda:0")
        compiled, cache = compile_with_cache(captured)
        x = native.ones((4,)).to("cuda:0")
        for offset in (1, 2, 1):
            value = base[offset:offset + 4]
            before_values = value.cpu().tolist()
            with patch(__name__ + ".CAPTURE", value):
                self.assertEqual(compiled(x).cpu().tolist(), [-v - 1 for v in before_values])
            self.assertEqual(value.cpu().tolist(), before_values)
        # Rebinding identity specializes even when metadata is unchanged.
        self.assertEqual(len(cache.graphs), 3)
        before = dict(cache.graphs)
        for value in (base[:3], base.reshape(4, 5).t(), native.ones((4,)),
                      native.ones((4,), requires_grad=True)):
            with patch(__name__ + ".CAPTURE", value), patch.object(
                    _compile_trace, "_execute_operation", side_effect=AssertionError("executed")):
                with self.assertRaises(NotImplementedError):
                    compiled(x)
            self.assertEqual(cache.graphs, before)

    def test_reject_before_execution_and_cache_insertion(self):
        cpu = native.ones((3, 4))
        cuda = cpu.to("cuda:0")
        for program in (lambda x: (-x).abs(), lambda x: (-x).relu(), lambda x: (-x).square(),
                        lambda x: (-x).detach(), lambda x: (-x).float()):
            for warmed in (False, True):
                compiled, cache = compile_with_cache(program)
                if warmed:
                    compiled(cpu)
                before = dict(cache.graphs)
                with patch.object(_compile_trace, "_execute_operation", side_effect=AssertionError("executed")):
                    with self.assertRaisesRegex(NotImplementedError, "CUDA unary"):
                        compiled(cuda)
                self.assertEqual(cache.graphs, before)
        closed = cuda
        def closure(x):
            return -closed + x
        for program, args in ((closure, (cuda,)), (negate, (cuda.t(),)),
                              (negate, (cuda[:, 1:],)), (compose, (cuda, cpu))):
            compiled, cache = compile_with_cache(program)
            with patch.object(_compile_trace, "_execute_operation", side_effect=AssertionError("executed")):
                with self.assertRaises(NotImplementedError):
                    compiled(*args)
            self.assertEqual(cache.graphs, {})
        meta = _compile_trace._metadata_from_native_tensor(cuda)
        for invalid in (replace(meta, requires_grad=True),
                        replace(meta, dtype=_compile_trace.CompileTraceDType("torch.float64")),
                        replace(meta, stride=(1, 3))):
            with self.assertRaises(NotImplementedError):
                _compile_trace._unary_output_metadata(invalid, "neg")
        dynamic, cache = compile_with_cache(compose, dynamic=True)
        x = native.ones((4,)).to("cuda:0")
        dynamic(x, x)
        before = dict(cache.graphs)
        with patch.object(_compile_trace, "_execute_operation", side_effect=AssertionError("executed")):
            with self.assertRaisesRegex(NotImplementedError, "same shape"):
                dynamic(x, native.ones((7,)).to("cuda:0"))
        self.assertEqual(cache.graphs, before)
        y = native.full((7,), 1.25).to("cuda:0")
        self.compare(dynamic(y, y), torch.full((7,), 2.5, device="cuda:0"))
        self.assertEqual(cache.graphs, before)
        compiled, cache = compile_with_cache(negate)
        with patch.object(_compile_trace._native, "_compile_trace_unary", side_effect=RuntimeError("launch failed")):
            with self.assertRaisesRegex(RuntimeError, "launch failed"):
                compiled(x)
        self.assertEqual(cache.graphs, {})
        self.assertEqual(compiled(x).cpu().tolist(), [-1.] * 4)

    def test_stream_completion_and_independent_storage(self):
        values = np.arange(65539, dtype=np.float32) / 16 - 7
        x, tx = upload(native, values, values.shape), upload(torch, values, values.shape)
        compiled, _ = compile_with_cache(compose)
        expected = torch.compile(compose, backend="eager", fullgraph=True)(tx, tx)
        stream, lib = torch.cuda.Stream(), runtime()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            result = compiled(x, x)
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(), values.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        self.compare(x, tx)
        lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.cudaMemset.restype = ctypes.c_int
        self.assertEqual(lib.cudaMemset(x.data_ptr(), 0, values.nbytes), 0)
        self.compare(result, expected)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CompileCudaNegDeviceTests(Comparison, unittest.TestCase):
    def test_ownership_captures_and_device_restoration(self):
        compiled, cache = compile_with_cache(compose)
        unary, unary_cache = compile_with_cache(negate)
        capture, capture_cache = compile_with_cache(captured)
        reference = torch.compile(compose, backend="eager", fullgraph=True)
        reference_unary = torch.compile(negate, backend="eager", fullgraph=True)
        previous, lib = torch.cuda.current_device(), runtime()
        try:
            for shape in ((), (0,), (257,)):
                x = [native.full(shape, 1.25).to(f"cuda:{i}") for i in (0, 1)]
                tx = [torch.full(shape, 1.25, device=f"cuda:{i}") for i in (0, 1)]
                for target in (0, 1, 0):
                    current = 1 - target
                    torch.cuda.set_device(current)
                    self.compare(unary(x[target]), reference_unary(tx[target]))
                    self.compare(compiled(x[target], x[target]), reference(tx[target], tx[target]))
                    with patch(__name__ + ".CAPTURE", x[target]):
                        self.compare(capture(x[target]), -tx[target] + -tx[target])
                        before = dict(capture_cache.graphs)
                        with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                            capture(x[current])
                        self.assertEqual(capture_cache.graphs, before)
                    before = dict(cache.graphs)
                    with patch.object(_compile_trace, "_execute_operation", side_effect=AssertionError("executed")):
                        with self.assertRaisesRegex(NotImplementedError, "matching devices"):
                            compiled(x[0], x[1])
                    self.assertEqual(cache.graphs, before)
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.assertEqual(torch.cuda.current_device(), current)
                    output = unary(x[target])
                    del output
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
            self.assertEqual(len(cache.graphs), 6)
            self.assertEqual(len(unary_cache.graphs), 6)
            self.assertEqual(len(capture_cache.graphs), 6)
        finally:
            torch.cuda.set_device(previous)
