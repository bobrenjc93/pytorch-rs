"""Held-out H100 reduction graphs; independent of fixed evaluator inputs."""
import ctypes
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_bytecode as bytecode, _compile_trace as trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_sum_lowering import program
from tests.test_cuda_add import available, runtime, torch, upload
from tests import test_cuda_sum_rows as eager_tests


def composed(x, y):
    rows = (x * 0.5 + y).sum(dim=-1, keepdim=True)
    return [rows, -rows + rows, x.sum(1)]


def late_unsupported(x):
    rows = x.sum(1)
    return rows.sum(1)


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaSumTests(unittest.TestCase):
    compare_sum = eager_tests.CudaSumRowsTests.compare_sum

    def test_held_out_forms_offsets_empty_and_changed_values(self):
        rng = np.random.default_rng(4852031)
        shapes = [(0, 0), (0, 97), (13, 0), (1, 1), (3, 31), (11, 33),
                  (6, 257), (2, 4099), (21, 1)]
        shapes += [tuple(map(int, rng.integers(2, 120, 2))) for _ in range(4)]
        for expression in ('x.sum(1)', 'x.sum(dim=-1, keepdim=False)',
                           'x.sum(1, True)', 'x.sum(-1, keepdim=True)',
                           'x.sum(keepdim=True, dim=1)'):
            fn = program(expression)
            for shape in shapes:
                for offset in (0, 9):
                    with self.subTest(expression=expression, shape=shape, offset=offset):
                        compiled, cache = compile_with_cache(fn)
                        torch._dynamo.reset()
                        reference = torch.compile(fn, backend='eager', fullgraph=True, dynamic=False)
                        for turn in range(2):
                            n = int(np.prod(shape))
                            values = rng.normal(size=n + offset).astype(np.float32)
                            x, tx = [upload(m, values, values.shape)[offset:].reshape(shape)
                                     for m in (native, torch)]
                            self.assertEqual(x.storage_offset(), offset)
                            def guard(frame, event, arg):
                                if event == 'call' and frame.f_code is fn.__code__:
                                    raise AssertionError('original Python body executed')
                            previous = sys.getprofile()
                            try:
                                sys.setprofile(guard)
                                if turn:
                                    with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('re-lowering')):
                                        result = compiled(x)
                                else:
                                    result = compiled(x)
                            finally:
                                sys.setprofile(previous)
                            self.compare_sum(result, reference(tx))
                            np.testing.assert_array_equal(x.cpu().tolist(), tx.cpu().tolist())
                            self.assertEqual(len(cache.graphs), 1)

    def test_composition_dynamic_shapes_and_native_whole_graph_validation(self):
        rng = np.random.default_rng(970541)
        compiled, cache = compile_with_cache(composed, dynamic=True)
        for shape in ((4, 23), (9, 65), (8, 0), (0, 9)):
            values = [rng.normal(size=np.prod(shape)).astype(np.float32) for _ in range(2)]
            args = [upload(native, v, shape) for v in values]
            refs = [upload(torch, v, shape) for v in values]
            actual = compiled(*args)
            for a, b in zip(actual, composed(*refs)):
                self.compare_sum(a, b)
        # Existing cache keys include concrete strides; distinct row widths
        # can specialize even with dynamic=True.
        self.assertEqual(len(cache.graphs), 4)
        x = native.ones((4, 23)).to('cuda:0')
        # Late invalid reduction rank must be planned before the first sum.
        with patch.object(trace._native, '_compile_trace_reduction', side_effect=AssertionError('launch')), \
                patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launch')):
            with self.assertRaisesRegex(NotImplementedError, 'rank-2'):
                native.compile(late_unsupported, backend='eager', fullgraph=True)(x)
        graph = next(iter(cache.graphs.values()))
        for bad in (replace(graph.operations[-1], reduction=(0, False)),
                    replace(graph.operations[-1], reduction=(1, 1)),
                    replace(graph.operations[-1], metadata=replace(graph.operations[-1].metadata, storage_offset=3))):
            broken = replace(graph, operations=(*graph.operations[:-1], bad))
            with patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launch')):
                with self.assertRaises(NotImplementedError):
                    broken.forward(x, x)
        # Independently exercise the native planner with malformed later nodes.
        for tail in (('sum', (1,), (1, False), (4,), (1,)),
                     ('sum', (0,), (0, False), (4,), (1,)),
                     ('sum', (0,), (1, 1), (4,), (1,)),
                     ('unknown', (0,), None, (4, 23), (23, 1))):
            with self.assertRaises((ValueError, NotImplementedError, RuntimeError)):
                trace._native._compile_trace_cuda_graph((x,), [
                    ('sum', (0,), (1, False), (4,), (1,)), tail])

    def test_dynamic_cache_hit_rejects_later_incompatible_add_before_launch(self):
        def reduce_add(x, y):
            return x.sum(1) + y
        compiled, cache = compile_with_cache(reduce_add, dynamic=True)
        x = native.ones((4, 23)).to('cuda:0')
        y = native.ones((4,)).to('cuda:0')
        self.assertEqual(compiled(x, y).cpu().tolist(), [24.] * 4)
        changed = native.ones((5, 23)).to('cuda:0')
        with patch.object(bytecode, 'lower_compile_graph', side_effect=AssertionError('re-lowering')), \
                patch.object(trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('launch')), \
                patch.object(trace._native, '_compile_trace_reduction', side_effect=AssertionError('launch')):
            with self.assertRaises(NotImplementedError):
                compiled(changed, y)
        self.assertEqual(len(cache.graphs), 1)
        self.assertEqual(compiled(x, y).cpu().tolist(), [24.] * 4)

    def test_rejections_and_live_method_mutation(self):
        fn = program('x.sum(1)')
        compiled, cache = compile_with_cache(fn)
        x = native.ones((4, 17)).to('cuda:0')
        compiled(x)
        for bad in (x.reshape(68), x.reshape(1, 4, 17), x.transpose(0, 1),
                    x[:, :8], native.ones((4, 17), requires_grad=True), native.ones((4, 17))):
            with self.subTest(shape=bad.shape), self.assertRaises((NotImplementedError, TypeError)):
                compiled(bad)
        for cold in (False, True):
            wrapper = native.compile(fn, backend='eager', fullgraph=True) if cold else compiled
            for replacement in (lambda *args, **kwargs: self.fail('patched sum executed'), property(lambda s: self.fail('descriptor executed'))):
                with patch.object(native.Tensor, 'sum', replacement):
                    with self.assertRaisesRegex(NotImplementedError, 'patched.*sum'):
                        wrapper(x)
        self.assertEqual(len(cache.graphs), 1)
        self.compare_sum(compiled(x), torch.ones((4, 17), device='cuda:0').sum(1))
        for dim, keep in ((0, False), (True, False), (1, 1), (1.0, False)):
            with self.assertRaises(NotImplementedError):
                trace._native._compile_trace_reduction(x, 'sum', dim, keep)

    def test_noncanonical_contiguous_singleton_strides(self):
        for shape in ((1, 17), (0, 17)):
            x = native.full(shape, -0.25).to('cuda:0').transpose(0, 1)
            tx = torch.full(shape, -0.25, device='cuda:0').transpose(0, 1)
            self.assertTrue(x.is_contiguous())
            for expression in ('x.sum(1)', 'x.sum(1, True)'):
                compiled = native.compile(program(expression), backend='eager', fullgraph=True)
                self.compare_sum(compiled(x), program(expression)(tx))

    def test_stream_completion_and_independent_output(self):
        x = native.full((5, 521), 0.25).to('cuda:0')
        compiled = native.compile(program('x.sum(1)'), backend='eager', fullgraph=True)
        lib, stream = runtime(), torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = compiled(x)
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty(5, device='cuda:0')
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(), 20,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, torch.full((5,), 130.25, device='cuda:0'))
        self.assertNotEqual(result.data_ptr(), x.data_ptr())


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CompileCudaSumDeviceTests(unittest.TestCase):
    compare_sum = eager_tests.CudaSumRowsTests.compare_sum

    def test_current_device_restored_for_single_and_composed_reductions(self):
        previous, lib = torch.cuda.current_device(), runtime()
        try:
            for expression in ('x.sum(1)', '-x.sum(1, True)'):
                compiled = native.compile(program(expression), backend='eager', fullgraph=True)
                for target in (0, 1, 0):
                    x = native.full((6, 37), 0.5).to(f'cuda:{target}')
                    tx = torch.full((6, 37), 0.5, device=f'cuda:{target}')
                    torch.cuda.set_device(1 - target)
                    self.compare_sum(compiled(x), program(expression)(tx))
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, 1 - target)
                    self.assertEqual(torch.cuda.current_device(), 1 - target)
                    del x
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, 1 - target)
        finally:
            torch.cuda.set_device(previous)
