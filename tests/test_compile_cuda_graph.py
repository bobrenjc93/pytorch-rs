"""Composed native graph execution, independent of scoring corpora."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import replace
import gc
import unittest
from unittest.mock import patch

import numpy as np
import torch_rs as native
from torch_rs import _compile_trace
from tests.test_compile_cuda_boundary import compile_with_cache
from tests.test_compile_cuda_mul_scalar import call_without_python, make_program
from tests.test_cuda_add import Comparison, available, runtime, torch, upload


def composed(x, y):
    z = -(x * 0.5) @ (y * -2)
    return [z, (z, x), z + z]


class GraphBridgeScopeTests(unittest.TestCase):
    def test_cpu_inputs_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, 'CUDA float32'):
            _compile_trace._native._compile_trace_cuda_graph((native.ones((2, 2)),), [])


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CudaGraphTests(Comparison, unittest.TestCase):
    def test_single_bridge_new_data_aliases_and_completion(self):
        compiled, _ = compile_with_cache(composed)
        lib = runtime()
        retained = []
        bridge = _compile_trace._native._compile_trace_cuda_graph
        with patch.object(_compile_trace._native, '_compile_trace_cuda_graph', wraps=bridge) as calls, \
             patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('per-node Python execution')):
            for v in (1., 3., -2.):
                base = native.full((3 + 7 * 7,), v).to('cuda:0')
                a = base[3:].reshape(7, 7)
                # Overlapping input views with different offsets.
                b = base[1:50].reshape(7, 7)
                out = call_without_python(compiled, {composed.__code__}, a, b)
                self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
                self.assertIs(out[0], out[1][0]); self.assertIs(out[1][1], a)
                self.assertEqual(out[0].cpu().tolist(), [[7 * v * v] * 7] * 7)
                self.assertEqual(out[2].cpu().tolist(), [[14 * v * v] * 7] * 7)
                self.assertEqual(base.cpu().tolist(), [v] * 52)
                self.assertNotIn(out[0].data_ptr(), [a.data_ptr(), b.data_ptr(), *(x.data_ptr() for x in retained)])
                retained.append(out[0])
                del base, a, b, out
                gc.collect()
            self.assertEqual(calls.call_count, 3)
        for out, v in zip(retained, (1., 3., -2.)):
            self.assertEqual(out.cpu().tolist(), [[7 * v * v] * 7] * 7)

    def test_late_validation_before_bridge_and_failure_recovery(self):
        fn = make_program('def program(x,y):\n    return (-x) @ y\n')
        a, b = native.ones((3, 7)).to('cuda:0'), native.ones((7, 5)).to('cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            compiled(a, b)
            graph = next(iter(cache.graphs.values()))
            last = graph.operations[-1]
            changes = ({'inputs': ('missing',)}, {'target': 'abs'}, {'scalar': 2},
                       {'metadata': replace(last.metadata, storage_offset=3)},
                       {'metadata': replace(last.metadata, device='cuda:1')},
                       {'metadata': replace(last.metadata, requires_grad=True)})
            bad_graphs = [replace(graph, operations=(*graph.operations[:-1], replace(last, **c))) for c in changes]
            bad_graphs += [replace(graph, output='missing'), replace(graph, output_metadata=None)]
            with patch.object(_compile_trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early launch')):
                for bad in bad_graphs:
                    with self.assertRaises((NotImplementedError, ValueError)):
                        bad.forward(a, b)
                with self.assertRaisesRegex(RuntimeError, 'dimension'):
                    compiled(a, native.ones((8, 5)).to('cuda:0'))
            self.assertEqual(compiled(a, b).cpu().tolist(), [[-7.] * 5] * 3)
            fresh, fresh_cache = compile_with_cache(fn, dynamic=dynamic)
            with patch.object(_compile_trace._native, '_compile_trace_cuda_graph', side_effect=RuntimeError('launch failed')):
                with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                    fresh(a, b)
            self.assertEqual(fresh_cache.graphs, {})
            self.assertEqual(fresh(a, b).cpu().tolist(), [[-7.] * 5] * 3)

    def test_native_bridge_preflights_all_nodes(self):
        a, b = native.ones((3, 7)).to('cuda:0'), native.ones((7, 5)).to('cuda:0')
        bridge = _compile_trace._native._compile_trace_cuda_graph
        first = ('neg', (0,), None, (3, 7), (7, 1))
        bad_nodes = [
            ('matmul', (2, 0), None, (3, 7), (7, 1)),
            ('matmul', (2, 1), None, (3, 5), (1, 3)),
            ('matmul', (2, 1), None, (3, 6), (6, 1)),
            ('add', (2, 99), None, (3, 7), (7, 1)),
            ('neg', (2,), 1., (3, 7), (7, 1)),
            ('mul_scalar', (2,), 2**64, (3, 7), (7, 1)),
            ('abs', (2,), None, (3, 7), (7, 1)),
        ]
        for last in bad_nodes:
            with self.subTest(last=last), self.assertRaises((NotImplementedError, RuntimeError, ValueError, OverflowError)):
                bridge((a, b), [first, last])
        out = bridge((a, b), [first, ('matmul', (2, 1), None, (3, 5), (5, 1))])
        self.assertEqual(out[-1].cpu().tolist(), [[-7.] * 5] * 3)

    def test_late_neg_declared_metadata_rejected_before_execution(self):
        fn = make_program('def program(x,y):\n    return -(x @ y)\n')
        a, b = native.ones((3, 7)).to('cuda:0'), native.ones((7, 5)).to('cuda:0')
        for dynamic in (False, True):
            compiled, cache = compile_with_cache(fn, dynamic=dynamic)
            compiled(a, b)
            graph = next(iter(cache.graphs.values()))
            node = graph.operations[-1]
            declarations = [None] + [replace(node.metadata, **change) for change in (
                {'device': 'cuda:1'}, {'storage_offset': 3}, {'requires_grad': True},
                {'dtype': _compile_trace.CompileTraceDType('torch.float64')},
                {'stride': (1, 3)}, {'shape': (15,), 'stride': (1,)},
            )]
            with patch.object(_compile_trace._native, '_compile_trace_cuda_graph', side_effect=AssertionError('early launch')):
                for metadata in declarations:
                    bad = replace(graph, operations=(*graph.operations[:-1], replace(node, metadata=metadata)))
                    with self.assertRaises((NotImplementedError, ValueError)):
                        bad.forward(a, b)
            self.assertEqual(compiled(a, b).cpu().tolist(), [[-7.] * 5] * 3)

    def test_composition_preserves_wide_overflow_and_signed_zero(self):
        # Compare bitwise with the same native operations executed eagerly:
        # moving/scaling a matmul could otherwise conceal changed FP rounding.
        compiled, _ = compile_with_cache(composed, limit=32)
        for k in (4, 1031, 65539):
            for pattern in ([2.**127, 2.**127, -2.**127, -2.**127],
                            [0., -0., np.inf, np.nan], [0.1, -0.2, 0.1, 0.1]):
                values = np.resize(np.array(pattern, dtype=np.float32), (3, k))
                a = upload(native, values.ravel(), values.shape)
                b = native.full((k, 5), 0.1).to('cuda:0')
                expected = composed(a, b)
                actual = call_without_python(compiled, {composed.__code__}, a, b)
                for x, y in zip((actual[0], actual[2]), (expected[0], expected[2])):
                    x, y = [np.asarray(t.cpu().tolist(), dtype=np.float32) for t in (x, y)]
                    np.testing.assert_array_equal(x.view(np.uint32), y.view(np.uint32))

    def test_threaded_intermediate_lifetimes(self):
        def run(value):
            a = native.full((17, 31), value).to('cuda:0')
            b = native.ones((31, 9)).to('cuda:0')
            compiled, _ = compile_with_cache(composed)
            return compiled(a, b)[0]
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(run, range(1, 10)))
        gc.collect()
        for value, result in enumerate(results, 1):
            self.assertEqual(result.cpu().tolist(), [[31. * value] * 9] * 17)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CudaGraphDeviceTests(unittest.TestCase):
    def test_composed_restoration_and_mixed_devices(self):
        compiled, _ = compile_with_cache(composed)
        previous = torch.cuda.current_device()
        try:
            for ordinal in (0, 1):
                torch.cuda.set_device(1 - ordinal)
                for m, k, n in ((3, 7, 5), (0, 7, 5), (3, 0, 5), (3, 7, 0)):
                    a = native.ones((m, k)).to(f'cuda:{ordinal}')
                    b = native.ones((k, n)).to(f'cuda:{ordinal}')
                    result = compiled(a, b)
                    self.assertEqual(result[0].cpu().tolist(), [[float(k)] * n] * m)
                    self.assertEqual(torch.cuda.current_device(), 1 - ordinal)
                    wrong = b.cpu().to(f'cuda:{1 - ordinal}')
                    with self.assertRaises(NotImplementedError):
                        compiled(a, wrong)
                    with self.assertRaises(NotImplementedError):
                        _compile_trace._native._compile_trace_cuda_graph((a, wrong), [])
                    self.assertEqual(torch.cuda.current_device(), 1 - ordinal)
        finally:
            torch.cuda.set_device(previous)
