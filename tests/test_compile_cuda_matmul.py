"""Captured cuBLAS products; independent of the fixed scoring workloads."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import replace
import gc
import os
from pathlib import Path
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
from tests import test_cuda_matmul as eager_cases


def product(x, y):
    return x @ y


class MatmulMetadataTests(unittest.TestCase):
    def test_scope_dimensions_and_fresh_output(self):
        for options, shape in (({'device': 'cpu'}, (2, 3)),
                               ({'requires_grad': True}, (2, 3)),
                               ({'stride': (1, 2)}, (2, 3)), ({}, (3,))):
            recorder = _compile_trace.CompileTraceRecorder()
            x = recorder.input(shape=shape, **({'device': 'cuda:0'} | options))
            y = recorder.input(shape=(3, 2), device='cuda:0')
            with self.assertRaises(NotImplementedError):
                x @ y
            self.assertEqual(recorder._operations, [])
        recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(shape=(2**40, 0), device='cuda:0')
        y = recorder.input(shape=(0, 2**40), device='cuda:0')
        with self.assertRaises(OverflowError):
            x @ y
        recorder = _compile_trace.CompileTraceRecorder()
        x = recorder.input(shape=(1, 3), stride=(1, 1), device='cuda:0', storage_offset=7)
        y = recorder.input(shape=(3, 1), stride=(1, 3), device='cuda:0', storage_offset=2)
        graph = recorder.finish(x @ y)
        self.assertEqual(graph.operations[0].target, 'matmul')
        self.assertEqual(graph.output_metadata.shape, (1, 1))
        self.assertEqual(graph.output_metadata.stride, (1, 1))
        self.assertEqual(graph.output_metadata.storage_offset, 0)
        with self.assertRaises(NotImplementedError):
            _compile_trace._native._compile_trace_binary(native.ones((2, 2)), native.ones((2, 2)), 'matmul')


# Mirror the eager numerical regressions through captured programs as well.
@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CompileCudaMatmulTests(Comparison, unittest.TestCase):
    def setUp(self):
        eager_cases.CudaMatmulTests.setUp(self)
        self.enterContext(torch._dynamo.config.patch(recompile_limit=128))
        self.compiled, self.cache = compile_with_cache(product, limit=4096)

    def compare_product(self, actual, expected):
        eager_cases.CudaMatmulTests.compare_product(self, actual, expected)

    def compiled_pair(self, a, b):
        return call_without_python(self.compiled, {product.__code__}, a, b)

    def test_generated_capture_forms_changed_inputs_and_composition(self):
        rng = np.random.default_rng(839127)
        shapes = [(0, 7, 3), (3, 0, 5), (2, 7, 0), (1, 1, 1), (17, 31, 9)]
        shapes += [tuple(map(int, rng.integers(2, 37, size=3))) for _ in range(5)]
        for expr in ('x @ y', 'x.matmul(y)', 'x.__matmul__(y)', 'm.matmul(x, y)',
                     'mm(x, y)', '-(x * 0.5) @ (y * -2)',
                     '(x @ y) + (x @ y)', '(x @ y) * 1.375'):
            fn = make_program(f'def program(x, y):\n    return {expr}\n', mm=native.matmul)
            ref_fn = make_program(f'def program(x, y):\n    return {expr}\n', torch, mm=torch.matmul)
            for fullgraph, dynamic in POLICIES:
                compiled, cache = compile_with_cache(fn, fullgraph, dynamic, limit=128)
                torch._dynamo.reset()
                reference = torch.compile(ref_fn, backend='eager', fullgraph=fullgraph, dynamic=dynamic)
                for m, k, n in shapes:
                    for offset in (0, 3):
                        for changed in range(2):
                            vals = [rng.normal(size=offset+s).astype(np.float32) for s in (m*k, k*n)]
                            bases = [upload(mod, v, v.shape) for mod in (native, torch) for v in vals]
                            args = [base[offset:].reshape(shape) for base, shape in
                                    zip(bases, ((m,k), (k,n), (m,k), (k,n)))]
                            with self.subTest(expr=expr, policy=(fullgraph,dynamic), shape=(m,k,n), offset=offset, changed=changed):
                                actual = call_without_python(compiled, {fn.__code__}, *args[:2])
                                expected = reference(*args[2:])
                                self.compare_product(actual, expected)
                                self.compare(bases[0], bases[2]); self.compare(bases[1], bases[3])
                                if actual.numel():
                                    second = call_without_python(compiled, {fn.__code__}, *args[:2])
                                    self.assertNotIn(actual.data_ptr(), [v.data_ptr() for v in (*args[:2], second)])
                self.assertTrue(all(any(op.target == 'matmul' for op in g.operations) for g in cache.graphs.values()))

    def test_singleton_empty_and_overlapping_views(self):
        values = np.arange(80, dtype=np.float32) / 8 - 3
        a, ta = [upload(mod, values, values.shape) for mod in (native, torch)]
        views = ((lambda x: x[3:14].reshape(11, 1).t(),
                  lambda x: x[7:29].reshape(11, 2)),
                 (lambda x: x[3:7].reshape(1, 4).t(),
                  lambda x: x[3:7].reshape(1, 4)),
                 (lambda x: x[80:].reshape(9, 0),
                  lambda x: x[80:].reshape(0, 7)),
                 (lambda x: x.reshape(8, 10)[8:8][:, 10:10],
                  lambda x: x[80:].reshape(0, 7)),
                 (lambda x: x[3:19].reshape(4, 4),
                  lambda x: x[3:19].reshape(4, 4)))
        for left, right in views:
            self.compare_product(self.compiled_pair(left(a), right(a)), left(ta) @ right(ta))
        self.compare(a, ta)
        # Huge metadata with no allocation/launch, including an empty inner axis.
        for shapes in (((0, 2**40), (2**40, 0)), ((0, 0), (0, 2**40))):
            x, y = [native.zeros(shape, device="cuda:0") for shape in shapes]
            tx, ty = [torch.zeros(shape, device="cuda:0") for shape in shapes]
            self.compare_product(self.compiled_pair(x, y), tx @ ty)

    def test_special_values_and_cancellation(self):
        special = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
                            0xff800000, 0x7fc12345], dtype=np.uint32).view(np.float32)
        # Inner=1 isolates IEEE multiplication and accumulation classifications.
        a, ta = [upload(mod, special, (len(special), 1)) for mod in (native, torch)]
        b, tb = [upload(mod, np.array([1., -2.], np.float32), (1, 2))
                 for mod in (native, torch)]
        self.compare(self.compiled_pair(a, b), ta @ tb)
        rng = np.random.default_rng(176359)
        for k in (3, 31, 257, 1031):
            values = rng.integers(-64, 65, size=5 * k).astype(np.float32) / 8
            weights = rng.integers(-64, 65, size=k * 7).astype(np.float32) / 8
            a, ta = [upload(mod, values, (5, k)) for mod in (native, torch)]
            b, tb = [upload(mod, weights, (k, 7)) for mod in (native, torch)]
            self.compare(self.compiled_pair(a, b), ta @ tb)

    def test_wide_decimal_products(self):
        # Root/evaluator counterexamples plus repeated non-binary fractions.
        widths = (4096, 8192, 16384, 32768, 65536, 65539, 131072, 262144, 1000000)
        for k in widths:
            for value, weight in ((0.1, 1.), (-0.1, 1.), (0.1, 0.1),
                                  (-0.1, 0.1), (1. / 3., -0.2)):
                with self.subTest(k=k, value=value, weight=weight):
                    a, ta = [mod.full((1, k), value).to("cuda:0") for mod in (native, torch)]
                    b, tb = [mod.full((k, 1), weight).to("cuda:0") for mod in (native, torch)]
                    self.compare_product(self.compiled_pair(a, b), ta @ tb)
                    self.compare(a, ta)
                    self.compare(b, tb)

    def test_generated_wide_products_and_overlapping_offsets(self):
        rng = np.random.default_rng(573219)
        for m, k, n in [(int(rng.integers(2, 6)), int(rng.integers(4096, 90001)),
                         int(rng.integers(2, 8))) for _ in range(6)]:
            for shared in (False, True):
                # Positive/negative decimals with nonzero mean also expose
                # accumulation drift in each row/column of generated products.
                av = rng.uniform(-0.2, 0.7, size=7 + max(m*k, k*n)).astype(np.float32)
                bv = av if shared else rng.uniform(-0.7, 0.2, size=av.size).astype(np.float32)
                bases = [upload(mod, v, v.shape) for mod in (native, torch) for v in (av, bv)]
                a, b, ta, tb = [base[3:3+size].reshape(shape) for base, size, shape in
                               zip(bases, (m*k, k*n, m*k, k*n), ((m,k), (k,n), (m,k), (k,n)))]
                if shared:
                    b = bases[0][5:5+k*n].reshape(k, n)
                    tb = bases[2][5:5+k*n].reshape(k, n)
                with self.subTest(shape=(m,k,n), shared=shared):
                    result, second = self.compiled_pair(a, b), self.compiled_pair(a, b)
                    self.compare_product(result, ta @ tb)
                    self.assertNotIn(result.data_ptr(), (a.data_ptr(), b.data_ptr(), second.data_ptr()))
                    self.compare(bases[0], bases[2])
                    self.compare(bases[1], bases[3])

    def test_intermediate_overflow_cancellation_and_nonfinite_classification(self):
        high = np.float32(2.**127)
        tiny = np.nextafter(np.float32(0), np.float32(1))
        patterns = ([high, high, -high, -high], [high, -high, high, -high],
                    [high, high, high, high], [high, high, -high, 0.],
                    [np.inf, 1., -np.inf, 0.], [np.nan, 1., 0., 0.],
                    [tiny, tiny, -tiny, tiny], [1.e20, 0.1, -1.e20, 0.1])
        for k in (4, 1031, 2048, 65539):
            for pattern in patterns:
                for m, n in ((1, 1), (3, 5)):
                    av = np.zeros((m, k), dtype=np.float32)
                    av[:, :4] = pattern
                    bv = np.ones((k, n), dtype=np.float32)
                    a, ta = [upload(mod, av.ravel(), av.shape) for mod in (native, torch)]
                    b, tb = [upload(mod, bv.ravel(), bv.shape) for mod in (native, torch)]
                    with self.subTest(k=k, pattern=pattern, shape=(m,n)):
                        actual, expected = self.compiled_pair(a, b), ta @ tb
                        self.compare_product(actual, expected)
                        x, y = [np.asarray(v.cpu().tolist(), dtype=np.float32) for v in (actual, expected)]
                        for classification in (np.isfinite, np.isposinf, np.isneginf, np.isnan):
                            np.testing.assert_array_equal(classification(x), classification(y))
                        # Subnormal and signed-zero results must not be hidden by atol.
                        small = np.isfinite(y) & (np.abs(y) < np.finfo(np.float32).tiny)
                        np.testing.assert_array_equal(x[small].view(np.uint32), y[small].view(np.uint32))
                        if k in (1031, 2048) and m == n == 1 and pattern is patterns[0]:
                            self.assertEqual(expected.item(), 0.)

    def test_completion_thread_and_allocation_lifetimes(self):
        a = native.full((13, 37), 1.25).to("cuda:0")
        b = native.full((37, 7), 0.5).to("cuda:0")
        expected = torch.full((13, 7), 37 * 1.25 * 0.5, device="cuda:0")
        lib = runtime()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = self.compiled_pair(a, b)
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(expected)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(),
                                                13 * 7 * 4, 3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.cudaMemset.restype = ctypes.c_int
        for value in (a, b):
            self.assertEqual(lib.cudaMemset(value.data_ptr(), 0, value.numel() * 4), 0)
        self.compare_product(result, expected)
        del a, b, result, value
        gc.collect()

        def produce(index):
            left = native.full((2, 13, 37), index / 8).to("cuda:0").select(0, 1)
            right = native.full((2, 37, 7), 0.5).to("cuda:0").select(0, 1)
            return (self.compiled_pair(left, right))[1:12]

        with ThreadPoolExecutor(max_workers=4) as workers:
            retained = list(workers.map(produce, range(32)))
        gc.collect()
        for index, value in enumerate(retained):
            self.assertEqual(value.cpu().tolist(), [[37 * index / 16] * 7] * 11)

    def test_guards_late_validation_recovery_and_captures(self):
        fn = make_program('def program(x, y):\n    return -(x @ y) + bias\n',
                          bias=native.ones((5,)).to('cuda:0'))
        for fullgraph, dynamic in POLICIES:
            compiled, cache = compile_with_cache(fn, fullgraph, dynamic, limit=128)
            for shape in ((3, 7, 5), (9, 7, 5)):
                m,k,n = shape
                args = (native.ones((m, k)).to('cuda:0'), native.ones((k, n)).to('cuda:0'))
                out = call_without_python(compiled, {fn.__code__}, *args)
                self.assertEqual(out.cpu().tolist(), [[1.-k]*n]*m)
            before = len(cache.graphs)
            with patch.object(_compile_bytecode, 'lower_compile_graph', side_effect=AssertionError('cache miss')):
                call_without_python(compiled, {fn.__code__}, *args)
            self.assertEqual(len(cache.graphs), before)
            for bias in (native.full((5,), 2.0).to('cuda:0'), native.ones((9, 5)).to('cuda:0')):
                fn.__globals__['bias'] = bias
                call_without_python(compiled, {fn.__code__}, *args)
            fn.__globals__['bias'] = native.ones((5,)).to('cuda:0')
            with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('early launch')):
                with self.assertRaisesRegex(RuntimeError, 'dimension'):
                    compiled(args[0], native.ones((8, 5)).to('cuda:0'))
            graph = next(reversed(cache.graphs.values()))
            node = graph.operations[0]
            for change in ({'inputs': ()}, {'scalar': 2}, {'metadata': replace(node.metadata, storage_offset=3)},
                           {'metadata': replace(node.metadata, device='cuda:1')},
                           {'metadata': replace(node.metadata, requires_grad=True)}):
                bad = replace(graph, operations=(replace(node, **change), *graph.operations[1:]))
                with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('early launch')):
                    with self.assertRaises((NotImplementedError, RuntimeError)):
                        bad.forward(*args)
            compiled2, cache2 = compile_with_cache(product, fullgraph, dynamic)
            with patch.object(_compile_trace._native, '_compile_trace_binary', side_effect=RuntimeError('launch failed')):
                with self.assertRaisesRegex(RuntimeError, 'launch failed'):
                    compiled2(*args)
            self.assertEqual(cache2.graphs, {})
            self.assertEqual(compiled2(*args).cpu().tolist(), [[7.]*5]*9)
            native.compiler.reset()
            self.assertEqual(cache2.graphs, {})

    def test_rejections_binding_guards_and_alias_containers(self):
        a, b = native.ones((3, 5)).to('cuda:0'), native.ones((5, 3)).to('cuda:0')
        for fullgraph in (True, False):
            compiled, _ = compile_with_cache(product, fullgraph)
            for x,y in ((a.cpu(),b), (a,b.cpu()), (a,b[:,1:]), (a.select(0,0),b), (a.reshape(1,3,5),b)):
                with self.assertRaises(NotImplementedError):
                    compiled(x,y)
            for expr in ('m.mm(x,y)', 'm.matmul(input=x, other=y)', 'x.matmul(other=y)',
                         '(x @ y).relu()', '(x @ y).sum()', 'x @ y.t()'):
                fn = make_program(f'def program(x,y):\n    return {expr}\n')
                with self.assertRaises(NotImplementedError):
                    native.compile(fn, backend='eager', fullgraph=fullgraph)(a,b)
            fn = make_program('def program(x,y):\n    z = m.matmul(x,y)\n    return [z, (z, x)]\n')
            compiled, cache = compile_with_cache(fn, fullgraph)
            out = call_without_python(compiled, {fn.__code__}, a,b)
            self.assertIs(out[0], out[1][0]); self.assertIs(out[1][1], a)
            with patch.object(native, 'matmul', lambda x,y: x):
                with self.assertRaises(NotImplementedError):
                    compiled(a,b)
            self.assertEqual(compiled(a,b)[0].cpu().tolist(), [[5.]*3]*3)
        for method in ('matmul', '__matmul__'):
            fn = make_program(f'def program(x,y):\n    return x.{method}(y)\n')
            compiled, _ = compile_with_cache(fn)
            compiled(a,b)
            with patch.object(native.Tensor, method, lambda x,y: x):
                with self.assertRaisesRegex(NotImplementedError, 'patched'):
                    compiled(a,b)
                with self.assertRaisesRegex(NotImplementedError, 'patched'):
                    native.compile(fn, backend='eager', fullgraph=True)(a,b)
            compiled(a,b)
        compiled, cache = compile_with_cache(product, limit=1)
        compiled(a,b)
        with self.assertRaises(NotImplementedError):
            compiled(native.ones((2, 5)).to('cuda:0'), b)
        self.assertEqual(len(cache.graphs),1)

    def test_isolated_installed_wheel_no_body_no_pytorch(self):
        script = '''
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
class Block:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('PyTorch forwarding')
sys.meta_path.insert(0, Block())
import torch_rs as m
from torch_rs import _compile_trace, _compile_bytecode
for module in (m, _compile_trace, _compile_bytecode, _compile_trace._native):
    path = pathlib.Path(module.__file__).resolve()
    assert path.is_relative_to(root / '.venv'), path
    print(path)
def program(x,y):
    return -(x @ y) * 0.5
for fullgraph in (False, True):
    compiled = m.compile(program, backend='eager', fullgraph=fullgraph)
    def forbid(frame, event, arg):
        if event == 'call' and frame.f_code is program.__code__:
            raise AssertionError('body replay')
    sys.setprofile(forbid)
    for value in (1., 3.):
        a = m.full((2,3), value).to('cuda:0')
        b = m.full((3,4), 2.).to('cuda:0')
        assert compiled(a,b).cpu().tolist() == [[-3.*value]*4]*2
    sys.setprofile(None)
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(Path.cwd())],
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CompileCudaMatmulDeviceTests(unittest.TestCase):
    def test_device_restore_cache_and_mixed_device_errors(self):
        lib = runtime()
        previous = torch.cuda.current_device()
        compiled, cache = compile_with_cache(product)
        try:
            for ordinal in (0,1):
                torch.cuda.set_device(1-ordinal)
                for m,k,n in ((3,7,5),(0,7,5),(3,0,5),(3,7,0)):
                    a = native.full((m, k), 2.0).to(f'cuda:{ordinal}')
                    b = native.ones((k, n)).to(f'cuda:{ordinal}')
                    out = call_without_python(compiled, {product.__code__}, a,b)
                    self.assertEqual(str(out.device), f'cuda:{ordinal}')
                    self.assertEqual(out.cpu().tolist(), [[2.*k]*n]*m)
                    wrong = b.cpu().to(f'cuda:{1-ordinal}')
                    with self.assertRaises(NotImplementedError):
                        compiled(a,wrong)
                    with self.assertRaises(NotImplementedError):
                        _compile_trace._native._compile_trace_binary(a,wrong,'matmul')
                    del a,b,out,wrong
                    gc.collect()
                    current = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(current)),0)
                    self.assertEqual(current.value, 1-ordinal)
            self.assertEqual({str(g.inputs[0].metadata.device) for g in cache.graphs.values()}, {'cuda:0','cuda:1'})
        finally:
            torch.cuda.set_device(previous)
