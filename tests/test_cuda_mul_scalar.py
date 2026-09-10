"""General eager CUDA scalar multiplication against stable CUDA PyTorch."""
import ctypes
import gc
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np
import torch_rs as native
from tests.test_cuda_add import Comparison, available, runtime, torch, upload
from tests.test_compile_cuda_boundary import compile_with_cache


FORMS = (
    lambda m, x, s: x * s,
    lambda m, x, s: s * x,
    lambda m, x, s: x.mul(s),
    lambda m, x, s: x.mul(other=s),
    lambda m, x, s: x.multiply(s),
    lambda m, x, s: m.mul(x, s),
    lambda m, x, s: m.mul(input=s, other=x),
    lambda m, x, s: m.multiply(x, s),
    lambda m, x, s: m.multiply(input=s, other=x),
)


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaMulScalarTests(Comparison, unittest.TestCase):
    def test_generated_shapes_and_all_public_forms(self):
        rng = np.random.default_rng(709321)
        shapes = [(), (0,), (2, 0, 3), (0, 2**40), (1,), (255,), (256,),
                  (257,), (65539,), (1_048_589,)]
        shapes += [tuple(int(n) for n in rng.integers(1, 14, size=int(rng.integers(1, 6))))
                   for _ in range(10)]
        for shape in shapes:
            data = rng.normal(size=int(np.prod(shape, dtype=np.int64))).astype(np.float32)
            x, tx = upload(native, data, shape), upload(torch, data, shape)
            for scalar in (1.375, -2.718281828, 0.0):
                for index, call in enumerate(FORMS):
                    with self.subTest(shape=shape, scalar=scalar, form=index):
                        out = call(native, x, scalar)
                        self.compare(out, call(torch, tx, scalar))
                        self.assertIsNot(out, x)
                        self.assertEqual(out.storage_offset(), 0)
                        if data.size:
                            self.assertNotEqual(out.data_ptr(), x.data_ptr())
                self.compare(x, tx)

    def test_ieee_values_offsets_and_empty_views(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                         0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
                         0x7f800000, 0xff800000, 0x7fc00000, 0xffc12345], dtype=np.uint32)
        data = np.tile(bits.view(np.float32), 30)
        base, ref = upload(native, data, (420,)), upload(torch, data, (420,))
        for view in (lambda x: x, lambda x: x[7:414].reshape(11, 37),
                     lambda x: x.select(0, 1), lambda x: x[-1:],
                     lambda x: x.reshape(1, 7, 60).transpose(0, 1),
                     lambda x: x.reshape(7, 60)[7:7][:, 60:60],
                     lambda x: x[420:].reshape(2, 0, 3)):
            x, tx = view(base), view(ref)
            self.assertTrue(x.is_contiguous())
            for scalar in (1.0, -1.0, 0.0, -0.0, 0.1, 1e-40, 1e40,
                           float('inf'), -float('inf'), float('nan')):
                with self.subTest(shape=tuple(x.shape), offset=x.storage_offset(), scalar=scalar):
                    self.compare(x * scalar, tx * scalar)
            self.compare(x, tx)

    def test_scalar_conversion_contracts(self):
        class IntSubclass(int):
            pass

        class FloatSubclass(float):
            pass

        x = native.tensor([1.25, -3.5, 0.0, -0.0]).to('cuda:0')
        tx = torch.tensor([1.25, -3.5, 0.0, -0.0], device='cuda:0')
        for scalar in (True, False, -7, 2**63, 2**64-1, -(2**63),
                       IntSubclass(3), FloatSubclass(-2.5), np.bool_(False),
                       np.int64(-3), np.uint32(7), np.float32(-0.), np.float64(1.1)):
            # Explicit methods avoid NumPy's own left-operand ufunc dispatch.
            for call in (lambda m, x, s: x.__mul__(s), lambda m, x, s: x.__rmul__(s),
                         *FORMS[2:]):
                with self.subTest(scalar=repr(scalar), type=type(scalar).__name__, call=call):
                    self.compare(call(native, x, scalar), call(torch, tx, scalar))
        for scalar in (2**64, -(2**63)-1, np.uint64(2**63), None, [], '2'):
            for call in FORMS[2:]:
                with self.subTest(rejected=repr(scalar), call=call):
                    try:
                        call(torch, tx, scalar)
                    except Exception as expected:
                        with self.assertRaises(type(expected)) as actual:
                            call(native, x, scalar)
                        self.assertEqual(str(actual.exception), str(expected))
                    else:
                        self.fail('expected reference scalar rejection')
        # Complex dtype promotion is deliberately outside native float32 support.
        for scalar in (1j, 2+0j, np.complex64(2j)):
            for call in FORMS[2:]:
                with self.assertRaises((NotImplementedError, TypeError)):
                    call(native, x, scalar)

    def test_completion_fresh_storage_and_lifetimes(self):
        data = np.random.default_rng(719).normal(size=65539).astype(np.float32)
        x, tx = upload(native, data, data.shape), upload(torch, data, data.shape)
        expected = tx * -1.375
        stream, lib = torch.cuda.Stream(), runtime()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            result = x * -1.375
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(), data.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        self.assertEqual(lib.cudaMemset(x.data_ptr(), 0, data.nbytes), 0)
        self.compare(result, expected)
        del x, result
        gc.collect()

        def produce(index):
            base = native.full((1031,), index / 8).to('cuda:0')
            return (base[1:1028] * -1.5)[1:17]

        with ThreadPoolExecutor(max_workers=4) as pool:
            retained = list(pool.map(produce, range(40)))
        for i, view in enumerate(retained):
            self.assertEqual(view.cpu().tolist(), [-1.5 * i / 8] * 16)
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(lambda: retained[-1] * 2).result()
        del retained
        gc.collect()
        self.assertEqual(result.cpu().tolist(), [-1.5 * 39 / 4] * 16)

    def test_generated_large_allocation_beyond_front_cache(self):
        # Full materialized comparison, including grid-stride iterations and tail.
        data = np.random.default_rng(89630).uniform(-3, 4, 17_000_003).astype(np.float32)
        x, tx = upload(native, data, data.shape), upload(torch, data, data.shape)
        result = x * -0.375
        del x
        gc.collect()
        replacement = native.full(data.shape, 7.5).to('cuda:0')
        self.compare(result, tx * -0.375)
        self.assertEqual(replacement[-3:].cpu().tolist(), [7.5] * 3)

    def test_unsupported_boundaries_and_overrides(self):
        x = native.ones((3, 5)).to('cuda:0')
        for view in (x.t(), x[:, 1:4], x.select(1, 2)):
            for call in FORMS:
                with self.assertRaisesRegex(NotImplementedError, 'CUDA scalar multiplication.*contiguous'):
                    call(native, view, 2.0)
        for shape in ((), (0,), (3, 5)):
            x = native.ones(shape).to('cuda:0')
            for call in FORMS[2:]:
                with self.assertRaises(NotImplementedError):
                    call(native, x, x)
            for fn in (native.mul, native.multiply):
                for out in (None, x):
                    with self.assertRaises((TypeError, NotImplementedError, RuntimeError)):
                        fn(x, 2.0, out=out)
            with self.assertRaises(NotImplementedError):
                x.requires_grad_()
            with self.assertRaises(NotImplementedError):
                (x * 2).backward()
            with native.no_grad(), self.assertRaises(NotImplementedError):
                native.ones(shape, requires_grad=True).to('cuda:0')
        for dtype in (torch.float64, torch.int64, torch.float16):
            with self.assertRaises((TypeError, NotImplementedError)):
                native.zeros((3,), device='cuda:0', dtype=dtype)
        with self.assertRaises(NotImplementedError):
            native.nn.functional.dropout(x, p=1, training=True)
        from torch_rs.overrides import TorchFunctionMode
        class Forward(TorchFunctionMode):
            def __torch_function__(self, fn, types, args=(), kwargs=None):
                return fn(*args, **(kwargs or {}))
        with Forward():
            result = native.mul(x, -1.25)
        self.assertEqual(result.cpu().tolist(), [[-1.25] * 5] * 3)
        class Override:
            @classmethod
            def __torch_function__(cls, fn, types, args=(), kwargs=None):
                return 'override'
        self.assertEqual(native.mul(x, Override()), 'override')

    def test_compiler_rejection_does_not_execute_or_cache(self):
        programs = (lambda x: x * x, lambda x: (x * 1.25) * x,
                    lambda x: x.mul(other=1.25))
        x = native.ones((257,)).to('cuda:0')
        for program in programs:
            for fullgraph in (False, True):
                compiled, cache = compile_with_cache(program, fullgraph)
                from torch_rs import _compile_trace
                with patch.object(_compile_trace, '_execute_operation', side_effect=AssertionError('executed graph')):
                    with self.assertRaises(NotImplementedError):
                        compiled(x)
                self.assertEqual(cache.graphs, {})

    def test_no_pytorch_forwarding(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('native multiplication imported PyTorch')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
x = m.tensor([99., 1.25, -2., 3.5]).to('cuda:0')[1:]
for result in (x * -2, -2 * x, x.mul(-2), x.multiply(-2), m.mul(x, -2), m.multiply(-2, x)):
    assert result.cpu().tolist() == [-2.5, 4., -7.]
    assert result.data_ptr() != x.data_ptr()
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CudaMulScalarDeviceTests(Comparison, unittest.TestCase):
    def test_allocation_failure_restores_device_and_recovers(self):
        lib, previous = runtime(), torch.cuda.current_device()
        try:
            torch.cuda.set_device(1)
            x = native.tensor([1.25, -2.5]).to('cuda:0')
            # Four exabytes cannot fit in device virtual memory: exercise the
            # actual shared allocator error without filling physical GPU memory.
            with self.assertRaisesRegex(RuntimeError, 'CUDA runtime error: cudaErrorMemoryAllocation'):
                native.zeros((2**60,), device='cuda:0')
            ordinal = ctypes.c_int()
            self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
            self.assertEqual(ordinal.value, 1)
            self.assertEqual((x * 1.5).cpu().tolist(), [1.875, -3.75])
            self.assertEqual(x.cpu().tolist(), [1.25, -2.5])
        finally:
            torch.cuda.set_device(previous)

    def test_ownership_restoration_and_mixed_device_rejection(self):
        lib, previous = runtime(), torch.cuda.current_device()
        try:
            for current, target in ((1, 0), (0, 1), (1, 0)):
                torch.cuda.set_device(current)
                for shape in ((), (0,), (3, 5)):
                    x = native.full(shape, -1.75).to(f'cuda:{target}')
                    result = x * 3.125
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.compare(result, torch.full(shape, -1.75, device=f'cuda:{target}') * 3.125)
                    if result.numel():
                        driver = ctypes.CDLL('libcuda.so.1')
                        driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
                        driver.cuPointerGetAttribute.restype = ctypes.c_int
                        for tensor in (x, result):
                            for attribute, expected in ((2, 2), (9, target), (8, 0)):
                                value = ctypes.c_int()
                                self.assertEqual(driver.cuPointerGetAttribute(
                                    ctypes.byref(value), attribute, tensor.data_ptr()), 0)
                                self.assertEqual(value.value, expected)
                    other = native.ones(shape).to(f'cuda:{current}')
                    with self.assertRaises(NotImplementedError):
                        x * other
                    if shape == (3, 5):
                        with self.assertRaisesRegex(NotImplementedError, 'contiguous'):
                            x.t() * 2
                    del x, result, other
                    gc.collect()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(previous)
