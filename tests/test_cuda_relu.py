"""Real CUDA ReLU differentials; IEEE payload tests never use Python floats."""
import ctypes
import gc
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch_rs as native
from tests.test_cuda_add import Comparison, available, runtime, torch, upload

EDGE_BITS = np.array([
    0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
    0x00800000, 0x80800000, 0x3fa00000, 0xbfa00000, 0x7f7fffff, 0xff7fffff,
    0x7f800000, 0xff800000, 0x7fc00000, 0xffc12345, 0x7f800001,
    0xff800001, 0x7fa12345, 0xffa12345, 0x7fffffff, 0xffffffff,
], dtype=np.uint32)


def copy_bits(destination, source, size, kind):
    lib = runtime()
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    assert lib.cudaMemcpy(destination, source, size, kind) == 0


def upload_bits(module, bits):
    x = module.zeros((bits.size,), device='cuda:0')
    torch.cuda.synchronize()
    copy_bits(x.data_ptr(), bits.ctypes.data, bits.nbytes, 1)
    return x


def download_bits(x):
    bits = np.empty(x.numel(), dtype=np.uint32)
    torch.cuda.synchronize()
    copy_bits(bits.ctypes.data, x.data_ptr(), bits.nbytes, 2)
    return bits


FORMS = (lambda m, x: x.relu(), lambda m, x: m.relu(x),
         lambda m, x: m.nn.functional.relu(x, inplace=False))


@unittest.skipUnless(available('0'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0')
class CudaReluTests(Comparison, unittest.TestCase):
    def test_seeded_shapes_aliases_and_fresh_canonical_storage(self):
        rng = np.random.default_rng(19781978)
        shapes = [(), (0,), (2, 0, 3), (0, 2**40), (1,), (255,), (256,),
                  (257,), (65539,), (1048589,), (3, 7), (2, 3, 1, 5, 2)]
        shapes += [tuple(int(n) for n in rng.integers(1, 8, size=rank)) for rank in range(1, 7)]
        for shape in shapes:
            values = rng.normal(size=int(np.prod(shape, dtype=np.int64))).astype(np.float32)
            x, tx = [upload(m, values, shape) for m in (native, torch)]
            for form in FORMS:
                with self.subTest(shape=shape, form=form):
                    result = form(native, x)
                    self.compare(result, form(torch, tx))
                    self.compare(x, tx)
                    self.assertIsNot(result, x)
                    self.assertEqual(result.storage_offset(), 0)
                    if values.size:
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_ieee_payloads_offsets_singleton_strides(self):
        bits = np.tile(EDGE_BITS, 30)
        base, reference = [upload_bits(m, bits) for m in (native, torch)]
        for view in (lambda x: x, lambda x: x[1:258], lambda x: x[4:260].reshape(16, 16),
                     lambda x: x.select(0, 1), lambda x: x[-1:],
                     lambda x: x[660:].reshape(2, 0, 3),
                     lambda x: x.reshape(1, 30, 22).transpose(0, 1)):
            x, tx = view(base), view(reference)
            self.assertTrue(x.is_contiguous())
            source = download_bits(x)
            magnitude = source & 0x7fffffff
            expected = np.where(((source >> 31) != 0) & (magnitude <= 0x7f800000),
                                np.uint32(0), source)
            for form in FORMS:
                actual, ref = form(native, x), form(torch, tx)
                np.testing.assert_array_equal(download_bits(ref), expected)
                np.testing.assert_array_equal(download_bits(actual), expected)
                self.assertEqual(actual.shape, tuple(ref.shape))
                self.assertEqual(actual.stride(), ref.stride())
                self.assertEqual(actual.storage_offset(), 0)
            np.testing.assert_array_equal(download_bits(x), source)

    def test_stream_completion_mutation_and_lifetime(self):
        values = np.arange(65539, dtype=np.float32) / 16 - 7
        x, tx = [upload(m, values, values.shape) for m in (native, torch)]
        expected = tx.relu()
        lib, stream = runtime(), torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            result = x.relu()
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(), values.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        zeros = np.zeros_like(values)
        copy_bits(x.data_ptr(), zeros.ctypes.data, values.nbytes, 1)
        self.compare(result, expected)
        retained = result[1:258]
        del x, result
        gc.collect()
        for _ in range(32):
            native.ones(values.shape).to('cuda:0').relu()
        self.compare(retained, expected[1:258])

    def test_cold_threads_and_large_grid_stride_lifetimes(self):
        def run(value):
            base = native.full((1031,), value / 8).to('cuda:0')
            return base[1:1028].relu()[1:17]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(run, range(-16, 32)))
        gc.collect()
        for value, result in zip(range(-16, 32), results):
            self.assertEqual(result.cpu().tolist(), [max(0, value / 8)] * 16)
        x = native.full((17_000_003,), 1.25).to('cuda:0')
        out = x.relu()
        del x
        gc.collect()
        replacement = native.full((17_000_003,), -7.5).to('cuda:0')
        self.assertEqual(out[:3].cpu().tolist(), [1.25] * 3)
        self.assertEqual(out[-3:].cpu().tolist(), [1.25] * 3)
        self.assertEqual(replacement.relu()[-3:].cpu().tolist(), [0.] * 3)

    def test_boundary_and_no_reference_forwarding(self):
        x = native.ones((3, 7)).to('cuda:0')
        for view in (x.t(), x[:, 1:4], x.select(1, 2)):
            for form in FORMS:
                with self.assertRaisesRegex(NotImplementedError, 'CUDA ReLU.*contiguous'):
                    form(native, view)
        for call in (lambda: x.relu(1), lambda: x.relu(inplace=True),
                     lambda: native.nn.functional.relu(x, inplace=True),
                     lambda: x.requires_grad_(), lambda: x.relu().backward(),
                     lambda: native.ones((2,), requires_grad=True).to('cuda:0')):
            with self.assertRaises((TypeError, NotImplementedError)):
                call()
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('reference import')
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
x = m.tensor([99., -2., 3.5]).to('cuda:0')[1:]
for result in (x.relu(), m.relu(x), m.nn.functional.relu(x)):
    assert result.cpu().tolist() == [0., 3.5]
assert 'torch' not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available('0,1'), 'requires real CUDA with CUDA_VISIBLE_DEVICES=0,1')
class CudaReluDeviceTests(Comparison, unittest.TestCase):
    def test_same_and_other_device_restoration(self):
        lib, previous = runtime(), torch.cuda.current_device()
        try:
            for current, target in ((0, 0), (1, 0), (0, 1), (1, 1), (1, 0)):
                torch.cuda.set_device(current)
                for shape in ((), (0,), (3, 7)):
                    x = native.full(shape, 1.25).to(f'cuda:{target}')
                    result = x.relu()
                    self.compare(result, torch.full(shape, 1.25, device=f'cuda:{target}').relu())
                    if shape == (3, 7):
                        with self.assertRaises(NotImplementedError):
                            x.t().relu()
                    del x, result
                    gc.collect()
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(previous)
