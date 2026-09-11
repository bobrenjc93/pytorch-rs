"""Native matrix row sums against real CUDA PyTorch; no scoring corpus changes."""
import ctypes
import gc
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch_rs as native
from tests.test_cuda_add import Comparison, available, runtime, torch, upload


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaSumRowsTests(Comparison, unittest.TestCase):
    def compare_sum(self, actual, expected):
        self.assertIs(type(actual), native.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), 0)
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(actual.is_leaf)
        self.assertFalse(actual.requires_grad)
        np.testing.assert_allclose(actual.cpu().tolist(), expected.cpu().tolist(),
                                   rtol=1e-5, atol=1e-4, equal_nan=True)

    def test_generated_rectangles_forms_and_preservation(self):
        rng = np.random.default_rng(925301)
        shapes = [(0, 0), (0, 2**40), (11, 0), (1, 1), (32777, 1)]
        shapes += [(9, width) for width in (7, 31, 32, 33, 127, 255, 256, 257, 1031, 65539)]
        shapes += [tuple(int(v) for v in rng.integers(1, 200, size=2)) for _ in range(16)]
        for shape in shapes:
            values = rng.normal(size=int(np.prod(shape))).astype(np.float32)
            x, tx = upload(native, values, shape), upload(torch, values, shape)
            for dim in (1, -1, (1,), [-1]):
                for keepdim in (False, True):
                    for method in (False, True):
                        with self.subTest(shape=shape, dim=dim, keepdim=keepdim, method=method):
                            result = (x.sum(dim=dim, keepdim=keepdim) if method else
                                      native.sum(input=x, dim=dim, keepdim=keepdim))
                            self.compare_sum(result, tx.sum(dim=dim, keepdim=keepdim))
                            self.assertIsNot(result, x)
                            if result.numel():
                                other = x.sum(dim=dim, keepdim=keepdim)
                                self.assertNotEqual(result.data_ptr(), x.data_ptr())
                                self.assertNotEqual(result.data_ptr(), other.data_ptr())
            self.compare(x, tx)

    def test_wide_same_sign_decimal_rows(self):
        for width in (65539, 1_000_000, 1_000_003):
            for value in (0.1, -0.1, 0.01, 1.1):
                # Include an offset view and non-power-of-two column tails.
                base = native.full((2, 2, width), value).to("cuda:0")
                reference = torch.full((2, 2, width), value, device="cuda:0")
                for x, tx in ((base.select(0, 0), reference.select(0, 0)),
                              (base.select(0, 1), reference.select(0, 1))):
                    for keepdim in (False, True):
                        with self.subTest(width=width, value=value, keepdim=keepdim,
                                          offset=x.storage_offset()):
                            result = x.sum(1, keepdim=keepdim)
                            expected = tx.sum(1, keepdim=keepdim)
                            # Use the existing reduction tolerances, unchanged.
                            self.compare_sum(result, expected)
                            self.compare_sum(native.sum(x, dim=-1, keepdim=keepdim), expected)
                            self.assertNotEqual(result.data_ptr(), x.data_ptr())
                self.compare(base, reference)

    def test_float32_cancellation_and_intermediate_overflow(self):
        maximum = np.finfo(np.float32).max
        patterns = [[1e8, 1, 1, -1e8]]
        for large in (np.float32(3e38), maximum):
            patterns += [[large, 0, large, -large],
                         [large, -large, large, -large]]
        for row_count in (1, 2, 17):
            for pattern in patterns:
                values = np.tile(np.array(pattern, dtype=np.float32), (row_count, 1))
                x, tx = [upload(m, values.ravel(), values.shape) for m in (native, torch)]
                for keepdim in (False, True):
                    with self.subTest(rows=row_count, pattern=pattern, keepdim=keepdim):
                        # Classification and cancellation must match exactly;
                        # tolerance cannot hide a finite-vs-overflow difference.
                        np.testing.assert_array_equal(x.sum(1, keepdim).cpu().tolist(),
                                                      tx.sum(1, keepdim).cpu().tolist())
                self.compare(x, tx)

    def test_reduction_geometry_vector_alignment_and_dynamic_range(self):
        rng = np.random.default_rng(718934)
        widths = (2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
                  127, 128, 129, 131, 255, 256, 257, 511, 512, 513,
                  1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193,
                  65535, 65536, 65539, 1_000_000, 1_000_003)
        for width in widths:
            row_counts = (1, 2, 17) if width > 65539 else (1, 2, 3, 15, 16, 17, 33)
            for rows in row_counts:
                for offset in range(4):
                    # Independent row starts hit all vector head/tail alignments.
                    values = rng.normal(size=rows * width + offset).astype(np.float32)
                    values *= np.exp2(rng.integers(-20, 21, size=values.size)).astype(np.float32)
                    x, tx = [upload(m, values, values.shape)[offset:].reshape(rows, width)
                             for m in (native, torch)]
                    for keepdim in (False, True):
                        with self.subTest(width=width, rows=rows, offset=offset, keepdim=keepdim):
                            self.compare_sum(x.sum(1, keepdim), tx.sum(1, keepdim))
                    self.compare(x, tx)

    def test_nonfinite_and_subnormal_values_across_reduction_trees(self):
        rng = np.random.default_rng(556712)
        special = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff,
                            0x7f7fffff, 0xff7fffff, 0x7f800000,
                            0xff800000, 0x7fc12345], dtype=np.uint32).view(np.float32)
        for width in (127, 128, 129, 513, 8193, 65539, 1_000_003):
            for offset in range(4):
                values = np.zeros((17, width), dtype=np.float32)
                # Keep some rows subnormal-only, some overflowing, some nonfinite.
                for row in range(17):
                    pool = special[:5] if row < 5 else special[5:7] if row < 10 else special
                    values[row] = rng.choice(pool, size=width)
                padded = np.concatenate((np.zeros(offset, np.float32), values.ravel()))
                x, tx = [upload(m, padded, padded.shape)[offset:].reshape(values.shape)
                         for m in (native, torch)]
                for keepdim in (False, True):
                    with self.subTest(width=width, offset=offset, keepdim=keepdim):
                        self.compare(x.sum(1, keepdim), tx.sum(1, keepdim))
                self.compare(x, tx)

    def test_offset_singleton_and_empty_views(self):
        values = np.arange(420, dtype=np.float32) / 16 - 5
        base, reference = upload(native, values, (420,)), upload(torch, values, (420,))
        views = (lambda x: x[7:414].reshape(11, 37),
                 lambda x: x[3:14].reshape(11, 1).t(),
                 lambda x: x[3:14].reshape(1, 11).t(),
                 lambda x: x.reshape(7, 60)[7:7][:, 60:60],
                 lambda x: x.reshape(7, 60)[:, 60:60],
                 lambda x: x[420:].reshape(9, 0))
        for view in views:
            x, tx = view(base), view(reference)
            self.assertTrue(x.is_contiguous())
            for keepdim in (False, True):
                self.compare_sum(x.sum(-1, keepdim), tx.sum(-1, keepdim))
            self.compare(base, reference)

    def test_cancellation_nonfinite_and_subnormal_values(self):
        rng = np.random.default_rng(773089)
        for width in (3, 31, 33, 257, 4099):
            # Exact binary fractions cancel across columns and warp iterations.
            half = rng.integers(-4096, 4096, size=(5, width // 2)).astype(np.float32) / 16
            values = np.concatenate((half, -half, np.full((5, 1), 0.125, np.float32)), axis=1)
            rng.shuffle(values, axis=1)
            x, tx = upload(native, values.ravel(), values.shape), upload(torch, values.ravel(), values.shape)
            self.compare_sum(x.sum(1), tx.sum(1))
        bits = np.array([[0, 0x80000000, 0, 0x80000000],
                         [1, 1, 1, 1], [0x80000001, 0x80000001, 0, 0],
                         [0x7f800000, 1, 0, 0], [0xff800000, 1, 0, 0],
                         [0x7f800000, 0xff800000, 0, 0],
                         [0x7fc12345, 0, 0, 0]], dtype=np.uint32)
        x, tx = [upload(m, bits.view(np.float32).ravel(), bits.shape) for m in (native, torch)]
        self.compare(x.sum(1), tx.sum(1))  # Includes subnormal and signed-zero bits.
        self.compare(x, tx)

    def test_completion_fresh_storage_and_thread_lifetimes(self):
        x = native.full((17, 1031), 1.25).to("cuda:0")
        expected = torch.full((17, 1031), 1.25, device="cuda:0").sum(1)
        lib = runtime()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = x.sum(1)
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(expected)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(), 17 * 4,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.cudaMemset.restype = ctypes.c_int
        self.assertEqual(lib.cudaMemset(x.data_ptr(), 0, 17 * 1031 * 4), 0)
        self.compare_sum(result, expected)
        del x
        gc.collect()

        def produce(index):
            x = native.full((2, 13, 37), index / 8).to("cuda:0").select(0, 1)
            return x.sum(1)[1:12]

        with ThreadPoolExecutor(max_workers=4) as workers:
            retained = list(workers.map(produce, range(32)))
        gc.collect()
        for index, value in enumerate(retained):
            self.assertEqual(value.cpu().tolist(), [37 * index / 8] * 11)

    def test_unsupported_boundaries_and_compiled_reductions(self):
        x = native.ones((3, 5)).to("cuda:0")
        for view in (x.t(), x[:, 1:4]):
            with self.assertRaisesRegex(NotImplementedError, "contiguous"):
                view.sum(1)
        for value in (x, native.ones((0, 5)).to("cuda:0"), native.ones((3, 0)).to("cuda:0")):
            for dim in (None, 0, -2, (), (0, 1)):
                with self.assertRaises(NotImplementedError):
                    native.sum(value, dim=dim)
            with self.assertRaises(NotImplementedError):
                value.requires_grad_()
            with self.assertRaises(NotImplementedError):
                value.sum(1).backward()
            with self.assertRaises(NotImplementedError):
                value.mean(dim=1)
        for shape in ((), (5,), (2, 3, 5)):
            with self.assertRaises(NotImplementedError):
                native.ones(shape).to("cuda:0").sum(-1)
        for dtype in (torch.float64, torch.int64):
            with self.assertRaises(TypeError):
                native.sum(x, dim=1, dtype=dtype)
        for fullgraph in (False, True):
            compiled = native.compile(lambda a: a.sum(1), backend="eager", fullgraph=fullgraph)
            self.assertEqual(compiled(x).cpu().tolist(), [5.] * 3)
            for program in (lambda a: a.sum(0), lambda a: native.sum(a, dim=-1, keepdim=True)):
                compiled = native.compile(program, backend="eager", fullgraph=fullgraph)
                with self.assertRaises(NotImplementedError):
                    compiled(x)

    def test_no_pytorch_forwarding(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("native row sum imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
x = m.tensor([99., 1.25, -2., 3.5, 0.25, 4., -1.]).to("cuda:0")[1:].reshape(2, 3)
assert x.sum(1).cpu().tolist() == [2.75, 3.25]
assert m.sum(x, dim=-1, keepdim=True).cpu().tolist() == [[2.75], [3.25]]
assert x.cpu().tolist() == [[1.25, -2., 3.5], [0.25, 4., -1.]]
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CudaSumRowsDeviceTests(unittest.TestCase):
    def test_device_guard_restores_current_device(self):
        lib = runtime()
        previous = torch.cuda.current_device()
        try:
            for current, target in ((1, 0), (0, 1)):
                torch.cuda.set_device(current)
                for shape in ((3, 37), (2, 1_000_003), (0, 37), (3, 0)):
                    x = native.full(shape, 1.25).to(f"cuda:{target}")
                    result = x.sum(1)
                    self.assertEqual(str(result.device), f"cuda:{target}")
                    self.assertEqual(result.cpu().tolist(), [1.25 * shape[1]] * shape[0])
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    with self.assertRaises(NotImplementedError):
                        x.sum(0)
                    del x, result
                    gc.collect()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(previous)
