"""Real CUDA regressions at TensorIterator's signed 32-bit byte boundary.

Sparse device fixtures avoid allocating multi-gigabyte Python/CPU tensors.
The largest pair of fixtures needs about 12 GiB of device memory.
"""
import ctypes
import gc
import unittest

import numpy as np
import torch_rs as native
from tests.test_cuda_add import available, runtime, torch


LIMIT = 1 << 29  # 1 + 4 * (numel - 1) <= INT32_MAX, not a value-domain limit.


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaSumRowsIndexingTests(unittest.TestCase):
    def require_memory(self, elements):
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(0)
        needed = 2 * elements * 4 + (1 << 30)
        if free < needed:
            self.skipTest(f"32-bit indexing fixtures need {needed} free CUDA bytes; have {free}")

    def compare_sparse(self, rows, width, offset, patches):
        lib = runtime()
        lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.cudaMemcpy.restype = ctypes.c_int
        count = rows * width + offset
        x = native.zeros((1, count), device="cuda:0").reshape(-1)[offset:].reshape(rows, width)
        tx = torch.zeros((1, count), device="cuda:0").reshape(-1)[offset:].reshape(rows, width)
        torch.cuda.synchronize()
        # Copy the same three/sparse cells directly to both allocations. No
        # native mutation API or reference forwarding is required for setup.
        for row, column, value in patches:
            cell = ctypes.c_float(value)
            for tensor in (x, tx):
                self.assertEqual(lib.cudaMemcpy(tensor.data_ptr() + (row * width + column) * 4,
                                                ctypes.byref(cell), 4, 1), 0)
        for keepdim in (False, True):
            with self.subTest(rows=rows, width=width, offset=offset, keepdim=keepdim):
                result = x.sum(1, keepdim)
                expected = tx.sum(1, keepdim)
                again = native.sum(x, dim=-1, keepdim=keepdim)
                self.assertEqual(result.shape, expected.shape)
                self.assertEqual(result.stride(), expected.stride())
                self.assertEqual(result.storage_offset(), 0)
                self.assertEqual(str(result.device), "cuda:0")
                self.assertFalse(result.requires_grad)
                self.assertNotEqual(result.data_ptr(), x.data_ptr())
                self.assertNotEqual(result.data_ptr(), again.data_ptr())
                # Exact cancellation/overflow classification, with no relaxed
                # tolerance even though each row has hundreds of millions of cells.
                np.testing.assert_array_equal(result.cpu().tolist(), expected.cpu().tolist())
                np.testing.assert_array_equal(again.cpu().tolist(), expected.cpu().tolist())
        for row, column, value in patches:
            cell = ctypes.c_float()
            self.assertEqual(lib.cudaMemcpy(ctypes.byref(cell),
                                            x.data_ptr() + (row * width + column) * 4, 4, 2), 0)
            np.testing.assert_array_equal(cell.value, np.float32(value))
        del result, expected, again, x, tx
        gc.collect()

    def test_reviewer_sparse_cancellation_below_at_and_above_byte_boundary(self):
        self.require_memory(LIMIT + 8)
        patches = [(0, 513002932, 1e8), (0, 279983480, -1e8), (0, 6175757, -1.)]
        for width in (LIMIT - 1, LIMIT, LIMIT + 1, LIMIT + 4, LIMIT + 5):
            for offset in range(4):
                self.compare_sparse(1, width, offset, patches)

    def test_row_partitions_reconfigure_geometry_and_keep_output_offsets(self):
        self.require_memory(3 * (LIMIT + 5))
        rng = np.random.default_rng(9102026)
        for rows, width in ((2, LIMIT // 2), (2, LIMIT // 2 + 1),
                            (3, LIMIT // 2 + 3), (3, LIMIT + 5)):
            patches = []
            for row in range(rows):
                indices = rng.choice(width, size=17, replace=False)
                values = rng.choice(np.array([1e8, -1e8, -1., 0.125, 3e38, -3e38], np.float32), size=17)
                patches.extend((row, int(i), float(v)) for i, v in zip(indices, values))
            for offset in (0, 3):
                self.compare_sparse(rows, width, offset, patches)

    def test_nested_odd_column_splits_accumulate_in_order(self):
        self.require_memory(2 * LIMIT + 6)
        width = 2 * LIMIT + 3
        # Exercise more than two partial sums, scalar alignment heads/tails,
        # cancellation and intermediate nonfinites at split endpoints.
        indices = (0, LIMIT // 2 - 1, LIMIT // 2, LIMIT - 1,
                   LIMIT, LIMIT + 1, 3 * LIMIT // 2, width - 1)
        for values in ((1e8, 1., -1e8, 1., -1., 0.125, 2., -0.125),
                       (3e38, 3e38, -3e38, 0., -3e38, 0., 1., 0.),
                       (float('inf'), 0., -float('inf'), 0., float('nan'), 0., 0., 0.)):
            for offset in (0, 1, 3):
                patches = [(0, i, v) for i, v in zip(indices, values)]
                self.compare_sparse(1, width, offset, patches)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CudaSumRowsIndexingDeviceTests(unittest.TestCase):
    def test_split_launches_restore_device_on_success_rejection_and_destruction(self):
        lib = runtime()
        previous = torch.cuda.current_device()
        try:
            for current, target in ((1, 0), (0, 1)):
                torch.cuda.set_device(current)
                if torch.cuda.mem_get_info(target)[0] < 3 * (1 << 30):
                    self.skipTest("split restoration fixture requires 3 GiB free on each device")
                x = native.zeros((1, LIMIT + 4), device=f"cuda:{target}")
                result = x.sum(1)
                self.assertEqual(result.cpu().tolist(), [0.])
                self.assertEqual(str(result.device), f"cuda:{target}")
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

