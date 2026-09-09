"""Exercise native storage with nonzero device data, aliases and operation errors."""
import ctypes
import gc
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch_rs as native
try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "requires CUDA reference hardware")
class NativeCudaViewTests(unittest.TestCase):
    def test_nonzero_views_copy_to_cpu_and_keep_storage_alive(self):
        original = native.zeros((60,), device="cuda:0")
        # Inject nonzero data through the CUDA ABI to verify real device reads.
        # This is test setup, not a production backend dependency on Python.
        from torch_rs._cuda_public_storage import _candidate_libraries
        runtime = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
        runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        runtime.cudaMemcpy.restype = ctypes.c_int
        values = np.arange(60, dtype=np.float32) * 0.25 - 7
        self.assertEqual(runtime.cudaMemcpy(original.data_ptr(), values.ctypes.data, values.nbytes, 1), 0)
        reference = torch.tensor(values, device="cuda:0")
        views = [lambda x: x[3:17], lambda x: x.select(0, 9),
                 lambda x: x.reshape(3, 4, 5).transpose(0, 2),
                 lambda x: x.reshape(12, 5)[:, 1:4],
                 lambda x: x.reshape(12, 5)[:, 5:5],
                 lambda x: x.reshape(12, 5)[12:12][:, 5:5],
                 lambda x: x.reshape(12, 5).select(1, 2)]
        actual_views = [view(original) for view in views]
        expected_views = [view(reference) for view in views]
        del original
        gc.collect()
        for a, b in zip(actual_views, expected_views):
            for actual in (a.cpu(), a.to("cpu"), native.as_tensor(a, device="cpu")):
                expected = b.cpu()
                self.assertEqual(actual.tolist(), expected.tolist())
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), 0)

    def test_unsupported_materialization_fails_at_operation(self):
        a = native.zeros((12,), device="cuda:0").reshape(3, 4).t()
        for operation in (lambda: a.clone(), lambda: a.contiguous(),
                          lambda: a.reshape(12), lambda: a.sum(0), lambda: a + a):
            with self.assertRaises(NotImplementedError):
                operation()
        self.assertEqual(a.cpu().tolist(), [[0.0] * 3] * 4)

    def test_allocation_reuse_and_threaded_lifetimes(self):
        def roundtrip(index):
            x = native.zeros((1024 + index % 3,), device="cuda:0")
            view = x[1:17]
            del x
            self.assertEqual(view.cpu().tolist(), [0.0] * 16)
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(roundtrip, range(128)))
