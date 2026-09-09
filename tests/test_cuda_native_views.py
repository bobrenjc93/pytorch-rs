"""Exercise native storage with nonzero device data, aliases and operation errors."""
import ctypes
import gc
import os
from pathlib import Path
import subprocess
import sys
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
    def _copy_values(self, tensor, values):
        # Inject nonzero data through the CUDA ABI to verify real device reads.
        # This is test setup, not a production backend dependency on Python.
        from torch_rs._cuda_public_storage import _candidate_libraries
        runtime = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
        runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        runtime.cudaMemcpy.restype = ctypes.c_int
        self.assertEqual(runtime.cudaMemcpy(tensor.data_ptr(), values.ctypes.data, values.nbytes, 1), 0)

    def test_nonzero_views_copy_to_cpu_and_keep_storage_alive(self):
        original = native.zeros((60,), device="cuda:0")
        values = np.arange(60, dtype=np.float32) * 0.25 - 7
        self._copy_values(original, values)
        reference = torch.tensor(values, device="cuda:0")
        views = [lambda x: x[3:17], lambda x: x.select(0, 9),
                 lambda x: x.reshape(3, 4, 5).transpose(0, 2),
                 lambda x: x.reshape(12, 5)[:, 1:4],
                 lambda x: x.reshape(12, 5)[:, 5:5],
                 lambda x: x.reshape(12, 5)[12:12][:, 5:5],
                 lambda x: x.reshape(12, 5).select(1, 2),
                 lambda x: x.reshape(3, 4, 5).select(2, 2),
                 lambda x: x.reshape(3, 4, 5).select(2, 2).t(),
                 lambda x: x.reshape(3, 4, 5).transpose(0, 1)[1:3].transpose(0, 2)[1:4],
                 lambda x: x.reshape(3, 4, 5).select(1, 2).reshape(3, 1, 5)]
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

    def test_bounded_staging_crosses_chunks_and_preserves_values(self):
        # Narrow and wider rows span multiple chunks, including a final partial chunk.
        for rows, columns in ((20003, 7), (1031, 257)):
            values = np.arange(rows * columns, dtype=np.float32) * 0.25 - 7
            original = native.zeros((values.size,), device="cuda:0")
            self._copy_values(original, values)
            a = original.reshape(rows, columns)
            b = torch.tensor(values, device="cuda:0").reshape(rows, columns)
            views = (lambda x: x.select(1, 2), lambda x: x[:, 1:3], lambda x: x[:, 1:3].t(),
                     lambda x: x[:, 1:-1], lambda x: x[:, 1:-1].t())
            for view in views:
                actual, expected = view(a).cpu(), view(b).cpu()
                self.assertEqual(actual.tolist(), expected.tolist())
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

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_widely_spaced_selection_copies_under_host_memory_limit(self):
        script = r"""
import ctypes
import os
import resource
import torch
import torch_rs as native
from torch_rs._cuda_public_storage import _candidate_libraries

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
torch.set_num_threads(1)
# Warm both contiguous and pitched transfer paths before limiting address space.
for module in (native, torch):
    warm = module.zeros((32,), device="cuda:0")
    warm.cpu()
    warm.reshape(2, 16).select(1, 0).cpu()

elements = 2**24
a = native.zeros((elements,), device="cuda:0")
b = torch.zeros((elements,), device="cuda:0")
runtime = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
runtime.cudaMemcpy.restype = ctypes.c_int
for index, value in ((0, 3.25), (2**23, -9.5)):
    scalar = ctypes.c_float(value)
    assert runtime.cudaMemcpy(a.data_ptr() + index * 4, ctypes.byref(scalar), 4, 1) == 0
    b[index] = value
a, b = a.reshape(2, 2**23).select(1, 0), b.reshape(2, 2**23).select(1, 0)
torch.cuda.synchronize()
original_limit = resource.getrlimit(resource.RLIMIT_AS)
with open("/proc/self/statm", encoding="ascii") as statm:
    virtual_bytes = int(statm.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")
limit = virtual_bytes + 8 * 1024 * 1024
if original_limit[1] != resource.RLIM_INFINITY and limit > original_limit[1]:
    os._exit(77)
try:
    resource.setrlimit(resource.RLIMIT_AS, (limit, original_limit[1]))
    for module, view in ((torch, b), (native, a)):
        for result in (view.cpu(), view.to("cpu"), module.as_tensor(view, device="cpu")):
            assert result.tolist() == [3.25, -9.5]
            assert tuple(result.shape) == (2,)
            assert result.stride() == (1,)
            assert result.storage_offset() == 0
finally:
    resource.setrlimit(resource.RLIMIT_AS, original_limit)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "MALLOC_MMAP_THRESHOLD_": "65536"},
        )
        if completed.returncode == 77:
            self.skipTest("existing hard address-space limit is too restrictive")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
