"""Public native float32 H2D copies, differentially checked on real CUDA."""
import ctypes
import gc
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch_rs as native

try:
    import torch
except ImportError:
    torch = None


def cuda_available(mask):
    return (os.environ.get("CUDA_VISIBLE_DEVICES") == mask and torch is not None
            and torch.cuda.is_available() and torch.cuda.device_count() >= len(mask.split(",")))


def metadata(x):
    return (tuple(x.shape), x.stride(), x.storage_offset(), x.data_ptr(),
            str(x.dtype), str(x.device), x.requires_grad, x.is_leaf)


@unittest.skipUnless(cuda_available("0"), "requires CUDA reference with CUDA_VISIBLE_DEVICES=0")
class CudaHostTransferTests(unittest.TestCase):
    def assert_transfer(self, a, b, call=lambda module, x: x.to("cuda:0")):
        before = metadata(a)
        actual, expected = call(native, a), call(torch, b)
        torch.cuda.synchronize(0)
        self.assertIs(type(actual), native.Tensor)
        self.assertIsNot(actual, a)
        self.assertEqual(metadata(a), before)
        np.testing.assert_array_equal(a.tolist(), b.tolist())
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertFalse(actual.requires_grad)
        self.assertEqual(actual.data_ptr() == 0, expected.data_ptr() == 0)
        for result in (actual.cpu(), actual.to("cpu"), native.as_tensor(actual, device="cpu")):
            reference = expected.cpu()
            np.testing.assert_array_equal(
                np.asarray(result.tolist(), dtype=np.float32).view(np.uint32),
                np.asarray(reference.tolist(), dtype=np.float32).view(np.uint32),
            )
            self.assertEqual(result.stride(), reference.stride())
        return actual

    def test_device_argument_forms_and_fresh_copies(self):
        a, b = native.tensor([1.25, -7.0]), torch.tensor([1.25, -7.0])
        calls = (
            lambda m, x: x.to("cuda:0"),
            lambda m, x: x.to(m.device("cuda:0")),
            lambda m, x: x.to(device="cuda:0"),
            lambda m, x: x.to(device=m.device("cuda", 0)),
            lambda m, x: x.to("cuda:0", m.float32, False, True),
            lambda m, x: x.to(device="cuda:0", dtype=m.float, copy=False),
            lambda m, x: x.to("cuda:0", memory_format=m.preserve_format),
            lambda m, x: x.to("cuda:0", memory_format=None),
            lambda m, x: x.to(m.zeros((0,), device="cuda:0")),
            lambda m, x: x.to(m.zeros((0,), device="cuda:0"), False, True),
        )
        outputs = [self.assert_transfer(a, b, call) for call in calls]
        self.assertEqual(len({x.data_ptr() for x in outputs}), len(outputs))
        for x in outputs:
            self.assertIs(x.to("cuda:0"), x)

    def test_nonzero_scalars_empty_dense_and_sparse_views(self):
        rng = np.random.default_rng(91843)
        values = rng.normal(size=120).astype(np.float32)
        a, b = native.tensor(values), torch.tensor(values)
        views = (
            lambda x: x,
            lambda x: x[3:17],
            lambda x: x.select(0, 9),
            lambda x: x.reshape(4, 5, 6).transpose(0, 2),
            lambda x: x.reshape(4, 5, 6).permute(2, 0, 1),
            lambda x: x.reshape(20, 6)[:, 1:4],
            lambda x: x.reshape(20, 6).select(1, 2),
            lambda x: x.reshape(4, 5, 6).select(2, 2).t(),
            lambda x: x.reshape(4, 5, 6).transpose(0, 1)[1:3].transpose(0, 2)[1:4],
            lambda x: x.reshape(4, 5, 6).select(1, 2).reshape(4, 1, 6),
            lambda x: x[120:120],
            lambda x: x.reshape(20, 6)[:, 6:6],
            lambda x: x.reshape(20, 6)[20:20][:, 6:6],
        )
        for index, view in enumerate(views):
            with self.subTest(view=index):
                self.assert_transfer(view(a), view(b))
        for shape in ((), (0,), (3, 0, 2), (0, 2**40)):
            self.assert_transfer(native.zeros(shape), torch.zeros(shape))
        for m in (native, torch):
            # A gradient buffer has shared CPU storage but no gradient tracking.
            leaf = m.tensor([2.5, -3.0], requires_grad=True)
            (leaf * 3).sum().backward()
            if m is native:
                gradient = leaf.grad
            else:
                self.assert_transfer(gradient, leaf.grad)
        special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40], dtype=np.float32)
        self.assert_transfer(native.tensor(special), torch.tensor(special))

    def test_gradient_buffer_upload_is_an_independent_snapshot(self):
        for module in (native, torch):
            leaf = module.tensor([1.25, -2.0], requires_grad=True)
            (leaf * 3).sum().backward()
            gradient = leaf.grad
            uploaded = gradient.to("cuda:0")
            (leaf * 2).sum().backward()
            self.assertEqual(gradient.tolist(), [5.0, 5.0])
            self.assertEqual(uploaded.cpu().tolist(), [3.0, 3.0])

    def test_source_and_view_lifetimes_and_allocation_reuse(self):
        a = native.tensor(np.arange(1033, dtype=np.float32))
        uploaded = a.to("cuda:0")
        pointer = uploaded.data_ptr()
        view = uploaded[3:19]
        del a, uploaded
        gc.collect()
        for index in range(40):
            replacement = native.full((1033,), -float(index + 1)).to("cuda:0")
            self.assertNotEqual(replacement.data_ptr(), pointer)
            self.assertEqual(replacement.cpu().tolist(), [-float(index + 1)] * 1033)
            del replacement
        self.assertEqual(view.cpu().tolist(), list(range(3, 19)))
        del view
        gc.collect()
        seen = set()
        for index in range(40):
            replacement = native.full((1033,), float(index)).to("cuda:0")
            seen.add(replacement.data_ptr())
            self.assertEqual(replacement.cpu().tolist(), [float(index)] * 1033)
            del replacement
        self.assertLess(len(seen), 40)
        self.assertEqual(native.zeros((1033,), device="cuda:0").cpu().tolist(), [0.0] * 1033)

    def test_unsupported_and_invalid_inputs_preserve_source(self):
        for shape in ((), (0,), (3,)):
            x = native.ones(shape)
            before = metadata(x)
            for call, error, message in (
                (lambda: x.to("cuda"), NotImplementedError, "unindexed CUDA"),
                (lambda: x.to(0), NotImplementedError, "ordinals"),
                (lambda: x.to("cuda:0", dtype=torch.float64), TypeError, "invalid combination"),
                (lambda: x.to("cuda:0", non_blocking=True), NotImplementedError, "non_blocking"),
                (lambda: x.to("cuda:0", memory_format=native.contiguous_format), NotImplementedError, "preserve_format"),
                (lambda: x.to("cuda:1"), RuntimeError, "CUDA runtime error"),
                (lambda: x.to("cuda:-1"), RuntimeError, "Invalid device string"),
                (lambda: x.to("cuda:0", device="cuda:0"), TypeError, "invalid combination"),
            ):
                with self.subTest(shape=shape, message=message):
                    with self.assertRaisesRegex(error, message):
                        call()
                    self.assertEqual(metadata(x), before)
            # A handled invalid ordinal must not poison another CUDA user.
            reference = torch.arange(6, device="cuda:0").reshape(2, 3).t()
            self.assertEqual(reference.cpu().tolist(), [[0, 3], [1, 4], [2, 5]])
            x.requires_grad_()
            for source in (x, x * 2):
                with self.assertRaisesRegex(NotImplementedError, "requires_grad"):
                    source.to("cuda:0")
                with native.no_grad():
                    with self.assertRaisesRegex(NotImplementedError, "requires_grad"):
                        source.to("cuda:0")
            uploaded = x.detach().to("cuda:0")
            for call in (lambda: uploaded.to("cuda:0", copy=True),
                         lambda: uploaded.to("cuda:1"), lambda: uploaded + uploaded,
                         lambda: uploaded.requires_grad_()):
                with self.assertRaises(NotImplementedError):
                    call()

    def test_no_installed_pytorch_forwarding(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("production path imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as native
x = native.tensor([[1.25, -2.0], [3.5, 4.0]]).t()
y = x.to("cuda:0")
assert y.stride() == (1, 2)
assert y.cpu().tolist() == [[1.25, 3.5], [-2.0, 4.0]]
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_sparse_view_upload_has_bounded_host_allocation(self):
        script = '''
import os, resource
import torch
import torch_rs as native
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
torch.set_num_threads(1)
a = native.full((2, 2**23), 3.25).select(1, 7)
b = torch.full((2, 2**23), 3.25).select(1, 7)
for source in (a, b):
    source.to("cuda:0").cpu()
torch.cuda.synchronize()
old = resource.getrlimit(resource.RLIMIT_AS)
with open("/proc/self/statm") as f:
    limit = int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE") + 8 * 1024 * 1024
if old[1] != resource.RLIM_INFINITY and limit > old[1]:
    os._exit(77)
try:
    resource.setrlimit(resource.RLIMIT_AS, (limit, old[1]))
    for source in (a, b):
        result = source.to("cuda:0")
        assert result.stride() == (1,)
        assert result.cpu().tolist() == [3.25, 3.25]
finally:
    resource.setrlimit(resource.RLIMIT_AS, old)
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
        if result.returncode == 77:
            self.skipTest("existing address-space limit is too restrictive")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(cuda_available("0,1"), "requires CUDA_VISIBLE_DEVICES=0,1")
class CudaHostTransferDeviceGuardTests(unittest.TestCase):
    def test_current_device_restored_on_success_failure_and_drop(self):
        from torch_rs._cuda_public_storage import _candidate_libraries
        runtime = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
        runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDevice.restype = ctypes.c_int
        previous = torch.cuda.current_device()
        try:
            for current, destination in ((1, 0), (0, 1)):
                torch.cuda.set_device(current)
                for shape in ((), (0,), (17,), (17 * 1024 * 1024,)):
                    source = native.full(shape, -3.25)
                    uploaded = source.to(f"cuda:{destination}")
                    ordinal = ctypes.c_int()
                    self.assertEqual(runtime.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.assertEqual(torch.cuda.current_device(), current)
                    reference = torch.full(shape, -3.25).to(f"cuda:{destination}")
                    actual_values = uploaded[:17].cpu() if len(shape) else uploaded.cpu()
                    expected_values = reference[:17].cpu() if len(shape) else reference.cpu()
                    np.testing.assert_array_equal(actual_values.tolist(), expected_values.tolist())
                    del reference
                    del uploaded  # Large storage bypasses the cache and invokes cudaFree.
                    gc.collect()
                    for device in ("cuda:2",):
                        with self.assertRaisesRegex(RuntimeError, "CUDA runtime error"):
                            source.to(device)
                    self.assertEqual(runtime.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.assertEqual(torch.cuda.current_device(), current)
        finally:
            torch.cuda.set_device(previous)
