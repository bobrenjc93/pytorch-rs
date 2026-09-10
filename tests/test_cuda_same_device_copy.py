"""Real-CUDA differentials for native contiguous float32 vector copies."""
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


def available(mask):
    return (os.environ.get("CUDA_VISIBLE_DEVICES") == mask and torch is not None
            and torch.cuda.is_available() and torch.cuda.device_count() >= len(mask.split(",")))


def metadata(x):
    return (tuple(x.shape), x.stride(), x.storage_offset(), x.data_ptr(),
            str(x.dtype), str(x.device), x.requires_grad, x.is_leaf)


@unittest.skipUnless(available("0"), "requires CUDA reference with CUDA_VISIBLE_DEVICES=0")
class SameDeviceCopyTests(unittest.TestCase):
    def assert_copy(self, actual, expected, call):
        before = metadata(actual)
        source_bits = np.asarray(actual.cpu().tolist(), dtype=np.float32).view(np.uint32)
        output, reference = call(native, actual), call(torch, expected)
        self.assertIs(type(output), native.Tensor)
        self.assertIsNot(output, actual)
        self.assertEqual(metadata(actual), before)
        self.assertEqual(tuple(output.shape), tuple(reference.shape))
        self.assertEqual(output.stride(), reference.stride())
        self.assertEqual(output.storage_offset(), 0)
        self.assertEqual(str(output.device), str(reference.device))
        self.assertEqual(str(output.dtype), str(reference.dtype))
        self.assertEqual(output.is_leaf, reference.is_leaf)
        self.assertFalse(output.requires_grad)
        if output.numel():
            self.assertNotEqual(output.data_ptr(), actual.data_ptr())
        else:
            self.assertEqual(output.data_ptr(), 0)
        np.testing.assert_array_equal(
            np.asarray(output.cpu().tolist(), dtype=np.float32).view(np.uint32),
            np.asarray(reference.cpu().tolist(), dtype=np.float32).view(np.uint32))
        np.testing.assert_array_equal(
            np.asarray(actual.cpu().tolist(), dtype=np.float32).view(np.uint32), source_bits)
        return output

    def test_generated_lengths_offsets_and_exact_values(self):
        rng = np.random.default_rng(923145)
        for n in [0, 1, 255, 256, 257, 65539, *rng.integers(2, 8192, size=16).tolist()]:
            values = rng.normal(size=n + 7).astype(np.float32)
            a = native.tensor(values).to("cuda:0")
            b = torch.tensor(values).to("cuda:0")
            for source, reference in ((a, b), (a[3:3+n], b[3:3+n])):
                with self.subTest(length=n, offset=source.storage_offset()):
                    outputs = [self.assert_copy(source, reference, call) for call in (
                        lambda m, x: x.clone(), lambda m, x: x.to("cuda:0", copy=True))]
                    if n:
                        self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())
                    self.assertIs(source.to("cuda:0"), source)
        special = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-40, -1e-40], dtype=np.float32)
        for call in (lambda m, x: x.clone(), lambda m, x: x.to("cuda:0", copy=True)):
            self.assert_copy(native.tensor(special).to("cuda:0"), torch.tensor(special).cuda(), call)

    def test_argument_forms_and_relaxed_contiguous_strides(self):
        a = native.tensor([1.25, -3.5, 9.0, 4.0]).to("cuda:0")
        b = torch.tensor([1.25, -3.5, 9.0, 4.0], device="cuda:0")
        # Singleton and empty strided vectors are contiguous in PyTorch.
        views = (lambda x: x, lambda x: x.reshape(1, 4).select(1, 1), lambda x: x[4:4],
                 lambda x: x.reshape(2, 2)[2:2].select(1, 1))
        for view in views:
            for call in (
                lambda m, x: x.clone(memory_format=m.preserve_format),
                lambda m, x: m.clone(x),
                lambda m, x: x.to(copy=True),
                lambda m, x: x.to(m.device("cuda", 0), m.float32, False, True),
                lambda m, x: x.to(device="cuda:0", copy=True, memory_format=m.preserve_format),
                lambda m, x: x.to(x, copy=True),
                lambda m, x: x.to(m.float32, copy=True),
            ):
                self.assert_copy(view(a), view(b), call)
        for source in (a, a.reshape(2, 2).select(1, 0), a.reshape(2, 2), a[0:0]):
            for call in (lambda x: x.to(), lambda x: x.to("cuda:0"),
                         lambda x: x.to(x), lambda x: x.to(dtype=native.float32, copy=False)):
                self.assertIs(call(source), source)

    def test_independence_source_mutation_and_allocation_reuse(self):
        from torch_rs._cuda_public_storage import _candidate_libraries
        runtime = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
        runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        runtime.cudaMemcpy.restype = ctypes.c_int
        values = np.arange(1033, dtype=np.float32)
        source = native.tensor(values).to("cuda:0")
        clone, copied = source.clone(), source.to("cuda:0", copy=True)
        clone_view = clone[3:19]
        held_pointer = clone.data_ptr()
        changed = np.full(1033, -9.0, dtype=np.float32)
        self.assertEqual(runtime.cudaMemcpy(source.data_ptr(), changed.ctypes.data, changed.nbytes, 1), 0)
        self.assertEqual(source.cpu().tolist(), changed.tolist())
        self.assertEqual(clone.cpu().tolist(), values.tolist())
        self.assertEqual(copied.cpu().tolist(), values.tolist())
        del source, clone
        gc.collect()
        pointers = set()
        for index in range(40):
            # Different contents expose stale cached allocations and incomplete copies.
            source = native.full((1033,), float(index)).to("cuda:0")
            output = source.clone() if index % 2 else source.to("cuda:0", copy=True)
            self.assertNotEqual(output.data_ptr(), held_pointer)
            pointers.add(output.data_ptr())
            del source
            self.assertEqual(output.cpu().tolist(), [float(index)] * 1033)
            del output
        self.assertLess(len(pointers), 40)
        self.assertEqual(clone_view.cpu().tolist(), values[3:19].tolist())
        self.assertEqual(copied.cpu().tolist(), values.tolist())

    def test_rejections_preserve_source(self):
        a = native.tensor([1.0, 2.0, 3.0, 4.0]).to("cuda:0")
        for source in (a.reshape(2, 2).select(1, 0), a.reshape(2, 2), a.select(0, 1)):
            for call in (lambda: source.clone(), lambda: source.to("cuda:0", copy=True)):
                with self.assertRaisesRegex(NotImplementedError, "contiguous rank-1"):
                    call()
        before = metadata(a)
        for call in (
            lambda: a.to("cuda:1", copy=True), lambda: a.to("cuda:1"),
            lambda: a.to("cuda:0", copy=True, non_blocking=True),
            lambda: a.requires_grad_(),
        ):
            with self.assertRaises(NotImplementedError):
                call()
        with self.assertRaises(TypeError):
            a.to("cuda:0", dtype=torch.float64, copy=True)
        for fmt in (native.contiguous_format, native.channels_last, native.channels_last_3d):
            for call in (lambda: a.clone(memory_format=fmt),
                         lambda: a.to("cuda:0", copy=True, memory_format=fmt)):
                with self.assertRaisesRegex(NotImplementedError, "preserve_format"):
                    call()
        self.assertEqual(metadata(a), before)
        self.assertEqual(a.cpu().tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_no_pytorch_forwarding(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("production path imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as native
source = native.tensor([1.25, -3.5, 7.0]).to("cuda:0")[1:]
a, b = source.clone(), source.to("cuda:0", copy=True)
assert a.cpu().tolist() == b.cpu().tolist() == [-3.5, 7.0]
assert len({source.data_ptr(), a.data_ptr(), b.data_ptr()}) == 3
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available("0,1"), "requires CUDA_VISIBLE_DEVICES=0,1")
class CopyDeviceGuardTests(unittest.TestCase):
    def test_current_device_restored_and_cross_device_rejected(self):
        previous = torch.cuda.current_device()
        try:
            for target, current in ((0, 1), (1, 0)):
                a = native.tensor([1.25, -3.5]).to(f"cuda:{target}")
                torch.cuda.set_device(current)
                for call in (lambda: a.clone(), lambda: a.to(f"cuda:{target}", copy=True)):
                    result = call()
                    self.assertEqual(torch.cuda.current_device(), current)
                    self.assertEqual(str(result.device), f"cuda:{target}")
                    self.assertEqual(result.cpu().tolist(), [1.25, -3.5])
                    del result
                    self.assertEqual(torch.cuda.current_device(), current)
                with self.assertRaisesRegex(NotImplementedError, "cross-device"):
                    a.to(f"cuda:{current}", copy=True)
                self.assertEqual(torch.cuda.current_device(), current)
        finally:
            torch.cuda.set_device(previous)
