"""Differential coverage for ordinary native CUDA tensors, independent of compile benchmarks."""
import ctypes
import gc
import os
import subprocess
import sys
import unittest
import warnings

import numpy as np
import torch_rs as native

try:
    import torch
except ImportError:
    torch = None


def available(mask):
    return (os.environ.get("CUDA_VISIBLE_DEVICES") == mask and torch is not None
            and torch.cuda.is_available() and torch.cuda.device_count() >= len(mask.split(",")))


def upload(module, values, shape, device="cuda:0"):
    return module.tensor(values.tolist(), dtype=module.float32).reshape(shape).to(device)


def runtime():
    from torch_rs._cuda_public_storage import _candidate_libraries
    lib = ctypes.CDLL(os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0])
    lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaGetDevice.restype = ctypes.c_int
    lib.cudaStreamQuery.argtypes = [ctypes.c_void_p]
    lib.cudaStreamQuery.restype = ctypes.c_int
    lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_int, ctypes.c_void_p]
    lib.cudaMemcpyAsync.restype = ctypes.c_int
    return lib


class Comparison:
    def compare(self, actual, expected):
        self.assertIs(type(actual), native.Tensor)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        a = np.asarray(actual.cpu().tolist(), dtype=np.float32)
        b = np.asarray(expected.cpu().tolist(), dtype=np.float32)
        # Match every non-NaN bit, including signed zero, subnormals and infinities.
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        valid = ~np.isnan(b)
        np.testing.assert_array_equal(a[valid].view(np.uint32), b[valid].view(np.uint32))


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaAddTests(Comparison, unittest.TestCase):
    def test_random_held_out_shapes_and_public_forms(self):
        rng = np.random.default_rng(541982)
        shapes = [(), (0,), (3, 0, 2), (0, 2**40), (1,), (257,), (65539,), (1_048_589,)]
        shapes += [tuple(int(x) for x in rng.integers(1, 20, size=int(rng.integers(1, 6))))
                   for _ in range(18)]
        calls = (
            lambda m, x, y: x + y,
            lambda m, x, y: x.add(y),
            lambda m, x, y: x.add(other=y, alpha=1.0),
            lambda m, x, y: x.add(1, y),
            lambda m, x, y: m.add(x, y),
            lambda m, x, y: m.add(input=x, other=y, alpha=1, out=None),
        )
        for shape in shapes:
            n = int(np.prod(shape, dtype=np.int64))
            a, b = [rng.normal(size=n).astype(np.float32) for _ in range(2)]
            x, y = upload(native, a, shape), upload(native, b, shape)
            tx, ty = upload(torch, a, shape), upload(torch, b, shape)
            for i, call in enumerate(calls):
                with self.subTest(shape=shape, call=i), warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    result = call(native, x, y)
                    self.compare(result, call(torch, tx, ty))
                    if n:
                        self.assertNotIn(result.data_ptr(), (x.data_ptr(), y.data_ptr()))
            self.compare(x, tx)
            self.compare(y, ty)

    def test_offsets_singletons_and_empty_views(self):
        rng = np.random.default_rng(23810)
        a, b = [rng.normal(size=420).astype(np.float32) for _ in range(2)]
        views = (
            lambda x: x[7:414].reshape(11, 37),
            lambda x: x.select(0, 31),
            lambda x: x.reshape(7, 1, 60).transpose(0, 1),
            lambda x: x.reshape(7, 60)[7:7][:, 60:60],
            lambda x: x.reshape(7, 60)[:, 60:60],
            lambda x: x[420:420].reshape(3, 0, 2),
        )
        for i, view in enumerate(views):
            with self.subTest(view=i):
                x, y = [view(upload(native, v, (420,))) for v in (a, b)]
                tx, ty = [view(upload(torch, v, (420,))) for v in (a, b)]
                self.assertTrue(x.is_contiguous())
                self.compare(native.add(x, y), torch.add(tx, ty))
        # Independent offsets, overlapping sources, identical source aliases.
        x, tx = upload(native, a, (420,)), upload(torch, a, (420,))
        for left, right in ((x[3:102], x[9:108]), (x[3:102], x[3:102].detach())):
            self.compare(left + right, tx[left.storage_offset():left.storage_offset()+99]
                         + tx[right.storage_offset():right.storage_offset()+99])

    def test_numerical_edges(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000,
                         0x3f800000, 0xbf800000, 0x4b800000], dtype=np.uint32)
        values = bits.view(np.float32)
        a, b = np.repeat(values, len(values)), np.tile(values, len(values))
        self.compare(upload(native, a, a.shape) + upload(native, b, b.shape),
                     upload(torch, a, a.shape) + upload(torch, b, b.shape))

    def test_source_output_lifetimes_chaining_and_reuse(self):
        rng = np.random.default_rng(61321)
        a = rng.normal(size=8197).astype(np.float32)
        x, tx = upload(native, a, a.shape), upload(torch, a, a.shape)
        keep = (x + x)[3:8189]
        expected = (tx + tx)[3:8189]
        for _ in range(40):
            # Dropped outputs re-enter the native cache; a retained view must not.
            discarded = native.zeros((8197,), device="cuda:0") + x
            self.assertNotEqual(discarded.data_ptr() + 12, keep.data_ptr())
            del discarded
        del x
        gc.collect()
        self.compare(keep, expected)
        for _ in range(12):
            keep = keep + keep
            expected = expected + expected
        self.compare(keep, expected)
        # Drop both sources before reading, including same-size reuse pressure.
        out = upload(native, a, a.shape) + upload(native, -a, a.shape)
        for _ in range(40):
            replacement = native.zeros((8197,), device="cuda:0")
            del replacement
        self.compare(out, tx + (-tx))

    def test_cached_allocation_on_fresh_host_thread(self):
        from concurrent.futures import ThreadPoolExecutor
        values = np.random.default_rng(51183).normal(size=1237).astype(np.float32)
        x, tx = upload(native, values, values.shape), upload(torch, values, values.shape)
        cached = x + x
        del cached
        # Native executes first: PyTorch must not initialize this thread's
        # primary context on its behalf. Both results cross thread boundaries.
        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(lambda: x + x).result()
            expected = worker.submit(lambda: tx + tx).result()
        self.compare(result, expected)

    def test_completion_before_independent_stream_read(self):
        lib = runtime()
        stream = torch.cuda.Stream()
        a = np.random.default_rng(516).normal(size=2_000_003).astype(np.float32)
        x, tx = upload(native, a, a.shape), upload(torch, a, a.shape)
        expected = tx + tx
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            out = x + x  # Native operations use and complete the legacy stream.
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), out.data_ptr(), a.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        self.compare(out, expected)

    def test_invalid_and_unsupported_inputs(self):
        for shape in ((), (0,), (2, 3)):
            x = native.ones(shape).to("cuda:0")
            cpu = native.ones(shape)
            for call in (lambda: x + cpu, lambda: cpu + x, lambda: x.add(cpu),
                         lambda: native.add(cpu, x), lambda: x + 1, lambda: 1 + x,
                         lambda: x.add(1), lambda: native.add(x, 1), lambda: native.add(1, x),
                         lambda: x.add(x, alpha=2), lambda: native.add(x, x, alpha=0),
                         lambda: x * x, lambda: x - x, lambda: x.sum(),
                         lambda: x.requires_grad_(), lambda: x.to(dtype=torch.float64)):
                with self.subTest(shape=shape, call=call):
                    with self.assertRaises((NotImplementedError, TypeError)):
                        call()
            for alpha in (True, None, "1", 1j):
                for module, v in ((native, x), (torch, torch.ones(shape, device="cuda:0"))):
                    with self.assertRaises((TypeError, RuntimeError)):
                        module.add(v, v, alpha=alpha)
            with self.assertRaises(RuntimeError):
                native.add(x, x, out=x)
            with self.assertRaises(TypeError):
                x.add(x, out=x)
            self.assertEqual(x.cpu().tolist(), cpu.tolist())
            with native.no_grad():
                with self.assertRaises(NotImplementedError):
                    native.ones(shape, requires_grad=True).to("cuda:0")
        x = native.ones((3, 5)).to("cuda:0")
        for left, right in ((x, native.ones((5,)).to("cuda:0")),
                            (x, x.reshape(5, 3)), (x.t(), x.t()),
                            (x[:, 1:4], x[:, 1:4])):
            for call in (lambda: left + right, lambda: left.add(right),
                         lambda: native.add(left, right)):
                with self.assertRaisesRegex(NotImplementedError, "unsupported CUDA addition"):
                    call()

    def test_without_pytorch_import(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("native addition imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
x = m.tensor([99., 1.25, -2., 3.5]).to("cuda:0")[1:]
y = m.tensor([-1., 4., 0.5]).to("cuda:0")
for z in (x+y, x.add(y), m.add(x,y,alpha=1.0)):
    assert z.cpu().tolist() == [0.25, 2., 4.]
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CudaAddDeviceTests(Comparison, unittest.TestCase):
    def test_device_restoration_success_failure_and_drop(self):
        lib = runtime()
        previous = torch.cuda.current_device()
        try:
            for current, target in ((1, 0), (0, 1)):
                torch.cuda.set_device(current)
                for shape in ((), (0,), (113,), (17 * 1024 * 1024,), (65 * 1024 * 1024,)):
                    x = native.full(shape, 3.25).to(f"cuda:{target}")
                    y = native.full(shape, -7.5).to(f"cuda:{target}")
                    result = x + y
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.assertEqual(torch.cuda.current_device(), current)
                    expected = torch.full(shape, 3.25, device=f"cuda:{target}") + torch.full(
                        shape, -7.5, device=f"cuda:{target}")
                    self.compare(result[:113] if shape else result,
                                 expected[:113] if shape else expected)
                    other_device = native.ones(shape).to(f"cuda:{current}")
                    for call in (lambda: x + other_device, lambda: other_device.add(x),
                                 lambda: native.add(x, other_device)):
                        with self.assertRaisesRegex(NotImplementedError, "mixed devices"):
                            call()
                    del x, y, result, other_device
                    gc.collect()  # Large allocations exercise releases outside the front cache.
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.assertEqual(torch.cuda.current_device(), current)
        finally:
            torch.cuda.set_device(previous)


    def test_cross_thread_pool_release_on_both_devices(self):
        from concurrent.futures import ThreadPoolExecutor
        lib = runtime()
        original = torch.cuda.current_device()

        def worker(target):
            current = 1 - target
            torch.cuda.set_device(current)
            x = native.full((17_000_003,), float(target + 1)).to(f"cuda:{target}")
            for _ in range(12):
                out = x + x
                ordinal = ctypes.c_int()
                self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                self.assertEqual(ordinal.value, current)
            del x
            self.assertEqual(torch.cuda.current_device(), current)
            return out

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(worker, (0, 1)))
        for target, result in enumerate(results):
            self.assertEqual(result[:13].cpu().tolist(), [float(2 * (target + 1))] * 13)
            self.assertEqual(torch.cuda.current_device(), original)
