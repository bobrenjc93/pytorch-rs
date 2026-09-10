"""Public native rank-2 CUDA matmul differentials; scoring inputs are unchanged."""
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
class CudaMatmulTests(Comparison, unittest.TestCase):
    def setUp(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def compare_product(self, actual, expected):
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

    def test_generated_shapes_offsets_public_forms_and_preservation(self):
        rng = np.random.default_rng(191378)
        shapes = [(0, 0, 0), (0, 7, 5), (3, 7, 0), (5, 0, 9), (1, 1, 1),
                  (1, 1031, 7), (17, 33, 1), (31, 257, 19), (513, 2, 2049)]
        shapes += [tuple(int(v) for v in rng.integers(1, 67, size=3)) for _ in range(16)]
        calls = (lambda m, a, b: m.matmul(input=a, other=b),
                 lambda m, a, b: a.matmul(b), lambda m, a, b: a @ b)
        for m, k, n in shapes:
            for offset in (0, 3):
                av, bv = [rng.normal(size=offset + size).astype(np.float32)
                          for size in (m * k, k * n)]
                bases = [upload(mod, values, values.shape)
                         for mod in (native, torch) for values in (av, bv)]
                a, b, ta, tb = [base[offset:].reshape(shape) for base, shape in
                               zip(bases, ((m, k), (k, n), (m, k), (k, n)))]
                expected = ta @ tb
                for call in calls:
                    with self.subTest(shape=(m, k, n), offset=offset, call=call):
                        result = call(native, a, b)
                        self.compare_product(result, expected)
                        if result.numel():
                            second = a @ b
                            self.assertNotIn(result.data_ptr(),
                                             (a.data_ptr(), b.data_ptr(), second.data_ptr()))
                self.compare(bases[0], bases[2])
                self.compare(bases[1], bases[3])

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
            self.compare_product(left(a) @ right(a), left(ta) @ right(ta))
        self.compare(a, ta)
        # Huge metadata with no allocation/launch, including an empty inner axis.
        for shapes in (((0, 2**40), (2**40, 0)), ((0, 0), (0, 2**40))):
            x, y = [native.zeros(shape, device="cuda:0") for shape in shapes]
            tx, ty = [torch.zeros(shape, device="cuda:0") for shape in shapes]
            self.compare_product(x @ y, tx @ ty)

    def test_special_values_and_cancellation(self):
        special = np.array([0, 0x80000000, 1, 0x80000001, 0x7f800000,
                            0xff800000, 0x7fc12345], dtype=np.uint32).view(np.float32)
        # Inner=1 isolates IEEE multiplication and accumulation classifications.
        a, ta = [upload(mod, special, (len(special), 1)) for mod in (native, torch)]
        b, tb = [upload(mod, np.array([1., -2.], np.float32), (1, 2))
                 for mod in (native, torch)]
        self.compare(a @ b, ta @ tb)
        rng = np.random.default_rng(176359)
        for k in (3, 31, 257, 1031):
            values = rng.integers(-64, 65, size=5 * k).astype(np.float32) / 8
            weights = rng.integers(-64, 65, size=k * 7).astype(np.float32) / 8
            a, ta = [upload(mod, values, (5, k)) for mod in (native, torch)]
            b, tb = [upload(mod, weights, (k, 7)) for mod in (native, torch)]
            self.compare(a @ b, ta @ tb)

    def test_completion_thread_and_allocation_lifetimes(self):
        a = native.full((13, 37), 1.25).to("cuda:0")
        b = native.full((37, 7), 0.5).to("cuda:0")
        expected = torch.full((13, 7), 37 * 1.25 * 0.5, device="cuda:0")
        lib = runtime()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = a @ b
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
            return (left @ right)[1:12]

        with ThreadPoolExecutor(max_workers=4) as workers:
            retained = list(workers.map(produce, range(32)))
        gc.collect()
        for index, value in enumerate(retained):
            self.assertEqual(value.cpu().tolist(), [[37 * index / 16] * 7] * 11)

    def test_invalid_inputs_and_compilation(self):
        a, b = native.ones((3, 5)).to("cuda:0"), native.ones((5, 3)).to("cuda:0")
        for left, right, reason in ((a.cpu(), b, "same CUDA device"),
                                    (a, b.cpu(), "same CUDA device"),
                                    (a, a.t(), "contiguous"),
                                    (b.t(), b, "contiguous"),
                                    (a, b[:, 1:], "contiguous"),
                                    (a.select(0, 0), b, "rank-2"),
                                    (a.reshape(1, 3, 5), b, "rank-2")):
            with self.subTest(reason=reason), self.assertRaisesRegex(NotImplementedError, reason):
                native.matmul(left, right)
        with self.assertRaisesRegex(RuntimeError, "shapes|dimension"):
            a @ a
        for value in (a, b, a @ b):
            with self.assertRaises(NotImplementedError):
                value.requires_grad_()
            with self.assertRaises(NotImplementedError):
                value.backward()
        for dtype in (torch.float64, torch.int64):
            with self.assertRaises((TypeError, NotImplementedError)):
                native.ones((3, 5), dtype=dtype, device="cuda:0")
        for fullgraph in (False, True):
            for program in (lambda x, y: x @ y, lambda x, y: x.matmul(y),
                            lambda x, y: native.matmul(x, y)):
                compiled = native.compile(program, backend="eager", fullgraph=fullgraph)
                with self.assertRaises(NotImplementedError):
                    compiled(a, b)

    def test_no_pytorch_forwarding(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("native matmul imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
x = m.tensor([99., 1., 2., 3., 4., 5., 6.]).to("cuda:0")[1:].reshape(2, 3)
y = m.tensor([1., 2., 3., 4., 5., 6.]).reshape(3, 2).to("cuda:0")
assert m.matmul(x, y).cpu().tolist() == [[22., 28.], [49., 64.]]
assert "torch" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CudaMatmulDeviceTests(unittest.TestCase):
    def test_device_guard_and_mixed_device_rejection(self):
        lib = runtime()
        previous = torch.cuda.current_device()
        try:
            for current, target in ((1, 0), (0, 1)):
                torch.cuda.set_device(current)
                for m, k, n in ((3, 37, 7), (0, 3, 5), (3, 0, 7), (3, 5, 0)):
                    a = native.full((m, k), 1.25).to(f"cuda:{target}")
                    b = native.full((k, n), 0.5).to(f"cuda:{target}")
                    result = a @ b
                    self.assertEqual(str(result.device), f"cuda:{target}")
                    self.assertEqual(result.cpu().tolist(), [[k * 1.25 * 0.5] * n] * m)
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    wrong = native.zeros((k, n), device=f"cuda:{current}")
                    with self.assertRaisesRegex(NotImplementedError, "same CUDA device"):
                        a @ wrong
                    del a, b, result, wrong
                    gc.collect()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(previous)
