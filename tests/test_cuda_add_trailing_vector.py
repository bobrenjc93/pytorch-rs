"""Public eager matrix/vector addition differentials on real NVIDIA hardware."""
import ctypes
import gc
import unittest

import numpy as np
import torch_rs as native

from tests.test_cuda_add import Comparison, available, runtime, torch, upload


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaAddTrailingVectorTests(Comparison, unittest.TestCase):
    def check_forms(self, x, y, tx, ty):
        for left, right, tl, tr in ((x, y, tx, ty), (y, x, ty, tx)):
            for call in (lambda m, a, b: a + b,
                         lambda m, a, b: a.add(b),
                         lambda m, a, b: m.add(input=a, other=b, alpha=1.0)):
                result = call(native, left, right)
                self.compare(result, call(torch, tl, tr))
                if x.numel():
                    self.assertNotIn(result.data_ptr(), (x.data_ptr(), y.data_ptr()))
                    other = call(native, left, right)
                    self.assertNotEqual(other.data_ptr(), result.data_ptr())
        self.compare(x, tx)
        self.compare(y, ty)

    def test_generated_rectangles_and_empty_dimensions(self):
        for seed in (78123, 931874):
            rng = np.random.default_rng(seed)
            shapes = [(0, 0), (0, 17), (23, 0), (1, 1),
                      (1, 259), (259, 1), (3, 257), (257, 3), (1049, 1001)]
            shapes += [tuple(map(int, rng.integers(2, 100, size=2))) for _ in range(12)]
            for rows, columns in shapes:
                with self.subTest(seed=seed, shape=(rows, columns)):
                    a = rng.normal(size=rows * columns).astype(np.float32)
                    b = rng.normal(size=columns).astype(np.float32)
                    self.check_forms(upload(native, a, (rows, columns)),
                                     upload(native, b, (columns,)),
                                     upload(torch, a, (rows, columns)),
                                     upload(torch, b, (columns,)))

    def test_offset_views_and_overlapping_inputs(self):
        a = np.random.default_rng(53197).normal(size=420).astype(np.float32)
        base, tb = upload(native, a, a.shape), upload(torch, a, a.shape)
        views = (
            (lambda x: x[7:414].reshape(11, 37), lambda x: x[19:56]),
            (lambda x: x[3:10].reshape(7, 1), lambda x: x[7:8]),
            (lambda x: x[3:10].reshape(7, 1).t(), lambda x: x[19:26]),
            (lambda x: x[3:10].reshape(1, 7).t(), lambda x: x[7:8]),
            (lambda x: x.reshape(7, 60)[7:7], lambda x: x[13:73]),
            (lambda x: x.reshape(7, 60)[:, 60:60], lambda x: x[420:420]),
            (lambda x: x.reshape(7, 60)[7:7][:, 60:60], lambda x: x[420:420]),
            (lambda x: x[420:420].reshape(7, 0).t(), lambda x: x[19:26]),
            (lambda x: x[420:420].reshape(0, 7).t(), lambda x: x[420:420]),
        )
        for index, (matrix, vector) in enumerate(views):
            with self.subTest(view=index):
                x, y, tx, ty = matrix(base), vector(base), matrix(tb), vector(tb)
                self.assertTrue(x.is_contiguous())
                self.assertTrue(y.is_contiguous())
                self.check_forms(x, y, tx, ty)
        self.compare(base, tb)

    def test_ieee_edges(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                         0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7fc00000,
                         0x3f800000, 0xbf800000, 0x4b800000], dtype=np.uint32)
        values = bits.view(np.float32)
        a = np.repeat(values, len(values))
        self.check_forms(upload(native, a, (len(values), len(values))),
                         upload(native, values, values.shape),
                         upload(torch, a, (len(values), len(values))),
                         upload(torch, values, values.shape))

    def test_lifetimes_reuse_and_fresh_thread_context(self):
        from concurrent.futures import ThreadPoolExecutor
        a = np.random.default_rng(19).normal(size=7 * 257).astype(np.float32)
        b = a[13:270]
        x, y = upload(native, a, (7, 257)), upload(native, b, b.shape)
        tx, ty = upload(torch, a, (7, 257)), upload(torch, b, b.shape)
        cached = x + y
        del cached
        with ThreadPoolExecutor(max_workers=1) as worker:
            keep = worker.submit(lambda: (y + x)[1:6]).result()
        expected = (ty + tx)[1:6]
        del x, y
        gc.collect()
        for _ in range(40):
            replacement = native.ones((7, 257)).to("cuda:0")
            vector = native.ones((257,)).to("cuda:0")
            discarded = replacement + vector
            del discarded, replacement, vector
        self.compare(keep, expected)

    def test_completion_before_independent_stream_read(self):
        lib = runtime()
        a = np.random.default_rng(193).normal(size=1049 * 1001).astype(np.float32)
        b = a[:1001]
        x, y = upload(native, a, (1049, 1001)), upload(native, b, b.shape)
        tx, ty = upload(torch, a, (1049, 1001)), upload(torch, b, b.shape)
        expected = tx + ty
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            out = y + x
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), out.data_ptr(), a.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)

    def test_unsupported_shapes_layouts_and_devices(self):
        x = native.ones((3, 7)).to("cuda:0")
        v = native.ones((7,)).to("cuda:0")
        pairs = [(x, native.ones(shape).to("cuda:0"))
                 for shape in ((), (1,), (3,), (1, 7), (3, 1), (2, 3, 7))]
        pairs += [(x.t(), native.ones((3,)).to("cuda:0")),
                  (x[:, 1:], native.ones((6,)).to("cuda:0")),
                  (x, native.ones((7, 2)).to("cuda:0").select(1, 0)),
                  (x, native.ones((7,))), (native.ones((3, 7)), v),
                  (x.reshape(1, 3, 7), v),
                  (native.ones((0, 7)).to("cuda:0"), native.ones((1, 7)).to("cuda:0"))]
        for a, b in pairs:
            for left, right in ((a, b), (b, a)):
                with self.subTest(left=left.shape, right=right.shape):
                    with self.assertRaisesRegex(NotImplementedError, "unsupported CUDA addition"):
                        native.add(left, right)
        with self.assertRaises(NotImplementedError):
            native.add(x, v, alpha=2)

    def test_compiler_still_rejects_broadcast_before_execution(self):
        from unittest.mock import patch
        from torch_rs import _compile_trace
        from tests.test_compile_cuda_boundary import add_inputs, compile_with_cache
        x, v = native.ones((3, 7)).to("cuda:0"), native.ones((7,)).to("cuda:0")
        for left, right in ((x, v), (v, x)):
            with self.assertRaisesRegex(NotImplementedError, "same shape"):
                _compile_trace._native._compile_trace_binary(left, right, "add")
            for fullgraph in (True, False):
                for warmed in (False, True):
                    compiled, cache = compile_with_cache(add_inputs, fullgraph)
                    if warmed:
                        compiled(x, x)
                    before = dict(cache.graphs)
                    with patch.object(_compile_trace._native, "_compile_trace_binary",
                                      side_effect=AssertionError("executed rejected broadcast")):
                        with self.assertRaisesRegex(NotImplementedError, "same shape"):
                            compiled(left, right)
                    self.assertEqual(cache.graphs, before)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CudaAddTrailingVectorDeviceTests(Comparison, unittest.TestCase):
    def test_same_device_guard_and_mixed_device_rejection(self):
        original = torch.cuda.current_device()
        lib = runtime()
        try:
            for current, target in ((1, 0), (0, 1)):
                torch.cuda.set_device(current)
                for shape in ((3, 7), (0, 7), (3, 0)):
                    x = native.full(shape, 2.5).to(f"cuda:{target}")
                    v = native.full((shape[1],), -1.25).to(f"cuda:{target}")
                    tx = torch.full(shape, 2.5, device=f"cuda:{target}")
                    tv = torch.full((shape[1],), -1.25, device=f"cuda:{target}")
                    other = native.ones((shape[1],)).to(f"cuda:{current}")
                    for a, b in ((x, v), (v, x)):
                        result = a + b
                        self.compare(result, tx + tv)
                        del result
                    for a, b in ((x, other), (other, x)):
                        with self.assertRaisesRegex(NotImplementedError, "mixed devices"):
                            a + b
                    del x, v, other
                    gc.collect()
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(original)
