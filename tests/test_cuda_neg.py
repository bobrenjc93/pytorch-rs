"""Public native CUDA negation, checked against real CUDA PyTorch execution."""
import ctypes
import gc
import subprocess
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch_rs as native

from tests.test_cuda_add import Comparison, available, runtime, torch, upload


FORMS = (
    lambda m, x: -x,
    lambda m, x: x.neg(),
    lambda m, x: x.negative(),
    lambda m, x: m.neg(x),
    lambda m, x: m.negative(input=x, out=None),
)


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaNegTests(Comparison, unittest.TestCase):
    def test_generated_shapes_public_forms_and_input_preservation(self):
        rng = np.random.default_rng(803192)
        shapes = [(), (0,), (3, 0, 2), (0, 2**40), (1,), (255,), (256,),
                  (257,), (65539,), (1_048_589,)]
        shapes += [tuple(int(n) for n in rng.integers(1, 17, size=int(rng.integers(1, 6))))
                   for _ in range(12)]
        for shape in shapes:
            values = rng.normal(size=int(np.prod(shape, dtype=np.int64))).astype(np.float32)
            x, tx = upload(native, values, shape), upload(torch, values, shape)
            for index, call in enumerate(FORMS):
                with self.subTest(shape=shape, form=index):
                    result = call(native, x)
                    self.compare(result, call(torch, tx))
                    self.compare(x, tx)
                    self.assertEqual(result.storage_offset(), 0)
                    self.assertIsNot(result, x)
                    if values.size:
                        self.assertNotEqual(result.data_ptr(), x.data_ptr())

    def test_ieee_bits_and_offset_views(self):
        bits = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                         0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff,
                         0x7f800000, 0xff800000, 0x7fc00000, 0xffc12345], dtype=np.uint32)
        values = np.tile(bits.view(np.float32), 37)
        base, reference = upload(native, values, values.shape), upload(torch, values, values.shape)
        for view in (lambda x: x, lambda x: x[1:258], lambda x: x[4:260].reshape(16, 16),
                     lambda x: x.select(0, 1), lambda x: x[-1:],
                     lambda x: x[len(values):].reshape(2, 0, 3),
                     lambda x: x.reshape(1, 37, 14).transpose(0, 1)):
            x, tx = view(base), view(reference)
            self.assertTrue(x.is_contiguous())
            for call in FORMS:
                self.compare(call(native, x), call(torch, tx))
                self.compare(x, tx)

    def test_completion_and_independent_output(self):
        values = np.arange(65539, dtype=np.float32) / 16 - 7
        x = upload(native, values, values.shape)
        tx = upload(torch, values, values.shape)
        expected = -tx
        stream = torch.cuda.Stream()
        torch.cuda.synchronize()
        lib = runtime()
        with torch.cuda.stream(stream):
            result = -x
            self.assertEqual(lib.cudaStreamQuery(ctypes.c_void_p(1)), 0)
            copied = torch.empty_like(tx)
            self.assertEqual(lib.cudaMemcpyAsync(copied.data_ptr(), result.data_ptr(), values.nbytes,
                                                3, stream.cuda_stream), 0)
        stream.synchronize()
        torch.testing.assert_close(copied, expected, rtol=0, atol=0)
        # Overwrite the input through the ABI: the result must own fresh storage.
        lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        lib.cudaMemset.restype = ctypes.c_int
        self.assertEqual(lib.cudaMemset(x.data_ptr(), 0, values.nbytes), 0)
        self.compare(result, expected)
        self.assertEqual(x[:4].cpu().tolist(), [0.] * 4)

    def test_lifetimes_reuse_and_fresh_thread_context(self):
        def produce(index):
            base = native.full((1031,), index / 8).to("cuda:0")
            view = base[1:1028]
            del base
            result = -view
            del view
            # Keep a view of the output after its Tensor owner is dropped.
            return result[1:17]

        with ThreadPoolExecutor(max_workers=4) as workers:
            retained = list(workers.map(produce, range(48)))
        gc.collect()
        for index in range(64):
            x = native.full((1031,), -index / 16).to("cuda:0")
            y = -x
            del x
            self.assertEqual((-y)[:4].cpu().tolist(), [-index / 16] * 4)
        for index, view in enumerate(retained):
            self.assertEqual(view.cpu().tolist(), [-index / 8] * 16)
        # This new host thread first consumes a CUDA tensor and a cached allocation.
        with ThreadPoolExecutor(max_workers=1) as worker:
            result = worker.submit(lambda: -retained[-1]).result()
        del retained
        gc.collect()
        self.assertEqual(result.cpu().tolist(), [47 / 8] * 16)

    def test_large_allocation_lifetime_outside_front_cache(self):
        # More than 64 MiB exercises the pool/free path and repeated grid strides.
        x = native.full((17_000_003,), 1.25).to("cuda:0")
        result = -x
        del x
        gc.collect()
        replacement = native.full((17_000_003,), 7.5).to("cuda:0")
        self.assertEqual(result[:3].cpu().tolist(), [-1.25] * 3)
        self.assertEqual(result[-3:].cpu().tolist(), [-1.25] * 3)
        self.assertEqual(replacement[-3:].cpu().tolist(), [7.5] * 3)

    def test_layout_autograd_rejection_and_compiled_negation(self):
        base = native.ones((3, 5)).to("cuda:0")
        for x in (base.t(), base[:, 1:4], base.select(1, 2)):
            for call in FORMS:
                with self.assertRaisesRegex(NotImplementedError, "CUDA negation.*contiguous"):
                    call(native, x)
        for shape in ((), (0,), (3, 5)):
            x = native.ones(shape).to("cuda:0")
            with self.assertRaises(NotImplementedError):
                x.requires_grad_()
            with self.assertRaises(NotImplementedError):
                (-x).backward()
            with self.assertRaises(NotImplementedError):
                native.ones(shape, requires_grad=True).to("cuda:0")
            self.assertFalse(x.requires_grad)
            for fullgraph in (True, False):
                compiled = native.compile(lambda a: -a, backend="eager", fullgraph=fullgraph)
                self.compare(compiled(x), -torch.ones(shape, device="cuda:0"))

    def test_no_pytorch_forwarding(self):
        script = '''
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("native negation imported PyTorch")
sys.meta_path.insert(0, BlockTorch())
import torch_rs as m
x = m.tensor([99., 1.25, -2., 3.5]).to("cuda:0")[1:]
for result in (-x, x.neg(), x.negative(), m.neg(x), m.negative(x)):
    assert result.cpu().tolist() == [-1.25, 2., -3.5]
    assert result.data_ptr() != x.data_ptr()
assert x.cpu().tolist() == [1.25, -2., 3.5]
assert "torch" not in sys.modules
'''
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                   text=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


@unittest.skipUnless(available("0,1"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0,1")
class CudaNegDeviceTests(Comparison, unittest.TestCase):
    def test_device_guard_and_context_cache(self):
        lib = runtime()
        previous = torch.cuda.current_device()
        try:
            for current, target in ((1, 0), (0, 1), (1, 0)):
                torch.cuda.set_device(current)
                for shape in ((), (0,), (3, 5)):
                    x = native.full(shape, 1.25).to(f"cuda:{target}")
                    result = -x
                    ordinal = ctypes.c_int()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
                    self.compare(result, -torch.full(shape, 1.25, device=f"cuda:{target}"))
                    if shape == (3, 5):
                        with self.assertRaisesRegex(NotImplementedError, "contiguous"):
                            -x.t()
                    del x, result
                    gc.collect()
                    self.assertEqual(lib.cudaGetDevice(ctypes.byref(ordinal)), 0)
                    self.assertEqual(ordinal.value, current)
        finally:
            torch.cuda.set_device(previous)
