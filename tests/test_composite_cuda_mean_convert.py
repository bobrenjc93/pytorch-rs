"""The composite's data helpers and CPU reductions compose with bounded CUDA add."""
import unittest

import torch_rs as native
from tests.test_cuda_add import Comparison, available, torch


class MeanConvertTests(unittest.TestCase):
    def test_convert_preserves_mean_autograd_and_python_scalars(self):
        x = native.tensor([[1., 3.], [5., 7.]], requires_grad=True)
        count = 2**100
        data = native.utils.data.default_convert({"x": x, "meta": (count, None, -0.0, 1j)})
        self.assertIs(data["x"], x)
        self.assertIs(data["meta"][0], count)
        self.assertIsNone(data["meta"][1])
        data["x"].mean(dim=-1).sum().backward()
        self.assertEqual(x.grad.tolist(), [[0.5, 0.5], [0.5, 0.5]])
        with self.assertRaises(TypeError):
            native.utils.data.default_collate([count, count])


@unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
class CudaMeanConvertTests(Comparison, unittest.TestCase):
    def test_cpu_reduction_upload_add_convert_and_readback(self):
        for shape in [(7, 13), (0, 13), (7, 0)]:
            with self.subTest(shape=shape):
                x = native.full(shape, 1.25)
                tx = torch.full(shape, 1.25)
                for dim in (0, 1, -1, -2):
                    reduced = x.mean(dim=dim, keepdim=True)
                    expected = tx.mean(dim=dim, keepdim=True).to("cuda:0")
                    data = native.utils.data.default_convert({"tensor": reduced.to("cuda:0"), "count": 7, "none": None})
                    result = native.add(data["tensor"], data["tensor"])
                    self.compare(result, expected + expected)
                    for call in (lambda: result.mean(), lambda: result.mean(dim=dim),
                                 lambda: native.mean(result, dim=dim), lambda: result.requires_grad_()):
                        with self.assertRaisesRegex(NotImplementedError, "CUDA|cuda|CPU"):
                            call()
                    with self.assertRaises(NotImplementedError):
                        native.utils.data.default_collate([result, result])
                    self.assertEqual(data["count"], 7)
                    self.assertIsNone(data["none"])

    def test_offset_views_retained_across_large_output_chains(self):
        # Exceeds the original 64 MiB cache limit; retain an alias while dropping
        # its base and intermediate outputs before a CPU read.
        n = 17 * 1024 * 1024 + 3
        base = native.full((n + 5,), 0.25).to("cuda:0")
        x = base[5:]
        keep = (x + x)[11:24]
        del base
        for _ in range(8):
            x = x + x
        self.assertEqual(keep.cpu().tolist(), [0.5] * 13)
        self.assertEqual(x[19:32].cpu().tolist(), [64.] * 13)

    def test_mixed_capacity_reuse_across_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        def worker(index):
            aliases = []
            for step in range(48):
                # Descending nearby sizes exercise excess allocation capacity;
                # larger jumps and held views pressure eviction and ownership.
                n = 16381 - (step % 16) * 173 + index * 19
                x = native.full((n,), float(index + 1)).to("cuda:0")
                out = x + x
                if step % 13 == 0:
                    aliases.append(out[7:20])
                del x, out
                zero = native.zeros((n - 3,), device="cuda:0")
                self.assertEqual(zero[-7:].cpu().tolist(), [0.] * 7)
            return aliases

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(worker, range(4)))
        for index, aliases in enumerate(results):
            for view in aliases:
                self.assertEqual(view.cpu().tolist(), [float(2 * (index + 1))] * 13)
