"""Cancellation, overflow, layout and backward regressions for dimension sums."""
import unittest
import numpy as np
import torch_rs as native
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "install the reference dependency group")
class RankTwoSumNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_cancellation_overflow_and_long_axes(self):
        patterns = ([1e8, 1, -1e8, 1], [3e38, 3e38, -3e38, -3e38],
                    [1, -1e8, 1, 1e8], [float("inf"), 1, -float("inf"), 1])
        for length in (4, 5, 7, 8, 16, 31, 32, 127, 512, 1024, 8193, 65536):
            for width in (1, 3, 5, 33):
                for pattern in patterns:
                    values = np.resize(np.array(pattern, dtype=np.float32), (length, width))
                    for transpose in (False, True):
                        a, b = native.tensor(values.tolist()), torch.tensor(values)
                        if transpose:
                            a, b = a.t(), b.t()
                        for dim in (0, 1):
                            for keepdim in (False, True):
                                with self.subTest(length=length, width=width, pattern=pattern,
                                                  transpose=transpose, dim=dim, keepdim=keepdim):
                                    for actual in (a.sum(dim, keepdim=keepdim),
                                                   native.sum(a, dim, keepdim=keepdim)):
                                        expected = b.sum(dim, keepdim=keepdim)
                                        np.testing.assert_equal(actual.tolist(), expected.tolist())

    def test_offset_views_and_weighted_backward(self):
        rng = np.random.default_rng(819)
        for dim in (0, 1):
            a = native.tensor(rng.normal(size=(13, 257)).astype(np.float32).tolist(), requires_grad=True)
            b = torch.tensor(a.tolist(), requires_grad=True)
            av, bv = a.narrow(1, 3, 249).t(), b.narrow(1, 3, 249).t()
            ar, br = av.sum(dim), bv.sum(dim)
            weights = np.linspace(-2, 3, len(ar), dtype=np.float32)
            np.testing.assert_allclose(ar.tolist(), br.tolist(), rtol=1e-5, atol=1e-5)
            (ar * native.tensor(weights.tolist())).sum().backward()
            (br * torch.tensor(weights)).sum().backward()
            np.testing.assert_equal(a.grad.tolist(), b.grad.tolist())
