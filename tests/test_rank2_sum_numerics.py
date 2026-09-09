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

    def test_outer_reduction_vector_remainders(self):
        rng = np.random.default_rng(251)
        for prefix in (0, 32, 64):
            for tail in range(1, 32):
                width = prefix + tail
                for count in (7, 16, 67, 513):
                    patterns = [np.resize([-1e8, 1., 1., 1., 1., 1e8, 1.], count),
                                rng.choice([1e8, -1e8, 1.], count),
                                rng.choice([3e38, -3e38], count)]
                    for pattern in patterns:
                        values = np.repeat(np.asarray(pattern, dtype=np.float32)[:, None], width + 2, axis=1)
                        a, b = native.tensor(values.tolist()), torch.tensor(values)
                        for offset in (False, True):
                            av, bv = (a[:, 1:-1], b[:, 1:-1]) if offset else (a[:, :width], b[:, :width])
                            for transpose in (False, True):
                                aa, bb = (av.t(), bv.t()) if transpose else (av, bv)
                                dim = int(transpose)
                                for keepdim in (False, True):
                                    with self.subTest(prefix=prefix, tail=tail, count=count,
                                                      offset=offset, transpose=transpose, keepdim=keepdim):
                                        expected = bb.sum(dim, keepdim=keepdim).tolist()
                                        np.testing.assert_equal(aa.sum(dim, keepdim=keepdim).tolist(), expected)
                                        np.testing.assert_equal(native.sum(aa, dim, keepdim=keepdim).tolist(), expected)

    def test_selected_views_with_two_nonunit_strides(self):
        rng = np.random.default_rng(1894)
        patterns = ([1e8, 1, -1e8, 1], [3e38, 3e38, -3e38, -3e38])
        for count in (5, 16, 31, 127, 512, 8193):
            for width in (4, 5, 8, 33):
                for pattern in patterns:
                    values = rng.choice(np.array(pattern, dtype=np.float32),
                                        size=(count, width, 3))
                    a, b = native.tensor(values.tolist()), torch.tensor(values)
                    for transpose in (False, True):
                        av, bv = a.select(2, 1), b.select(2, 1)
                        if transpose:
                            av, bv = av.t(), bv.t()
                        for dim in (0, 1):
                            for keepdim in (False, True):
                                with self.subTest(count=count, width=width, pattern=pattern,
                                                  transpose=transpose, dim=dim, keepdim=keepdim):
                                    expected = bv.sum(dim, keepdim=keepdim).tolist()
                                    np.testing.assert_equal(av.sum(dim, keepdim=keepdim).tolist(), expected)
                                    np.testing.assert_equal(native.sum(av, dim, keepdim=keepdim).tolist(), expected)

        for dim in (0, 1):
            values = rng.normal(size=(13, 37, 3)).astype(np.float32)
            a = native.tensor(values.tolist(), requires_grad=True)
            b = torch.tensor(values, requires_grad=True)
            av, bv = a.select(2, 1).t(), b.select(2, 1).t()
            weights = rng.normal(size=av.shape[1 - dim]).astype(np.float32)
            (av.sum(dim) * native.tensor(weights.tolist())).sum().backward()
            (bv.sum(dim) * torch.tensor(weights)).sum().backward()
            np.testing.assert_equal(a.grad.tolist(), b.grad.tolist())

    def test_long_outer_vector_tails(self):
        rng = np.random.default_rng(8193251)
        for count in (4097, 65537):
            for width in (8, 15, 40, 63):
                values = rng.choice(np.array([-1e8, 1., 1e8], dtype=np.float32),
                                    size=(count, width))
                a, b = native.tensor(values.tolist()), torch.tensor(values)
                for av, bv, dim in ((a, b, 0), (a.t(), b.t(), 1)):
                    with self.subTest(count=count, width=width, dim=dim):
                        np.testing.assert_equal(av.sum(dim).tolist(), bv.sum(dim).tolist())
                        np.testing.assert_equal(native.sum(av, dim, keepdim=True).tolist(),
                                                torch.sum(bv, dim, keepdim=True).tolist())
