import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UnsqueezeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("unsqueeze differentials require pinned PyTorch 2.13.0")

    def make_sources(self, module):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = module.tensor(values.tolist())
        return (
            ("scalar", module.tensor(-0.0)),
            ("empty", module.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("noncontiguous", base.transpose(0, 3)[1]),
            ("leading-zero-empty", module.zeros((0, 3))),
        )

    def assert_matches(self, actual, expected, actual_source, expected_source, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.data_ptr(), actual_source.data_ptr())
            self.assertEqual(expected.data_ptr(), expected_source.data_ptr())
            self.assertEqual(
                actual.is_set_to(actual_source), expected.is_set_to(expected_source)
            )
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.detach().cpu().numpy()
            )

    def test_endpoint_shapes_strides_offsets_and_aliasing_match_pytorch_2_13(self):
        for (case, actual_source), (_, expected_source) in zip(
            self.make_sources(torch),
            self.make_sources(reference_torch),
            strict=True,
        ):
            rank = actual_source.ndim
            dims = (0, -(rank + 1), rank, -1)
            for dim in dims:
                self.assert_matches(
                    actual_source.unsqueeze(dim),
                    expected_source.unsqueeze(dim),
                    actual_source,
                    expected_source,
                    (case, "method", dim),
                )
                self.assert_matches(
                    torch.unsqueeze(actual_source, dim),
                    reference_torch.unsqueeze(expected_source, dim),
                    actual_source,
                    expected_source,
                    (case, "top-level", dim),
                )
                self.assert_matches(
                    torch.unsqueeze(input=actual_source, dim=dim),
                    reference_torch.unsqueeze(input=expected_source, dim=dim),
                    actual_source,
                    expected_source,
                    (case, "keywords", dim),
                )

    def test_index_protocol_autograd_empty_and_no_grad_match_pytorch_2_13(self):
        class IndexOnly:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        self.assertEqual(
            torch.zeros((2, 3, 4)).unsqueeze(IndexOnly(-1)).shape,
            reference_torch.zeros((2, 3, 4)).unsqueeze(IndexOnly(-1)).shape,
        )

        values = np.arange(6, dtype=np.float32).reshape(2, 3)
        weights = np.linspace(-2.0, 3.0, num=6, dtype=np.float32).reshape(1, 2, 3)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = torch.unsqueeze(actual_leaf, 0)
        expected = reference_torch.unsqueeze(expected_leaf, 0)
        self.assertEqual(
            (actual.requires_grad, actual.is_leaf),
            (expected.requires_grad, expected.is_leaf),
        )
        self.assert_matches(actual, expected, actual_leaf, expected_leaf, "autograd-view")

        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.unsqueeze(-1).sum().backward()
        expected_empty.unsqueeze(-1).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad),
            expected_empty.grad.detach().cpu().numpy(),
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.unsqueeze(-1)
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.unsqueeze(-1)
        self.assertEqual(
            (actual_untracked.requires_grad, actual_untracked.is_leaf),
            (expected_untracked.requires_grad, expected_untracked.is_leaf),
        )
        self.assert_matches(
            actual_untracked,
            expected_untracked,
            actual_leaf,
            expected_leaf,
            "no-grad-view",
        )


if __name__ == "__main__":
    unittest.main()
