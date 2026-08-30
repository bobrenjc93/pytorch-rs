import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class Rank7LogicalIterationReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "rank-7 logical-iteration differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=True):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_values = self.tensor_values(actual).reshape(-1)
            expected_values = self.tensor_values(expected).reshape(-1)
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.view(np.uint32),
                    expected_values.view(np.uint32),
                )
            else:
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    def assert_scalar_matches(self, actual, expected, *, case, exact_bits=True):
        self.assert_tensor_matches(actual, expected, case=case, exact_bits=exact_bits)
        self.assertEqual(actual.shape, ())
        self.assertEqual(expected.shape, ())

    @staticmethod
    def make_rank7_cases(module):
        offset_values = np.arange(1920, dtype=np.float32).reshape(
            2, 2, 3, 4, 5, 2, 2, 2
        )
        offset = module.tensor(
            offset_values.tolist(),
            dtype=module.float32,
        )[1].permute(3, 1, 5, 0, 4, 2, 6)

        singleton_values = np.linspace(0.0, 31.0, num=48, dtype=np.float32).reshape(
            2, 1, 3, 2, 1, 2, 2
        )
        singleton = module.tensor(
            singleton_values.tolist(),
            dtype=module.float32,
        ).permute(2, 0, 3, 5, 4, 6, 1)

        empty = module.zeros((2, 0, 3, 4, 5, 2, 2), dtype=module.float32).permute(
            4, 2, 0, 6, 5, 3, 1
        )

        return (
            ("offset permutation", offset),
            ("singleton permutation", singleton),
            ("empty permutation", empty),
        )

    def test_rank7_values_contiguous_and_unary_match_pytorch_2_13(self):
        actual_cases = self.make_rank7_cases(torch)
        expected_cases = self.make_rank7_cases(reference_torch)

        for (case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            self.assert_tensor_matches(actual, expected, case=(case, "view"))
            self.assert_scalar_matches(
                actual.sum(), expected.sum(), case=(case, "sum"), exact_bits=False
            )
            self.assert_tensor_matches(
                actual.contiguous(),
                expected.contiguous(),
                case=(case, "contiguous"),
            )
            self.assert_tensor_matches(
                actual.negative(),
                expected.negative(),
                case=(case, "negative"),
            )
            self.assert_tensor_matches(
                actual.sqrt(),
                expected.sqrt(),
                case=(case, "sqrt"),
                exact_bits=False,
            )

    def test_rank7_view_autograd_matches_pytorch_2_13(self):
        values = np.linspace(1.0, 7.0, num=1920, dtype=np.float32).reshape(
            2, 2, 3, 4, 5, 2, 2, 2
        )
        weights = np.linspace(-3.0, 4.0, num=960, dtype=np.float32).reshape(
            5, 3, 2, 2, 2, 4, 2
        )

        for operation in ("contiguous", "negative"):
            actual_leaf = torch.tensor(
                values.tolist(), dtype=torch.float32, requires_grad=True
            )
            expected_leaf = reference_torch.tensor(
                values,
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            actual_weights = torch.tensor(weights.tolist(), dtype=torch.float32)
            expected_weights = reference_torch.tensor(
                weights, dtype=reference_torch.float32
            )

            actual_view = actual_leaf[1].permute(3, 1, 5, 0, 4, 2, 6)
            expected_view = expected_leaf[1].permute(3, 1, 5, 0, 4, 2, 6)
            actual_output = getattr(actual_view, operation)()
            expected_output = getattr(expected_view, operation)()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(operation, "forward")
            )

            (actual_output * actual_weights).sum().backward()
            (expected_output * expected_weights).sum().backward()
            self.assert_tensor_matches(
                actual_leaf.grad,
                expected_leaf.grad,
                case=(operation, "leaf grad"),
            )

    def test_rank7_no_grad_boundaries_match_pytorch_2_13(self):
        values = np.linspace(1.0, 7.0, num=1920, dtype=np.float32).reshape(
            2, 2, 3, 4, 5, 2, 2, 2
        )
        actual_leaf = torch.tensor(values.tolist(), dtype=torch.float32, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_view = actual_leaf[1].permute(3, 1, 5, 0, 4, 2, 6)
        expected_view = expected_leaf[1].permute(3, 1, 5, 0, 4, 2, 6)

        with torch.no_grad():
            actual_outputs = (actual_view.contiguous(), actual_view.negative())
        with reference_torch.no_grad():
            expected_outputs = (expected_view.contiguous(), expected_view.negative())

        for operation, actual, expected in zip(
            ("contiguous", "negative"),
            actual_outputs,
            expected_outputs,
            strict=True,
        ):
            self.assert_tensor_matches(actual, expected, case=operation)


if __name__ == "__main__":
    unittest.main()
