import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UnaryNegationReferenceTests(unittest.TestCase):
    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_matches(self, actual, expected, *, case):
        self.assert_metadata_matches(actual, expected, case=case)
        with self.subTest(case=case):
            actual_bits = np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32)
            expected_bits = (
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            )
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_values_layouts_signed_zero_and_non_finites_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_cases = [("scalar", torch.tensor(2.5))]
        expected_cases = [("scalar", reference_torch.tensor(2.5))]
        for shape in ((0,), (1, 0), (0, 1), (1, 0, 1), (2, 0, 3)):
            actual_cases.append((shape, torch.zeros(shape)))
            expected_cases.append((shape, reference_torch.zeros(shape)))

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist())
        expected_base = reference_torch.tensor(values)
        actual_dense_view = actual_base.transpose(0, 2)
        expected_dense_view = expected_base.transpose(0, 2)
        actual_cases.extend(
            (
                ("transposed dense view", actual_dense_view),
                ("offset non-dense view", actual_dense_view[1]),
            )
        )
        expected_cases.extend(
            (
                ("transposed dense view", expected_dense_view),
                ("offset non-dense view", expected_dense_view[1]),
            )
        )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        special_values = memoryview(special_bits.view(np.float32))
        actual_cases.append(
            ("signed zeros and non-finites", torch.tensor(special_values))
        )
        expected_cases.append(
            (
                "signed zeros and non-finites",
                reference_torch.tensor(special_values),
            )
        )

        for (actual_case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases
        ):
            self.assertEqual(actual_case, expected_case)
            self.assert_matches(-actual, -expected, case=actual_case)

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        weights = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_weights = torch.tensor(weights)
        expected_weights = reference_torch.tensor(weights)

        actual_output = -actual_leaf.transpose(0, 1)
        expected_output = -expected_leaf.transpose(0, 1)
        self.assert_matches(actual_output, expected_output, case="tracked view")
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_leaf.grad.numpy().view(np.uint32),
        )

        actual_empty = torch.tensor([[]], requires_grad=True)
        expected_empty = reference_torch.tensor([[]], requires_grad=True)
        actual_empty_output = -actual_empty
        expected_empty_output = -expected_empty
        self.assert_matches(
            actual_empty_output,
            expected_empty_output,
            case="tracked empty",
        )
        actual_empty_output.sum().backward()
        expected_empty_output.sum().backward()
        self.assert_matches(
            actual_empty.grad,
            expected_empty.grad,
            case="empty gradient",
        )

        with torch.no_grad():
            actual_untracked = -actual_leaf.transpose(0, 1)
        with reference_torch.no_grad():
            expected_untracked = -expected_leaf.transpose(0, 1)
        self.assert_matches(
            actual_untracked,
            expected_untracked,
            case="no_grad view",
        )
        self.assertTrue((-actual_leaf).requires_grad)
        self.assertTrue((-expected_leaf).requires_grad)

    def test_extreme_empty_layouts_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for shape in (
            (0, sys.maxsize, 3),
            (0, 2, sys.maxsize, sys.maxsize),
            (0, 1, 2, 1 << 61),
            (2, 0, sys.maxsize),
        ):
            actual = torch.zeros((0,)).reshape(shape)
            expected = reference_torch.zeros((0,)).reshape(shape)
            with self.subTest(shape=shape):
                try:
                    expected_output = -expected
                except Exception as expected_error:
                    with self.assertRaises(type(expected_error)) as actual_raised:
                        -actual
                    self.assertEqual(
                        str(actual_raised.exception), str(expected_error)
                    )
                else:
                    actual_output = -actual
                    self.assert_metadata_matches(
                        actual_output, expected_output, case=shape
                    )
                    self.assertEqual(
                        actual_output.tolist(), expected_output.tolist()
                    )


if __name__ == "__main__":
    unittest.main()
