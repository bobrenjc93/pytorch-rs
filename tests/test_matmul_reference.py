import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MatmulReferenceTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

    def actual_tensor(self, values):
        if values.size == 0:
            return torch.zeros(values.shape)
        return torch.tensor(values.tolist())

    def reference_tensor(self, values):
        return reference_torch.tensor(values, dtype=reference_torch.float32)

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.cpu().numpy(),
                rtol=2.0e-5,
                atol=2.0e-5,
                equal_nan=True,
            )

    def assert_error_matches(self, actual_call, expected_call, *, case):
        with self.subTest(case=case):
            with self.assertRaises(Exception) as actual_raised:
                actual_call()
            with self.assertRaises(Exception) as expected_raised:
                expected_call()
            self.assertEqual(
                type(actual_raised.exception).__name__,
                type(expected_raised.exception).__name__,
            )
            self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_ordinary_and_broadcasted_batches_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xBA7C_4213)
        shapes = (
            ((2, 3), (3, 4)),
            ((3, 2, 4), (3, 4, 5)),
            ((2, 1, 3, 4), (1, 5, 4, 2)),
            ((3, 4), (2, 1, 4, 5)),
            ((2, 3, 4), (4, 5)),
            ((1, 2, 1, 3, 4), (1, 3, 4, 2)),
        )
        for case, (left_shape, right_shape) in enumerate(shapes):
            left_values = rng.normal(size=left_shape).astype(np.float32)
            right_values = rng.normal(size=right_shape).astype(np.float32)
            actual_left = self.actual_tensor(left_values)
            actual_right = self.actual_tensor(right_values)
            expected_left = self.reference_tensor(left_values)
            expected_right = self.reference_tensor(right_values)

            self.assert_matches(
                actual_left @ actual_right,
                expected_left @ expected_right,
                case=case,
            )

    def test_transposed_and_indexed_views_match_pytorch_2_13(self):
        left_values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        right_values = np.arange(120, dtype=np.float32).reshape(2, 5, 4, 3)
        actual_left = self.actual_tensor(left_values)[1].transpose(0, 1)
        actual_right = self.actual_tensor(right_values)[1].transpose(0, 2)
        expected_left = self.reference_tensor(left_values)[1].transpose(0, 1)
        expected_right = self.reference_tensor(right_values)[1].transpose(0, 2)
        self.assertGreater(actual_left.storage_offset(), 0)
        self.assertGreater(actual_right.storage_offset(), 0)
        self.assert_matches(
            actual_left @ actual_right,
            expected_left @ expected_right,
            case="indexed matrix and batch transposes",
        )

        left_values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        right_values = np.arange(144, dtype=np.float32).reshape(2, 3, 6, 4)
        actual_left = self.actual_tensor(left_values).transpose(0, 1).mT
        actual_right = self.actual_tensor(right_values).transpose(0, 1).mT
        expected_left = self.reference_tensor(left_values).transpose(0, 1).mT
        expected_right = self.reference_tensor(right_values).transpose(0, 1).mT
        self.assert_matches(
            actual_left @ actual_right,
            expected_left @ expected_right,
            case="transposed batches and matrices",
        )

        left_values = np.arange(24, dtype=np.float32).reshape(2, 4, 3, 1)
        right_values = np.arange(60, dtype=np.float32).reshape(1, 4, 5, 3)
        actual_left = self.actual_tensor(left_values).mT
        actual_right = self.actual_tensor(right_values).mT
        expected_left = self.reference_tensor(left_values).mT
        expected_right = self.reference_tensor(right_values).mT
        self.assert_matches(
            actual_left @ actual_right,
            expected_left @ expected_right,
            case="broadcast transposed matrix views",
        )

    def test_empty_batch_row_inner_and_column_dimensions_match_pytorch_2_13(self):
        shapes = (
            ((0, 2, 3), (1, 3, 4)),
            ((2, 0, 3), (1, 3, 4)),
            ((2, 3, 0), (1, 0, 4)),
            ((2, 3, 4), (1, 4, 0)),
            ((1, 2, 3), (0, 3, 4)),
            ((0, 3, 0), (0, 2)),
        )
        for case, (left_shape, right_shape) in enumerate(shapes):
            self.assert_matches(
                torch.zeros(left_shape) @ torch.zeros(right_shape),
                reference_torch.zeros(left_shape) @ reference_torch.zeros(right_shape),
                case=case,
            )

    def test_invalid_batch_inner_and_scalar_shapes_match_pytorch_2_13(self):
        shapes = (
            ((2, 3, 4), (3, 4, 5)),
            ((2, 2, 3, 4), (3, 4, 5)),
            ((2, 3), (4, 2)),
            ((2, 3, 4), (2, 5, 6)),
            ((3, 4), (2, 5, 6)),
            ((2, 3, 4), (5, 6)),
        )
        for case, (left_shape, right_shape) in enumerate(shapes):
            self.assert_error_matches(
                lambda left_shape=left_shape, right_shape=right_shape: torch.zeros(left_shape)
                @ torch.zeros(right_shape),
                lambda left_shape=left_shape, right_shape=right_shape: reference_torch.zeros(
                    left_shape
                )
                @ reference_torch.zeros(right_shape),
                case=case,
            )

        self.assert_error_matches(
            lambda: torch.tensor(1.0) @ torch.zeros((2, 2)),
            lambda: reference_torch.tensor(1.0) @ reference_torch.zeros((2, 2)),
            case="scalar",
        )


if __name__ == "__main__":
    unittest.main()
