import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MatmulReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.cpu().numpy(),
                rtol=3.0e-5,
                atol=2.0e-5,
                equal_nan=True,
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def pair(self, values, shape):
        array = np.asarray(values, dtype=np.float32).reshape(shape)
        return (
            torch.tensor(array.item() if not shape else array.tolist()),
            reference_torch.tensor(array, dtype=reference_torch.float32),
        )

    def test_seeded_ordinary_and_empty_rank_combinations_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x6D61746D756C_213)
        rank_pairs = ((1, 1), (2, 1), (1, 2), (2, 2))
        for case in range(48):
            left_rank, right_rank = rank_pairs[case % len(rank_pairs)]
            rows, inner, columns = (int(value) for value in rng.integers(0, 9, size=3))
            left_shape = (inner,) if left_rank == 1 else (rows, inner)
            right_shape = (inner,) if right_rank == 1 else (inner, columns)
            left_values = rng.normal(size=left_shape).astype(np.float32)
            right_values = rng.normal(size=right_shape).astype(np.float32)
            if left_values.size:
                actual_left = torch.tensor(left_values.tolist())
                expected_left = reference_torch.tensor(left_values)
            else:
                actual_left = torch.zeros(left_shape)
                expected_left = reference_torch.zeros(left_shape)
            if right_values.size:
                actual_right = torch.tensor(right_values.tolist())
                expected_right = reference_torch.tensor(right_values)
            else:
                actual_right = torch.zeros(right_shape)
                expected_right = reference_torch.zeros(right_shape)

            self.assert_matches(
                actual_left @ actual_right,
                expected_left @ expected_right,
                case=(case, left_shape, right_shape),
            )

    def test_special_values_cancellation_and_views_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        special_cases = (
            ([np.inf], [0.0]),
            ([np.nan], [1.0]),
            ([np.inf, -np.inf], [1.0, 1.0]),
            ([-0.0], [1.0]),
        )
        for case, (left, right) in enumerate(special_cases):
            actual_left, expected_left = self.pair(left, (len(left),))
            actual_right, expected_right = self.pair(right, (len(right),))
            self.assert_matches(
                actual_left @ actual_right,
                expected_left @ expected_right,
                case=("special", case),
            )

        cancellation = np.tile(np.array([1000.25, -1000.0], dtype=np.float32), 33)[:65]
        actual_vector, expected_vector = self.pair(cancellation, (65,))
        actual_ones = torch.ones((65,))
        expected_ones = reference_torch.ones((65,), dtype=reference_torch.float32)
        self.assert_matches(
            actual_vector @ actual_ones,
            expected_vector @ expected_ones,
            case="cancellation",
        )

        actual_vectors = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]).T
        expected_vectors = reference_torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
        ).T
        actual_matrix = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).T
        expected_matrix = reference_torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).T
        for case, (actual_left, actual_right, expected_left, expected_right) in enumerate(
            (
                (actual_vectors[0], actual_vectors[1], expected_vectors[0], expected_vectors[1]),
                (actual_matrix, actual_vectors[1], expected_matrix, expected_vectors[1]),
                (actual_vectors[0], actual_matrix.T, expected_vectors[0], expected_matrix.T),
            )
        ):
            self.assert_matches(
                actual_left @ actual_right,
                expected_left @ expected_right,
                case=("view", case),
            )

    def test_rank_and_inner_dimension_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        cases = (
            ((), ()),
            ((), (2,)),
            ((2,), ()),
            ((2,), (3,)),
            ((2, 3), (4,)),
            ((3,), (4, 2)),
            ((2, 3), (4, 2)),
            ((0,), (1,)),
            ((2, 0), (1,)),
            ((0,), (1, 2)),
        )
        for left_shape, right_shape in cases:
            self.assert_error_matches(
                lambda left_shape=left_shape, right_shape=right_shape: torch.zeros(left_shape)
                @ torch.zeros(right_shape),
                lambda left_shape=left_shape, right_shape=right_shape: reference_torch.zeros(
                    left_shape
                )
                @ reference_torch.zeros(right_shape),
            )
