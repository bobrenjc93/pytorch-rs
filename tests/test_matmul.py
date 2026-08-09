import unittest

import numpy as np
import torch_rs as torch


class MatmulTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, shape):
        self.assertEqual(actual.shape, shape)
        expected_stride = []
        stride = 1
        for dimension in reversed(shape):
            expected_stride.insert(0, stride)
            stride *= max(dimension, 1)
        self.assertEqual(actual.stride(), tuple(expected_stride))
        self.assertEqual(actual.storage_offset(), 0)
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected, dtype=np.float32).reshape(shape),
            rtol=1.0e-6,
            atol=1.0e-6,
            equal_nan=True,
        )

    def test_all_rank_one_and_rank_two_combinations(self):
        vector = torch.tensor([1.0, 2.0, 3.0])
        other_vector = torch.tensor([4.0, 5.0, 6.0])
        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        self.assert_tensor(vector @ other_vector, 32.0, ())
        self.assert_tensor(matrix @ other_vector, [32.0, 77.0], (2,))
        self.assert_tensor(vector @ right_matrix, [22.0, 28.0], (2,))
        self.assert_tensor(matrix @ right_matrix, [[22.0, 28.0], [49.0, 64.0]], (2, 2))

    def test_empty_dimensions_have_pytorch_shapes_and_positive_zero(self):
        cases = (
            (torch.zeros((0,)), torch.zeros((0,)), (), 0.0),
            (torch.zeros((2, 0)), torch.zeros((0,)), (2,), [0.0, 0.0]),
            (torch.zeros((0,)), torch.zeros((0, 3)), (3,), [0.0, 0.0, 0.0]),
            (torch.zeros((0, 3)), torch.zeros((3,)), (0,), []),
            (torch.zeros((2, 0)), torch.zeros((0, 0)), (2, 0), []),
        )
        for left, right, shape, expected in cases:
            with self.subTest(left=left.shape, right=right.shape):
                output = left @ right
                self.assert_tensor(output, expected, shape)
                if shape == ():
                    self.assertEqual(
                        np.float32(output.item()).view(np.uint32).item(),
                        np.float32(0.0).view(np.uint32).item(),
                    )

    def test_non_contiguous_and_offset_views_use_logical_values(self):
        vectors = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]).T
        left_vector = vectors[0]
        right_vector = vectors[1]
        self.assertEqual(left_vector.stride(), (2,))
        self.assertEqual(right_vector.storage_offset(), 1)
        self.assert_tensor(left_vector @ right_vector, 140.0, ())

        left_matrix = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).T
        self.assert_tensor(left_matrix @ right_vector, [140.0, 320.0], (2,))

        right_matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).T
        short_vector = torch.tensor([[2.0, 99.0], [3.0, 99.0]]).T[0]
        self.assertEqual(short_vector.stride(), (2,))
        self.assert_tensor(short_vector @ right_matrix, [8.0, 18.0], (2,))

    def test_special_values_and_cancellation_use_float32_accumulation(self):
        cancellation = torch.tensor([1.0e20, 1.0, -1.0e20]) @ torch.ones((3,))
        self.assertEqual(cancellation.item(), 0.0)

        signed_zero = torch.tensor([-0.0]) @ torch.ones((1,))
        self.assertEqual(
            np.float32(signed_zero.item()).view(np.uint32).item(),
            np.float32(0.0).view(np.uint32).item(),
        )

        for left, right in (
            ([np.inf], [0.0]),
            ([np.nan], [1.0]),
            ([np.inf, -np.inf], [1.0, 1.0]),
        ):
            with self.subTest(left=left, right=right):
                self.assertTrue(np.isnan((torch.tensor(left) @ torch.tensor(right)).item()))

    def test_rank_and_mismatch_errors_match_pytorch_messages(self):
        cases = (
            ((), (2,), "both arguments to matmul need to be at least 1D, but they are 0D and 1D"),
            ((2,), (), "both arguments to matmul need to be at least 1D, but they are 1D and 0D"),
            ((2,), (3,), "inconsistent tensor size, expected tensor [2] and src [3] to have the same number of elements, but got 2 and 3 elements respectively"),
            ((2, 3), (4,), "size mismatch, got input (2), mat (2x3), vec (4)"),
            ((3,), (4, 2), "mat1 and mat2 shapes cannot be multiplied (1x3 and 4x2)"),
            ((2, 3), (4, 2), "mat1 and mat2 shapes cannot be multiplied (2x3 and 4x2)"),
        )
        for left_shape, right_shape, message in cases:
            with self.subTest(left=left_shape, right=right_shape):
                with self.assertRaises(RuntimeError) as raised:
                    torch.zeros(left_shape) @ torch.zeros(right_shape)
                self.assertEqual(str(raised.exception), message)
