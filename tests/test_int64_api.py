import math
import operator
import sys
import unittest

import numpy as np
import torch_rs as torch


class Int64ApiTests(unittest.TestCase):
    def assert_array_equal(self, tensor, expected, dtype):
        actual = np.asarray(tensor)
        expected = np.asarray(expected, dtype=dtype)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual, expected)

    def test_dtype_descriptors_and_tensor_inference(self):
        self.assertIs(torch.long, torch.int64)
        self.assertIsInstance(torch.int64, torch.dtype)
        self.assertEqual(str(torch.int64), "torch.int64")
        self.assertEqual(repr(torch.long), "torch.int64")

        cases = (
            ([1, -2, 3], torch.int64, [1, -2, 3]),
            ((1, 2.5, 3), torch.float32, [1.0, 2.5, 3.0]),
            ([1.0, 2.0], torch.float32, [1.0, 2.0]),
            ([], torch.float32, []),
            ([[], []], torch.float32, [[], []]),
        )
        for data, dtype, expected in cases:
            with self.subTest(data=data):
                tensor = torch.tensor(data)
                self.assertIs(tensor.dtype, dtype)
                self.assertEqual(tensor.tolist(), expected)

        for data in ([True], [1, False], [[True], [False]]):
            with self.subTest(data=data):
                with self.assertRaisesRegex(RuntimeError, "bool tensor storage"):
                    torch.tensor(data)

        for data in ("12", b"12", bytearray(b"12"), [object()]):
            with self.subTest(data=data):
                with self.assertRaises(TypeError):
                    torch.tensor(data)

    def test_explicit_tensor_dtype_converts_supported_python_values(self):
        integers = torch.tensor([1.9, -2.9, True, False], dtype=torch.int64)
        self.assertIs(integers.dtype, torch.int64)
        self.assertEqual(integers.tolist(), [1, -2, 1, 0])

        floats = torch.tensor([1, -2, True, False], dtype=torch.float32)
        self.assertIs(floats.dtype, torch.float32)
        self.assertEqual(floats.tolist(), [1.0, -2.0, 1.0, 0.0])

        for value in (2**63, -(2**63) - 1, math.inf, -math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises((RuntimeError, OverflowError)):
                    torch.tensor([value], dtype=torch.int64)

    def test_creation_functions_materialize_int64_storage(self):
        cases = (
            (torch.zeros((2, 2), dtype=torch.int64), [[0, 0], [0, 0]]),
            (torch.ones((2, 2), dtype=torch.long), [[1, 1], [1, 1]]),
            (torch.full((2, 2), -7, dtype=torch.int64), [[-7, -7], [-7, -7]]),
            (torch.full((2,), 2.9, dtype=torch.int64), [2, 2]),
        )
        for tensor, expected in cases:
            with self.subTest(expected=expected):
                self.assertIs(tensor.dtype, torch.int64)
                self.assert_array_equal(tensor, expected, np.int64)
                self.assertEqual(tensor.device, torch.device("cpu"))

        empty = torch.ones((2, 0, 3), dtype=torch.int64)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.tolist(), [[], []])
        self.assertEqual(empty.stride(), (3, 3, 1))

        oversized = sys.maxsize // np.dtype(np.int64).itemsize + 1
        with self.assertRaisesRegex(RuntimeError, "exceeds the platform capacity"):
            torch.zeros((oversized,), dtype=torch.int64)

    def test_same_and_mixed_dtype_broadcasting_matches_numpy(self):
        left = torch.tensor([[1], [2]], dtype=torch.int64)
        right = torch.tensor([[10, 20, 30]], dtype=torch.int64)
        floating = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float32)

        for operation in (operator.add, operator.sub, operator.mul):
            with self.subTest(operation=operation, dtype="int64"):
                actual = operation(left, right)
                expected = operation(
                    np.array([[1], [2]], dtype=np.int64),
                    np.array([[10, 20, 30]], dtype=np.int64),
                )
                self.assertIs(actual.dtype, torch.int64)
                self.assert_array_equal(actual, expected, np.int64)

            with self.subTest(operation=operation, dtype="mixed"):
                actual = operation(left, floating)
                expected = operation(
                    np.array([[1], [2]], dtype=np.float32),
                    np.array([[0.5, 1.5, 2.5]], dtype=np.float32),
                )
                self.assertIs(actual.dtype, torch.float32)
                self.assert_array_equal(actual, expected, np.float32)

        divided = left / right
        self.assertIs(divided.dtype, torch.float32)
        expected = np.array([[1], [2]], dtype=np.float32) / np.array(
            [[10, 20, 30]], dtype=np.float32
        )
        self.assert_array_equal(divided, expected, np.float32)

    def test_integer_and_floating_scalars_promote_like_tensors(self):
        tensor = torch.tensor([-2, 0, 4])
        integer_cases = (
            (tensor + 3, [1, 3, 7]),
            (3 + tensor, [1, 3, 7]),
            (tensor - 3, [-5, -3, 1]),
            (3 - tensor, [5, 3, -1]),
            (tensor * 3, [-6, 0, 12]),
        )
        for actual, expected in integer_cases:
            self.assertIs(actual.dtype, torch.int64)
            self.assertEqual(actual.tolist(), expected)

        for actual, expected in (
            (tensor + 0.5, [-1.5, 0.5, 4.5]),
            (tensor * np.float32(0.5), [-1.0, 0.0, 2.0]),
            (tensor / 2, [-1.0, 0.0, 2.0]),
            (2 / tensor, [-1.0, math.inf, 0.5]),
        ):
            self.assertIs(actual.dtype, torch.float32)
            np.testing.assert_allclose(
                np.asarray(actual), np.asarray(expected, dtype=np.float32), equal_nan=True
            )

    def test_integer_overflow_wraps_in_native_kernels(self):
        maximum = torch.tensor([2**63 - 1])
        minimum = torch.tensor([-(2**63)])
        self.assertEqual((maximum + 1).tolist(), [-(2**63)])
        self.assertEqual((minimum - 1).tolist(), [2**63 - 1])
        self.assertEqual((maximum * 2).tolist(), [-2])
        self.assertEqual(torch.tensor([2**63 - 1, 1]).sum().item(), -(2**63))

        with self.assertRaises(RuntimeError):
            maximum + 2**63

    def test_relu_sum_views_and_items_keep_int64(self):
        source = torch.tensor([[-3, 0], [4, 5]])
        view = source.reshape(4)
        relu = view.relu()
        reduction = view.sum()

        for tensor in (source, view, relu, reduction):
            self.assertIs(tensor.dtype, torch.int64)
        self.assertEqual(view.tolist(), [-3, 0, 4, 5])
        self.assertEqual(relu.tolist(), [0, 0, 4, 5])
        self.assertEqual(reduction.item(), 6)
        self.assertIs(type(reduction.item()), int)
        self.assertIs(type(torch.tensor(1.5).item()), float)

    def test_rank_two_matmul_handles_integer_mixed_and_empty_inputs(self):
        left = torch.tensor([[1, 2, 3], [4, 5, 6]])
        right = torch.tensor([[7, 8], [9, 10], [11, 12]])
        integer = left @ right
        self.assertIs(integer.dtype, torch.int64)
        self.assertEqual(integer.tolist(), [[58, 64], [139, 154]])

        mixed_right = torch.tensor([[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]])
        mixed = left @ mixed_right
        self.assertIs(mixed.dtype, torch.float32)
        self.assert_array_equal(mixed, [[11.0, 14.0], [24.5, 32.0]], np.float32)

        empty = torch.zeros((2, 0), dtype=torch.int64) @ torch.ones(
            (0, 3), dtype=torch.int64
        )
        self.assertIs(empty.dtype, torch.int64)
        self.assertEqual(empty.tolist(), [[0, 0, 0], [0, 0, 0]])

        overflow = torch.tensor([[2**63 - 1]]) @ torch.tensor([[2]])
        self.assertEqual(overflow.item(), -2)

    def test_numpy_tolist_and_repr_reflect_physical_dtype(self):
        tensor = torch.tensor([[1, 2], [3, 4]])
        array = np.asarray(tensor)
        self.assertEqual(array.dtype, np.dtype(np.int64))
        self.assertEqual(array.tolist(), tensor.tolist())
        array[0, 0] = 99
        self.assertEqual(tensor.tolist()[0][0], 1)

        converted = np.asarray(tensor, dtype=np.float32)
        self.assertEqual(converted.dtype, np.dtype(np.float32))
        self.assertIn("dtype=torch.int64", repr(tensor))
        self.assertNotIn("1.0", repr(tensor))


if __name__ == "__main__":
    unittest.main()
