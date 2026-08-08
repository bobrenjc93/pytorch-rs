import math
import operator
import sys
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch


class PythonApiBaselineTests(unittest.TestCase):
    def assert_tensor_values(self, actual, expected, shape):
        self.assertEqual(actual.shape, shape)
        actual_values = np.asarray(actual.tolist(), dtype=np.float32).reshape(-1)
        expected_values = np.asarray(expected, dtype=np.float32).reshape(-1)
        self.assertEqual(actual_values.size, expected_values.size)
        for actual_value, expected_value in zip(actual_values, expected_values):
            if np.isnan(expected_value):
                self.assertTrue(np.isnan(actual_value))
            else:
                actual_bits = actual_value.view(np.uint32).item()
                expected_bits = expected_value.view(np.uint32).item()
                self.assertEqual(actual_bits, expected_bits)

    def test_readme_style_tensor_expression(self):
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        y = torch.ones([2, 2])
        result = (x + y).relu()

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[0.0, 3.0], [4.0, 0.0]])

    def test_scalar_reduction_and_item(self):
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).sum()
        self.assertEqual(value.shape, ())
        self.assertEqual(value.item(), 10.0)

    def test_matrix_multiplication_operator(self):
        left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        output = left @ right
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(output.tolist(), [[58.0, 64.0], [139.0, 154.0]])

    def test_binary_arithmetic_broadcasts_trailing_dimensions(self):
        left = torch.tensor([[[1.0, 2.0, 4.0]], [[8.0, 16.0, 32.0]]])
        right = torch.tensor([[1.0], [2.0], [4.0]])
        cases = (
            (
                operator.add,
                [
                    [[2.0, 3.0, 5.0], [3.0, 4.0, 6.0], [5.0, 6.0, 8.0]],
                    [
                        [9.0, 17.0, 33.0],
                        [10.0, 18.0, 34.0],
                        [12.0, 20.0, 36.0],
                    ],
                ],
            ),
            (
                operator.sub,
                [
                    [[0.0, 1.0, 3.0], [-1.0, 0.0, 2.0], [-3.0, -2.0, 0.0]],
                    [
                        [7.0, 15.0, 31.0],
                        [6.0, 14.0, 30.0],
                        [4.0, 12.0, 28.0],
                    ],
                ],
            ),
            (
                operator.mul,
                [
                    [[1.0, 2.0, 4.0], [2.0, 4.0, 8.0], [4.0, 8.0, 16.0]],
                    [
                        [8.0, 16.0, 32.0],
                        [16.0, 32.0, 64.0],
                        [32.0, 64.0, 128.0],
                    ],
                ],
            ),
            (
                operator.truediv,
                [
                    [[1.0, 2.0, 4.0], [0.5, 1.0, 2.0], [0.25, 0.5, 1.0]],
                    [
                        [8.0, 16.0, 32.0],
                        [4.0, 8.0, 16.0],
                        [2.0, 4.0, 8.0],
                    ],
                ],
            ),
        )

        for operation, expected in cases:
            with self.subTest(operation=operation):
                self.assert_tensor_values(operation(left, right), expected, (2, 3, 3))

    def test_binary_arithmetic_broadcasts_scalars_and_zero_dimensions(self):
        scalar = torch.tensor(2.0)
        matrix = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        self.assert_tensor_values(matrix + scalar, [[3.0, 5.0], [7.0, 9.0]], (2, 2))
        self.assert_tensor_values(scalar - matrix, [[1.0, -1.0], [-3.0, -5.0]], (2, 2))

        empty = torch.zeros((2, 0, 3))
        row = torch.ones((1, 1, 3))
        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation):
                self.assert_tensor_values(operation(empty, row), [[], []], (2, 0, 3))

        self.assertEqual((torch.zeros((0,)) + torch.ones((1,))).shape, (0,))

        large_empty = torch.full((sys.maxsize, 0), 1.0)
        large_output = large_empty + torch.tensor(2.0)
        self.assertEqual(large_output.shape, (sys.maxsize, 0))
        self.assertEqual(large_output.numel(), 0)

    def test_python_real_scalar_and_reverse_arithmetic(self):
        tensor = torch.tensor([1.0, -2.0, 4.0])
        cases = (
            (tensor + 2, [3.0, 0.0, 6.0]),
            (2 + tensor, [3.0, 0.0, 6.0]),
            (tensor - 2.0, [-1.0, -4.0, 2.0]),
            (2.0 - tensor, [1.0, 4.0, -2.0]),
            (tensor * np.float32(2.0), [2.0, -4.0, 8.0]),
            (np.float32(2.0) * tensor, [2.0, -4.0, 8.0]),
            (tensor / 2, [0.5, -1.0, 2.0]),
            (2 / tensor, [2.0, -1.0, 0.5]),
            (tensor + True, [2.0, -1.0, 5.0]),
        )
        for actual, expected in cases:
            with self.subTest(expected=expected):
                self.assert_tensor_values(actual, expected, (3,))

        zero = torch.tensor(0.0)
        self.assertEqual((zero + (-(2**63))).item(), -9223372036854775808.0)
        self.assertEqual((zero + (2**64 - 1)).item(), 18446744073709551616.0)
        self.assertEqual(
            (zero + np.uint64(2**64 - 1)).item(),
            18446744073709551616.0,
        )

    def test_python_bool_subtraction_matches_pytorch_errors(self):
        tensor = torch.tensor([1.0, 2.0])
        for operation in (
            lambda: tensor - True,
            lambda: False - tensor,
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(RuntimeError, "bool tensor is not supported"):
                    operation()

    def test_unsupported_operands_use_python_reflected_dispatch(self):
        class ReflectedArithmetic:
            def __init__(self):
                self.calls = []

            def reflected(self, name, tensor):
                self.calls.append(name)
                return name, tensor

            def __radd__(self, tensor):
                return self.reflected("add", tensor)

            def __rsub__(self, tensor):
                return self.reflected("sub", tensor)

            def __rmul__(self, tensor):
                return self.reflected("mul", tensor)

            def __rtruediv__(self, tensor):
                return self.reflected("truediv", tensor)

        tensor = torch.tensor([1.0])
        value = ReflectedArithmetic()
        for operation, expected_name in (
            (operator.add, "add"),
            (operator.sub, "sub"),
            (operator.mul, "mul"),
            (operator.truediv, "truediv"),
        ):
            with self.subTest(operation=operation):
                name, reflected_tensor = operation(tensor, value)
                self.assertEqual(name, expected_name)
                self.assertIs(reflected_tensor, tensor)
        self.assertEqual(value.calls, ["add", "sub", "mul", "truediv"])

    def test_recognized_scalar_errors_do_not_fall_back_to_reflection(self):
        class OverflowingInteger(int):
            def __new__(cls):
                instance = super().__new__(cls, 2**64)
                instance.reflected = False
                return instance

            def __rmul__(self, tensor):
                self.reflected = True
                return tensor

        value = OverflowingInteger()
        with self.assertRaises(OverflowError):
            torch.ones((1,)) * value
        self.assertFalse(value.reflected)

    def test_scalar_division_preserves_non_finite_and_signed_zero_results(self):
        tensor = torch.tensor([1.0, -1.0, 0.0, -0.0])
        self.assert_tensor_values(
            tensor / -0.0,
            [-math.inf, math.inf, math.nan, math.nan],
            (4,),
        )
        self.assert_tensor_values(
            -0.0 / tensor,
            [-0.0, 0.0, math.nan, math.nan],
            (4,),
        )
        self.assert_tensor_values(
            tensor + math.nan,
            [math.nan, math.nan, math.nan, math.nan],
            (4,),
        )

    def test_scalar_arithmetic_rejects_non_real_and_out_of_range_values(self):
        tensor = torch.ones((2,))
        for value in (object(), Decimal("1.0"), 1 + 2j, [1.0]):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    operator.add(tensor, value)
                with self.assertRaises(TypeError):
                    operator.add(value, tensor)

        for value in (-(2**63) - 1, 2**64):
            with self.subTest(value=value):
                with self.assertRaises(OverflowError):
                    tensor * value

    def test_subtraction_and_division_cover_general_same_shapes(self):
        cases = (
            (torch.tensor(7.0), torch.tensor(2.0), (), 5.0, 3.5),
            (
                torch.tensor([[[12.0, -8.0]], [[3.0, 0.5]]]),
                torch.tensor([[[3.0, 2.0]], [[-1.5, 0.25]]]),
                (2, 1, 2),
                [[[9.0, -10.0]], [[4.5, 0.25]]],
                [[[4.0, -4.0]], [[-2.0, 2.0]]],
            ),
            (
                torch.full((2, 0, 3), 1.0),
                torch.full((2, 0, 3), 2.0),
                (2, 0, 3),
                [[], []],
                [[], []],
            ),
        )

        for left, right, shape, expected_sub, expected_div in cases:
            with self.subTest(shape=shape):
                self.assert_tensor_values(left - right, expected_sub, shape)
                self.assert_tensor_values(left / right, expected_div, shape)

    def test_subtraction_and_division_match_pytorch_special_values(self):
        cases = (
            (
                operator.sub,
                [math.nan, math.inf, -math.inf, math.inf, -math.inf, -0.0, 0.0],
                [1.0, math.inf, -math.inf, -math.inf, math.inf, 0.0, -0.0],
            ),
            (
                operator.truediv,
                [
                    math.nan,
                    math.inf,
                    -math.inf,
                    math.inf,
                    -math.inf,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    0.0,
                    -0.0,
                ],
                [
                    1.0,
                    math.inf,
                    -math.inf,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.0,
                    2.0,
                    2.0,
                    -2.0,
                    -2.0,
                ],
            ),
        )

        expected = (
            [math.nan, math.nan, math.nan, math.inf, -math.inf, -0.0, 0.0],
            [
                math.nan,
                math.nan,
                math.nan,
                math.inf,
                -math.inf,
                math.inf,
                -math.inf,
                -math.inf,
                math.inf,
                0.0,
                -0.0,
                -0.0,
                0.0,
            ],
        )
        for (operation, left, right), expected_values in zip(cases, expected):
            with self.subTest(operation=operation):
                self.assert_tensor_values(
                    operation(torch.tensor(left), torch.tensor(right)),
                    expected_values,
                    (len(expected_values),),
                )

    def test_binary_arithmetic_rejects_incompatible_shapes(self):
        left = torch.zeros([2, 2])
        right = torch.zeros([3])

        for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
            with self.subTest(operation=operation):
                with self.assertRaises(RuntimeError):
                    operation(left, right)

    def test_ragged_input_is_rejected(self):
        with self.assertRaises(ValueError):
            torch.tensor([[1.0], [2.0, 3.0]])

    def test_full_handles_scalar_empty_and_multidimensional_shapes(self):
        scalar = torch.full([], -2.5)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.numel(), 1)
        self.assertEqual(scalar.item(), -2.5)

        empty = torch.full([2, 0, 3], 7.0)
        self.assertEqual(empty.shape, (2, 0, 3))
        self.assertEqual(empty.numel(), 0)
        self.assertEqual(empty.tolist(), [[], []])

        matrix = torch.full((2, 3), 1.25)
        self.assertEqual(matrix.shape, (2, 3))
        self.assertEqual(matrix.tolist(), [[1.25] * 3] * 2)

    def test_tolist_maps_zero_element_list_capacity_overflow_to_memory_error(self):
        tensor = torch.full((sys.maxsize, 0), 1.0)
        self.assertEqual(tensor.numel(), 0)

        with self.assertRaises(MemoryError):
            tensor.tolist()

    def test_full_preserves_nan_and_infinities(self):
        nan_values = torch.full([2], math.nan).tolist()
        self.assertTrue(all(math.isnan(value) for value in nan_values))
        self.assertEqual(torch.full([2], math.inf).tolist(), [math.inf, math.inf])
        self.assertEqual(torch.full([2], -math.inf).tolist(), [-math.inf, -math.inf])

    def test_full_accepts_pytorch_keyword_names(self):
        result = torch.full(size=[2], fill_value=3.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [3.0, 3.0])

    def test_full_rejects_negative_sizes_as_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "negative dimension -1"):
            torch.full([-1], 3.0)

    def test_full_rejects_storage_capacity_overflow(self):
        oversized = sys.maxsize // 4 + 1
        with self.assertRaisesRegex(RuntimeError, "exceeds the platform capacity"):
            torch.full([oversized], 1.0)

    def test_full_rejects_finite_fill_value_overflow(self):
        for fill_value in (1e40, -1e40):
            with self.subTest(fill_value=fill_value):
                with self.assertRaisesRegex(RuntimeError, "float32 without overflow"):
                    torch.full((2,), fill_value)

    def test_full_maps_shape_product_overflow_to_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.full((2**62, 4), 1.0)

    def test_full_rejects_invalid_size_arguments(self):
        for size in ([True], (False,), range(2)):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    torch.full(size, 3.0)

    def test_full_accepts_index_protocol_dimensions(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        dimension = IndexDimension(2)
        result = torch.full([dimension], 3.0)
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.tolist(), [3.0, 3.0])
        self.assertEqual(dimension.calls, 1)

    def test_full_normalizes_invalid_index_dimensions_to_type_error(self):
        class FailingIndex:
            def __index__(self):
                raise RuntimeError("index conversion failed")

        for dimension in (2**63, -(2**63) - 1, FailingIndex()):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(TypeError, "size element at index 0"):
                    torch.full([dimension], 3.0)

    def test_full_accepts_scalar_tensor_fill_value(self):
        result = torch.full((2,), torch.tensor(3.0))
        self.assertEqual(result.tolist(), [3.0, 3.0])

        with self.assertRaises(TypeError):
            torch.full((2,), torch.tensor([3.0]))

    def test_full_accepts_real_numpy_scalar_fill_values(self):
        cases = (
            (np.longdouble(1.25), [1.25, 1.25]),
            (np.float32(1.25), [1.25, 1.25]),
            (np.int64(3), [3.0, 3.0]),
            (np.bool_(True), [1.0, 1.0]),
        )
        for fill_value, expected in cases:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((2,), fill_value).tolist(), expected)

    def test_full_rejects_zero_dimensional_buffer_fill_values(self):
        array = np.array(3.0)
        for fill_value in (array, memoryview(array)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((2,), fill_value)

    def test_full_enforces_numpy_integer_signed_boundary(self):
        accepted = (
            np.int64(-(2**63)),
            np.int64(2**63 - 1),
            np.uint64(2**63 - 1),
        )
        for fill_value in accepted:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((1,), fill_value).numel(), 1)

        for fill_value in (np.uint64(2**63), np.uint64(2**64 - 1)):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((1,), fill_value)

    def test_full_rejects_non_scalar_numeric_coercions(self):
        class FloatLike:
            def __init__(self):
                self.calls = 0

            def __float__(self):
                self.calls += 1
                return 3.0

        float_like = FloatLike()
        for fill_value in (Decimal("3.0"), float_like):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(TypeError):
                    torch.full((2,), fill_value)
        self.assertEqual(float_like.calls, 0)

    def test_full_converts_integer_fill_values_without_double_rounding(self):
        class IntWithFloat(int):
            def __new__(cls, value):
                instance = super().__new__(cls, value)
                instance.float_calls = 0
                return instance

            def __float__(self):
                self.float_calls += 1
                return 0.0

        fill_value = IntWithFloat(9007199791611905)
        result = torch.full((1,), fill_value)
        self.assertEqual(result.item(), 9007200328482816.0)
        self.assertEqual(fill_value.float_calls, 0)

    def test_full_enforces_python_integer_scalar_boundaries(self):
        accepted = (
            (-(2**63), -9223372036854775808.0),
            (2**64 - 1, 18446744073709551616.0),
        )
        for fill_value, expected in accepted:
            with self.subTest(fill_value=fill_value):
                self.assertEqual(torch.full((1,), fill_value).item(), expected)

        for fill_value in (-(2**63) - 1, 2**64):
            with self.subTest(fill_value=fill_value):
                with self.assertRaises(OverflowError):
                    torch.full((1,), fill_value)

    def test_full_matches_pytorch_validation_order(self):
        with self.assertRaises(TypeError):
            torch.full([-1], object())

        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.full((2**62, 4), 1e40)

    def test_full_validates_strides_for_empty_shapes(self):
        large = 2**62
        for size in ((0, large, 2), (2, 0, large, 2), (1, large, 2, 0)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                    torch.full(size, 1.0)


if __name__ == "__main__":
    unittest.main()
