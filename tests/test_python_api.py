import math
import sys
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch


class PythonApiBaselineTests(unittest.TestCase):
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
