import sys
import unittest

import numpy as np
import torch_rs as torch


class TransposeTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, shape, stride):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected, dtype=np.float32), equal_nan=True
        )

    def test_method_and_top_level_transpose_swap_only_metadata(self):
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(2, 1, 0)

        for view in (
            source.transpose(0, -1),
            torch.transpose(source, dim0=-3, dim1=2),
            torch.transpose(input=source, dim0=0, dim1=2),
        ):
            with self.subTest(view=view):
                self.assert_tensor(view, expected, (4, 3, 2), (1, 4, 12))
                self.assertEqual(view.storage_offset(), source.storage_offset())
                self.assertIs(view.dtype, source.dtype)
                self.assertEqual(view.device, source.device)
                self.assertFalse(view.is_contiguous())

        restored = source.transpose(0, 2).transpose(-1, 0)
        self.assert_tensor(restored, np.arange(24).reshape(2, 3, 4), source.shape, source.stride())
        duplicate = source.transpose(1, -2)
        self.assert_tensor(duplicate, np.arange(24).reshape(2, 3, 4), source.shape, source.stride())

    def test_is_contiguous_accepts_keyword_only_memory_format(self):
        contiguous = torch.zeros((2, 3, 4, 5))
        channels_last = torch.zeros((1, 1, 2, 2)).transpose(1, 3)
        channels_last_3d = (
            torch.zeros((2, 4, 5, 6, 3))
            .transpose(1, 4)
            .transpose(2, 4)
            .transpose(3, 4)
        )
        cases = (
            (contiguous, torch.preserve_format, True),
            (contiguous, torch.contiguous_format, True),
            (contiguous, torch.channels_last, False),
            (channels_last, torch.contiguous_format, False),
            (channels_last, torch.channels_last, True),
            (channels_last_3d, torch.channels_last, False),
            (channels_last_3d, torch.channels_last_3d, True),
        )
        for tensor, memory_format, expected in cases:
            with self.subTest(shape=tensor.shape, memory_format=memory_format):
                self.assertEqual(
                    tensor.is_contiguous(memory_format=memory_format), expected
                )

        self.assertTrue(contiguous.is_contiguous())
        with self.assertRaisesRegex(TypeError, "takes 0 positional arguments"):
            contiguous.is_contiguous(torch.contiguous_format)
        for invalid in (None, object(), "contiguous_format"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    TypeError,
                    "argument 'memory_format' must be torch.memory_format",
                ):
                    contiguous.is_contiguous(memory_format=invalid)

    def test_scalar_dimensions_and_dimension_errors_match_pytorch(self):
        scalar = torch.tensor(2.5)
        for dim0, dim1 in ((0, 0), (-1, -1), (0, -1), (-1, 0)):
            with self.subTest(dim0=dim0, dim1=dim1):
                view = scalar.transpose(dim0, dim1)
                self.assertEqual(view.shape, ())
                self.assertEqual(view.stride(), ())
                self.assertEqual(view.item(), 2.5)

        for dimension in (-2, 1):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"expected to be in range of \[-1, 0\], but got {dimension}",
                ):
                    scalar.transpose(dimension, 0)

        tensor = torch.zeros((2, 3, 4))
        for dimension in (-4, 3):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                    tensor.transpose(0, dimension)

    def test_dimensions_accept_pytorch_integer_types_and_check_overflow(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(tensor.transpose(IntSubclass(0), np.int64(-1)).shape, (4, 3, 2))
        self.assertEqual(torch.transpose(tensor, np.uint32(1), 2).shape, (2, 4, 3))

        for dimension in (True, False, 1.0, "1", None, IndexOnly()):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(TypeError, "must be int"):
                    tensor.transpose(dimension, 0)

        for dimension in (2**100, -(2**100)):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                    tensor.transpose(dimension, 0)

    def test_transpose_type_errors_preserve_argument_call_context(self):
        tensor = torch.zeros((2, 3))
        cases = (
            (
                lambda: tensor.transpose(1.5, 0),
                "transpose(): argument 'dim0' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.transpose(0, 1.5),
                "transpose(): argument 'dim1' (position 2) must be int, not float",
            ),
            (
                lambda: tensor.transpose(dim0=1.5, dim1=0),
                "transpose(): argument 'dim0' must be int, not float",
            ),
            (
                lambda: tensor.transpose(0, dim1=1.5),
                "transpose(): argument 'dim1' must be int, not float",
            ),
            (
                lambda: torch.transpose(tensor, 1.5, 0),
                "transpose(): argument 'dim0' (position 2) must be int, not float",
            ),
            (
                lambda: torch.transpose(tensor, 0, 1.5),
                "transpose(): argument 'dim1' (position 3) must be int, not float",
            ),
            (
                lambda: torch.transpose(input=tensor, dim0=1.5, dim1=0),
                "transpose(): argument 'dim0' must be int, not float",
            ),
            (
                lambda: torch.transpose(tensor, 0, dim1=1.5),
                "transpose(): argument 'dim1' must be int, not float",
            ),
            (
                lambda: tensor.transpose(np.float64(1.5), 0),
                "transpose(): argument 'dim0' (position 1) must be int, not numpy.float64",
            ),
            (
                lambda: tensor.transpose(dim0=np.bool_(True), dim1=0),
                "transpose(): argument 'dim0' must be int, not numpy.bool",
            ),
            (
                lambda: torch.transpose(tensor, np.float32(1.5), 0),
                "transpose(): argument 'dim0' (position 2) must be int, not numpy.float32",
            ),
        )
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

    def test_python_binding_reports_missing_duplicate_and_extra_arguments(self):
        tensor = torch.zeros((2, 3))
        invalid_calls = (
            lambda: tensor.transpose(),
            lambda: tensor.transpose(0),
            lambda: tensor.transpose(0, 1, 2),
            lambda: tensor.transpose(0, dim0=1, dim1=0),
            lambda: torch.transpose(),
            lambda: torch.transpose(tensor, 0),
            lambda: torch.transpose(tensor, 0, 1, 2),
            lambda: torch.transpose(tensor, 0, dim0=1, dim1=0),
            lambda: torch.transpose(tensor, 0, 1, input=tensor),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        gap_cases = (
            (
                lambda: tensor.transpose(dim1=0),
                'transpose() missing 2 required positional argument: "dim0", "dim1"',
            ),
            (
                lambda: tensor.transpose(dim1=0, unexpected=None),
                'transpose() missing 2 required positional argument: "dim0", "dim1"',
            ),
            (
                lambda: torch.transpose(dim0=0),
                'transpose() missing 3 required positional argument: "input", "dim0", "dim1"',
            ),
            (
                lambda: torch.transpose(input=tensor, dim1=0),
                'transpose() missing 2 required positional argument: "dim0", "dim1"',
            ),
        )
        for call, expected in gap_cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

    def test_stride_aware_consumers_and_materialized_outputs(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())
        view = source.transpose(0, 2)
        expected = values.transpose(2, 1, 0)

        self.assertEqual(view.tolist(), expected.tolist())
        copied_array = np.asarray(view)
        np.testing.assert_array_equal(copied_array, expected)
        copied_array[0, 0, 0] = -99.0
        self.assertEqual(view[0][0][0].item(), 0.0)

        indexed = view[1]
        self.assertEqual(indexed.shape, (3, 2))
        self.assertEqual(indexed.stride(), (4, 12))
        self.assertEqual(indexed.storage_offset(), 1)
        self.assertEqual(indexed[2, 1].item(), 21.0)

        cloned = indexed.clone()
        self.assert_tensor(cloned, expected[1], (3, 2), (1, 3))
        self.assertEqual(cloned.storage_offset(), 0)
        contiguous_clone = torch.clone(indexed, memory_format=torch.contiguous_format)
        self.assert_tensor(contiguous_clone, expected[1], (3, 2), (2, 1))

        same_shape = view.reshape(view.shape)
        self.assert_tensor(same_shape, expected, view.shape, view.stride())
        flattened = view.reshape(-1)
        self.assert_tensor(flattened, expected.reshape(-1), (24,), (1,))
        appended = view.reshape(*view.shape, 1)
        self.assert_tensor(appended, expected.reshape(4, 3, 2, 1), (4, 3, 2, 1), (1, 4, 12, 12))

        operations = (
            (view.relu(), np.maximum(expected, 0)),
            (view.sin(), np.sin(expected)),
            (view.exp(), np.exp(expected)),
            (view + 2.0, expected + 2.0),
            (view + view, expected + expected),
            (view + torch.tensor([10.0, 20.0]), expected + [10.0, 20.0]),
        )
        for actual, operation_expected in operations:
            with self.subTest(actual=actual):
                self.assertEqual(actual.stride(), view.stride())
                self.assertEqual(actual.storage_offset(), 0)
                np.testing.assert_allclose(
                    np.asarray(actual), operation_expected, rtol=2.0e-6, atol=1.0e-6
                )
        self.assertEqual(view.sum().item(), np.float32(expected.sum()).item())

    def test_rank_two_matmul_uses_both_input_stride_tables(self):
        left_values = np.arange(6, dtype=np.float32).reshape(2, 3)
        right_values = np.arange(8, dtype=np.float32).reshape(4, 2)
        left = torch.tensor(left_values.tolist()).transpose(0, 1)
        right = torch.tensor(right_values.tolist()).transpose(0, 1)
        output = left @ right

        self.assert_tensor(
            output,
            left_values.T @ right_values.T,
            (3, 4),
            (4, 1),
        )

    def test_pointwise_outputs_canonicalize_singleton_channels_last_strides(self):
        source = torch.tensor(
            np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2).tolist()
        )
        view = source.transpose(1, 3)
        self.assertEqual(view.shape, (1, 2, 2, 1))
        self.assertEqual(view.stride(), (4, 1, 2, 4))

        for output in (view.relu(), view.sin(), view + view):
            with self.subTest(output=output):
                self.assertEqual(output.stride(), (4, 1, 2, 2))
                self.assertEqual(output.reshape(output.shape).stride(), (2, 1, 2, 2))

    def test_reflected_division_uses_unary_layout_and_checks_stride_overflow(self):
        view = torch.tensor([[1.0, 2.0]]).transpose(0, 1)
        self.assertEqual(view.shape, (2, 1))
        self.assertEqual(view.stride(), (1, 2))
        self.assertEqual((view / 1.0).stride(), (1, 2))

        reflected = 1.0 / view
        self.assert_tensor(reflected, [[1.0], [0.5]], (2, 1), (1, 1))

        empty_cases = (
            (torch.zeros((1, 0)).transpose(0, 1), (1, 0)),
            (torch.zeros((1, 0, 1)), (0, 1, 0)),
            (torch.zeros((2, 0, 3)).transpose(0, 2), (2, 2, 1)),
        )
        for empty, expected_stride in empty_cases:
            with self.subTest(shape=empty.shape, stride=empty.stride()):
                output = 1.0 / empty
                self.assertEqual(output.shape, empty.shape)
                self.assertEqual(output.stride(), expected_stride)
                self.assertEqual(output.numel(), 0)

        large = torch.zeros((2, 0, sys.maxsize))
        self.assertEqual((1.0 / large).stride(), (0, 1, 0))
        extreme = large.transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            1.0 / extreme

    def test_empty_and_extreme_shapes_preserve_swapped_strides(self):
        cases = (
            (torch.zeros((2, 0, 3)), (0, 2), (3, 0, 2), (1, 3, 3)),
            (torch.zeros((1, 0)), (0, 1), (0, 1), (1, 1)),
            (
                torch.zeros((2, 0, sys.maxsize)),
                (0, 2),
                (sys.maxsize, 0, 2),
                (1, sys.maxsize, sys.maxsize),
            ),
        )
        for source, dimensions, shape, stride in cases:
            with self.subTest(shape=shape):
                view = source.transpose(*dimensions)
                self.assertEqual(view.shape, shape)
                self.assertEqual(view.stride(), stride)
                self.assertEqual(view.numel(), 0)
                self.assertTrue(view.is_contiguous())

        offset = torch.zeros((sys.maxsize, 0))[sys.maxsize - 1].transpose(0, 0)
        self.assertEqual(offset.storage_offset(), sys.maxsize - 1)
        self.assertEqual(offset.tolist(), [])
        self.assertEqual(offset.clone().storage_offset(), 0)

        overflow_order = torch.zeros((sys.maxsize, 0, 2, 2))
        for operation in (
            lambda: overflow_order.transpose(1, 3),
            lambda: overflow_order.transpose(-3, -1),
            lambda: torch.transpose(overflow_order, 1, 3),
        ):
            with self.subTest(operation=operation):
                with self.assertRaisesRegex(
                    RuntimeError, "^numel: integer multiplication overflow$"
                ):
                    operation()
        self.assertEqual(overflow_order.transpose(1, 1).numel(), 0)


if __name__ == "__main__":
    unittest.main()
