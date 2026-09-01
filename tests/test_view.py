import gc
import inspect
import operator
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


INCOMPATIBLE_LAYOUT = (
    "view size is not compatible with input tensor's size and stride "
    "(at least one dimension spans across two contiguous subspaces). "
    "Use .reshape(...) instead."
)


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


class StatefulIndexDimension:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __index__(self):
        value = self.values[self.calls]
        self.calls += 1
        return value


class TupleIndexDimension(tuple):
    def __new__(cls, values, index_value):
        instance = super().__new__(cls, values)
        instance.index_value = index_value
        instance.calls = 0
        return instance

    def __index__(self):
        self.calls += 1
        return self.index_value


class ListIndexDimension(list):
    def __init__(self, values, index_value):
        super().__init__(values)
        self.index_value = index_value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.index_value


class TensorViewTests(unittest.TestCase):
    def shape_forms(self, shape):
        return (
            ("tuple", tuple(shape), False),
            ("list", list(shape), False),
            ("Size", torch.Size(shape), False),
            ("keyword", tuple(shape), True),
        )

    def assert_view_result(
        self,
        result,
        source,
        *,
        expected_shape,
        expected_stride,
        expected_offset,
    ):
        direct = source.reshape(expected_shape)
        self.assertIsNot(result, source)
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.stride(), expected_stride)
        self.assertEqual(result.storage_offset(), expected_offset)
        self.assertEqual(result.is_contiguous(), direct.is_contiguous())
        self.assertEqual(result.requires_grad, direct.requires_grad)
        self.assertEqual(result.is_leaf, direct.is_leaf)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertTrue(result.is_set_to(direct))

    def make_layout_cases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        noncontiguous = base.transpose(0, 1)
        return (
            ("scalar", torch.tensor(-0.0), (), (), (), 0),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 0),
                (2, 0),
                (1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1),
                (0, 1),
                (1, 0),
                0,
            ),
            (
                "contiguous",
                base,
                (6, 4),
                (6, 4),
                (4, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                (2, 6),
                (2, 6),
                (6, 1),
                12,
            ),
            (
                "noncontiguous-same-shape",
                noncontiguous,
                (3, 2, 4),
                (3, 2, 4),
                (4, 12, 1),
                0,
            ),
            (
                "noncontiguous-compatible-split",
                noncontiguous,
                (3, 2, 2, 2),
                (3, 2, 2, 2),
                (4, 12, 2, 1),
                0,
            ),
        )

    def make_dtype_layout_cases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        return (
            ("scalar", torch.tensor(-0.0, requires_grad=True)),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
            ),
            ("contiguous-offset", (leaf * 2.0)[1]),
            ("noncontiguous-offset", (leaf * 3.0).transpose(0, 2)[1]),
        )

    def test_same_dtype_overload_returns_fresh_detached_storage_aliases(self):
        forms = (
            ("float32 positional", torch.float32, False),
            ("float positional", torch.float, False),
            ("float32 keyword", torch.float32, True),
            ("float keyword", torch.float, True),
        )
        for case, source in self.make_dtype_layout_cases():
            source_contract = (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.requires_grad,
                source.is_leaf,
                source.data_ptr(),
                np.asarray(source.detach()).copy(),
            )
            results = []
            for form, dtype, keyword in forms:
                with self.subTest(case=case, form=form):
                    result = (
                        source.view(dtype=dtype) if keyword else source.view(dtype)
                    )
                    self.assertIsNot(result, source)
                    self.assertTrue(result.is_set_to(source))
                    self.assertEqual(result.shape, source.shape)
                    self.assertEqual(result.stride(), source.stride())
                    self.assertEqual(
                        result.storage_offset(), source.storage_offset()
                    )
                    self.assertEqual(result.is_contiguous(), source.is_contiguous())
                    self.assertIs(result.dtype, source.dtype)
                    self.assertEqual(result.device, source.device)
                    self.assertEqual(result.data_ptr(), source.data_ptr())
                    self.assertFalse(result.requires_grad)
                    self.assertTrue(result.is_leaf)
                    self.assertFalse((result + 1.0).requires_grad)
                    np.testing.assert_array_equal(
                        np.asarray(result), source_contract[-1]
                    )
                    results.append(result)

            for index, result in enumerate(results):
                for other in results[index + 1 :]:
                    self.assertIsNot(result, other)
            self.assertEqual(
                (
                    source.shape,
                    source.stride(),
                    source.storage_offset(),
                    source.requires_grad,
                    source.is_leaf,
                    source.data_ptr(),
                ),
                source_contract[:-1],
            )
            np.testing.assert_array_equal(
                np.asarray(source.detach()), source_contract[-1]
            )

    def test_same_dtype_alias_outlives_temporary_source_owners(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retain_positional():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return (temporary * 2.0).transpose(0, 2)[1].view(torch.float32)

        def retain_keyword():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return (temporary * 3.0).transpose(0, 2)[1].view(dtype=torch.float)

        positional = retain_positional()
        keyword = retain_keyword()
        gc.collect()

        np.testing.assert_array_equal(
            np.asarray(positional), (values * 2.0).transpose(2, 1, 0)[1]
        )
        np.testing.assert_array_equal(
            np.asarray(keyword), (values * 3.0).transpose(2, 1, 0)[1]
        )

    def test_tuple_list_size_and_keyword_delegate_to_native_view(self):
        for (
            case,
            source,
            shape,
            expected_shape,
            expected_stride,
            expected_offset,
        ) in self.make_layout_cases():
            for form, argument, keyword in self.shape_forms(shape):
                with self.subTest(case=case, form=form):
                    result = (
                        source.view(size=argument)
                        if keyword
                        else source.view(argument)
                    )
                    self.assert_view_result(
                        result,
                        source,
                        expected_shape=expected_shape,
                        expected_stride=expected_stride,
                        expected_offset=expected_offset,
                    )

    def test_single_integer_and_index_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            ("scalar", torch.tensor(-0.0), 1, (1,), (1,), 0),
            ("inferred", base, -1, (24,), (1,), 0),
            ("offset", base[1], IntSubclass(12), (12,), (1,), 12),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                np.int64(-1),
                (0,),
                (1,),
                1,
            ),
            (
                "compatible-noncontiguous",
                torch.tensor(
                    np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
                ).transpose(0, 1)[0],
                IndexDimension(2),
                (2,),
                (3,),
                0,
            ),
        )
        for case, source, dimension, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(dimension)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        stateful = StatefulIndexDimension((24, 1, 24))
        result = base.view(stateful)
        self.assertEqual(result.shape, (24,))
        self.assertEqual(stateful.calls, 3)

    def test_two_positional_dimensions_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            ("contiguous-inferred", base, (6, -1), (6, 4), (4, 1), 0),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(6)),
                (2, 6),
                (6, 1),
                12,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (IndexDimension(2), 0),
                (2, 0),
                (1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1),
                (0, 1),
                (1, 0),
                0,
            ),
            (
                "noncontiguous-offset-inferred",
                base.transpose(0, 1)[1],
                (2, IndexDimension(-1)),
                (2, 4),
                (12, 1),
                4,
            ),
        )
        for case, source, dimensions, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(*dimensions)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        result = torch.zeros((6,)).view(first, second)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual((first.calls, second.calls), (3, 1))

    def test_three_positional_dimensions_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            (
                "contiguous-inferred",
                base,
                (2, -1, 2),
                (2, 6, 2),
                (12, 2, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(2), IndexDimension(3)),
                (2, 2, 3),
                (6, 3, 1),
                12,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1),
                (2, 0, 1),
                (1, 1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1, 1),
                (0, 1, 1),
                (1, 1, 1),
                0,
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 2),
                (2, 2, 2),
                (12, 2, 1),
                4,
            ),
        )
        for case, source, dimensions, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(*dimensions)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        third = StatefulIndexDimension((4,))
        result = torch.zeros((24,)).view(first, second, third)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertEqual((first.calls, second.calls, third.calls), (3, 1, 1))

    def test_four_positional_dimensions_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            (
                "contiguous-inferred",
                base,
                (2, -1, 2, 1),
                (2, 6, 2, 1),
                (12, 2, 1, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(1), 2, IndexDimension(3)),
                (2, 1, 2, 3),
                (6, 6, 3, 1),
                12,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1, 1),
                (2, 0, 1, 1),
                (1, 1, 1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1, 1, 1),
                (0, 1, 1, 1),
                (1, 1, 1, 1),
                0,
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 1, 2),
                (2, 2, 1, 2),
                (12, 2, 2, 1),
                4,
            ),
            (
                "noncontiguous-compatible-split",
                base.transpose(0, 1),
                (3, 2, 2, 2),
                (3, 2, 2, 2),
                (4, 12, 2, 1),
                0,
            ),
        )
        for case, source, dimensions, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(*dimensions)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        third = StatefulIndexDimension((4,))
        fourth = StatefulIndexDimension((2,))
        result = torch.zeros((48,)).view(first, second, third, fourth)
        self.assertEqual(result.shape, (2, 3, 4, 2))
        self.assertEqual(
            (first.calls, second.calls, third.calls, fourth.calls),
            (3, 1, 1, 1),
        )

    def test_five_positional_dimensions_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            (
                "contiguous-inferred",
                base,
                (2, -1, 2, 1, 1),
                (2, 6, 2, 1, 1),
                (12, 2, 1, 1, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(1), 1, 2, IndexDimension(3)),
                (2, 1, 1, 2, 3),
                (6, 6, 6, 3, 1),
                12,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1, 1, 1),
                (2, 0, 1, 1, 1),
                (1, 1, 1, 1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1, 1, 1, 1),
                (0, 1, 1, 1, 1),
                (1, 1, 1, 1, 1),
                0,
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 1, 1, 2),
                (2, 2, 1, 1, 2),
                (12, 2, 2, 2, 1),
                4,
            ),
            (
                "noncontiguous-compatible-split",
                base.transpose(0, 1),
                (3, 2, 2, 1, 2),
                (3, 2, 2, 1, 2),
                (4, 12, 2, 2, 1),
                0,
            ),
        )
        for case, source, dimensions, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(*dimensions)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        third = StatefulIndexDimension((4,))
        fourth = StatefulIndexDimension((2,))
        fifth = StatefulIndexDimension((2,))
        result = torch.zeros((96,)).view(first, second, third, fourth, fifth)
        self.assertEqual(result.shape, (2, 3, 4, 2, 2))
        self.assertEqual(
            (first.calls, second.calls, third.calls, fourth.calls, fifth.calls),
            (3, 1, 1, 1, 1),
        )

    def test_six_positional_dimensions_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            (
                "contiguous-inferred",
                base,
                (2, -1, 2, 1, 1, 1),
                (2, 6, 2, 1, 1, 1),
                (12, 2, 1, 1, 1, 1),
                0,
            ),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(1), 1, 1, 2, IndexDimension(3)),
                (2, 1, 1, 1, 2, 3),
                (6, 6, 6, 6, 3, 1),
                12,
            ),
            (
                "empty-offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1, 1, 1, 1),
                (2, 0, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1),
                1,
            ),
            (
                "empty-same-shape",
                torch.zeros((0, 1)) + 1,
                (0, 1, 1, 1, 1, 1),
                (0, 1, 1, 1, 1, 1),
                (1, 1, 1, 1, 1, 1),
                0,
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 1, 1, 1, 2),
                (2, 2, 1, 1, 1, 2),
                (12, 2, 2, 2, 2, 1),
                4,
            ),
            (
                "noncontiguous-compatible-split",
                base.transpose(0, 1),
                (3, 2, 2, 1, 1, 2),
                (3, 2, 2, 1, 1, 2),
                (4, 12, 2, 2, 2, 1),
                0,
            ),
        )
        for case, source, dimensions, shape, stride, offset in cases:
            with self.subTest(case=case):
                result = source.view(*dimensions)
                self.assert_view_result(
                    result,
                    source,
                    expected_shape=shape,
                    expected_stride=stride,
                    expected_offset=offset,
                )

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        third = StatefulIndexDimension((4,))
        fourth = StatefulIndexDimension((2,))
        fifth = StatefulIndexDimension((2,))
        sixth = StatefulIndexDimension((2,))
        result = torch.zeros((192,)).view(
            first, second, third, fourth, fifth, sixth
        )
        self.assertEqual(result.shape, (2, 3, 4, 2, 2, 2))
        self.assertEqual(
            (
                first.calls,
                second.calls,
                third.calls,
                fourth.calls,
                fifth.calls,
                sixth.calls,
            ),
            (3, 1, 1, 1, 1, 1),
        )

    def test_seven_or_more_positional_dimensions_delegate_to_native_view(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        cases = (
            ("contiguous-seven", base, (2, 3, 1, 1, 1, 1, 4)),
            ("contiguous-eight", base, (1, 2, 3, 1, 1, 1, 1, 4)),
            ("contiguous-inferred", base, (2, -1, 1, 1, 1, 1, 4)),
            ("contiguous-offset", base[1], (1, 3, 1, 1, 1, 1, 4)),
            ("empty", torch.zeros((0,)), (2, 0, 3, 1, 1, 1, 1)),
            (
                "integer-protocol",
                base,
                (
                    IntSubclass(2),
                    np.int64(3),
                    np.uint32(1),
                    IndexDimension(1),
                    1,
                    1,
                    IndexDimension(4),
                ),
            ),
        )
        for case, source, dimensions in cases:
            with self.subTest(case=case):
                result = source.view(*dimensions)
                direct = source.reshape(tuple(result.shape))
                self.assertIsNot(result, source)
                self.assertEqual(result.shape, direct.shape)
                self.assertEqual(result.stride(), direct.stride())
                self.assertEqual(result.storage_offset(), direct.storage_offset())
                self.assertEqual(result.is_contiguous(), direct.is_contiguous())
                self.assertEqual(result.requires_grad, direct.requires_grad)
                self.assertEqual(result.is_leaf, direct.is_leaf)
                self.assertEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(result.is_set_to(direct))
                np.testing.assert_array_equal(np.asarray(result), np.asarray(direct))

        first = StatefulIndexDimension((2, 1, 2))
        second = StatefulIndexDimension((3,))
        third = StatefulIndexDimension((1,))
        fourth = StatefulIndexDimension((1,))
        fifth = StatefulIndexDimension((1,))
        sixth = StatefulIndexDimension((1,))
        seventh = StatefulIndexDimension((4,))
        result = torch.zeros((24,)).view(
            first, second, third, fourth, fifth, sixth, seventh
        )
        self.assertEqual(result.shape, (2, 3, 1, 1, 1, 1, 4))
        self.assertEqual(
            (
                first.calls,
                second.calls,
                third.calls,
                fourth.calls,
                fifth.calls,
                sixth.calls,
                seventh.calls,
            ),
            (3, 1, 1, 1, 1, 1, 1),
        )

    def test_inferred_and_extreme_empty_shapes_preserve_aliasing(self):
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        for form, argument, keyword in self.shape_forms((2, -1, 2)):
            with self.subTest(kind="inferred", form=form):
                result = source.view(size=argument) if keyword else source.view(argument)
                self.assertEqual(result.shape, (2, 6, 2))
                self.assertEqual(result.stride(), (12, 2, 1))
                self.assertEqual(result.data_ptr(), source.data_ptr())
                np.testing.assert_array_equal(
                    np.asarray(result), np.asarray(source).reshape(2, 6, 2)
                )

        maximum = sys.maxsize
        empty = torch.zeros((0,))
        for form, argument, keyword in self.shape_forms((0, maximum, maximum)):
            with self.subTest(kind="extreme-empty", form=form):
                result = empty.view(size=argument) if keyword else empty.view(argument)
                self.assertEqual(result.shape, (0, maximum, maximum))
                self.assertEqual(result.stride(), (1, maximum, 1))
                self.assertEqual(result.storage_offset(), 0)
                self.assertEqual(result.numel(), 0)
                self.assertEqual(result.tolist(), [])
                self.assertEqual(result.data_ptr(), empty.data_ptr())

        inferred_empty = empty.view((-1,))
        self.assertEqual(inferred_empty.shape, (0,))
        self.assertEqual(inferred_empty.data_ptr(), empty.data_ptr())

    def test_incompatible_layout_and_shape_errors_never_copy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist()).transpose(0, 1)

        for shape in ((6, 4), [6, 4], torch.Size((6, 4))):
            with self.subTest(shape_type=type(shape).__name__):
                with self.assertRaisesRegex(
                    RuntimeError, f"^{re.escape(INCOMPATIBLE_LAYOUT)}$"
                ):
                    source.view(shape)

        reshaped = source.reshape((6, 4))
        self.assertNotEqual(reshaped.data_ptr(), source.data_ptr())
        self.assertFalse(reshaped.is_set_to(source))

        cases = (
            ((2, 2), "shape '[2, 2]' is invalid for input of size 6"),
            ((-1, -1), "only one dimension can be inferred"),
            ((2, -2), "invalid shape dimension -2 at index 1 of shape [2, -2]"),
        )
        for shape, message in cases:
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.zeros((6,)).view(shape)

        ambiguous = (
            "cannot reshape tensor of 0 elements into shape [0, -1] because the "
            "unspecified dimension size -1 can be any value and is ambiguous"
        )
        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(ambiguous)}$"):
            torch.zeros((0,)).view((0, -1))

        with self.assertRaisesRegex(
            RuntimeError, r"^shape '\[5\]' is invalid for input of size 6$"
        ):
            torch.zeros((6,)).view(5)
        with self.assertRaisesRegex(
            RuntimeError, f"^{re.escape(INCOMPATIBLE_LAYOUT)}$"
        ):
            source.view(-1)

        variadic_cases = (
            (lambda: source.view(6, 4), INCOMPATIBLE_LAYOUT),
            (lambda: source.view(3, 4, 2), INCOMPATIBLE_LAYOUT),
            (lambda: source.view(1, 6, 2, 2), INCOMPATIBLE_LAYOUT),
            (lambda: source.view(1, 1, 6, 2, 2), INCOMPATIBLE_LAYOUT),
            (lambda: source.view(1, 1, 1, 6, 2, 2), INCOMPATIBLE_LAYOUT),
            (lambda: source.view(1, 1, 1, 1, 6, 2, 2), INCOMPATIBLE_LAYOUT),
            (
                lambda: torch.zeros((6,)).view(2, 2),
                "shape '[2, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.zeros((6,)).view(1, 2, 2),
                "shape '[1, 2, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.zeros((6,)).view(1, 2, 2, 2),
                "shape '[1, 2, 2, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.zeros((6,)).view(1, 1, 2, 2, 2),
                "shape '[1, 1, 2, 2, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.zeros((6,)).view(1, 1, 1, 2, 2, 2),
                "shape '[1, 1, 1, 2, 2, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.zeros((6,)).view(1, 1, 1, 1, 1, 2, 2),
                "shape '[1, 1, 1, 1, 1, 2, 2]' is invalid for input of size 6",
            ),
            (
                lambda: torch.zeros((6,)).view(-1, -1),
                "only one dimension can be inferred",
            ),
            (
                lambda: torch.zeros((6,)).view(-1, 1, -1),
                "only one dimension can be inferred",
            ),
            (
                lambda: torch.zeros((6,)).view(-1, 1, 1, -1),
                "only one dimension can be inferred",
            ),
            (
                lambda: torch.zeros((6,)).view(-1, 1, 1, 1, -1),
                "only one dimension can be inferred",
            ),
            (
                lambda: torch.zeros((6,)).view(-1, 1, 1, 1, 1, -1),
                "only one dimension can be inferred",
            ),
            (
                lambda: torch.zeros((6,)).view(-1, 1, 1, 1, 1, 1, -1),
                "only one dimension can be inferred",
            ),
            (
                lambda: torch.zeros((6,)).view(2, -2),
                "invalid shape dimension -2 at index 1 of shape [2, -2]",
            ),
            (
                lambda: torch.zeros((6,)).view(1, -2, 3),
                "invalid shape dimension -2 at index 1 of shape [1, -2, 3]",
            ),
            (
                lambda: torch.zeros((6,)).view(1, -2, 1, 3),
                "invalid shape dimension -2 at index 1 of shape [1, -2, 1, 3]",
            ),
            (
                lambda: torch.zeros((6,)).view(1, -2, 1, 1, 3),
                "invalid shape dimension -2 at index 1 of shape [1, -2, 1, 1, 3]",
            ),
            (
                lambda: torch.zeros((6,)).view(1, -2, 1, 1, 1, 3),
                "invalid shape dimension -2 at index 1 of shape [1, -2, 1, 1, 1, 3]",
            ),
            (
                lambda: torch.zeros((6,)).view(1, -2, 1, 1, 1, 1, 3),
                "invalid shape dimension -2 at index 1 of shape [1, -2, 1, 1, 1, 1, 3]",
            ),
            (
                lambda: torch.zeros((0,)).view(0, -1),
                ambiguous,
            ),
            (
                lambda: torch.zeros((0,)).view(2, 0, -1),
                "cannot reshape tensor of 0 elements into shape [2, 0, -1] "
                "because the unspecified dimension size -1 can be any value "
                "and is ambiguous",
            ),
            (
                lambda: torch.zeros((0,)).view(2, 0, 1, -1),
                "cannot reshape tensor of 0 elements into shape [2, 0, 1, -1] "
                "because the unspecified dimension size -1 can be any value "
                "and is ambiguous",
            ),
            (
                lambda: torch.zeros((0,)).view(2, 0, 1, 1, -1),
                "cannot reshape tensor of 0 elements into shape [2, 0, 1, 1, -1] "
                "because the unspecified dimension size -1 can be any value "
                "and is ambiguous",
            ),
            (
                lambda: torch.zeros((0,)).view(2, 0, 1, 1, 1, -1),
                "cannot reshape tensor of 0 elements into shape [2, 0, 1, 1, 1, -1] "
                "because the unspecified dimension size -1 can be any value "
                "and is ambiguous",
            ),
            (
                lambda: torch.zeros((0,)).view(2, 0, 1, 1, 1, 1, -1),
                "cannot reshape tensor of 0 elements into shape [2, 0, 1, 1, 1, 1, -1] "
                "because the unspecified dimension size -1 can be any value "
                "and is ambiguous",
            ),
        )
        for call, message in variadic_cases:
            with self.subTest(variadic_error=message):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    call()

    def test_dimension_conversion_matches_the_sequence_overload(self):
        tensor = torch.zeros((6,))
        for shape in (
            (IntSubclass(2), np.int64(3)),
            [IndexDimension(2), np.uint32(3)],
            torch.Size((2, 3)),
            (1, True, 6),
        ):
            with self.subTest(shape=shape):
                result = tensor.view(shape)
                self.assertEqual(result.numel(), 6)
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        with self.assertRaises(TypeError):
            tensor.view((True, 6))
        with self.assertRaisesRegex(
            TypeError,
            r"^view\(\): argument 'size' failed to unpack the object at pos 2 "
            r'with error "type must be tuple of ints,but got float"$',
        ):
            tensor.view((2, 3.0))
        with self.assertRaisesRegex(TypeError, "Overflow when unpacking long long"):
            tensor.view((2**63, 1))

    def test_two_positional_dimension_conversion_matches_pytorch_parsing(self):
        tensor = torch.zeros((6,))
        cases = (
            (IntSubclass(2), np.int64(3)),
            (IndexDimension(2), np.uint32(3)),
            (2, IndexDimension(3)),
        )
        for first, second in cases:
            with self.subTest(
                first_type=type(first).__name__, second_type=type(second).__name__
            ):
                result = tensor.view(first, second)
                self.assertEqual(result.shape, (2, 3))
                self.assertEqual(result.stride(), (3, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        bool_result = torch.zeros((1,)).view(1, True)
        self.assertEqual(bool_result.shape, (1, 1))
        invalid_first = (
            "view() received an invalid combination of arguments - got "
            "(bool, int), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_first)}$"):
            tensor.view(True, 6)
        with self.assertRaisesRegex(
            TypeError,
            r"^view\(\): argument 'size' failed to unpack the object at pos 2 "
            r'with error "type must be tuple of ints,but got float"$',
        ):
            tensor.view(2, 3.0)
        with self.assertRaisesRegex(TypeError, "pos 1.*Overflow when unpacking long long"):
            tensor.view(2**63, 1)
        with self.assertRaisesRegex(TypeError, "pos 2.*Overflow when unpacking long long"):
            tensor.view(1, 2**63)

    def test_three_positional_dimension_conversion_matches_pytorch_parsing(self):
        tensor = torch.zeros((24,))
        cases = (
            (IntSubclass(2), np.int64(3), np.uint32(4)),
            (IndexDimension(2), 3, IndexDimension(4)),
            (2, IndexDimension(3), 4),
        )
        for first, second, third in cases:
            with self.subTest(
                first_type=type(first).__name__,
                second_type=type(second).__name__,
                third_type=type(third).__name__,
            ):
                result = tensor.view(first, second, third)
                self.assertEqual(result.shape, (2, 3, 4))
                self.assertEqual(result.stride(), (12, 4, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        for dimensions in ((1, True, 24), (1, 24, True)):
            with self.subTest(dimensions=dimensions):
                result = tensor.view(*dimensions)
                self.assertEqual(result.numel(), 24)
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        invalid_first = (
            "view() received an invalid combination of arguments - got "
            "(bool, int, int), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_first)}$"):
            tensor.view(True, 1, 24)
        for dimensions, position in (((2, 3.0, 4), 2), ((2, 3, 4.0), 3)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^view\(\): argument 'size' failed to unpack the object at pos {position} "
                    r'with error "type must be tuple of ints,but got float"$',
                ):
                    tensor.view(*dimensions)
        for position, dimensions in enumerate(
            ((2**63, 1, 1), (1, 2**63, 1), (1, 1, 2**63)), start=1
        ):
            with self.subTest(overflow_position=position), self.assertRaisesRegex(
                TypeError,
                rf"pos {position}.*Overflow when unpacking long long",
            ):
                tensor.view(*dimensions)

    def test_four_positional_dimension_conversion_matches_pytorch_parsing(self):
        tensor = torch.zeros((48,))
        cases = (
            (IntSubclass(2), np.int64(3), np.uint32(4), IndexDimension(2)),
            (IndexDimension(2), 3, IndexDimension(4), 2),
            (2, IndexDimension(3), 4, np.int64(2)),
        )
        for first, second, third, fourth in cases:
            with self.subTest(
                first_type=type(first).__name__,
                second_type=type(second).__name__,
                third_type=type(third).__name__,
                fourth_type=type(fourth).__name__,
            ):
                result = tensor.view(first, second, third, fourth)
                self.assertEqual(result.shape, (2, 3, 4, 2))
                self.assertEqual(result.stride(), (24, 8, 2, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        for dimensions in ((1, True, 1, 48), (1, 1, 48, True)):
            with self.subTest(dimensions=dimensions):
                result = tensor.view(*dimensions)
                self.assertEqual(result.numel(), 48)
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        invalid_first = (
            "view() received an invalid combination of arguments - got "
            "(bool, int, int, int), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_first)}$"):
            tensor.view(True, 1, 1, 48)
        for dimensions, position in (
            ((2, 3.0, 4, 2), 2),
            ((2, 3, 4.0, 2), 3),
            ((2, 3, 4, 2.0), 4),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^view\(\): argument 'size' failed to unpack the object at pos {position} "
                    r'with error "type must be tuple of ints,but got float"$',
                ):
                    tensor.view(*dimensions)
        for position, dimensions in enumerate(
            (
                (2**63, 1, 1, 1),
                (1, 2**63, 1, 1),
                (1, 1, 2**63, 1),
                (1, 1, 1, 2**63),
            ),
            start=1,
        ):
            with self.subTest(overflow_position=position), self.assertRaisesRegex(
                TypeError,
                rf"pos {position}.*Overflow when unpacking long long",
            ):
                tensor.view(*dimensions)

    def test_five_positional_dimension_conversion_matches_pytorch_parsing(self):
        tensor = torch.zeros((96,))
        cases = (
            (
                IntSubclass(2),
                np.int64(3),
                np.uint32(4),
                IndexDimension(2),
                2,
            ),
            (IndexDimension(2), 3, IndexDimension(4), 2, np.int64(2)),
            (2, IndexDimension(3), 4, np.int64(2), IndexDimension(2)),
        )
        for first, second, third, fourth, fifth in cases:
            with self.subTest(
                first_type=type(first).__name__,
                second_type=type(second).__name__,
                third_type=type(third).__name__,
                fourth_type=type(fourth).__name__,
                fifth_type=type(fifth).__name__,
            ):
                result = tensor.view(first, second, third, fourth, fifth)
                self.assertEqual(result.shape, (2, 3, 4, 2, 2))
                self.assertEqual(result.stride(), (48, 16, 4, 2, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        for dimensions in ((1, True, 1, 1, 96), (1, 1, 1, 96, True)):
            with self.subTest(dimensions=dimensions):
                result = tensor.view(*dimensions)
                self.assertEqual(result.numel(), 96)
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        invalid_first = (
            "view() received an invalid combination of arguments - got "
            "(bool, int, int, int, int), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_first)}$"):
            tensor.view(True, 1, 1, 1, 96)
        for dimensions, position in (
            ((2, 3.0, 4, 2, 2), 2),
            ((2, 3, 4.0, 2, 2), 3),
            ((2, 3, 4, 2.0, 2), 4),
            ((2, 3, 4, 2, 2.0), 5),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^view\(\): argument 'size' failed to unpack the object at pos {position} "
                    r'with error "type must be tuple of ints,but got float"$',
                ):
                    tensor.view(*dimensions)
        for position, dimensions in enumerate(
            (
                (2**63, 1, 1, 1, 1),
                (1, 2**63, 1, 1, 1),
                (1, 1, 2**63, 1, 1),
                (1, 1, 1, 2**63, 1),
                (1, 1, 1, 1, 2**63),
            ),
            start=1,
        ):
            with self.subTest(overflow_position=position), self.assertRaisesRegex(
                TypeError,
                rf"pos {position}.*Overflow when unpacking long long",
            ):
                tensor.view(*dimensions)

    def test_six_positional_dimension_conversion_matches_pytorch_parsing(self):
        tensor = torch.zeros((192,))
        cases = (
            (
                IntSubclass(2),
                np.int64(3),
                np.uint32(4),
                IndexDimension(2),
                2,
                np.int64(2),
            ),
            (IndexDimension(2), 3, IndexDimension(4), 2, np.int64(2), 2),
            (
                2,
                IndexDimension(3),
                4,
                np.int64(2),
                IndexDimension(2),
                2,
            ),
        )
        for first, second, third, fourth, fifth, sixth in cases:
            with self.subTest(
                first_type=type(first).__name__,
                second_type=type(second).__name__,
                third_type=type(third).__name__,
                fourth_type=type(fourth).__name__,
                fifth_type=type(fifth).__name__,
                sixth_type=type(sixth).__name__,
            ):
                result = tensor.view(first, second, third, fourth, fifth, sixth)
                self.assertEqual(result.shape, (2, 3, 4, 2, 2, 2))
                self.assertEqual(result.stride(), (96, 32, 8, 4, 2, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        for dimensions in ((1, True, 1, 1, 1, 192), (1, 1, 1, 1, 192, True)):
            with self.subTest(dimensions=dimensions):
                result = tensor.view(*dimensions)
                self.assertEqual(result.numel(), 192)
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        invalid_first = (
            "view() received an invalid combination of arguments - got "
            "(bool, int, int, int, int, int), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_first)}$"):
            tensor.view(True, 1, 1, 1, 1, 192)
        for dimensions, position in (
            ((2, 3.0, 4, 2, 2, 2), 2),
            ((2, 3, 4.0, 2, 2, 2), 3),
            ((2, 3, 4, 2.0, 2, 2), 4),
            ((2, 3, 4, 2, 2.0, 2), 5),
            ((2, 3, 4, 2, 2, 2.0), 6),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^view\(\): argument 'size' failed to unpack the object at pos {position} "
                    r'with error "type must be tuple of ints,but got float"$',
                ):
                    tensor.view(*dimensions)
        for position, dimensions in enumerate(
            (
                (2**63, 1, 1, 1, 1, 1),
                (1, 2**63, 1, 1, 1, 1),
                (1, 1, 2**63, 1, 1, 1),
                (1, 1, 1, 2**63, 1, 1),
                (1, 1, 1, 1, 2**63, 1),
                (1, 1, 1, 1, 1, 2**63),
            ),
            start=1,
        ):
            with self.subTest(overflow_position=position), self.assertRaisesRegex(
                TypeError,
                rf"pos {position}.*Overflow when unpacking long long",
            ):
                tensor.view(*dimensions)

    def test_seven_or_more_positional_dimension_conversion_matches_pytorch_parsing(self):
        tensor = torch.zeros((24,))
        cases = (
            (
                IntSubclass(2),
                np.int64(3),
                np.uint32(1),
                IndexDimension(1),
                1,
                1,
                IndexDimension(4),
            ),
            (IndexDimension(2), 3, 1, 1, 1, 1, np.int64(4)),
        )
        for dimensions in cases:
            with self.subTest(dimensions=tuple(type(item).__name__ for item in dimensions)):
                result = tensor.view(*dimensions)
                self.assertEqual(result.shape, (2, 3, 1, 1, 1, 1, 4))
                self.assertEqual(result.stride(), (12, 4, 4, 4, 4, 4, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())

        bool_result = torch.zeros((1,)).view(1, True, 1, 1, 1, 1, 1)
        self.assertEqual(bool_result.shape, (1, 1, 1, 1, 1, 1, 1))
        invalid_first = (
            "view() received an invalid combination of arguments - got "
            "(bool, int, int, int, int, int, int), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(invalid_first)}$"):
            tensor.view(True, 1, 1, 1, 1, 1, 24)
        with self.assertRaisesRegex(
            TypeError,
            r"^view\(\): argument 'size' failed to unpack the object at pos 7 "
            r'with error "type must be tuple of ints,but got float"$',
        ):
            tensor.view(2, 3, 1, 1, 1, 1, 4.0)
        with self.assertRaisesRegex(TypeError, "pos 7.*Overflow when unpacking long long"):
            tensor.view(2, 3, 1, 1, 1, 1, 2**63)

    def test_two_positional_dimensions_prefer_dual_sequence_contents(self):
        tensor = torch.zeros((6,))
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            with self.subTest(dimension_type=dimension_type.__name__):
                sequence = dimension_type((6,), 2)
                result = tensor.view(sequence, 3)
                self.assertEqual(result.shape, (6,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(sequence.calls, 1)

                fallback = dimension_type((2.0, 3), 2)
                result = tensor.view(fallback, 3)
                self.assertEqual(result.shape, (2, 3))
                self.assertEqual(result.stride(), (3, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(fallback.calls, 3)

    def test_three_positional_dimensions_prefer_dual_sequence_contents(self):
        tensor = torch.zeros((24,))
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            with self.subTest(dimension_type=dimension_type.__name__):
                sequence = dimension_type((24,), 2)
                result = tensor.view(sequence, 3, 4)
                self.assertEqual(result.shape, (24,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(sequence.calls, 1)

                fallback = dimension_type((2.0, 3), 2)
                result = tensor.view(fallback, 3, 4)
                self.assertEqual(result.shape, (2, 3, 4))
                self.assertEqual(result.stride(), (12, 4, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(fallback.calls, 3)

    def test_four_positional_dimensions_prefer_dual_sequence_contents(self):
        tensor = torch.zeros((48,))
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            with self.subTest(dimension_type=dimension_type.__name__):
                sequence = dimension_type((48,), 2)
                result = tensor.view(sequence, 3, 4, 2)
                self.assertEqual(result.shape, (48,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(sequence.calls, 1)

                fallback = dimension_type((2.0, 3), 2)
                result = tensor.view(fallback, 3, 4, 2)
                self.assertEqual(result.shape, (2, 3, 4, 2))
                self.assertEqual(result.stride(), (24, 8, 2, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(fallback.calls, 3)

    def test_five_positional_dimensions_prefer_dual_sequence_contents(self):
        tensor = torch.zeros((96,))
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            with self.subTest(dimension_type=dimension_type.__name__):
                sequence = dimension_type((96,), 2)
                result = tensor.view(sequence, 3, 4, 2, 2)
                self.assertEqual(result.shape, (96,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(sequence.calls, 1)

                fallback = dimension_type((2.0, 3), 2)
                result = tensor.view(fallback, 3, 4, 2, 2)
                self.assertEqual(result.shape, (2, 3, 4, 2, 2))
                self.assertEqual(result.stride(), (48, 16, 4, 2, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(fallback.calls, 3)

    def test_six_positional_dimensions_prefer_dual_sequence_contents(self):
        tensor = torch.zeros((192,))
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            with self.subTest(dimension_type=dimension_type.__name__):
                sequence = dimension_type((192,), 2)
                result = tensor.view(sequence, 3, 4, 2, 2, 2)
                self.assertEqual(result.shape, (192,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(sequence.calls, 1)

                fallback = dimension_type((2.0, 3), 2)
                result = tensor.view(fallback, 3, 4, 2, 2, 2)
                self.assertEqual(result.shape, (2, 3, 4, 2, 2, 2))
                self.assertEqual(result.stride(), (96, 32, 8, 4, 2, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(fallback.calls, 3)

    def test_seven_or_more_positional_dimensions_prefer_dual_sequence_contents(self):
        tensor = torch.zeros((24,))
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            with self.subTest(dimension_type=dimension_type.__name__):
                sequence = dimension_type((24,), 2)
                result = tensor.view(sequence, 3, 1, 1, 1, 1, 4)
                self.assertEqual(result.shape, (24,))
                self.assertEqual(result.stride(), (1,))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(sequence.calls, 1)

                fallback = dimension_type((2.0, 3), 2)
                result = tensor.view(fallback, 3, 1, 1, 1, 1, 4)
                self.assertEqual(result.shape, (2, 3, 1, 1, 1, 1, 4))
                self.assertEqual(result.stride(), (12, 4, 4, 4, 4, 4, 1))
                self.assertEqual(result.data_ptr(), tensor.data_ptr())
                self.assertEqual(fallback.calls, 3)

    def test_operator_index_poisoning_cannot_change_shape_parsing(self):
        tensor = torch.zeros((6,))
        original_index = operator.index
        try:
            operator.index = lambda value: {2: 1, 3: 6}.get(value, value)

            result = tensor.view((2, 3))
            self.assertEqual(result.shape, (2, 3))
            self.assertEqual(result.stride(), (3, 1))
            self.assertEqual(result.data_ptr(), tensor.data_ptr())
            variadic = tensor.view(2, 3)
            self.assertEqual(variadic.shape, (2, 3))
            self.assertEqual(variadic.stride(), (3, 1))
            self.assertEqual(variadic.data_ptr(), tensor.data_ptr())
            three_variadic = torch.zeros((24,)).view(2, 3, 4)
            self.assertEqual(three_variadic.shape, (2, 3, 4))
            self.assertEqual(three_variadic.stride(), (12, 4, 1))
            four_variadic = torch.zeros((48,)).view(2, 3, 4, 2)
            self.assertEqual(four_variadic.shape, (2, 3, 4, 2))
            self.assertEqual(four_variadic.stride(), (24, 8, 2, 1))
            five_variadic = torch.zeros((96,)).view(2, 3, 4, 2, 2)
            self.assertEqual(five_variadic.shape, (2, 3, 4, 2, 2))
            self.assertEqual(five_variadic.stride(), (48, 16, 4, 2, 1))
            six_variadic = torch.zeros((192,)).view(2, 3, 4, 2, 2, 2)
            self.assertEqual(six_variadic.shape, (2, 3, 4, 2, 2, 2))
            self.assertEqual(six_variadic.stride(), (96, 32, 8, 4, 2, 1))
            seven_variadic = torch.zeros((24,)).view(2, 3, 1, 1, 1, 1, 4)
            self.assertEqual(seven_variadic.shape, (2, 3, 1, 1, 1, 1, 4))
            self.assertEqual(seven_variadic.stride(), (12, 4, 4, 4, 4, 4, 1))
            flattened = tensor.view(-1)
            self.assertEqual(flattened.shape, (6,))
            self.assertEqual(flattened.stride(), (1,))
            self.assertEqual(flattened.data_ptr(), tensor.data_ptr())
            with self.assertRaises(TypeError):
                tensor.view((2, 3.0))
        finally:
            operator.index = original_index

    def test_autograd_repeated_backward_and_no_grad_use_view_semantics(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        result = source.view(3, -1)

        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.stride(), (1, 3))
        self.assertEqual(result.storage_offset(), 0)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assertEqual(result.data_ptr(), source.data_ptr())

        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        (result * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]),
        )

        flat_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        flat = flat_leaf.view(-1)
        self.assertEqual(flat.shape, (6,))
        self.assertEqual(flat.stride(), (1,))
        self.assertFalse(flat.is_leaf)
        weights = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        (flat * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(flat_leaf.grad),
            np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
        )

        three_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        three_source = three_leaf.transpose(0, 1)[1]
        three_result = three_source.view(2, 2, 2)
        self.assertEqual(three_result.shape, (2, 2, 2))
        self.assertEqual(three_result.stride(), (12, 2, 1))
        self.assertEqual(three_result.storage_offset(), 4)
        self.assertTrue(three_result.requires_grad)
        self.assertFalse(three_result.is_leaf)
        self.assertEqual(three_result.data_ptr(), three_source.data_ptr())
        three_weights = torch.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2).tolist()
        )
        (three_result * three_weights).sum().backward()
        expected_three_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_three_grad[:, 1, :] = np.arange(1, 9, dtype=np.float32).reshape(
            2, 4
        )
        np.testing.assert_array_equal(np.asarray(three_leaf.grad), expected_three_grad)

        four_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        four_source = four_leaf.transpose(0, 1)[1]
        four_result = four_source.view(2, 2, 1, 2)
        self.assertEqual(four_result.shape, (2, 2, 1, 2))
        self.assertEqual(four_result.stride(), (12, 2, 2, 1))
        self.assertEqual(four_result.storage_offset(), 4)
        self.assertTrue(four_result.requires_grad)
        self.assertFalse(four_result.is_leaf)
        self.assertEqual(four_result.data_ptr(), four_source.data_ptr())
        four_weights = torch.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 1, 2).tolist()
        )
        (four_result * four_weights).sum().backward()
        expected_four_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_four_grad[:, 1, :] = np.arange(1, 9, dtype=np.float32).reshape(
            2, 4
        )
        np.testing.assert_array_equal(np.asarray(four_leaf.grad), expected_four_grad)

        five_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        five_source = five_leaf.transpose(0, 1)[1]
        five_result = five_source.view(2, 2, 1, 1, 2)
        self.assertEqual(five_result.shape, (2, 2, 1, 1, 2))
        self.assertEqual(five_result.stride(), (12, 2, 2, 2, 1))
        self.assertEqual(five_result.storage_offset(), 4)
        self.assertTrue(five_result.requires_grad)
        self.assertFalse(five_result.is_leaf)
        self.assertEqual(five_result.data_ptr(), five_source.data_ptr())
        five_weights = torch.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 1, 1, 2).tolist()
        )
        (five_result * five_weights).sum().backward()
        expected_five_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_five_grad[:, 1, :] = np.arange(1, 9, dtype=np.float32).reshape(
            2, 4
        )
        np.testing.assert_array_equal(np.asarray(five_leaf.grad), expected_five_grad)

        six_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        six_source = six_leaf.transpose(0, 1)[1]
        six_result = six_source.view(2, 2, 1, 1, 1, 2)
        self.assertEqual(six_result.shape, (2, 2, 1, 1, 1, 2))
        self.assertEqual(six_result.stride(), (12, 2, 2, 2, 2, 1))
        self.assertEqual(six_result.storage_offset(), 4)
        self.assertTrue(six_result.requires_grad)
        self.assertFalse(six_result.is_leaf)
        self.assertEqual(six_result.data_ptr(), six_source.data_ptr())
        six_weights = torch.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 1, 1, 1, 2).tolist()
        )
        (six_result * six_weights).sum().backward()
        expected_six_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_six_grad[:, 1, :] = np.arange(1, 9, dtype=np.float32).reshape(
            2, 4
        )
        np.testing.assert_array_equal(np.asarray(six_leaf.grad), expected_six_grad)

        high_rank_leaf = torch.tensor(
            np.arange(24, dtype=np.float32).tolist(),
            requires_grad=True,
        )
        high_rank_result = high_rank_leaf.view(2, 3, 1, 1, 1, 1, 4)
        self.assertEqual(high_rank_result.shape, (2, 3, 1, 1, 1, 1, 4))
        self.assertEqual(high_rank_result.stride(), (12, 4, 4, 4, 4, 4, 1))
        self.assertTrue(high_rank_result.requires_grad)
        self.assertFalse(high_rank_result.is_leaf)
        self.assertEqual(high_rank_result.data_ptr(), high_rank_leaf.data_ptr())
        high_rank_result.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(high_rank_leaf.grad),
            np.ones((24,), dtype=np.float32),
        )

        repeated_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        repeated_loss = repeated_leaf.transpose(0, 1).view([3, 2]).sum()
        repeated_loss.backward()
        repeated_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(repeated_leaf.grad),
            np.full((2, 3), 2.0, dtype=np.float32),
        )

        no_grad_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_result = no_grad_source.view(3, 2)

        self.assertTrue(no_grad_result.requires_grad)
        self.assertTrue(no_grad_result.is_leaf)
        self.assertEqual(no_grad_result.stride(), (1, 3))
        self.assertEqual(no_grad_result.data_ptr(), no_grad_source.data_ptr())
        self.assertIsNone(no_grad_leaf.grad)

    def test_tensorbase_descriptor_metadata_matches_the_native_method(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "view")
        bound = tensor.view

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "view")
        self.assertEqual(descriptor.__qualname__, "TensorBase.view")
        self.assertEqual(bound.__name__, "view")
        self.assertEqual(bound.__qualname__, "Tensor.view")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor), "<method 'view' of 'torch._C.TensorBase' objects>"
        )
        self.assertIs(torch.Tensor.view, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        for callable_object in (descriptor, bound):
            self.assertTrue(callable_object.__doc__.startswith("\nview(*shape) -> Tensor\n"))
            self.assertIn(".. method:: view(dtype) -> Tensor", callable_object.__doc__)
            self.assertTrue(
                callable_object.__doc__.endswith(
                    ">>> x.view(torch.uint8).size()\n    torch.Size([4, 16])\n"
                )
            )
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor, (2, 1)).shape, (2, 1))
        self.assertEqual(descriptor(tensor, -1).shape, (2,))
        self.assertEqual(descriptor(tensor, 2, 1).shape, (2, 1))
        self.assertEqual(descriptor(tensor, 1, 1, 2).shape, (1, 1, 2))
        self.assertEqual(descriptor(tensor, 1, 1, 1, 2).shape, (1, 1, 1, 2))
        self.assertEqual(
            descriptor(tensor, 1, 1, 1, 1, 2).shape, (1, 1, 1, 1, 2)
        )
        self.assertEqual(
            descriptor(tensor, 1, 1, 1, 1, 1, 2).shape, (1, 1, 1, 1, 1, 2)
        )
        self.assertEqual(
            descriptor(tensor, 1, 1, 1, 1, 1, 1, 2).shape,
            (1, 1, 1, 1, 1, 1, 2),
        )
        self.assertEqual(descriptor(tensor, size=[2, 1]).shape, (2, 1))

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        descriptor = inspect.getattr_static(torch.Tensor, "view")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        size = torch.Size((2, 3))
        tuple_index = TupleIndexDimension((6,), 2)
        list_index = ListIndexDimension((6,), 2)
        cases = (
            ("tuple", lambda: tensor.view((2, 3)), (tensor, (2, 3)), None),
            ("list", lambda: tensor.view([2, 3]), (tensor, [2, 3]), None),
            ("Size", lambda: tensor.view(size), (tensor, size), None),
            ("integer", lambda: tensor.view(-1), (tensor, -1), None),
            ("two integers", lambda: tensor.view(2, 3), (tensor, 2, 3), None),
            (
                "three integers",
                lambda: tensor.view(1, 2, 3),
                (tensor, 1, 2, 3),
                None,
            ),
            (
                "four integers",
                lambda: tensor.view(1, 1, 2, 3),
                (tensor, 1, 1, 2, 3),
                None,
            ),
            (
                "five integers",
                lambda: tensor.view(1, 1, 1, 2, 3),
                (tensor, 1, 1, 1, 2, 3),
                None,
            ),
            (
                "six integers",
                lambda: tensor.view(1, 1, 1, 1, 2, 3),
                (tensor, 1, 1, 1, 1, 2, 3),
                None,
            ),
            (
                "seven integers",
                lambda: tensor.view(1, 1, 1, 1, 1, 2, 3),
                (tensor, 1, 1, 1, 1, 1, 2, 3),
                None,
            ),
            (
                "tuple/index",
                lambda: tensor.view(tuple_index, 3),
                (tensor, tuple_index, 3),
                None,
            ),
            (
                "list/index",
                lambda: tensor.view(list_index, 3),
                (tensor, list_index, 3),
                None,
            ),
            (
                "keyword",
                lambda: tensor.view(size=(2, 3)),
                (tensor,),
                {"size": (2, 3)},
            ),
            (
                "dtype positional",
                lambda: tensor.view(torch.float32),
                (tensor, torch.float32),
                None,
            ),
            (
                "dtype alias positional",
                lambda: tensor.view(torch.float),
                (tensor, torch.float),
                None,
            ),
            (
                "dtype keyword",
                lambda: tensor.view(dtype=torch.float32),
                (tensor,),
                {"dtype": torch.float32},
            ),
            (
                "dtype alias keyword",
                lambda: tensor.view(dtype=torch.float),
                (tensor,),
                {"dtype": torch.float},
            ),
        )
        for case, call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with self.subTest(case=case), mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)
        self.assertEqual((tuple_index.calls, list_index.calls), (1, 1))

        deferred = RecordingMode(marker)
        with deferred:
            self.assertIs(tensor.view((2, 3.0)), marker)
        self.assertEqual(len(deferred.calls), 1)

        variadic_deferred = RecordingMode(marker)
        with variadic_deferred:
            self.assertIs(tensor.view(2, 3.0), marker)
        self.assertEqual(len(variadic_deferred.calls), 1)

        three_variadic_deferred = RecordingMode(marker)
        with three_variadic_deferred:
            self.assertIs(tensor.view(1, 2, 3.0), marker)
        self.assertEqual(len(three_variadic_deferred.calls), 1)
        _, _, args, kwargs = three_variadic_deferred.calls[0]
        self.assertEqual(args, (tensor, 1, 2, 3.0))
        self.assertIsNone(kwargs)

        four_variadic_deferred = RecordingMode(marker)
        with four_variadic_deferred:
            self.assertIs(tensor.view(1, 1, 2, 3.0), marker)
        self.assertEqual(len(four_variadic_deferred.calls), 1)
        _, _, args, kwargs = four_variadic_deferred.calls[0]
        self.assertEqual(args, (tensor, 1, 1, 2, 3.0))
        self.assertIsNone(kwargs)

        five_variadic_deferred = RecordingMode(marker)
        with five_variadic_deferred:
            self.assertIs(tensor.view(1, 1, 1, 2, 3.0), marker)
        self.assertEqual(len(five_variadic_deferred.calls), 1)
        _, _, args, kwargs = five_variadic_deferred.calls[0]
        self.assertEqual(args, (tensor, 1, 1, 1, 2, 3.0))
        self.assertIsNone(kwargs)

        six_variadic_deferred = RecordingMode(marker)
        with six_variadic_deferred:
            self.assertIs(tensor.view(1, 1, 1, 1, 2, 3.0), marker)
        self.assertEqual(len(six_variadic_deferred.calls), 1)
        _, _, args, kwargs = six_variadic_deferred.calls[0]
        self.assertEqual(args, (tensor, 1, 1, 1, 1, 2, 3.0))
        self.assertIsNone(kwargs)

        seven_variadic_deferred = RecordingMode(marker)
        with seven_variadic_deferred:
            self.assertIs(tensor.view(1, 1, 1, 1, 1, 2, 3.0), marker)
        self.assertEqual(len(seven_variadic_deferred.calls), 1)
        _, _, args, kwargs = seven_variadic_deferred.calls[0]
        self.assertEqual(args, (tensor, 1, 1, 1, 1, 1, 2, 3.0))
        self.assertIsNone(kwargs)

        variadic_invalid = RecordingMode(marker)
        with variadic_invalid, self.assertRaises(TypeError):
            tensor.view(2.0, 3)
        self.assertEqual(variadic_invalid.calls, [])

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.view(range(2))
        self.assertEqual(invalid.calls, [])

        invalid_size_dtype = RecordingMode(marker)
        with invalid_size_dtype, self.assertRaises(TypeError):
            tensor.view(size=torch.float32)
        self.assertEqual(invalid_size_dtype.calls, [])

        mixed_dimension = StatefulIndexDimension((6, 6))
        invalid_mixed = RecordingMode(marker)
        with invalid_mixed, self.assertRaises(TypeError):
            tensor.view(mixed_dimension, dtype=torch.float32)
        self.assertEqual(mixed_dimension.calls, 2)
        self.assertEqual(invalid_mixed.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(size=[2, 3])
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"size": [2, 3]})
        self.assertEqual(forwarded.shape, (2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(1, 2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 1, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (1, 2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(1, 1, 2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 1, 1, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (1, 1, 2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(1, 1, 1, 2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 1, 1, 1, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (1, 1, 1, 2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(1, 1, 1, 1, 2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 1, 1, 1, 1, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (1, 1, 1, 1, 2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(1, 1, 1, 1, 1, 2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 1, 1, 1, 1, 1, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (1, 1, 1, 1, 1, 2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(-1)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, -1))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (6,))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(2, 3)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor, 2, 3))
            self.assertIsNone(kwargs)
        self.assertEqual(forwarded.shape, (2, 3))
        self.assertEqual(forwarded.data_ptr(), tensor.data_ptr())

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(dtype=torch.float)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"dtype": torch.float})
        self.assertIsNot(forwarded, tensor)
        self.assertTrue(forwarded.is_set_to(tensor))
        self.assertFalse(forwarded.requires_grad)
        self.assertTrue(forwarded.is_leaf)

        declining = RecordingMode(NotImplemented)
        with self.assertRaises(TypeError) as raised:
            with declining:
                tensor.view((2, 3))
        self.assertTrue(
            str(raised.exception).startswith(
                "Multiple dispatch failed for 'torch.Tensor.view'; all "
                "__torch_function__ handlers returned NotImplemented:"
            )
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_unsupported_shape_and_mixed_overloads_do_not_mutate_the_source(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        original = (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.requires_grad,
            tensor.is_leaf,
            np.asarray(tensor).copy(),
        )
        calls = (
            lambda: tensor.view(size=-1),
            lambda: tensor.view(size=torch.float32),
            lambda: tensor.view(True),
            lambda: tensor.view(torch.float32, 6),
            lambda: tensor.view(torch.float32, size=(6,)),
            lambda: tensor.view(dtype=torch.float32, size=(6,)),
            lambda: tensor.view(torch.float32, dtype=torch.float32),
            lambda: tensor.view((6,), dtype=torch.float32),
        )
        for call in calls:
            with self.subTest(call=call), self.assertRaises(TypeError):
                call()

        keyword_overload = (
            "view() received an invalid combination of arguments - got "
            "(int, int, size=tuple), but expected one of:\n"
            " * (torch.dtype dtype)\n"
            " * (tuple of ints size)\n"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(keyword_overload)}$"):
            tensor.view(2, 3, size=(2, 3))

        three_dimension_keyword_overload = keyword_overload.replace(
            "(int, int, size=tuple)", "(int, int, int, size=tuple)"
        )
        with self.assertRaisesRegex(
            TypeError, f"^{re.escape(three_dimension_keyword_overload)}$"
        ):
            tensor.view(1, 2, 3, size=(1, 2, 3))

        four_dimension_keyword_overload = keyword_overload.replace(
            "(int, int, size=tuple)", "(int, int, int, int, size=tuple)"
        )
        with self.assertRaisesRegex(
            TypeError, f"^{re.escape(four_dimension_keyword_overload)}$"
        ):
            tensor.view(1, 1, 2, 3, size=(1, 1, 2, 3))

        five_dimension_keyword_overload = keyword_overload.replace(
            "(int, int, size=tuple)", "(int, int, int, int, int, size=tuple)"
        )
        with self.assertRaisesRegex(
            TypeError, f"^{re.escape(five_dimension_keyword_overload)}$"
        ):
            tensor.view(1, 1, 1, 2, 3, size=(1, 1, 1, 2, 3))

        six_dimension_keyword_overload = keyword_overload.replace(
            "(int, int, size=tuple)", "(int, int, int, int, int, int, size=tuple)"
        )
        with self.assertRaisesRegex(
            TypeError, f"^{re.escape(six_dimension_keyword_overload)}$"
        ):
            tensor.view(1, 1, 1, 1, 2, 3, size=(1, 1, 1, 1, 2, 3))

        seven_dimension_keyword_overload = keyword_overload.replace(
            "(int, int, size=tuple)",
            "(int, int, int, int, int, int, int, size=tuple)",
        )
        with self.assertRaisesRegex(
            TypeError, f"^{re.escape(seven_dimension_keyword_overload)}$"
        ):
            tensor.view(1, 1, 1, 1, 1, 2, 3, size=(1, 1, 1, 1, 1, 2, 3))

        mixed_dimension = StatefulIndexDimension((6, 6))
        with self.assertRaises(TypeError):
            tensor.view(mixed_dimension, dtype=torch.float32)
        self.assertEqual(mixed_dimension.calls, 2)
        self.assertEqual(
            (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
            original[:-1],
        )
        np.testing.assert_array_equal(np.asarray(tensor), original[-1])


if __name__ == "__main__":
    unittest.main()
