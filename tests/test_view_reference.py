import gc
import inspect
import operator
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorViewReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("view differentials require pinned PyTorch 2.13.0")

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        noncontiguous = base.transpose(0, 1)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32), ()),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (2, 0),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1),
            ),
            ("contiguous", base, (6, 4)),
            ("contiguous-offset", base[1], (2, 6)),
            ("noncontiguous-same-shape", noncontiguous, (3, 2, 4)),
            (
                "noncontiguous-compatible-split",
                noncontiguous,
                (3, 2, 2, 2),
            ),
        )

    def make_dtype_layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        return (
            (
                "scalar",
                module.tensor(
                    -0.0, dtype=module.float32, requires_grad=True
                ),
            ),
            (
                "empty-offset",
                module.zeros(
                    (2, 0, 3), dtype=module.float32, requires_grad=True
                ).transpose(0, 2)[1],
            ),
            ("contiguous-offset", (leaf * 2.0)[1]),
            ("noncontiguous-offset", (leaf * 3.0).transpose(0, 2)[1]),
        )

    def dtype_view_observation(self, module, source, *, alias, keyword):
        dtype = module.float if alias else module.float32
        result = source.view(dtype=dtype) if keyword else source.view(dtype)
        return (
            result is not source,
            result.is_set_to(source),
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            self.tensor_array(result, module).copy(),
        )

    def test_same_dtype_views_match_pytorch_2_13(self):
        actual_cases = self.make_dtype_layout_cases(torch)
        expected_cases = self.make_dtype_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source = actual_case
            expected_name, expected_source = expected_case
            self.assertEqual(case, expected_name)
            for alias in (False, True):
                for keyword in (False, True):
                    with self.subTest(
                        case=case, alias=alias, keyword=keyword
                    ):
                        actual = self.dtype_view_observation(
                            torch,
                            actual_source,
                            alias=alias,
                            keyword=keyword,
                        )
                        expected = self.dtype_view_observation(
                            reference_torch,
                            expected_source,
                            alias=alias,
                            keyword=keyword,
                        )
                        self.assertEqual(actual[:-1], expected[:-1])
                        np.testing.assert_array_equal(actual[-1], expected[-1])

            self.assertTrue(actual_source.requires_grad)
            self.assertTrue(expected_source.requires_grad)

    def test_same_dtype_view_lifetimes_match_pytorch_2_13(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retain(module, *, keyword):
            temporary = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
            )
            source = (temporary * 3.0).transpose(0, 2)[1]
            if keyword:
                return source.view(dtype=module.float)
            return source.view(module.float32)

        retained = []
        for keyword in (False, True):
            retained.append(
                (
                    retain(torch, keyword=keyword),
                    retain(reference_torch, keyword=keyword),
                )
            )
        gc.collect()

        for actual, expected in retained:
            self.assertEqual(
                (
                    tuple(actual.shape),
                    actual.stride(),
                    actual.storage_offset(),
                    actual.requires_grad,
                    actual.is_leaf,
                ),
                (
                    tuple(expected.shape),
                    expected.stride(),
                    expected.storage_offset(),
                    expected.requires_grad,
                    expected.is_leaf,
                ),
            )
            np.testing.assert_array_equal(np.asarray(actual), expected.numpy())

    def shape_argument(self, module, form, shape):
        if form == "tuple" or form == "keyword":
            return tuple(shape)
        if form == "list":
            return list(shape)
        if form == "Size":
            return module.Size(shape)
        raise AssertionError(form)

    def view_observation(self, module, source, shape, form):
        argument = self.shape_argument(module, form, shape)
        result = (
            source.view(size=argument)
            if form == "keyword"
            else source.view(argument)
        )
        direct = source.reshape(tuple(result.shape))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def test_shapes_strides_offsets_aliasing_and_values_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, shape = actual_case
            expected_name, expected_source, expected_shape = expected_case
            self.assertEqual((case, shape), (expected_name, expected_shape))
            for form in ("tuple", "list", "Size", "keyword"):
                with self.subTest(case=case, form=form):
                    actual = self.view_observation(torch, actual_source, shape, form)
                    expected = self.view_observation(
                        reference_torch, expected_source, shape, form
                    )
                    self.assertEqual(actual[:-1], expected[:-1])
                    np.testing.assert_array_equal(actual[-1], expected[-1])

    def single_view_observation(self, module, source, dimension):
        result = source.view(dimension)
        direct = source.reshape(tuple(result.shape))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def single_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32), 1),
            ("inferred", base, -1),
            ("offset", base[1], IntSubclass(12)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                np.int64(-1),
            ),
            (
                "compatible-noncontiguous",
                module.tensor(
                    np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
                    dtype=module.float32,
                ).transpose(0, 1)[0],
                IndexDimension(2),
            ),
        )

    def test_single_integer_shapes_and_views_match_pytorch_2_13(self):
        actual_cases = self.single_view_cases(torch)
        expected_cases = self.single_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimension = actual_case
            expected_name, expected_source, expected_dimension = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.single_view_observation(
                    torch, actual_source, actual_dimension
                )
                expected = self.single_view_observation(
                    reference_torch, expected_source, expected_dimension
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

    def positional_view_observation(self, module, source, dimensions):
        result = source.view(*dimensions)
        direct = source.reshape(tuple(result.shape))
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.is_contiguous(),
            result.requires_grad,
            result.is_leaf,
            str(result.dtype),
            str(result.device),
            result.data_ptr() == source.data_ptr(),
            result.is_set_to(direct),
            self.tensor_array(result, module).copy(),
        )

    def two_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("contiguous-inferred", base, (6, -1)),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(6)),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (IndexDimension(2), 0),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1),
            ),
            (
                "noncontiguous-offset-inferred",
                base.transpose(0, 1)[1],
                (2, IndexDimension(-1)),
            ),
        )

    def test_two_positional_dimensions_match_pytorch_2_13(self):
        actual_cases = self.two_view_cases(torch)
        expected_cases = self.two_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimensions = actual_case
            expected_name, expected_source, expected_dimensions = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.positional_view_observation(
                    torch, actual_source, actual_dimensions
                )
                expected = self.positional_view_observation(
                    reference_torch, expected_source, expected_dimensions
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 1, 2))
            second = StatefulIndexDimension((3,))
            result = module.zeros((6,), dtype=module.float32).view(first, second)
            outcomes.append(
                (tuple(result.shape), result.stride(), first.calls, second.calls)
            )
        self.assertEqual(outcomes[0], outcomes[1])

        invalid_outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 3.0))
            try:
                module.zeros((6,), dtype=module.float32).view(first, 3)
            except Exception as error:
                invalid_outcomes.append((type(error), str(error), first.calls))
            else:
                self.fail(f"{module.__name__} accepted a nonintegral dimension")
        self.assertEqual(invalid_outcomes[0], invalid_outcomes[1])

    def three_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("contiguous-inferred", base, (2, -1, 2)),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(2), IndexDimension(3)),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1, 1),
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 2),
            ),
        )

    def test_three_positional_dimensions_match_pytorch_2_13(self):
        actual_cases = self.three_view_cases(torch)
        expected_cases = self.three_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimensions = actual_case
            expected_name, expected_source, expected_dimensions = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.positional_view_observation(
                    torch, actual_source, actual_dimensions
                )
                expected = self.positional_view_observation(
                    reference_torch, expected_source, expected_dimensions
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 1, 2))
            second = StatefulIndexDimension((3,))
            third = StatefulIndexDimension((4,))
            result = module.zeros((24,), dtype=module.float32).view(
                first, second, third
            )
            outcomes.append(
                (
                    tuple(result.shape),
                    result.stride(),
                    first.calls,
                    second.calls,
                    third.calls,
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def four_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("contiguous-inferred", base, (2, -1, 2, 1)),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(1), 2, IndexDimension(3)),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1, 1),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1, 1, 1),
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 1, 2),
            ),
            (
                "noncontiguous-compatible-split",
                base.transpose(0, 1),
                (3, 2, 2, 2),
            ),
        )

    def test_four_positional_dimensions_match_pytorch_2_13(self):
        actual_cases = self.four_view_cases(torch)
        expected_cases = self.four_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimensions = actual_case
            expected_name, expected_source, expected_dimensions = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.positional_view_observation(
                    torch, actual_source, actual_dimensions
                )
                expected = self.positional_view_observation(
                    reference_torch, expected_source, expected_dimensions
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 1, 2))
            second = StatefulIndexDimension((3,))
            third = StatefulIndexDimension((4,))
            fourth = StatefulIndexDimension((2,))
            result = module.zeros((48,), dtype=module.float32).view(
                first, second, third, fourth
            )
            outcomes.append(
                (
                    tuple(result.shape),
                    result.stride(),
                    first.calls,
                    second.calls,
                    third.calls,
                    fourth.calls,
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def five_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("contiguous-inferred", base, (2, -1, 2, 1, 1)),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(1), 1, 2, IndexDimension(3)),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1, 1, 1),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1, 1, 1, 1),
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 1, 1, 2),
            ),
            (
                "noncontiguous-compatible-split",
                base.transpose(0, 1),
                (3, 2, 2, 1, 2),
            ),
        )

    def test_five_positional_dimensions_match_pytorch_2_13(self):
        actual_cases = self.five_view_cases(torch)
        expected_cases = self.five_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimensions = actual_case
            expected_name, expected_source, expected_dimensions = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.positional_view_observation(
                    torch, actual_source, actual_dimensions
                )
                expected = self.positional_view_observation(
                    reference_torch, expected_source, expected_dimensions
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 1, 2))
            second = StatefulIndexDimension((3,))
            third = StatefulIndexDimension((4,))
            fourth = StatefulIndexDimension((2,))
            fifth = StatefulIndexDimension((2,))
            result = module.zeros((96,), dtype=module.float32).view(
                first, second, third, fourth, fifth
            )
            outcomes.append(
                (
                    tuple(result.shape),
                    result.stride(),
                    first.calls,
                    second.calls,
                    third.calls,
                    fourth.calls,
                    fifth.calls,
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def six_view_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("contiguous-inferred", base, (2, -1, 2, 1, 1, 1)),
            (
                "contiguous-offset",
                base[1],
                (IntSubclass(2), np.int64(1), 1, 1, 2, IndexDimension(3)),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                (IndexDimension(2), 0, 1, 1, 1, 1),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                (0, 1, 1, 1, 1, 1),
            ),
            (
                "noncontiguous-offset-split",
                base.transpose(0, 1)[1],
                (2, 2, 1, 1, 1, 2),
            ),
            (
                "noncontiguous-compatible-split",
                base.transpose(0, 1),
                (3, 2, 2, 1, 1, 2),
            ),
        )

    def test_six_positional_dimensions_match_pytorch_2_13(self):
        actual_cases = self.six_view_cases(torch)
        expected_cases = self.six_view_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_dimensions = actual_case
            expected_name, expected_source, expected_dimensions = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                actual = self.positional_view_observation(
                    torch, actual_source, actual_dimensions
                )
                expected = self.positional_view_observation(
                    reference_torch, expected_source, expected_dimensions
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        outcomes = []
        for module in (torch, reference_torch):
            first = StatefulIndexDimension((2, 1, 2))
            second = StatefulIndexDimension((3,))
            third = StatefulIndexDimension((4,))
            fourth = StatefulIndexDimension((2,))
            fifth = StatefulIndexDimension((2,))
            sixth = StatefulIndexDimension((2,))
            result = module.zeros((192,), dtype=module.float32).view(
                first, second, third, fourth, fifth, sixth
            )
            outcomes.append(
                (
                    tuple(result.shape),
                    result.stride(),
                    first.calls,
                    second.calls,
                    third.calls,
                    fourth.calls,
                    fifth.calls,
                    sixth.calls,
                )
            )
        self.assertEqual(outcomes[0], outcomes[1])

    def test_dual_sequence_index_overload_resolution_matches_pytorch_2_13(self):
        for dimension_type in (TupleIndexDimension, ListIndexDimension):
            outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((6,), dtype=module.float32)
                sequence = dimension_type((6,), 2)
                result = source.view(sequence, 3)
                outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(dimension_type=dimension_type.__name__):
                self.assertEqual(outcomes[0], outcomes[1])

            fallback_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((6,), dtype=module.float32)
                sequence = dimension_type((2.0, 3), 2)
                result = source.view(sequence, 3)
                fallback_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(fallback_type=dimension_type.__name__):
                self.assertEqual(fallback_outcomes[0], fallback_outcomes[1])

            three_dimension_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((24,), dtype=module.float32)
                sequence = dimension_type((24,), 2)
                result = source.view(sequence, 3, 4)
                three_dimension_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                three_dimension_type=dimension_type.__name__
            ):
                self.assertEqual(
                    three_dimension_outcomes[0], three_dimension_outcomes[1]
                )

            three_dimension_fallback_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((24,), dtype=module.float32)
                sequence = dimension_type((2.0, 3), 2)
                result = source.view(sequence, 3, 4)
                three_dimension_fallback_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                three_dimension_fallback_type=dimension_type.__name__
            ):
                self.assertEqual(
                    three_dimension_fallback_outcomes[0],
                    three_dimension_fallback_outcomes[1],
                )

            four_dimension_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((48,), dtype=module.float32)
                sequence = dimension_type((48,), 2)
                result = source.view(sequence, 3, 4, 2)
                four_dimension_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                four_dimension_type=dimension_type.__name__
            ):
                self.assertEqual(
                    four_dimension_outcomes[0], four_dimension_outcomes[1]
                )

            four_dimension_fallback_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((48,), dtype=module.float32)
                sequence = dimension_type((2.0, 3), 2)
                result = source.view(sequence, 3, 4, 2)
                four_dimension_fallback_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                four_dimension_fallback_type=dimension_type.__name__
            ):
                self.assertEqual(
                    four_dimension_fallback_outcomes[0],
                    four_dimension_fallback_outcomes[1],
                )

            five_dimension_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((96,), dtype=module.float32)
                sequence = dimension_type((96,), 2)
                result = source.view(sequence, 3, 4, 2, 2)
                five_dimension_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                five_dimension_type=dimension_type.__name__
            ):
                self.assertEqual(
                    five_dimension_outcomes[0], five_dimension_outcomes[1]
                )

            five_dimension_fallback_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((96,), dtype=module.float32)
                sequence = dimension_type((2.0, 3), 2)
                result = source.view(sequence, 3, 4, 2, 2)
                five_dimension_fallback_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                five_dimension_fallback_type=dimension_type.__name__
            ):
                self.assertEqual(
                    five_dimension_fallback_outcomes[0],
                    five_dimension_fallback_outcomes[1],
                )

            six_dimension_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((192,), dtype=module.float32)
                sequence = dimension_type((192,), 2)
                result = source.view(sequence, 3, 4, 2, 2, 2)
                six_dimension_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                six_dimension_type=dimension_type.__name__
            ):
                self.assertEqual(
                    six_dimension_outcomes[0], six_dimension_outcomes[1]
                )

            six_dimension_fallback_outcomes = []
            for module in (torch, reference_torch):
                source = module.zeros((192,), dtype=module.float32)
                sequence = dimension_type((2.0, 3), 2)
                result = source.view(sequence, 3, 4, 2, 2, 2)
                six_dimension_fallback_outcomes.append(
                    (
                        tuple(result.shape),
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr() == source.data_ptr(),
                        sequence.calls,
                    )
                )
            with self.subTest(
                six_dimension_fallback_type=dimension_type.__name__
            ):
                self.assertEqual(
                    six_dimension_fallback_outcomes[0],
                    six_dimension_fallback_outcomes[1],
                )

    def test_inference_extreme_empty_and_view_errors_match_pytorch_2_13(self):
        actual_source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=torch.float32,
        )
        expected_source = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=reference_torch.float32,
        )
        for form in ("tuple", "list", "Size", "keyword"):
            with self.subTest(kind="inferred", form=form):
                actual = self.view_observation(
                    torch, actual_source, (2, -1, 2), form
                )
                expected = self.view_observation(
                    reference_torch, expected_source, (2, -1, 2), form
                )
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])

        maximum = sys.maxsize
        actual_empty = torch.zeros((0,), dtype=torch.float32)
        expected_empty = reference_torch.zeros(
            (0,), dtype=reference_torch.float32
        )
        for form in ("tuple", "list", "Size", "keyword"):
            with self.subTest(kind="extreme-empty", form=form):
                actual_argument = self.shape_argument(
                    torch, form, (0, maximum, maximum)
                )
                expected_argument = self.shape_argument(
                    reference_torch, form, (0, maximum, maximum)
                )
                actual_result = (
                    actual_empty.view(size=actual_argument)
                    if form == "keyword"
                    else actual_empty.view(actual_argument)
                )
                expected_result = (
                    expected_empty.view(size=expected_argument)
                    if form == "keyword"
                    else expected_empty.view(expected_argument)
                )
                self.assertEqual(
                    (
                        tuple(actual_result.shape),
                        actual_result.stride(),
                        actual_result.storage_offset(),
                        actual_result.numel(),
                        actual_result.data_ptr() == actual_empty.data_ptr(),
                        actual_result.tolist(),
                    ),
                    (
                        tuple(expected_result.shape),
                        expected_result.stride(),
                        expected_result.storage_offset(),
                        expected_result.numel(),
                        expected_result.data_ptr() == expected_empty.data_ptr(),
                        expected_result.tolist(),
                    ),
                )

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_noncontiguous = torch.tensor(
            values.tolist(), dtype=torch.float32
        ).transpose(0, 1)
        expected_noncontiguous = reference_torch.tensor(
            values.tolist(), dtype=reference_torch.float32
        ).transpose(0, 1)
        error_cases = (
            (
                lambda: actual_noncontiguous.view(6, 4),
                lambda: expected_noncontiguous.view(6, 4),
            ),
            (
                lambda: actual_noncontiguous.view(3, 4, 2),
                lambda: expected_noncontiguous.view(3, 4, 2),
            ),
            (
                lambda: actual_noncontiguous.view(1, 6, 2, 2),
                lambda: expected_noncontiguous.view(1, 6, 2, 2),
            ),
            (
                lambda: actual_noncontiguous.view(1, 1, 6, 2, 2),
                lambda: expected_noncontiguous.view(1, 1, 6, 2, 2),
            ),
            (
                lambda: actual_noncontiguous.view(1, 1, 1, 6, 2, 2),
                lambda: expected_noncontiguous.view(1, 1, 1, 6, 2, 2),
            ),
            (
                lambda: actual_noncontiguous.view((6, 4)),
                lambda: expected_noncontiguous.view((6, 4)),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(2, 2),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(2, 2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(1, 2, 2),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, 2, 2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(1, 2, 2, 2),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, 2, 2, 2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    1, 1, 2, 2, 2
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, 1, 2, 2, 2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    1, 1, 1, 2, 2, 2
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, 1, 1, 2, 2, 2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view((2, 2)),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view((2, 2)),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(-1, -1),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(-1, -1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(-1, 1, -1),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(-1, 1, -1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    -1, 1, 1, -1
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(-1, 1, 1, -1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    -1, 1, 1, 1, -1
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(-1, 1, 1, 1, -1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    -1, 1, 1, 1, 1, -1
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(-1, 1, 1, 1, 1, -1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view((-1, -1)),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view((-1, -1)),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(2, -2),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(2, -2),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(1, -2, 3),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, -2, 3),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(1, -2, 1, 3),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, -2, 1, 3),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    1, -2, 1, 1, 3
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, -2, 1, 1, 3),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(
                    1, -2, 1, 1, 1, 3
                ),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(1, -2, 1, 1, 1, 3),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view((2, -2)),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view((2, -2)),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(0, -1),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(0, -1),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(2, 0, -1),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(2, 0, -1),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(2, 0, 1, -1),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(2, 0, 1, -1),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(
                    2, 0, 1, 1, -1
                ),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(2, 0, 1, 1, -1),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(
                    2, 0, 1, 1, 1, -1
                ),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(2, 0, 1, 1, 1, -1),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view((0, -1)),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view((0, -1)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        single_error_cases = (
            (
                lambda: actual_noncontiguous.view(-1),
                lambda: expected_noncontiguous.view(-1),
            ),
            (
                lambda: torch.zeros((6,), dtype=torch.float32).view(5),
                lambda: reference_torch.zeros(
                    (6,), dtype=reference_torch.float32
                ).view(5),
            ),
            (
                lambda: torch.zeros((0,), dtype=torch.float32).view(1),
                lambda: reference_torch.zeros(
                    (0,), dtype=reference_torch.float32
                ).view(1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(single_error_cases):
            with self.subTest(single_error_case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_six_positional_resource_limits_match_pytorch_2_13(self):
        maximum = sys.maxsize
        actual_empty = torch.zeros((0,), dtype=torch.float32)
        expected_empty = reference_torch.zeros(
            (0,), dtype=reference_torch.float32
        )
        actual_one = torch.ones((1,), dtype=torch.float32)
        expected_one = reference_torch.ones(
            (1,), dtype=reference_torch.float32
        )
        for actual_call, expected_call in (
            (
                lambda: actual_empty.view(maximum, maximum, 0, 1, 1, 1),
                lambda: expected_empty.view(maximum, maximum, 0, 1, 1, 1),
            ),
            (
                lambda: actual_empty.view(1, 1, 1, 1, maximum, maximum),
                lambda: expected_empty.view(1, 1, 1, 1, maximum, maximum),
            ),
            (
                lambda: actual_one.view(1, 1, 1, 1, maximum, maximum),
                lambda: expected_one.view(1, 1, 1, 1, maximum, maximum),
            ),
            (
                lambda: torch.ones((4,), dtype=torch.float32).view(
                    maximum, 2, maximum, 1, 1, 2
                ),
                lambda: reference_torch.ones(
                    (4,), dtype=reference_torch.float32
                ).view(maximum, 2, maximum, 1, 1, 2),
            ),
            (
                lambda: torch.ones((6,), dtype=torch.float32).view(
                    maximum, 2, maximum, 1, 3, -1
                ),
                lambda: reference_torch.ones(
                    (6,), dtype=reference_torch.float32
                ).view(maximum, 2, maximum, 1, 3, -1),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        result = source.view(3, -1)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def three_dimension_autograd_outcome(self, module):
        leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)[1]
        result = source.view(2, 2, 2)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2).tolist(),
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def four_dimension_autograd_outcome(self, module):
        leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)[1]
        result = source.view(2, 2, 1, 2)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 1, 2).tolist(),
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def five_dimension_autograd_outcome(self, module):
        leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)[1]
        result = source.view(2, 2, 1, 1, 2)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 1, 1, 2).tolist(),
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def six_dimension_autograd_outcome(self, module):
        leaf = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)[1]
        result = source.view(2, 2, 1, 1, 1, 2)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            np.arange(1, 9, dtype=np.float32).reshape(2, 2, 1, 1, 1, 2).tolist(),
            dtype=module.float32,
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def repeated_backward_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        loss = leaf.transpose(0, 1).view([3, 2]).sum()
        loss.backward()
        loss.backward()
        return self.tensor_array(leaf.grad, module).copy()

    def single_autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        result = leaf.view(-1)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == leaf.data_ptr(),
        )
        weights = module.tensor(
            [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=module.float32
        )
        (result * weights).sum().backward()
        return metadata, self.tensor_array(leaf.grad, module).copy()

    def no_grad_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        with module.no_grad():
            result = source.view(3, 2)
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
            leaf.grad,
        )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        actual_metadata, actual_grad = self.autograd_outcome(torch)
        expected_metadata, expected_grad = self.autograd_outcome(reference_torch)
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_grad, expected_grad)
        actual_three_metadata, actual_three_grad = (
            self.three_dimension_autograd_outcome(torch)
        )
        expected_three_metadata, expected_three_grad = (
            self.three_dimension_autograd_outcome(reference_torch)
        )
        self.assertEqual(actual_three_metadata, expected_three_metadata)
        np.testing.assert_array_equal(actual_three_grad, expected_three_grad)
        actual_four_metadata, actual_four_grad = (
            self.four_dimension_autograd_outcome(torch)
        )
        expected_four_metadata, expected_four_grad = (
            self.four_dimension_autograd_outcome(reference_torch)
        )
        self.assertEqual(actual_four_metadata, expected_four_metadata)
        np.testing.assert_array_equal(actual_four_grad, expected_four_grad)
        actual_five_metadata, actual_five_grad = (
            self.five_dimension_autograd_outcome(torch)
        )
        expected_five_metadata, expected_five_grad = (
            self.five_dimension_autograd_outcome(reference_torch)
        )
        self.assertEqual(actual_five_metadata, expected_five_metadata)
        np.testing.assert_array_equal(actual_five_grad, expected_five_grad)
        actual_six_metadata, actual_six_grad = (
            self.six_dimension_autograd_outcome(torch)
        )
        expected_six_metadata, expected_six_grad = (
            self.six_dimension_autograd_outcome(reference_torch)
        )
        self.assertEqual(actual_six_metadata, expected_six_metadata)
        np.testing.assert_array_equal(actual_six_grad, expected_six_grad)
        np.testing.assert_array_equal(
            self.repeated_backward_outcome(torch),
            self.repeated_backward_outcome(reference_torch),
        )
        actual_single_metadata, actual_single_grad = self.single_autograd_outcome(
            torch
        )
        expected_single_metadata, expected_single_grad = self.single_autograd_outcome(
            reference_torch
        )
        self.assertEqual(actual_single_metadata, expected_single_metadata)
        np.testing.assert_array_equal(actual_single_grad, expected_single_grad)
        self.assertEqual(
            self.no_grad_outcome(torch), self.no_grad_outcome(reference_torch)
        )

    def descriptor_contract(self, module):
        tensor = module.tensor([1.0, 2.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "view")
        bound = tensor.view
        contract = []
        for callable_object, expected_type in (
            (descriptor, types.MethodDescriptorType),
            (bound, types.BuiltinMethodType),
        ):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                signature_error = type(error).__name__
            else:
                signature_error = None
            contract.append(
                (
                    type(callable_object) is expected_type,
                    callable_object.__name__,
                    callable_object.__qualname__,
                    callable_object.__doc__,
                    callable_object.__text_signature__,
                    getattr(callable_object, "__module__", "missing"),
                    signature_error,
                )
            )
        return (
            tuple(contract),
            descriptor.__objclass__.__name__,
            descriptor.__objclass__.__module__,
            repr(descriptor),
            descriptor is module.Tensor.view,
            descriptor.__get__(None, module.Tensor) is descriptor,
            tuple(descriptor(tensor, (2, 1)).shape),
            tuple(descriptor(tensor, -1).shape),
            tuple(descriptor(tensor, 2, 1).shape),
            tuple(descriptor(tensor, 1, 1, 2).shape),
            tuple(descriptor(tensor, 1, 1, 1, 2).shape),
            tuple(descriptor(tensor, 1, 1, 1, 1, 2).shape),
            tuple(descriptor(tensor, 1, 1, 1, 1, 1, 2).shape),
            tuple(descriptor(tensor, size=[2, 1]).shape),
        )

    def test_tensorbase_descriptor_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=module.float32
        )
        descriptor = inspect.getattr_static(module.Tensor, "view")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        def normalize(value):
            if value is tensor:
                return "self"
            if value is module.float32:
                return "float32"
            if isinstance(value, list):
                return "list", tuple(value)
            if isinstance(value, tuple):
                return type(value).__name__, tuple(value)
            return value

        def normalize_call(call):
            function, dispatch_types, args, kwargs = call
            return (
                function is descriptor,
                function.__qualname__,
                dispatch_types,
                tuple(normalize(argument) for argument in args),
                {key: normalize(value) for key, value in kwargs.items()}
                if kwargs is not None
                else None,
            )

        records = []
        tuple_index = TupleIndexDimension((6,), 2)
        list_index = ListIndexDimension((6,), 2)
        calls = (
            lambda: tensor.view((2, 3)),
            lambda: tensor.view([2, 3]),
            lambda: tensor.view(module.Size((2, 3))),
            lambda: tensor.view(-1),
            lambda: tensor.view(2, 3),
            lambda: tensor.view(1, 2, 3),
            lambda: tensor.view(1, 1, 2, 3),
            lambda: tensor.view(1, 1, 1, 2, 3),
            lambda: tensor.view(1, 1, 1, 1, 2, 3),
            lambda: tensor.view(tuple_index, 3),
            lambda: tensor.view(list_index, 3),
            lambda: tensor.view(size=(2, 3)),
            lambda: tensor.view(module.float32),
            lambda: tensor.view(module.float),
            lambda: tensor.view(dtype=module.float32),
            lambda: tensor.view(dtype=module.float),
        )
        for call in calls:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            records.append((result is marker, tuple(map(normalize_call, mode.calls))))

        deferred = RecordingMode(marker)
        with deferred:
            deferred_result = tensor.view((2, 3.0))

        variadic_deferred = RecordingMode(marker)
        with variadic_deferred:
            variadic_deferred_result = tensor.view(2, 3.0)

        three_variadic_deferred = RecordingMode(marker)
        with three_variadic_deferred:
            three_variadic_deferred_result = tensor.view(1, 2, 3.0)

        four_variadic_deferred = RecordingMode(marker)
        with four_variadic_deferred:
            four_variadic_deferred_result = tensor.view(1, 1, 2, 3.0)

        five_variadic_deferred = RecordingMode(marker)
        with five_variadic_deferred:
            five_variadic_deferred_result = tensor.view(1, 1, 1, 2, 3.0)

        six_variadic_deferred = RecordingMode(marker)
        with six_variadic_deferred:
            six_variadic_deferred_result = tensor.view(1, 1, 1, 1, 2, 3.0)

        invalid = RecordingMode(marker)
        try:
            with invalid:
                tensor.view(range(2))
        except Exception as error:
            invalid_error = type(error).__name__
        else:
            self.fail(f"{module.__name__} accepted a range shape")

        invalid_size_dtype = RecordingMode(marker)
        try:
            with invalid_size_dtype:
                tensor.view(size=module.float32)
        except Exception as error:
            invalid_size_dtype_error = (type(error).__name__, str(error))
        else:
            self.fail(f"{module.__name__} accepted a dtype through size=")

        mixed_dimension = StatefulIndexDimension((6, 6))
        invalid_mixed = RecordingMode(marker)
        try:
            with invalid_mixed:
                tensor.view(mixed_dimension, dtype=module.float32)
        except Exception as error:
            invalid_mixed_error = (type(error).__name__, str(error))
        else:
            self.fail(f"{module.__name__} accepted a mixed view overload")

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view(size=[2, 3])
        sequence_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                integer_forwarded = tensor.view(-1)
        integer_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                variadic_forwarded = tensor.view(2, 3)
        variadic_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                three_variadic_forwarded = tensor.view(1, 2, 3)
        three_variadic_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                four_variadic_forwarded = tensor.view(1, 1, 2, 3)
        four_variadic_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                five_variadic_forwarded = tensor.view(1, 1, 1, 2, 3)
        five_variadic_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                six_variadic_forwarded = tensor.view(1, 1, 1, 1, 2, 3)
        six_variadic_order = tuple(order)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                dtype_forwarded = tensor.view(dtype=module.float)
        dtype_order = tuple(order)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor.view((2, 3))
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "dual_sequence_index_calls": (tuple_index.calls, list_index.calls),
            "deferred": (
                deferred_result is marker,
                tuple(map(normalize_call, deferred.calls)),
            ),
            "variadic_deferred": (
                variadic_deferred_result is marker,
                tuple(map(normalize_call, variadic_deferred.calls)),
            ),
            "three_variadic_deferred": (
                three_variadic_deferred_result is marker,
                tuple(map(normalize_call, three_variadic_deferred.calls)),
            ),
            "four_variadic_deferred": (
                four_variadic_deferred_result is marker,
                tuple(map(normalize_call, four_variadic_deferred.calls)),
            ),
            "five_variadic_deferred": (
                five_variadic_deferred_result is marker,
                tuple(map(normalize_call, five_variadic_deferred.calls)),
            ),
            "six_variadic_deferred": (
                six_variadic_deferred_result is marker,
                tuple(map(normalize_call, six_variadic_deferred.calls)),
            ),
            "invalid": invalid_error,
            "invalid_calls": len(invalid.calls),
            "invalid_size_dtype": invalid_size_dtype_error,
            "invalid_size_dtype_calls": len(invalid_size_dtype.calls),
            "invalid_mixed": invalid_mixed_error,
            "invalid_mixed_index_calls": mixed_dimension.calls,
            "invalid_mixed_mode_calls": len(invalid_mixed.calls),
            "forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in sequence_order
            ),
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "integer_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in integer_order
            ),
            "integer_forwarded": (
                tuple(integer_forwarded.shape),
                integer_forwarded.stride(),
                integer_forwarded.storage_offset(),
                integer_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "variadic_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in variadic_order
            ),
            "variadic_forwarded": (
                tuple(variadic_forwarded.shape),
                variadic_forwarded.stride(),
                variadic_forwarded.storage_offset(),
                variadic_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "three_variadic_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in three_variadic_order
            ),
            "three_variadic_forwarded": (
                tuple(three_variadic_forwarded.shape),
                three_variadic_forwarded.stride(),
                three_variadic_forwarded.storage_offset(),
                three_variadic_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "four_variadic_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in four_variadic_order
            ),
            "four_variadic_forwarded": (
                tuple(four_variadic_forwarded.shape),
                four_variadic_forwarded.stride(),
                four_variadic_forwarded.storage_offset(),
                four_variadic_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "five_variadic_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in five_variadic_order
            ),
            "five_variadic_forwarded": (
                tuple(five_variadic_forwarded.shape),
                five_variadic_forwarded.stride(),
                five_variadic_forwarded.storage_offset(),
                five_variadic_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "six_variadic_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in six_variadic_order
            ),
            "six_variadic_forwarded": (
                tuple(six_variadic_forwarded.shape),
                six_variadic_forwarded.stride(),
                six_variadic_forwarded.storage_offset(),
                six_variadic_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "dtype_forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in dtype_order
            ),
            "dtype_forwarded": (
                dtype_forwarded is not tensor,
                dtype_forwarded.is_set_to(tensor),
                tuple(dtype_forwarded.shape),
                dtype_forwarded.stride(),
                dtype_forwarded.storage_offset(),
                dtype_forwarded.requires_grad,
                dtype_forwarded.is_leaf,
                dtype_forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )

    def test_sequence_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        shape_factories = (
            lambda module: (IntSubclass(2), np.int64(3)),
            lambda module: [IndexDimension(2), np.uint32(3)],
            lambda module: module.Size((2, 3)),
            lambda module: (1, True, 6),
        )
        for factory in shape_factories:
            actual_result = actual.view(factory(torch))
            expected_result = expected.view(factory(reference_torch))
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        self.assert_error_matches(
            lambda: actual.view((2, 3.0)),
            lambda: expected.view((2, 3.0)),
        )
        with self.assertRaises(TypeError) as actual_overflow:
            actual.view((2**63, 1))
        with self.assertRaises(TypeError) as expected_overflow:
            expected.view((2**63, 1))
        for error in (actual_overflow.exception, expected_overflow.exception):
            self.assertIn("failed to unpack the object at pos 1", str(error))
            self.assertIn("Overflow when unpacking long long", str(error))

    def test_single_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        for dimension in (IntSubclass(6), np.int64(6), IndexDimension(6)):
            with self.subTest(dimension_type=type(dimension).__name__):
                actual_result = actual.view(dimension)
                expected_result = expected.view(dimension)
                self.assertEqual(
                    (
                        tuple(actual_result.shape),
                        actual_result.stride(),
                        actual_result.data_ptr() == actual.data_ptr(),
                    ),
                    (
                        tuple(expected_result.shape),
                        expected_result.stride(),
                        expected_result.data_ptr() == expected.data_ptr(),
                    ),
                )

        with self.assertRaises(TypeError) as actual_overflow:
            actual.view(2**63)
        with self.assertRaises(TypeError) as expected_overflow:
            expected.view(2**63)
        for error in (actual_overflow.exception, expected_overflow.exception):
            self.assertIn("failed to unpack the object at pos 1", str(error))
            self.assertIn("Overflow when unpacking long long", str(error))

        outcomes = []
        for module in (torch, reference_torch):
            dimension = StatefulIndexDimension((6, 1, 6))
            result = module.zeros((6,), dtype=module.float32).view(dimension)
            outcomes.append((tuple(result.shape), result.stride(), dimension.calls))
        self.assertEqual(outcomes[0], outcomes[1])

    def test_two_positional_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        dimension_factories = (
            lambda: (IntSubclass(2), np.int64(3)),
            lambda: (IndexDimension(2), np.uint32(3)),
            lambda: (2, IndexDimension(3)),
        )
        for factory in dimension_factories:
            actual_dimensions = factory()
            expected_dimensions = factory()
            actual_result = actual.view(*actual_dimensions)
            expected_result = expected.view(*expected_dimensions)
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        actual_bool = torch.zeros((1,), dtype=torch.float32).view(1, True)
        expected_bool = reference_torch.zeros(
            (1,), dtype=reference_torch.float32
        ).view(1, True)
        self.assertEqual(
            (tuple(actual_bool.shape), actual_bool.stride()),
            (tuple(expected_bool.shape), expected_bool.stride()),
        )
        self.assert_error_matches(
            lambda: actual.view(True, 6),
            lambda: expected.view(True, 6),
        )
        self.assert_error_matches(
            lambda: actual.view(2.0, 3),
            lambda: expected.view(2.0, 3),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3.0),
            lambda: expected.view(2, 3.0),
        )
        for position, actual_dimensions, expected_dimensions in (
            (1, (2**63, 1), (2**63, 1)),
            (2, (1, 2**63), (1, 2**63)),
        ):
            with self.assertRaises(TypeError) as actual_overflow:
                actual.view(*actual_dimensions)
            with self.assertRaises(TypeError) as expected_overflow:
                expected.view(*expected_dimensions)
            for error in (actual_overflow.exception, expected_overflow.exception):
                self.assertIn(
                    f"failed to unpack the object at pos {position}", str(error)
                )
                self.assertIn("Overflow when unpacking long long", str(error))

    def test_three_positional_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((24,), dtype=torch.float32)
        expected = reference_torch.zeros(
            (24,), dtype=reference_torch.float32
        )
        dimension_factories = (
            lambda: (IntSubclass(2), np.int64(3), np.uint32(4)),
            lambda: (IndexDimension(2), 3, IndexDimension(4)),
            lambda: (2, IndexDimension(3), 4),
        )
        for factory in dimension_factories:
            actual_dimensions = factory()
            expected_dimensions = factory()
            actual_result = actual.view(*actual_dimensions)
            expected_result = expected.view(*expected_dimensions)
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        for dimensions in ((1, True, 24), (1, 24, True)):
            with self.subTest(dimensions=dimensions):
                actual_result = actual.view(*dimensions)
                expected_result = expected.view(*dimensions)
                self.assertEqual(
                    (tuple(actual_result.shape), actual_result.stride()),
                    (tuple(expected_result.shape), expected_result.stride()),
                )
        self.assert_error_matches(
            lambda: actual.view(True, 1, 24),
            lambda: expected.view(True, 1, 24),
        )
        self.assert_error_matches(
            lambda: actual.view(2.0, 3, 4),
            lambda: expected.view(2.0, 3, 4),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3.0, 4),
            lambda: expected.view(2, 3.0, 4),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4.0),
            lambda: expected.view(2, 3, 4.0),
        )
        for position, dimensions in enumerate(
            ((2**63, 1, 1), (1, 2**63, 1), (1, 1, 2**63)), start=1
        ):
            with self.assertRaises(TypeError) as actual_overflow:
                actual.view(*dimensions)
            with self.assertRaises(TypeError) as expected_overflow:
                expected.view(*dimensions)
            for error in (actual_overflow.exception, expected_overflow.exception):
                self.assertIn(
                    f"failed to unpack the object at pos {position}", str(error)
                )
                self.assertIn("Overflow when unpacking long long", str(error))

    def test_four_positional_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((48,), dtype=torch.float32)
        expected = reference_torch.zeros(
            (48,), dtype=reference_torch.float32
        )
        dimension_factories = (
            lambda: (
                IntSubclass(2),
                np.int64(3),
                np.uint32(4),
                IndexDimension(2),
            ),
            lambda: (IndexDimension(2), 3, IndexDimension(4), 2),
            lambda: (2, IndexDimension(3), 4, np.int64(2)),
        )
        for factory in dimension_factories:
            actual_dimensions = factory()
            expected_dimensions = factory()
            actual_result = actual.view(*actual_dimensions)
            expected_result = expected.view(*expected_dimensions)
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        for dimensions in ((1, True, 1, 48), (1, 1, 48, True)):
            with self.subTest(dimensions=dimensions):
                actual_result = actual.view(*dimensions)
                expected_result = expected.view(*dimensions)
                self.assertEqual(
                    (tuple(actual_result.shape), actual_result.stride()),
                    (tuple(expected_result.shape), expected_result.stride()),
                )
        self.assert_error_matches(
            lambda: actual.view(True, 1, 1, 48),
            lambda: expected.view(True, 1, 1, 48),
        )
        self.assert_error_matches(
            lambda: actual.view(2.0, 3, 4, 2),
            lambda: expected.view(2.0, 3, 4, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3.0, 4, 2),
            lambda: expected.view(2, 3.0, 4, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4.0, 2),
            lambda: expected.view(2, 3, 4.0, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4, 2.0),
            lambda: expected.view(2, 3, 4, 2.0),
        )
        for position, dimensions in enumerate(
            (
                (2**63, 1, 1, 1),
                (1, 2**63, 1, 1),
                (1, 1, 2**63, 1),
                (1, 1, 1, 2**63),
            ),
            start=1,
        ):
            with self.assertRaises(TypeError) as actual_overflow:
                actual.view(*dimensions)
            with self.assertRaises(TypeError) as expected_overflow:
                expected.view(*dimensions)
            for error in (actual_overflow.exception, expected_overflow.exception):
                self.assertIn(
                    f"failed to unpack the object at pos {position}", str(error)
                )
                self.assertIn("Overflow when unpacking long long", str(error))

    def test_five_positional_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((96,), dtype=torch.float32)
        expected = reference_torch.zeros(
            (96,), dtype=reference_torch.float32
        )
        dimension_factories = (
            lambda: (
                IntSubclass(2),
                np.int64(3),
                np.uint32(4),
                IndexDimension(2),
                2,
            ),
            lambda: (IndexDimension(2), 3, IndexDimension(4), 2, np.int64(2)),
            lambda: (2, IndexDimension(3), 4, np.int64(2), IndexDimension(2)),
        )
        for factory in dimension_factories:
            actual_dimensions = factory()
            expected_dimensions = factory()
            actual_result = actual.view(*actual_dimensions)
            expected_result = expected.view(*expected_dimensions)
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        for dimensions in ((1, True, 1, 1, 96), (1, 1, 1, 96, True)):
            with self.subTest(dimensions=dimensions):
                actual_result = actual.view(*dimensions)
                expected_result = expected.view(*dimensions)
                self.assertEqual(
                    (tuple(actual_result.shape), actual_result.stride()),
                    (tuple(expected_result.shape), expected_result.stride()),
                )
        self.assert_error_matches(
            lambda: actual.view(True, 1, 1, 1, 96),
            lambda: expected.view(True, 1, 1, 1, 96),
        )
        self.assert_error_matches(
            lambda: actual.view(2.0, 3, 4, 2, 2),
            lambda: expected.view(2.0, 3, 4, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3.0, 4, 2, 2),
            lambda: expected.view(2, 3.0, 4, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4.0, 2, 2),
            lambda: expected.view(2, 3, 4.0, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4, 2.0, 2),
            lambda: expected.view(2, 3, 4, 2.0, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4, 2, 2.0),
            lambda: expected.view(2, 3, 4, 2, 2.0),
        )
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
            with self.assertRaises(TypeError) as actual_overflow:
                actual.view(*dimensions)
            with self.assertRaises(TypeError) as expected_overflow:
                expected.view(*dimensions)
            for error in (actual_overflow.exception, expected_overflow.exception):
                self.assertIn(
                    f"failed to unpack the object at pos {position}", str(error)
                )
                self.assertIn("Overflow when unpacking long long", str(error))

    def test_six_positional_dimension_conversion_matches_pytorch_2_13(self):
        actual = torch.zeros((192,), dtype=torch.float32)
        expected = reference_torch.zeros(
            (192,), dtype=reference_torch.float32
        )
        dimension_factories = (
            lambda: (
                IntSubclass(2),
                np.int64(3),
                np.uint32(4),
                IndexDimension(2),
                2,
                IndexDimension(2),
            ),
            lambda: (
                IndexDimension(2),
                3,
                IndexDimension(4),
                2,
                np.int64(2),
                2,
            ),
            lambda: (
                2,
                IndexDimension(3),
                4,
                np.int64(2),
                IndexDimension(2),
                np.uint32(2),
            ),
        )
        for factory in dimension_factories:
            actual_dimensions = factory()
            expected_dimensions = factory()
            actual_result = actual.view(*actual_dimensions)
            expected_result = expected.view(*expected_dimensions)
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )

        for dimensions in (
            (1, True, 1, 1, 1, 192),
            (1, 1, 1, 1, 192, True),
        ):
            with self.subTest(dimensions=dimensions):
                actual_result = actual.view(*dimensions)
                expected_result = expected.view(*dimensions)
                self.assertEqual(
                    (tuple(actual_result.shape), actual_result.stride()),
                    (tuple(expected_result.shape), expected_result.stride()),
                )
        self.assert_error_matches(
            lambda: actual.view(True, 1, 1, 1, 1, 192),
            lambda: expected.view(True, 1, 1, 1, 1, 192),
        )
        self.assert_error_matches(
            lambda: actual.view(2.0, 3, 4, 2, 2, 2),
            lambda: expected.view(2.0, 3, 4, 2, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3.0, 4, 2, 2, 2),
            lambda: expected.view(2, 3.0, 4, 2, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4.0, 2, 2, 2),
            lambda: expected.view(2, 3, 4.0, 2, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4, 2.0, 2, 2),
            lambda: expected.view(2, 3, 4, 2.0, 2, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4, 2, 2.0, 2),
            lambda: expected.view(2, 3, 4, 2, 2.0, 2),
        )
        self.assert_error_matches(
            lambda: actual.view(2, 3, 4, 2, 2, 2.0),
            lambda: expected.view(2, 3, 4, 2, 2, 2.0),
        )
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
            with self.assertRaises(TypeError) as actual_overflow:
                actual.view(*dimensions)
            with self.assertRaises(TypeError) as expected_overflow:
                expected.view(*dimensions)
            for error in (actual_overflow.exception, expected_overflow.exception):
                self.assertIn(
                    f"failed to unpack the object at pos {position}", str(error)
                )
                self.assertIn("Overflow when unpacking long long", str(error))

    def test_operator_index_poisoning_matches_pytorch_2_13(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        original_index = operator.index
        try:
            operator.index = lambda value: {2: 1, 3: 6}.get(value, value)

            actual_result = actual.view((2, 3))
            expected_result = expected.view((2, 3))
            self.assertEqual(
                (
                    tuple(actual_result.shape),
                    actual_result.stride(),
                    actual_result.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_result.shape),
                    expected_result.stride(),
                    expected_result.data_ptr() == expected.data_ptr(),
                ),
            )
            actual_variadic = actual.view(2, 3)
            expected_variadic = expected.view(2, 3)
            self.assertEqual(
                (
                    tuple(actual_variadic.shape),
                    actual_variadic.stride(),
                    actual_variadic.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_variadic.shape),
                    expected_variadic.stride(),
                    expected_variadic.data_ptr() == expected.data_ptr(),
                ),
            )
            actual_four_variadic = torch.zeros(
                (48,), dtype=torch.float32
            ).view(2, 3, 4, 2)
            expected_four_variadic = reference_torch.zeros(
                (48,), dtype=reference_torch.float32
            ).view(2, 3, 4, 2)
            self.assertEqual(
                (
                    tuple(actual_four_variadic.shape),
                    actual_four_variadic.stride(),
                ),
                (
                    tuple(expected_four_variadic.shape),
                    expected_four_variadic.stride(),
                ),
            )
            actual_five_variadic = torch.zeros(
                (96,), dtype=torch.float32
            ).view(2, 3, 4, 2, 2)
            expected_five_variadic = reference_torch.zeros(
                (96,), dtype=reference_torch.float32
            ).view(2, 3, 4, 2, 2)
            self.assertEqual(
                (
                    tuple(actual_five_variadic.shape),
                    actual_five_variadic.stride(),
                ),
                (
                    tuple(expected_five_variadic.shape),
                    expected_five_variadic.stride(),
                ),
            )
            actual_six_variadic = torch.zeros(
                (192,), dtype=torch.float32
            ).view(2, 3, 4, 2, 2, 2)
            expected_six_variadic = reference_torch.zeros(
                (192,), dtype=reference_torch.float32
            ).view(2, 3, 4, 2, 2, 2)
            self.assertEqual(
                (
                    tuple(actual_six_variadic.shape),
                    actual_six_variadic.stride(),
                ),
                (
                    tuple(expected_six_variadic.shape),
                    expected_six_variadic.stride(),
                ),
            )
            self.assert_error_matches(
                lambda: actual.view((2, 3.0)),
                lambda: expected.view((2, 3.0)),
            )
            actual_flattened = actual.view(-1)
            expected_flattened = expected.view(-1)
            self.assertEqual(
                (
                    tuple(actual_flattened.shape),
                    actual_flattened.stride(),
                    actual_flattened.data_ptr() == actual.data_ptr(),
                ),
                (
                    tuple(expected_flattened.shape),
                    expected_flattened.stride(),
                    expected_flattened.data_ptr() == expected.data_ptr(),
                ),
            )
        finally:
            operator.index = original_index

    def test_deliberately_unsupported_overloads_remain_outside_the_binding(self):
        actual = torch.zeros((6,), dtype=torch.float32)
        expected = reference_torch.zeros((6,), dtype=reference_torch.float32)
        for actual_call, expected_call in (
            (
                lambda: actual.view(1, 1, 1, 1, 1, 2, 3),
                lambda: expected.view(1, 1, 1, 1, 1, 2, 3),
            ),
            (
                lambda: actual.view(1, 1, 1, 1, 1, 1, 2, 3),
                lambda: expected.view(1, 1, 1, 1, 1, 1, 2, 3),
            ),
        ):
            with self.assertRaises(TypeError):
                actual_call()
            self.assertEqual(expected_call().numel(), 6)

        mixed_calls = (
            (
                lambda: actual.view(size=torch.float32),
                lambda: expected.view(size=reference_torch.float32),
            ),
            (
                lambda: actual.view(torch.float32, 6),
                lambda: expected.view(reference_torch.float32, 6),
            ),
            (
                lambda: actual.view(torch.float32, size=(6,)),
                lambda: expected.view(reference_torch.float32, size=(6,)),
            ),
            (
                lambda: actual.view(dtype=torch.float32, size=(6,)),
                lambda: expected.view(
                    dtype=reference_torch.float32, size=(6,)
                ),
            ),
            (
                lambda: actual.view(torch.float32, dtype=torch.float32),
                lambda: expected.view(
                    reference_torch.float32, dtype=reference_torch.float32
                ),
            ),
            (
                lambda: actual.view((6,), dtype=torch.float32),
                lambda: expected.view((6,), dtype=reference_torch.float32),
            ),
            (
                lambda: actual.view(2, 3, size=(2, 3)),
                lambda: expected.view(2, 3, size=(2, 3)),
            ),
            (
                lambda: actual.view(1, 2, 3, size=(1, 2, 3)),
                lambda: expected.view(1, 2, 3, size=(1, 2, 3)),
            ),
            (
                lambda: actual.view(1, 1, 2, 3, size=(1, 1, 2, 3)),
                lambda: expected.view(1, 1, 2, 3, size=(1, 1, 2, 3)),
            ),
            (
                lambda: actual.view(1, 1, 1, 2, 3, size=(1, 1, 1, 2, 3)),
                lambda: expected.view(1, 1, 1, 2, 3, size=(1, 1, 1, 2, 3)),
            ),
            (
                lambda: actual.view(
                    1,
                    1,
                    1,
                    1,
                    2,
                    3,
                    size=(1, 1, 1, 1, 2, 3),
                ),
                lambda: expected.view(
                    1,
                    1,
                    1,
                    1,
                    2,
                    3,
                    size=(1, 1, 1, 1, 2, 3),
                ),
            ),
        )
        for actual_call, expected_call in mixed_calls:
            self.assert_error_matches(actual_call, expected_call)

        actual_dimension = StatefulIndexDimension((6, 6))
        expected_dimension = StatefulIndexDimension((6, 6))
        self.assert_error_matches(
            lambda: actual.view(actual_dimension, dtype=torch.float32),
            lambda: expected.view(
                expected_dimension, dtype=reference_torch.float32
            ),
        )
        self.assertEqual(actual_dimension.calls, expected_dimension.calls)
        self.assertEqual(actual_dimension.calls, 2)

        original = (
            tuple(actual.shape),
            actual.stride(),
            actual.storage_offset(),
            actual.data_ptr(),
            np.asarray(actual).copy(),
        )
        for dtype in (reference_torch.float64, reference_torch.int32):
            for keyword in (False, True):
                with self.subTest(dtype=dtype, keyword=keyword):
                    with self.assertRaises(TypeError):
                        if keyword:
                            actual.view(dtype=dtype)
                        else:
                            actual.view(dtype)

        with self.assertRaises(TypeError):
            actual.view(size=-1)
        with self.assertRaises(TypeError):
            expected.view(size=-1)
        with self.assertRaises(TypeError):
            actual.view(True)
        with self.assertRaises(TypeError):
            expected.view(True)
        self.assertEqual(
            (
                tuple(actual.shape),
                actual.stride(),
                actual.storage_offset(),
                actual.data_ptr(),
            ),
            original[:-1],
        )
        np.testing.assert_array_equal(np.asarray(actual), original[-1])


if __name__ == "__main__":
    unittest.main()
