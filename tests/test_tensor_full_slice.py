import gc
import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorFullSliceIndexTests(unittest.TestCase):
    def layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("vector", torch.tensor([-0.0, 1.0])),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("offset-noncontiguous", base.transpose(0, 3)[1]),
        )

    def double_full_slice_layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("matrix", torch.tensor([[-0.0, 1.0], [2.0, 3.0]])),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset", base[1]),
            ("offset-noncontiguous", base.transpose(0, 3)[1]),
        )

    def higher_rank_full_slice_layout_cases(self, rank):
        shape = tuple(range(2, rank + 2))
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        base_shape = (2, *shape)
        base_values = np.arange(np.prod(base_shape), dtype=np.float32).reshape(
            base_shape
        )
        base = torch.tensor(base_values.tolist())
        empty_shape = (shape[0], 0, *shape[2:])
        return (
            ("contiguous", torch.tensor(values.tolist())),
            ("empty", torch.zeros(empty_shape)),
            ("offset", base[1]),
            ("offset-noncontiguous", base.transpose(0, rank)[1]),
        )

    def full_slice_ellipsis_indices(self, slice_count):
        full_slices = (slice(None),) * slice_count
        return tuple(
            (*full_slices[:position], Ellipsis, *full_slices[position:])
            for position in range(slice_count + 1)
        )

    def full_slice_ellipsis_layout_cases(self, slice_count):
        if slice_count == 1:
            return self.layout_cases()
        if slice_count == 2:
            return self.double_full_slice_layout_cases()
        return self.higher_rank_full_slice_layout_cases(slice_count)

    def leading_integer_full_slice_layout_cases(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        base = torch.tensor(values.tolist())
        return (
            ("contiguous", torch.tensor([[-0.0, 1.0], [2.0, 3.0]]), -1),
            ("empty", torch.zeros((2, 0, 3)), -1),
            ("offset", base[1], 1),
            ("offset-noncontiguous", base[1].transpose(0, 1), -1),
        )

    def two_leading_integer_full_slice_layout_cases(self):
        contiguous_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        contiguous_values[1, 1, 0] = -0.0
        base_values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        base = torch.tensor(base_values.tolist())
        return (
            (
                "contiguous",
                torch.tensor(contiguous_values.tolist()),
                (-1, -2),
            ),
            ("empty", torch.zeros((2, 3, 0, 4)), (-1, -2)),
            ("offset", base[1], (-2, -1)),
            ("offset-noncontiguous", base[1].transpose(0, 2), (-1, -2)),
            (
                "wrapping-empty",
                torch.zeros((sys.maxsize, 1, 0, 3)),
                (sys.maxsize - 1, 0),
            ),
        )

    def three_leading_integer_full_slice_layout_cases(self):
        contiguous_values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        contiguous_values[1, 1, 1, 0] = -0.0
        base_values = np.arange(720, dtype=np.float32).reshape(2, 3, 4, 5, 6)
        base = torch.tensor(base_values.tolist())
        return (
            (
                "contiguous",
                torch.tensor(contiguous_values.tolist()),
                (-1, -2, -3),
            ),
            ("empty", torch.zeros((2, 3, 4, 0, 5)), (-1, -2, -3)),
            ("offset", base[1], (-2, -1, -3)),
            (
                "offset-noncontiguous",
                base[1].transpose(0, 3),
                (-1, -2, -3),
            ),
            (
                "wrapping-empty",
                torch.zeros((sys.maxsize, 1, 1, 0, 3)),
                (sys.maxsize - 1, 0, 0),
            ),
        )

    def four_leading_integer_full_slice_layout_cases(self):
        contiguous_values = np.arange(720, dtype=np.float32).reshape(
            2, 3, 4, 5, 6
        )
        contiguous_values[1, 1, 1, 1, 0] = -0.0
        base_values = np.arange(5040, dtype=np.float32).reshape(
            2, 3, 4, 5, 6, 7
        )
        base = torch.tensor(base_values.tolist())
        return (
            (
                "contiguous",
                torch.tensor(contiguous_values.tolist()),
                (-1, -2, -3, -4),
            ),
            (
                "empty",
                torch.zeros((2, 3, 4, 5, 0, 6)),
                (-1, -2, -3, -4),
            ),
            ("offset", base[1], (-2, -1, -3, -4)),
            (
                "offset-noncontiguous",
                base[1].transpose(0, 4),
                (-1, -2, -3, -4),
            ),
            (
                "wrapping-empty",
                torch.zeros((sys.maxsize, 1, 1, 1, 0, 3)),
                (sys.maxsize - 1, 0, 0, 0),
            ),
        )

    def assert_metadata_alias(self, source, alias):
        self.assertIsNot(alias, source)
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertTrue(alias.is_set_to(source))
        self.assertEqual(alias.data_ptr(), source.data_ptr())
        self.assertIs(alias.dtype, source.dtype)
        self.assertEqual(alias.device, source.device)
        self.assertEqual(alias.tolist(), source.tolist())

    def assert_same_view(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertTrue(actual.is_set_to(expected))
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def test_full_slice_returns_a_distinct_exact_metadata_alias(self):
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[:]
                self.assert_metadata_alias(source, alias)

        signed_zero = torch.tensor([-0.0])[:]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000)

    def test_singleton_full_slice_tuple_returns_a_distinct_exact_metadata_alias(self):
        index = (slice(None),)
        for case, source in self.layout_cases():
            with self.subTest(case=case):
                alias = source[index]
                self.assert_metadata_alias(source, alias)

        signed_zero = torch.tensor([-0.0])[index]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000)

    def test_double_full_slice_tuple_returns_a_distinct_exact_metadata_alias(self):
        index = (slice(None), slice(None))
        for case, source in self.double_full_slice_layout_cases():
            with self.subTest(case=case):
                alias = source[index]
                self.assert_metadata_alias(source, alias)

        signed_zero = torch.tensor([[-0.0]])[index]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000)

    def test_three_or_more_full_slice_tuples_return_exact_metadata_aliases(self):
        for count in (3, 4):
            index = (slice(None),) * count
            for case, source in self.higher_rank_full_slice_layout_cases(count):
                with self.subTest(count=count, case=case):
                    alias = source[index]
                    self.assert_metadata_alias(source, alias)

            signed_zero = torch.tensor(
                np.full((1,) * count, -0.0, dtype=np.float32).tolist()
            )[index]
            self.assertEqual(
                np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000
            )

    def test_full_slice_ellipsis_tuples_return_exact_metadata_aliases(self):
        for slice_count in range(1, 5):
            cases = self.full_slice_ellipsis_layout_cases(slice_count)
            signed_zero_source = torch.tensor(
                np.full((1,) * slice_count, -0.0, dtype=np.float32).tolist()
            )
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                for case, source in cases:
                    with self.subTest(
                        slice_count=slice_count,
                        position=position,
                        case=case,
                    ):
                        self.assert_metadata_alias(source, source[index])

                signed_zero = signed_zero_source[index]
                self.assertEqual(
                    np.asarray(signed_zero).view(np.uint32).item(), 0x8000_0000
                )

    def test_leading_integer_full_slice_reuses_the_integer_view(self):
        for case, source, index in self.leading_integer_full_slice_layout_cases():
            with self.subTest(case=case):
                self.assert_same_view(source[index, :], source[index])

    def test_leading_integer_full_slice_accepts_integer_protocol_values(self):
        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        source = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        cases = (
            (IntegerSubclass(1), 1),
            (np.int64(-1), 1),
            (np.uint64(0), 0),
            (IndexValue(-1), 1),
        )
        for index, normalized in cases:
            with self.subTest(index=repr(index)):
                self.assert_same_view(source[index, :], source[normalized])
        self.assertEqual(cases[-1][0].calls, 1)

    def test_two_leading_integer_full_slice_reuses_the_integer_tuple_view(self):
        for case, source, indices in (
            self.two_leading_integer_full_slice_layout_cases()
        ):
            with self.subTest(case=case):
                selected = source[indices[0], indices[1], :]
                self.assert_same_view(selected, source[indices])
                if case == "wrapping-empty":
                    self.assertEqual(selected.storage_offset(), sys.maxsize - 5)

    def test_two_leading_integer_full_slice_accepts_integer_protocol_values(self):
        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first_dynamic = IndexValue(-1)
        second_dynamic = IndexValue(-2)
        source = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )
        cases = (
            ((IntegerSubclass(1), np.int64(-1)), (1, 2)),
            ((np.uint64(0), second_dynamic), (0, 1)),
            ((first_dynamic, IntegerSubclass(0)), (1, 0)),
        )
        for indices, normalized in cases:
            with self.subTest(indices=repr(indices)):
                self.assert_same_view(
                    source[indices[0], indices[1], :], source[normalized]
                )
        self.assertEqual(first_dynamic.calls, 1)
        self.assertEqual(second_dynamic.calls, 1)

    def test_three_leading_integer_full_slice_reuses_the_integer_tuple_view(self):
        for case, source, indices in (
            self.three_leading_integer_full_slice_layout_cases()
        ):
            with self.subTest(case=case):
                selected = source[indices[0], indices[1], indices[2], :]
                self.assert_same_view(selected, source[indices])
                if case == "wrapping-empty":
                    self.assertEqual(selected.storage_offset(), sys.maxsize - 5)

    def test_three_leading_integer_full_slice_accepts_integer_protocol_values(self):
        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first_dynamic = IndexValue(-1)
        second_dynamic = IndexValue(-2)
        third_dynamic = IndexValue(-3)
        source = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )
        cases = (
            (
                (IntegerSubclass(1), np.int64(-1), np.uint64(2)),
                (1, 2, 2),
            ),
            (
                (np.uint64(0), second_dynamic, IntegerSubclass(-1)),
                (0, 1, 3),
            ),
            (
                (first_dynamic, IntegerSubclass(0), third_dynamic),
                (1, 0, 1),
            ),
        )
        for indices, normalized in cases:
            with self.subTest(indices=repr(indices)):
                self.assert_same_view(
                    source[indices[0], indices[1], indices[2], :],
                    source[normalized],
                )
        self.assertEqual(first_dynamic.calls, 1)
        self.assertEqual(second_dynamic.calls, 1)
        self.assertEqual(third_dynamic.calls, 1)

    def test_four_leading_integer_full_slice_reuses_the_integer_tuple_view(self):
        for case, source, indices in (
            self.four_leading_integer_full_slice_layout_cases()
        ):
            with self.subTest(case=case):
                selected = source[
                    indices[0], indices[1], indices[2], indices[3], :
                ]
                self.assert_same_view(selected, source[indices])
                if case == "wrapping-empty":
                    self.assertEqual(selected.storage_offset(), sys.maxsize - 5)

    def test_four_leading_integer_full_slice_accepts_integer_protocol_values(self):
        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first_dynamic = IndexValue(-1)
        second_dynamic = IndexValue(-2)
        third_dynamic = IndexValue(-3)
        fourth_dynamic = IndexValue(-4)
        source = torch.tensor(
            np.arange(720, dtype=np.float32).reshape(2, 3, 4, 5, 6).tolist()
        )
        cases = (
            (
                (
                    IntegerSubclass(1),
                    np.int64(-1),
                    np.uint64(2),
                    IntegerSubclass(-4),
                ),
                (1, 2, 2, 1),
            ),
            (
                (
                    np.uint64(0),
                    second_dynamic,
                    IntegerSubclass(-1),
                    fourth_dynamic,
                ),
                (0, 1, 3, 1),
            ),
            (
                (
                    first_dynamic,
                    IntegerSubclass(0),
                    third_dynamic,
                    np.uint64(3),
                ),
                (1, 0, 1, 3),
            ),
        )
        for indices, normalized in cases:
            with self.subTest(indices=repr(indices)):
                self.assert_same_view(
                    source[
                        indices[0],
                        indices[1],
                        indices[2],
                        indices[3],
                        :,
                    ],
                    source[normalized],
                )
        self.assertEqual(first_dynamic.calls, 1)
        self.assertEqual(second_dynamic.calls, 1)
        self.assertEqual(third_dynamic.calls, 1)
        self.assertEqual(fourth_dynamic.calls, 1)

    def test_scalar_full_slice_raises_the_exact_pytorch_error(self):
        with self.assertRaises(IndexError) as raised:
            torch.tensor(-0.0)[:]
        self.assertEqual(
            str(raised.exception),
            "slice() cannot be applied to a 0-dim tensor.",
        )

    def test_scalar_singleton_full_slice_tuple_raises_too_many_indices(self):
        with self.assertRaises(IndexError) as raised:
            torch.tensor(-0.0)[(slice(None),)]
        self.assertEqual(
            str(raised.exception),
            "too many indices for tensor of dimension 0",
        )

    def test_double_full_slice_tuple_requires_rank_two(self):
        index = (slice(None), slice(None))
        for dimensions, source in (
            (0, torch.tensor(-0.0)),
            (1, torch.tensor([-0.0, 1.0])),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(IndexError) as raised:
                    source[index]
                self.assertEqual(
                    str(raised.exception),
                    f"too many indices for tensor of dimension {dimensions}",
                )

    def test_three_or_more_full_slice_tuples_require_sufficient_rank(self):
        for count in (3, 4):
            index = (slice(None),) * count
            for dimensions in range(count):
                with self.subTest(count=count, dimensions=dimensions):
                    with self.assertRaises(IndexError) as raised:
                        torch.zeros((2,) * dimensions)[index]
                    self.assertEqual(
                        str(raised.exception),
                        f"too many indices for tensor of dimension {dimensions}",
                    )

    def test_full_slice_ellipsis_tuples_count_only_slices_against_rank(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                for dimensions in range(slice_count):
                    with self.subTest(
                        slice_count=slice_count,
                        position=position,
                        dimensions=dimensions,
                    ):
                        with self.assertRaises(IndexError) as raised:
                            torch.zeros((2,) * dimensions)[index]
                        self.assertEqual(
                            str(raised.exception),
                            f"too many indices for tensor of dimension {dimensions}",
                        )

    def assert_autograd_gradient_node_and_no_grad_leaf_status(
        self, index, node_name, diagnostic_rank=1
    ):
        rank = max(diagnostic_rank, 2)
        shape = (2,) * rank
        values = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        alias = leaf[index]
        self.assert_metadata_alias(leaf, alias)
        self.assertTrue(alias.requires_grad)
        self.assertFalse(alias.is_leaf)
        weight_values = np.arange(10, 10 + np.prod(shape), dtype=np.float32).reshape(
            shape
        )
        weights = torch.tensor(weight_values.tolist())
        (alias * weights).sum().backward()
        self.assertEqual(leaf.grad.tolist(), weights.tolist())

        diagnostic_values = np.full(
            (1,) * diagnostic_rank, 2.0, dtype=np.float32
        ).tolist()
        diagnostic_pattern = (
            r"\[" * diagnostic_rank + r"2\." + r"\]" * diagnostic_rank
        )
        diagnostic_leaf = torch.tensor(diagnostic_values, requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            rf"tensor\({diagnostic_pattern}, grad_fn=<{node_name}>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[index], training=False
            )

        no_grad_leaf = torch.tensor(values.tolist(), requires_grad=True)
        no_grad_source = no_grad_leaf.transpose(0, 1)
        with torch.no_grad():
            no_grad_alias = no_grad_source[index]
        self.assert_metadata_alias(no_grad_source, no_grad_alias)
        self.assertTrue(no_grad_alias.requires_grad)
        self.assertTrue(no_grad_alias.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)

    def test_slice_autograd_gradient_node_and_no_grad_leaf_status(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            slice(None), "SliceBackward0"
        )

    def test_singleton_full_slice_tuple_uses_alias_autograd_semantics(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            (slice(None),), "AliasBackward0"
        )

    def test_double_full_slice_tuple_uses_alias_autograd_semantics(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            (slice(None), slice(None)), "AliasBackward0", diagnostic_rank=2
        )

    def test_three_or_more_full_slice_tuples_use_alias_autograd_semantics(self):
        for count in (3, 4):
            with self.subTest(count=count):
                self.assert_autograd_gradient_node_and_no_grad_leaf_status(
                    (slice(None),) * count,
                    "AliasBackward0",
                    diagnostic_rank=count,
                )

    def test_full_slice_ellipsis_tuples_use_alias_autograd_semantics(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                with self.subTest(slice_count=slice_count, position=position):
                    self.assert_autograd_gradient_node_and_no_grad_leaf_status(
                        index,
                        "AliasBackward0",
                        diagnostic_rank=slice_count,
                    )

    def test_empty_tuple_uses_alias_autograd_semantics(self):
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            (), "AliasBackward0"
        )

    def test_leading_integer_full_slice_uses_integer_autograd_semantics(self):
        leaf = torch.tensor([float(value) for value in range(48)], requires_grad=True)
        source = (leaf * 2.0).reshape(2, 2, 3, 4)[1].transpose(0, 1)
        selected = source[-2, :]
        self.assert_same_view(selected, source[-2])
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)

        (selected.transpose(0, 1) * 3.0).sum().backward()
        expected_gradient = [0.0] * 48
        for index in (*range(28, 32), *range(40, 44)):
            expected_gradient[index] = 6.0
        self.assertEqual(leaf.grad.tolist(), expected_gradient)

        diagnostic_leaf = torch.tensor([[2.0]], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"tensor\(\[2\.\], grad_fn=<SelectBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[0, :], training=False
            )

        no_grad_source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = no_grad_source[1, :]
        self.assert_same_view(untracked, no_grad_source[1])
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty[-1, :].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_two_leading_integer_full_slice_uses_integer_autograd_semantics(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        source = (leaf * 2.0).reshape(2, 3, 4, 5)[1].transpose(0, 2)
        selected = source[-1, -2, :]
        self.assert_same_view(selected, source[-1, -2])
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)

        weights = torch.tensor([3.0, 5.0, 7.0])
        (selected * weights).sum().backward()
        expected_gradient = np.zeros_like(values)
        expected_gradient[1, :, -2, -1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(values.shape), expected_gradient
        )

        diagnostic_leaf = torch.tensor([[[2.0]]], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"tensor\(\[2\.\], grad_fn=<SelectBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[0, 0, :], training=False
            )

        no_grad_source = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            requires_grad=True,
        )
        with torch.no_grad():
            untracked = no_grad_source[1, 0, :]
        self.assert_same_view(untracked, no_grad_source[1, 0])
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

        empty = torch.zeros((2, 3, 0, 4), requires_grad=True)
        empty[-1, -2, :].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 3, 0, 4))
        self.assertEqual(empty.grad.tolist(), [[[], [], []], [[], [], []]])

        wrapping_empty = torch.zeros(
            (sys.maxsize, 1, 0, 3), requires_grad=True
        )
        wrapping_selected = wrapping_empty[sys.maxsize - 1, 0, :]
        self.assertEqual(wrapping_selected.shape, (0, 3))
        self.assertEqual(wrapping_selected.stride(), (3, 1))
        self.assertEqual(wrapping_selected.storage_offset(), sys.maxsize - 5)
        self.assertTrue(wrapping_selected.requires_grad)
        self.assertFalse(wrapping_selected.is_leaf)
        wrapping_selected.sum().backward()
        self.assertEqual(wrapping_empty.grad.shape, wrapping_empty.shape)
        self.assertEqual(wrapping_empty.grad.stride(), wrapping_empty.stride())
        self.assertEqual(wrapping_empty.grad.storage_offset(), 0)

    def test_three_leading_integer_full_slice_uses_integer_autograd_semantics(self):
        values = np.arange(720, dtype=np.float32).reshape(2, 3, 4, 5, 6)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        source = (leaf * 2.0).reshape(2, 3, 4, 5, 6)[1].transpose(0, 3)
        selected = source[-1, -2, -3, :]
        self.assert_same_view(selected, source[-1, -2, -3])
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)

        weights = torch.tensor([3.0, 5.0, 7.0])
        (selected * weights).sum().backward()
        expected_gradient = np.zeros_like(values)
        expected_gradient[1, :, -2, -3, -1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(values.shape), expected_gradient
        )

        diagnostic_leaf = torch.tensor([[[[2.0]]]], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"tensor\(\[2\.\], grad_fn=<SelectBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[0, 0, 0, :], training=False
            )

        no_grad_values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
        no_grad_source = torch.tensor(
            no_grad_values.tolist(), requires_grad=True
        ).transpose(0, 3)
        with torch.no_grad():
            untracked = no_grad_source[1, 0, 1, :]
        self.assert_same_view(untracked, no_grad_source[1, 0, 1])
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

        empty = torch.zeros((2, 3, 4, 0, 5), requires_grad=True)
        empty[-1, -2, -3, :].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 3, 4, 0, 5))
        self.assertEqual(empty.grad.numel(), 0)

        wrapping_empty = torch.zeros(
            (sys.maxsize, 1, 1, 0, 3), requires_grad=True
        )
        wrapping_selected = wrapping_empty[sys.maxsize - 1, 0, 0, :]
        self.assertEqual(wrapping_selected.shape, (0, 3))
        self.assertEqual(wrapping_selected.stride(), (3, 1))
        self.assertEqual(wrapping_selected.storage_offset(), sys.maxsize - 5)
        self.assertTrue(wrapping_selected.requires_grad)
        self.assertFalse(wrapping_selected.is_leaf)
        wrapping_selected.sum().backward()
        self.assertEqual(wrapping_empty.grad.shape, wrapping_empty.shape)
        self.assertEqual(wrapping_empty.grad.stride(), wrapping_empty.stride())
        self.assertEqual(wrapping_empty.grad.storage_offset(), 0)

    def test_four_leading_integer_full_slice_uses_integer_autograd_semantics(self):
        values = np.arange(5040, dtype=np.float32).reshape(2, 3, 4, 5, 6, 7)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        source = (leaf * 2.0).reshape(2, 3, 4, 5, 6, 7)[1].transpose(0, 4)
        selected = source[-1, -2, -3, -4, :]
        self.assert_same_view(selected, source[-1, -2, -3, -4])
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)

        weights = torch.tensor([3.0, 5.0, 7.0])
        (selected * weights).sum().backward()
        expected_gradient = np.zeros_like(values)
        expected_gradient[1, :, -2, -3, -4, -1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(values.shape), expected_gradient
        )

        diagnostic_leaf = torch.tensor([[[[[2.0]]]]], requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"tensor\(\[2\.\], grad_fn=<SelectBackward0>\)$",
        ):
            torch.nn.functional.dropout(
                None, p=diagnostic_leaf[0, 0, 0, 0, :], training=False
            )

        no_grad_values = np.arange(32, dtype=np.float32).reshape(2, 2, 2, 2, 2)
        no_grad_source = torch.tensor(
            no_grad_values.tolist(), requires_grad=True
        ).transpose(0, 4)
        with torch.no_grad():
            untracked = no_grad_source[1, 0, 1, 0, :]
        self.assert_same_view(untracked, no_grad_source[1, 0, 1, 0])
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

        empty = torch.zeros((2, 3, 4, 5, 0, 6), requires_grad=True)
        empty[-1, -2, -3, -4, :].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 3, 4, 5, 0, 6))
        self.assertEqual(empty.grad.numel(), 0)

        wrapping_empty = torch.zeros(
            (sys.maxsize, 1, 1, 1, 0, 3), requires_grad=True
        )
        wrapping_selected = wrapping_empty[sys.maxsize - 1, 0, 0, 0, :]
        self.assertEqual(wrapping_selected.shape, (0, 3))
        self.assertEqual(wrapping_selected.stride(), (3, 1))
        self.assertEqual(wrapping_selected.storage_offset(), sys.maxsize - 5)
        self.assertTrue(wrapping_selected.requires_grad)
        self.assertFalse(wrapping_selected.is_leaf)
        wrapping_selected.sum().backward()
        self.assertEqual(wrapping_empty.grad.shape, wrapping_empty.shape)
        self.assertEqual(wrapping_empty.grad.stride(), wrapping_empty.stride())
        self.assertEqual(wrapping_empty.grad.storage_offset(), 0)

    def assert_storage_and_autograd_survive_source_lifetime(
        self, index, source_rank=2
    ):
        input_shape = tuple(range(2, source_rank + 3))
        values = np.arange(np.prod(input_shape), dtype=np.float32).reshape(
            input_shape
        )
        input_strides = tuple(
            int(np.prod(input_shape[dimension + 1 :]))
            for dimension in range(len(input_shape))
        )
        expected_shape = (*input_shape[1:-1], input_shape[0])
        expected_stride = (*input_strides[1:-1], input_strides[0])
        expected_values = np.swapaxes(values, 0, source_rank)[1]

        def retained_view():
            source = torch.tensor(values.tolist()).transpose(0, source_rank)[1]
            return source[index]

        surviving = retained_view()
        gc.collect()
        self.assertEqual(surviving.shape, expected_shape)
        self.assertEqual(surviving.stride(), expected_stride)
        self.assertEqual(surviving.storage_offset(), 1)
        np.testing.assert_array_equal(np.asarray(surviving), expected_values)

        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_autograd_view():
            source = (leaf * 2.0).transpose(0, source_rank)[1]
            return source[index]

        tracked = retained_autograd_view()
        gc.collect()
        weight_values = np.arange(
            1, np.prod(tracked.shape) + 1, dtype=np.float32
        ).reshape(tracked.shape)
        weights = torch.tensor(weight_values.tolist())
        (tracked * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[..., 1] = 2.0 * np.moveaxis(weight_values, -1, 0)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_slice_storage_and_autograd_survive_source_lifetime(self):
        self.assert_storage_and_autograd_survive_source_lifetime(slice(None))

    def test_singleton_full_slice_tuple_survives_source_lifetime(self):
        self.assert_storage_and_autograd_survive_source_lifetime((slice(None),))

    def test_double_full_slice_tuple_survives_source_lifetime(self):
        self.assert_storage_and_autograd_survive_source_lifetime(
            (slice(None), slice(None))
        )

    def test_three_or_more_full_slice_tuples_survive_source_lifetime(self):
        for count in (3, 4):
            with self.subTest(count=count):
                self.assert_storage_and_autograd_survive_source_lifetime(
                    (slice(None),) * count, source_rank=count
                )

    def test_full_slice_ellipsis_tuples_survive_source_lifetime(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                with self.subTest(slice_count=slice_count, position=position):
                    self.assert_storage_and_autograd_survive_source_lifetime(
                        index, source_rank=max(slice_count, 2)
                    )

    def test_leading_integer_full_slice_survives_source_lifetime(self):
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_view():
            source = (leaf * 2.0)[1].transpose(0, 1)
            return source[-1, :]

        selected = retained_view()
        gc.collect()
        self.assertEqual(selected.shape, (2, 4))
        self.assertEqual(selected.stride(), (12, 1))
        self.assertEqual(selected.storage_offset(), 32)
        np.testing.assert_array_equal(
            np.asarray(selected), 2.0 * values[1, :, -1, :]
        )

        weights = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        (selected * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[1, :, -1, :] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_two_leading_integer_full_slice_survives_source_lifetime(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_view():
            source = (leaf * 2.0)[1].transpose(0, 2)
            return source[-1, -2, :]

        selected = retained_view()
        gc.collect()
        self.assertEqual(selected.shape, (3,))
        self.assertEqual(selected.stride(), (20,))
        self.assertEqual(selected.storage_offset(), 74)
        np.testing.assert_array_equal(
            np.asarray(selected), 2.0 * values[1, :, -2, -1]
        )

        weights = torch.tensor([1.0, 2.0, 3.0])
        (selected * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[1, :, -2, -1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_three_leading_integer_full_slice_survives_source_lifetime(self):
        values = np.arange(720, dtype=np.float32).reshape(2, 3, 4, 5, 6)
        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_view():
            source = (leaf * 2.0)[1].transpose(0, 3)
            return source[-1, -2, -3, :]

        selected = retained_view()
        gc.collect()
        self.assertEqual(selected.shape, (3,))
        self.assertEqual(selected.stride(), (120,))
        self.assertEqual(selected.storage_offset(), 437)
        np.testing.assert_array_equal(
            np.asarray(selected), 2.0 * values[1, :, -2, -3, -1]
        )

        weights = torch.tensor([1.0, 2.0, 3.0])
        (selected * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[1, :, -2, -3, -1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_four_leading_integer_full_slice_survives_source_lifetime(self):
        values = np.arange(5040, dtype=np.float32).reshape(2, 3, 4, 5, 6, 7)
        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_view():
            source = (leaf * 2.0)[1].transpose(0, 4)
            return source[-1, -2, -3, -4, :]

        selected = retained_view()
        gc.collect()
        self.assertEqual(selected.shape, (3,))
        self.assertEqual(selected.stride(), (840,))
        self.assertEqual(selected.storage_offset(), 3044)
        np.testing.assert_array_equal(
            np.asarray(selected), 2.0 * values[1, :, -2, -3, -4, -1]
        )

        weights = torch.tensor([1.0, 2.0, 3.0])
        (selected * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[1, :, -2, -3, -4, -1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def assert_dispatches_through_tensorbase_mode_before_indexing(self, index):
        index_rank = len(index) if isinstance(index, tuple) else 1
        shape = (2,) * max(index_rank, 2)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        source = torch.tensor(values.tolist())
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        dispatch_types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [mode]
            )

        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
        descriptor = inspect.getattr_static(torch.Tensor, "__getitem__")
        self.assertIs(type(function), types.WrapperDescriptorType)
        self.assertIs(function, descriptor)
        self.assertEqual(function.__qualname__, "TensorBase.__getitem__")
        self.assertEqual(function.__objclass__.__name__, "TensorBase")
        self.assertEqual(function.__objclass__.__module__, "torch._C")
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual(handler_stack, ())
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        scalar = torch.tensor(1.0)
        mode.calls.clear()
        with mode:
            scalar_result = scalar[index]
        self.assertIs(scalar_result, marker)
        self.assertEqual(len(mode.calls), 1)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_metadata_alias(source, forwarded)

    def test_full_slice_dispatches_through_tensorbase_mode_before_indexing(self):
        self.assert_dispatches_through_tensorbase_mode_before_indexing(slice(None))

    def test_singleton_full_slice_tuple_dispatches_with_original_index(self):
        self.assert_dispatches_through_tensorbase_mode_before_indexing(
            (slice(None),)
        )

    def test_double_full_slice_tuple_dispatches_with_original_index(self):
        self.assert_dispatches_through_tensorbase_mode_before_indexing(
            (slice(None), slice(None))
        )

    def test_three_or_more_full_slice_tuples_dispatch_with_original_index(self):
        for count in (3, 4):
            with self.subTest(count=count):
                self.assert_dispatches_through_tensorbase_mode_before_indexing(
                    (slice(None),) * count
                )

    def test_full_slice_ellipsis_tuples_dispatch_with_original_index(self):
        for slice_count in range(1, 5):
            for position, index in enumerate(
                self.full_slice_ellipsis_indices(slice_count)
            ):
                with self.subTest(slice_count=slice_count, position=position):
                    self.assert_dispatches_through_tensorbase_mode_before_indexing(
                        index
                    )

    def test_leading_integer_full_slice_dispatches_with_original_tuple(self):
        source = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        marker = object()

        class IndexValue:
            def __init__(self):
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return 1

        integer = IndexValue()
        index = (integer, slice(None))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, inspect.getattr_static(torch.Tensor, "__getitem__"))
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual(integer.calls, 0)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_same_view(forwarded, source[1])
        self.assertEqual(integer.calls, 1)

    def test_two_leading_integer_full_slice_dispatches_with_original_tuple(self):
        source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        marker = object()

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first = IndexValue(-1)
        second = IndexValue(-2)
        index = (first, second, slice(None))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, inspect.getattr_static(torch.Tensor, "__getitem__"))
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual(first.calls, 0)
        self.assertEqual(second.calls, 0)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_same_view(forwarded, source[-1, -2])
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_three_leading_integer_full_slice_dispatches_with_original_tuple(self):
        source = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )
        marker = object()

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first = IndexValue(-1)
        second = IndexValue(-2)
        third = IndexValue(-3)
        index = (first, second, third, slice(None))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, inspect.getattr_static(torch.Tensor, "__getitem__"))
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual(first.calls, 0)
        self.assertEqual(second.calls, 0)
        self.assertEqual(third.calls, 0)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_same_view(forwarded, source[-1, -2, -3])
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(third.calls, 1)

    def test_four_leading_integer_full_slice_dispatches_with_original_tuple(self):
        source = torch.tensor(
            np.arange(720, dtype=np.float32).reshape(2, 3, 4, 5, 6).tolist()
        )
        marker = object()

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first = IndexValue(-1)
        second = IndexValue(-2)
        third = IndexValue(-3)
        fourth = IndexValue(-4)
        index = (first, second, third, fourth, slice(None))

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = source[index]
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, inspect.getattr_static(torch.Tensor, "__getitem__"))
        self.assertEqual(dispatch_types, ())
        self.assertEqual(len(args), 2)
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual(first.calls, 0)
        self.assertEqual(second.calls, 0)
        self.assertEqual(third.calls, 0)
        self.assertEqual(fourth.calls, 0)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_same_view(forwarded, source[-1, -2, -3, -4])
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(third.calls, 1)
        self.assertEqual(fourth.calls, 1)

    def test_tuple_subclasses_are_normalized_through_overridden_iteration(self):
        source = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

        class IntegerRemapTuple(tuple):
            def __iter__(self):
                return iter((0,))

        selected = source[IntegerRemapTuple((slice(None),))]
        self.assertEqual(selected.tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(selected.shape, (3,))
        self.assertEqual(selected.stride(), (1,))
        self.assertEqual(selected.storage_offset(), 0)

        class FullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None),))

        alias = source[FullSliceRemapTuple((0,))]
        self.assert_metadata_alias(source, alias)

        class DoubleFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None), slice(None)))

        double_alias = source[DoubleFullSliceRemapTuple((0,))]
        self.assert_metadata_alias(source, double_alias)

        higher_rank_source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )

        class TripleFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None),) * 3)

        triple_alias = higher_rank_source[TripleFullSliceRemapTuple((0,))]
        self.assert_metadata_alias(higher_rank_source, triple_alias)

        class FullSliceEllipsisRemapTuple(tuple):
            def __iter__(self):
                return iter((slice(None), Ellipsis, slice(None)))

        mixed_alias = source[FullSliceEllipsisRemapTuple((0,))]
        self.assert_metadata_alias(source, mixed_alias)

        class LeadingIntegerFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((1, slice(None)))

        selected_with_slice = source[LeadingIntegerFullSliceRemapTuple((0,))]
        self.assert_same_view(selected_with_slice, source[1])

        class TwoLeadingIntegerFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((1, 2, slice(None)))

        selected_with_two_integers = higher_rank_source[
            TwoLeadingIntegerFullSliceRemapTuple((0,))
        ]
        self.assert_same_view(selected_with_two_integers, higher_rank_source[1, 2])

        rank_four_source = torch.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )

        class ThreeLeadingIntegerFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((1, 2, 3, slice(None)))

        selected_with_three_integers = rank_four_source[
            ThreeLeadingIntegerFullSliceRemapTuple((0,))
        ]
        self.assert_same_view(
            selected_with_three_integers, rank_four_source[1, 2, 3]
        )

        rank_five_source = torch.tensor(
            np.arange(720, dtype=np.float32).reshape(2, 3, 4, 5, 6).tolist()
        )

        class FourLeadingIntegerFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((1, 2, 3, 4, slice(None)))

        selected_with_four_integers = rank_five_source[
            FourLeadingIntegerFullSliceRemapTuple((0,))
        ]
        self.assert_same_view(
            selected_with_four_integers, rank_five_source[1, 2, 3, 4]
        )

        class EmptyRemapTuple(tuple):
            def __iter__(self):
                return iter(())

        empty_index = EmptyRemapTuple((0,))
        empty_alias = source[empty_index]
        self.assert_metadata_alias(source, empty_alias)
        self.assert_autograd_gradient_node_and_no_grad_leaf_status(
            empty_index, "AliasBackward0"
        )

        class IterationErrorTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration exploded")

        with self.assertRaisesRegex(RuntimeError, "^tuple iteration exploded$"):
            source[IterationErrorTuple((slice(None),))]

    def test_existing_indices_and_unsupported_slice_forms_are_unchanged(self):
        tensor = torch.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[4.0, 5.0], [6.0, 7.0]],
            ]
        )
        indexed = tensor[-1, 0]
        self.assertEqual(indexed.tolist(), [4.0, 5.0])
        self.assertEqual(indexed.stride(), (1,))
        self.assertEqual(indexed.storage_offset(), 4)
        self.assert_metadata_alias(tensor, tensor[...])

        unsupported = (
            slice(1, None),
            slice(None, -1),
            slice(None, None, 1),
            slice(None, None, 2),
            (slice(1, None),),
            (slice(None, -1),),
            (slice(None, None, 1),),
            (slice(None, None, 2),),
            (slice(None), 0),
            (0, slice(1, None)),
            (0, slice(None, -1)),
            (0, slice(None, None, 1)),
            (0, slice(None, None, 2)),
            (0, 0, slice(1, None)),
            (0, 0, slice(None, -1)),
            (0, 0, slice(None, None, 1)),
            (0, 0, slice(None, None, 2)),
            (slice(1, None), slice(None)),
            (slice(None), slice(None, -1)),
            (slice(None, None, 1), slice(None)),
            (slice(None), slice(None, None, 2)),
            (slice(1, None), Ellipsis),
            (Ellipsis, slice(None, -1)),
            (slice(None, None, 1), Ellipsis),
            (Ellipsis, slice(None, None, 2)),
            (slice(None), 0, Ellipsis),
            (Ellipsis, 0, slice(None)),
            (slice(None), None, Ellipsis),
            (Ellipsis, None, slice(None)),
            (slice(None), Ellipsis, Ellipsis),
            (slice(1, None), slice(None), slice(None)),
            (slice(None), slice(None, -1), slice(None)),
            (slice(None), slice(None, None, 1), slice(None)),
            (slice(None), slice(None, None, 2), slice(None)),
            (0, slice(None), slice(None)),
            (slice(None), 0, slice(None)),
            (0, None, slice(None)),
            (0, Ellipsis, slice(None)),
            (True, 0, slice(None)),
            (0, True, slice(None)),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

        rank_four_tensor = torch.zeros((2, 3, 4, 5))
        unsupported_four_item_tuples = (
            (0, 0, 0, slice(1, None)),
            (0, 0, 0, slice(None, -1)),
            (0, 0, 0, slice(None, None, 1)),
            (0, 0, 0, slice(None, None, 2)),
            (0, 0, slice(None), slice(None)),
            (0, slice(None), 0, slice(None)),
            (slice(None), 0, 0, slice(None)),
            (0, 0, None, slice(None)),
            (0, 0, Ellipsis, slice(None)),
            (0, 0, 0, Ellipsis),
            (True, 0, 0, slice(None)),
            (0, True, 0, slice(None)),
            (0, 0, True, slice(None)),
        )
        for index in unsupported_four_item_tuples:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    rank_four_tensor[index]

        rank_five_tensor = torch.zeros((2, 3, 4, 5, 6))
        unsupported_five_item_tuples = (
            (0, 0, 0, 0, slice(1, None)),
            (0, 0, 0, 0, slice(None, -1)),
            (0, 0, 0, 0, slice(None, None, 1)),
            (0, 0, 0, 0, slice(None, None, 2)),
            (0, 0, 0, slice(None), slice(None)),
            (0, 0, slice(None), 0, slice(None)),
            (0, slice(None), 0, 0, slice(None)),
            (slice(None), 0, 0, 0, slice(None)),
            (0, 0, 0, None, slice(None)),
            (0, 0, 0, Ellipsis, slice(None)),
            (0, 0, 0, 0, Ellipsis),
            (True, 0, 0, 0, slice(None)),
            (0, True, 0, 0, slice(None)),
            (0, 0, True, 0, slice(None)),
            (0, 0, 0, True, slice(None)),
        )
        for index in unsupported_five_item_tuples:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    rank_five_tensor[index]

    def test_leading_integer_full_slice_preserves_rank_and_bounds_errors(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        rank_one_index = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "too many indices for tensor of dimension 1"
        ):
            torch.zeros((2,))[rank_one_index, :]
        self.assertEqual(rank_one_index.calls, 0)

        out_of_bounds = IndexValue(2)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            torch.zeros((2, 3))[out_of_bounds, :]
        self.assertEqual(out_of_bounds.calls, 1)

    def test_two_leading_integer_full_slice_preserves_integer_errors(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        for dimensions in range(3):
            first = IndexValue(0)
            second = IndexValue(0)
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"too many indices for tensor of dimension {dimensions}",
                ):
                    torch.zeros((2,) * dimensions)[first, second, :]
                self.assertEqual(first.calls, 0)
                self.assertEqual(second.calls, 0)

        first_out_of_bounds = IndexValue(2)
        skipped_second = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            torch.zeros((2, 3, 4))[first_out_of_bounds, skipped_second, :]
        self.assertEqual(first_out_of_bounds.calls, 1)
        self.assertEqual(skipped_second.calls, 0)

        valid_first = IndexValue(0)
        second_out_of_bounds = IndexValue(3)
        with self.assertRaisesRegex(
            IndexError, "index 3 is out of bounds for dimension 1 with size 3"
        ):
            torch.zeros((2, 3, 4))[valid_first, second_out_of_bounds, :]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(second_out_of_bounds.calls, 1)

        invalid_second = IndexValue(1.5)
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2, 3, 4))[0, invalid_second, :]
        self.assertEqual(invalid_second.calls, 1)

        with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
            torch.zeros((2, 3, 4))[0, 2**100, :]

        wrapping_first = IndexValue(sys.maxsize - 1)
        wrapping_second_out_of_bounds = IndexValue(1)
        with self.assertRaisesRegex(
            IndexError, "index 1 is out of bounds for dimension 1 with size 1"
        ):
            torch.zeros((sys.maxsize, 1, 0, 3))[
                wrapping_first, wrapping_second_out_of_bounds, :
            ]
        self.assertEqual(wrapping_first.calls, 1)
        self.assertEqual(wrapping_second_out_of_bounds.calls, 1)

    def test_three_leading_integer_full_slice_preserves_integer_errors(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        for dimensions in range(4):
            first = IndexValue(0)
            second = IndexValue(0)
            third = IndexValue(0)
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"too many indices for tensor of dimension {dimensions}",
                ):
                    torch.zeros((2,) * dimensions)[first, second, third, :]
                self.assertEqual(first.calls, 0)
                self.assertEqual(second.calls, 0)
                self.assertEqual(third.calls, 0)

        first_out_of_bounds = IndexValue(2)
        skipped_second = IndexValue(0)
        skipped_third = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            torch.zeros((2, 3, 4, 5))[
                first_out_of_bounds, skipped_second, skipped_third, :
            ]
        self.assertEqual(first_out_of_bounds.calls, 1)
        self.assertEqual(skipped_second.calls, 0)
        self.assertEqual(skipped_third.calls, 0)

        valid_first = IndexValue(0)
        second_out_of_bounds = IndexValue(3)
        skipped_third = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 3 is out of bounds for dimension 1 with size 3"
        ):
            torch.zeros((2, 3, 4, 5))[
                valid_first, second_out_of_bounds, skipped_third, :
            ]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(second_out_of_bounds.calls, 1)
        self.assertEqual(skipped_third.calls, 0)

        valid_first = IndexValue(0)
        valid_second = IndexValue(0)
        third_out_of_bounds = IndexValue(4)
        with self.assertRaisesRegex(
            IndexError, "index 4 is out of bounds for dimension 2 with size 4"
        ):
            torch.zeros((2, 3, 4, 5))[
                valid_first, valid_second, third_out_of_bounds, :
            ]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(valid_second.calls, 1)
        self.assertEqual(third_out_of_bounds.calls, 1)

        invalid_third = IndexValue(1.5)
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2, 3, 4, 5))[0, 0, invalid_third, :]
        self.assertEqual(invalid_third.calls, 1)

        with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
            torch.zeros((2, 3, 4, 5))[0, 0, 2**100, :]

        wrapping_first = IndexValue(sys.maxsize - 1)
        wrapping_second = IndexValue(0)
        wrapping_third_out_of_bounds = IndexValue(1)
        with self.assertRaisesRegex(
            IndexError, "index 1 is out of bounds for dimension 2 with size 1"
        ):
            torch.zeros((sys.maxsize, 1, 1, 0, 3))[
                wrapping_first,
                wrapping_second,
                wrapping_third_out_of_bounds,
                :,
            ]
        self.assertEqual(wrapping_first.calls, 1)
        self.assertEqual(wrapping_second.calls, 1)
        self.assertEqual(wrapping_third_out_of_bounds.calls, 1)

    def test_four_leading_integer_full_slice_preserves_integer_errors(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        for dimensions in range(5):
            first = IndexValue(0)
            second = IndexValue(0)
            third = IndexValue(0)
            fourth = IndexValue(0)
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"too many indices for tensor of dimension {dimensions}",
                ):
                    torch.zeros((2,) * dimensions)[
                        first, second, third, fourth, :
                    ]
                self.assertEqual(first.calls, 0)
                self.assertEqual(second.calls, 0)
                self.assertEqual(third.calls, 0)
                self.assertEqual(fourth.calls, 0)

        first_out_of_bounds = IndexValue(2)
        skipped_second = IndexValue(0)
        skipped_third = IndexValue(0)
        skipped_fourth = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            torch.zeros((2, 3, 4, 5, 6))[
                first_out_of_bounds,
                skipped_second,
                skipped_third,
                skipped_fourth,
                :,
            ]
        self.assertEqual(first_out_of_bounds.calls, 1)
        self.assertEqual(skipped_second.calls, 0)
        self.assertEqual(skipped_third.calls, 0)
        self.assertEqual(skipped_fourth.calls, 0)

        valid_first = IndexValue(0)
        second_out_of_bounds = IndexValue(3)
        skipped_third = IndexValue(0)
        skipped_fourth = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 3 is out of bounds for dimension 1 with size 3"
        ):
            torch.zeros((2, 3, 4, 5, 6))[
                valid_first,
                second_out_of_bounds,
                skipped_third,
                skipped_fourth,
                :,
            ]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(second_out_of_bounds.calls, 1)
        self.assertEqual(skipped_third.calls, 0)
        self.assertEqual(skipped_fourth.calls, 0)

        valid_first = IndexValue(0)
        valid_second = IndexValue(0)
        third_out_of_bounds = IndexValue(4)
        skipped_fourth = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 4 is out of bounds for dimension 2 with size 4"
        ):
            torch.zeros((2, 3, 4, 5, 6))[
                valid_first,
                valid_second,
                third_out_of_bounds,
                skipped_fourth,
                :,
            ]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(valid_second.calls, 1)
        self.assertEqual(third_out_of_bounds.calls, 1)
        self.assertEqual(skipped_fourth.calls, 0)

        valid_first = IndexValue(0)
        valid_second = IndexValue(0)
        valid_third = IndexValue(0)
        fourth_out_of_bounds = IndexValue(5)
        with self.assertRaisesRegex(
            IndexError, "index 5 is out of bounds for dimension 3 with size 5"
        ):
            torch.zeros((2, 3, 4, 5, 6))[
                valid_first,
                valid_second,
                valid_third,
                fourth_out_of_bounds,
                :,
            ]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(valid_second.calls, 1)
        self.assertEqual(valid_third.calls, 1)
        self.assertEqual(fourth_out_of_bounds.calls, 1)

        invalid_fourth = IndexValue(1.5)
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2, 3, 4, 5, 6))[0, 0, 0, invalid_fourth, :]
        self.assertEqual(invalid_fourth.calls, 1)

        with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
            torch.zeros((2, 3, 4, 5, 6))[0, 0, 0, 2**100, :]

        wrapping_first = IndexValue(sys.maxsize - 1)
        wrapping_second = IndexValue(0)
        wrapping_third = IndexValue(0)
        wrapping_fourth_out_of_bounds = IndexValue(1)
        with self.assertRaisesRegex(
            IndexError, "index 1 is out of bounds for dimension 3 with size 1"
        ):
            torch.zeros((sys.maxsize, 1, 1, 1, 0, 3))[
                wrapping_first,
                wrapping_second,
                wrapping_third,
                wrapping_fourth_out_of_bounds,
                :,
            ]
        self.assertEqual(wrapping_first.calls, 1)
        self.assertEqual(wrapping_second.calls, 1)
        self.assertEqual(wrapping_third.calls, 1)
        self.assertEqual(wrapping_fourth_out_of_bounds.calls, 1)


if __name__ == "__main__":
    unittest.main()
