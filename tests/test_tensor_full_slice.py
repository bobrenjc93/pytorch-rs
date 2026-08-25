import gc
import inspect
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

    def two_leading_integers_full_slice_layout_cases(self):
        contiguous_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        contiguous_values[0, 0, 0] = -0.0
        base_values = np.arange(240, dtype=np.float32).reshape(2, 2, 3, 4, 5)
        base = torch.tensor(base_values.tolist())
        return (
            ("contiguous", torch.tensor(contiguous_values.tolist()), 0, 0),
            ("empty", torch.zeros((2, 3, 0, 4)), -1, -2),
            ("offset", base[1], 1, -1),
            ("offset-noncontiguous", base[1].transpose(0, 2), -1, 1),
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

    def test_two_leading_integers_full_slice_reuses_the_integer_tuple_view(self):
        for (
            case,
            source,
            first,
            second,
        ) in self.two_leading_integers_full_slice_layout_cases():
            with self.subTest(case=case):
                self.assert_same_view(source[first, second, :], source[first, second])

        signed_zero = self.two_leading_integers_full_slice_layout_cases()[0][1][0, 0, :]
        self.assertEqual(np.asarray(signed_zero).view(np.uint32)[0], 0x8000_0000)

    def test_two_leading_integers_full_slice_accepts_integer_protocol_values(self):
        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        dynamic_first = IndexValue(-1)
        dynamic_second = IndexValue(-2)
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        cases = (
            (IntegerSubclass(1), np.int64(-1), (1, 2)),
            (np.uint64(0), dynamic_second, (0, 1)),
            (dynamic_first, IntegerSubclass(0), (1, 0)),
        )
        for first, second, normalized in cases:
            with self.subTest(first=repr(first), second=repr(second)):
                self.assert_same_view(source[first, second, :], source[normalized])
        self.assertEqual(dynamic_first.calls, 1)
        self.assertEqual(dynamic_second.calls, 1)

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
        diagnostic_pattern = r"\[" * diagnostic_rank + r"2\." + r"\]" * diagnostic_rank
        diagnostic_leaf = torch.tensor(diagnostic_values, requires_grad=True)
        with self.assertRaisesRegex(
            ValueError,
            r"^dropout probability has to be between 0 and 1, but got "
            rf"tensor\({diagnostic_pattern}, grad_fn=<{node_name}>\)$",
        ):
            torch.nn.functional.dropout(None, p=diagnostic_leaf[index], training=False)

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
        self.assert_autograd_gradient_node_and_no_grad_leaf_status((), "AliasBackward0")

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
            torch.nn.functional.dropout(None, p=diagnostic_leaf[0, :], training=False)

        no_grad_source = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            untracked = no_grad_source[1, :]
        self.assert_same_view(untracked, no_grad_source[1])
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty[-1, :].sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_two_leading_integers_full_slice_uses_integer_autograd_semantics(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        source = (leaf * 2.0).transpose(0, 2)
        selected = source[-1, -2, :]
        self.assert_same_view(selected, source[-1, -2])
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)

        weights = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        (selected * weights).sum().backward()
        expected_gradient = np.zeros_like(values)
        expected_gradient[:, -2, -1, :] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected_gradient)

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

    def assert_storage_and_autograd_survive_source_lifetime(self, index, source_rank=2):
        input_shape = tuple(range(2, source_rank + 3))
        values = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)
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
        np.testing.assert_array_equal(np.asarray(selected), 2.0 * values[1, :, -1, :])

        weights = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        (selected * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[1, :, -1, :] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected)

    def test_two_leading_integers_full_slice_survives_source_lifetime(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        leaf = torch.tensor(values.tolist(), requires_grad=True)

        def retained_view():
            source = (leaf * 2.0).transpose(0, 2)
            return source[-1, -2, :]

        selected = retained_view()
        gc.collect()
        self.assertEqual(selected.shape, (2, 5))
        self.assertEqual(selected.stride(), (60, 1))
        self.assertEqual(selected.storage_offset(), 35)
        np.testing.assert_array_equal(np.asarray(selected), 2.0 * values[:, -2, -1, :])

        weights = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]])
        (selected * weights).sum().backward()
        expected = np.zeros_like(values)
        expected[:, -2, -1, :] = 2.0 * np.asarray(weights)
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
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [mode])

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
        self.assert_dispatches_through_tensorbase_mode_before_indexing((slice(None),))

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

    def test_two_leading_integers_full_slice_dispatches_with_original_tuple(self):
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
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

        class TwoLeadingIntegersFullSliceRemapTuple(tuple):
            def __iter__(self):
                return iter((1, -1, slice(None)))

        rank_three_source = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        selected_with_two_integers = rank_three_source[
            TwoLeadingIntegersFullSliceRemapTuple((0,))
        ]
        self.assert_same_view(selected_with_two_integers, rank_three_source[1, -1])

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
            (slice(None), 0, slice(None)),
            (0, slice(None), 0),
            (slice(None), 0, 0),
            (0, 0, Ellipsis),
            (0, 0, None),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]

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

    def test_two_leading_integers_full_slice_preserves_rank_and_bounds_errors(self):
        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        rank_two_first = IndexValue(0)
        rank_two_second = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "too many indices for tensor of dimension 2"
        ):
            torch.zeros((2, 3))[rank_two_first, rank_two_second, :]
        self.assertEqual(rank_two_first.calls, 0)
        self.assertEqual(rank_two_second.calls, 0)

        first_out_of_bounds = IndexValue(2)
        unconverted_second = IndexValue(0)
        with self.assertRaisesRegex(
            IndexError, "index 2 is out of bounds for dimension 0 with size 2"
        ):
            torch.zeros((2, 3, 4))[first_out_of_bounds, unconverted_second, :]
        self.assertEqual(first_out_of_bounds.calls, 1)
        self.assertEqual(unconverted_second.calls, 0)

        valid_first = IndexValue(-1)
        second_out_of_bounds = IndexValue(3)
        with self.assertRaisesRegex(
            IndexError, "index 3 is out of bounds for dimension 1 with size 3"
        ):
            torch.zeros((2, 3, 4))[valid_first, second_out_of_bounds, :]
        self.assertEqual(valid_first.calls, 1)
        self.assertEqual(second_out_of_bounds.calls, 1)

    def test_two_leading_integers_full_slice_preserves_invalid_index_errors(self):
        source = torch.zeros((2, 3, 4))

        def error_contract(index):
            try:
                source[index]
            except Exception as error:
                return type(error), str(error)
            self.fail(f"index unexpectedly succeeded: {index!r}")

        for leading_indices in (
            (True, 0),
            (0, False),
            (1.5, 0),
            (0, object()),
            (2**80, 0),
            (0, 2**80),
        ):
            with self.subTest(index=repr(leading_indices)):
                self.assertEqual(
                    error_contract((*leading_indices, slice(None))),
                    error_contract(leading_indices),
                )


if __name__ == "__main__":
    unittest.main()
