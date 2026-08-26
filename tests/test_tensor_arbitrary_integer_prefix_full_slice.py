import gc
import inspect
import sys
import unittest

import numpy as np
import torch_rs as torch


class TensorArbitraryIntegerPrefixFullSliceTests(unittest.TestCase):
    def assert_same_view(self, actual, expected):
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertTrue(actual.is_set_to(expected))
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)

    def six_leading_integer_layout_cases(self):
        shape = (2,) * 6 + (3,)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        values[1, 0, 1, 0, 1, 0, 0] = -0.0
        base_values = np.arange(2 * np.prod(shape), dtype=np.float32).reshape(
            (2, *shape)
        )
        base = torch.tensor(base_values.tolist())
        return (
            (
                "contiguous",
                torch.tensor(values.tolist()),
                (1, 0, 1, 0, 1, 0),
            ),
            (
                "empty",
                torch.zeros((2,) * 6 + (0, 3)),
                (1, 0, 1, 0, 1, 0),
            ),
            (
                "offset-noncontiguous",
                base[1].transpose(0, 6),
                (2, 1, 0, 1, 0, 1),
            ),
            (
                "wrapping-empty",
                torch.zeros((sys.maxsize, 1, 1, 1, 1, 1, 0, 3)),
                (sys.maxsize - 1, 0, 0, 0, 0, 0),
            ),
        )

    def test_six_plus_leading_integer_full_slice_reuses_integer_views(self):
        for case, source, indices in self.six_leading_integer_layout_cases():
            with self.subTest(case=case):
                selected = source[indices + (slice(None),)]
                self.assert_same_view(selected, source[indices])
                if case == "wrapping-empty":
                    self.assertEqual(selected.storage_offset(), sys.maxsize - 5)

        maximum_rank_source = torch.zeros((1,) * 64)
        maximum_prefix = (0,) * 63
        self.assert_same_view(
            maximum_rank_source[maximum_prefix + (slice(None),)],
            maximum_rank_source[maximum_prefix],
        )

    def test_six_plus_prefix_accepts_each_integer_protocol_value_once(self):
        class IntegerSubclass(int):
            pass

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        first_dynamic = IndexValue(1)
        third_dynamic = IndexValue(1)
        fifth_dynamic = IndexValue(1)
        indices = (
            first_dynamic,
            np.int64(0),
            third_dynamic,
            IntegerSubclass(0),
            fifth_dynamic,
            np.uint64(0),
        )
        source = torch.tensor(
            np.arange(192, dtype=np.float32).reshape((2,) * 6 + (3,)).tolist()
        )
        self.assert_same_view(
            source[indices + (slice(None),)],
            source[1, 0, 1, 0, 1, 0],
        )
        self.assertEqual(
            [first_dynamic.calls, third_dynamic.calls, fifth_dynamic.calls],
            [1, 1, 1],
        )

    def test_six_leading_integer_full_slice_preserves_autograd_and_lifetime(self):
        shape = (2,) * 8
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        indices = (1, 0, 1, 0, 1, 0)

        def retained_view():
            source = (leaf * 2.0).reshape(shape)[1].transpose(0, 6)
            selected = source[indices + (slice(None),)]
            self.assert_same_view(selected, source[indices])
            return selected

        selected = retained_view()
        gc.collect()
        self.assertEqual(selected.shape, (2,))
        self.assertEqual(selected.stride(), (64,))
        self.assertEqual(selected.storage_offset(), 149)
        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(selected),
            2.0 * values[1, :, 0, 1, 0, 1, 0, 1],
        )

        weights = torch.tensor([3.0, 5.0])
        (selected * weights).sum().backward()
        expected_gradient = np.zeros_like(values)
        expected_gradient[1, :, 0, 1, 0, 1, 0, 1] = 2.0 * np.asarray(weights)
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(shape), expected_gradient
        )

        empty = torch.zeros((2,) * 6 + (0, 3), requires_grad=True)
        empty[indices + (slice(None),)].sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.stride(), empty.stride())
        self.assertEqual(empty.grad.storage_offset(), 0)
        self.assertEqual(empty.grad.numel(), 0)

    def test_six_plus_indexing_dispatches_with_the_original_tuple(self):
        source = torch.zeros((2,) * 7)
        marker = object()

        class IndexValue:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        dynamic = [IndexValue(value) for value in (1, 0, 1, 0, 1, 0)]
        index = (*dynamic, slice(None))

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
        self.assertIs(args[0], source)
        self.assertIs(args[1], index)
        self.assertIsNone(kwargs)
        self.assertEqual([item.calls for item in dynamic], [0] * 6)

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with ForwardingMode():
            forwarded = source[index]
        self.assert_same_view(forwarded, source[1, 0, 1, 0, 1, 0])
        self.assertEqual([item.calls for item in dynamic], [1] * 6)

    def test_six_plus_prefix_preserves_validation_order_for_other_indices(self):
        events = []

        class IndexValue:
            def __init__(self, label, value):
                self.label = label
                self.value = value

            def __index__(self):
                events.append(self.label)
                return self.value

        excessive = tuple(IndexValue(index, 0) for index in range(6))
        with self.assertRaisesRegex(
            IndexError, "too many indices for tensor of dimension 6"
        ):
            torch.zeros((2,) * 6)[excessive + (slice(None),)]
        self.assertEqual(events, [])

        events.clear()
        out_of_bounds = tuple(
            IndexValue(index, 4 if index == 2 else 0) for index in range(6)
        )
        with self.assertRaisesRegex(
            IndexError, "index 4 is out of bounds for dimension 2 with size 4"
        ):
            torch.zeros((2, 3, 4, 5, 6, 7, 8))[
                out_of_bounds + (slice(None),)
            ]
        self.assertEqual(events, [0, 1, 2])

        events.clear()
        invalid = tuple(
            IndexValue(index, 1.5 if index == 3 else 0) for index in range(6)
        )
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2,) * 7)[invalid + (slice(None),)]
        self.assertEqual(events, [0, 1, 2, 3])

        events.clear()
        non_full_slice = tuple(IndexValue(index, 0) for index in range(6))
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2,) * 7)[non_full_slice + (slice(None, None, 1),)]
        self.assertEqual(events, list(range(6)))

        events.clear()
        mixed = tuple(IndexValue(index, 0) for index in range(6))
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2,) * 7)[(mixed[0], None, *mixed[2:], slice(None))]
        self.assertEqual(events, [0])

        events.clear()
        trailing_ellipsis = tuple(IndexValue(index, 0) for index in range(6))
        with self.assertRaisesRegex(IndexError, "only integers"):
            torch.zeros((2,) * 7)[trailing_ellipsis + (Ellipsis,)]
        self.assertEqual(events, list(range(6)))

    def test_tuple_subclass_can_expand_to_six_integer_indices_and_a_full_slice(self):
        source = torch.tensor(
            np.arange(192, dtype=np.float32).reshape((2,) * 6 + (3,)).tolist()
        )

        class RemappedTuple(tuple):
            def __iter__(self):
                return iter((1, 0, 1, 0, 1, 0, slice(None)))

        self.assert_same_view(
            source[RemappedTuple((0,))],
            source[1, 0, 1, 0, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
