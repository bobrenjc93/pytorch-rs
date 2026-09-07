import unittest

import numpy as np
import torch_rs as torch


CAT_ALIASES = ("concat", "concatenate")


class TopLevelCatTests(unittest.TestCase):
    @staticmethod
    def contiguous_stride(shape):
        strides = []
        stride = 1
        for dimension in reversed(shape):
            strides.append(stride)
            stride *= max(dimension, 1)
        return tuple(reversed(strides))

    def assert_cat_matches(
        self,
        actual,
        expected_values,
        *,
        case,
        shape=None,
        stride=None,
        requires_grad=False,
        is_leaf=True,
    ):
        expected = np.asarray(expected_values, dtype=np.float32)
        shape = tuple(expected.shape if shape is None else shape)
        stride = self.contiguous_stride(shape) if stride is None else tuple(stride)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, shape)
            self.assertEqual(actual.stride(), stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertTrue(actual.is_contiguous())
            self.assertEqual(actual.requires_grad, requires_grad)
            self.assertEqual(actual.is_leaf, is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.reshape(-1).view(np.uint32),
            )

    def test_list_tuple_empty_operands_order_metadata_and_fresh_storage(self):
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        offset_noncontiguous = base.transpose(0, 1)[1]
        self.assertEqual(offset_noncontiguous.tolist(), [2.0, 5.0])
        self.assertEqual(offset_noncontiguous.stride(), (3,))
        self.assertEqual(offset_noncontiguous.storage_offset(), 1)

        empty = torch.tensor([])
        tail = torch.tensor([-0.0, 7.5])
        result = torch.cat([offset_noncontiguous, empty, tail], dim=0)
        self.assert_cat_matches(result, [2.0, 5.0, -0.0, 7.5], case="list")
        self.assertFalse(result.is_set_to(offset_noncontiguous))
        self.assertFalse(result.is_set_to(tail))
        self.assertNotEqual(result.data_ptr(), offset_noncontiguous.data_ptr())
        self.assertNotEqual(result.data_ptr(), tail.data_ptr())

        tuple_result = torch.cat((torch.tensor([]), torch.tensor([3.25])), dim=-1)
        self.assert_cat_matches(tuple_result, [3.25], case="tuple dim -1")

        keyword_result = torch.cat(
            tensors=[torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
            out=None,
        )
        self.assert_cat_matches(keyword_result, [8.0, 9.0], case="keywords")

        axis_result = torch.cat([torch.tensor([14.0]), torch.tensor([15.0])], axis=0)
        self.assert_cat_matches(axis_result, [14.0, 15.0], case="axis alias")

        single = torch.tensor([11.0, 13.0])
        single_result = torch.cat([single])
        self.assert_cat_matches(single_result, [11.0, 13.0], case="single tensor")
        self.assertFalse(single_result.is_set_to(single))
        self.assertNotEqual(single_result.data_ptr(), single.data_ptr())

    def test_rank_two_row_column_empty_and_noncontiguous_inputs(self):
        base_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(base_values.tolist())
        offset_noncontiguous = base[1].transpose(0, 1)
        expected_view = base_values[1].transpose(1, 0)
        self.assertEqual(offset_noncontiguous.shape, (4, 3))
        self.assertEqual(offset_noncontiguous.stride(), (1, 4))
        self.assertEqual(offset_noncontiguous.storage_offset(), 12)

        empty_rows = torch.zeros((0, 3))
        row_tail = torch.tensor([[-0.0, 100.0, 101.0]])
        rows = torch.cat([offset_noncontiguous, empty_rows, row_tail], dim=0)
        self.assert_cat_matches(
            rows,
            np.concatenate(
                [
                    expected_view,
                    np.zeros((0, 3), dtype=np.float32),
                    np.asarray([[-0.0, 100.0, 101.0]], dtype=np.float32),
                ],
                axis=0,
            ),
            shape=(5, 3),
            stride=(3, 1),
            case="rank-2 rows dim 0",
        )
        self.assertFalse(rows.is_set_to(offset_noncontiguous))
        self.assertFalse(rows.is_set_to(row_tail))
        self.assertNotEqual(rows.data_ptr(), offset_noncontiguous.data_ptr())
        self.assertNotEqual(rows.data_ptr(), row_tail.data_ptr())

        rows_negative_dim = torch.cat((empty_rows, offset_noncontiguous), dim=-2)
        self.assert_cat_matches(
            rows_negative_dim,
            expected_view,
            shape=(4, 3),
            stride=(3, 1),
            case="rank-2 rows dim -2",
        )

        empty_columns = torch.zeros((4, 0))
        column_tail = torch.tensor([[-0.0], [200.0], [201.0], [202.0]])
        columns = torch.cat([offset_noncontiguous, empty_columns, column_tail], dim=1)
        self.assert_cat_matches(
            columns,
            np.concatenate(
                [
                    expected_view,
                    np.zeros((4, 0), dtype=np.float32),
                    np.asarray([[-0.0], [200.0], [201.0], [202.0]], dtype=np.float32),
                ],
                axis=1,
            ),
            shape=(4, 4),
            stride=(4, 1),
            case="rank-2 columns dim 1",
        )
        self.assertFalse(columns.is_set_to(offset_noncontiguous))
        self.assertFalse(columns.is_set_to(column_tail))
        self.assertNotEqual(columns.data_ptr(), offset_noncontiguous.data_ptr())
        self.assertNotEqual(columns.data_ptr(), column_tail.data_ptr())

        columns_negative_dim = torch.cat((empty_columns, offset_noncontiguous), dim=-1)
        self.assert_cat_matches(
            columns_negative_dim,
            expected_view,
            shape=(4, 3),
            stride=(3, 1),
            case="rank-2 columns dim -1",
        )

        empty_column_result = torch.cat(
            [torch.zeros((2, 0)), torch.zeros((2, 0))],
            dim=1,
        )
        self.assert_cat_matches(
            empty_column_result,
            np.empty((2, 0), dtype=np.float32),
            shape=(2, 0),
            stride=(1, 1),
            case="rank-2 empty columns",
        )

        single = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        single_result = torch.cat([single], dim=-1)
        self.assert_cat_matches(
            single_result,
            [[1.0, 2.0], [3.0, 4.0]],
            shape=(2, 2),
            stride=(2, 1),
            case="rank-2 single tensor",
        )
        self.assertFalse(single_result.is_set_to(single))
        self.assertNotEqual(single_result.data_ptr(), single.data_ptr())

    def test_rank_two_concat_aliases_share_supported_cat_path(self):
        for name in CAT_ALIASES:
            function = getattr(torch, name)
            with self.subTest(alias=name, axis="dim 0"):
                result = function(
                    [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])],
                    dim=0,
                )
                self.assert_cat_matches(
                    result,
                    [[1.0, 2.0], [3.0, 4.0]],
                    shape=(2, 2),
                    stride=(2, 1),
                    case=name,
                )

            with self.subTest(alias=name, axis="axis 1"):
                result = function(
                    [torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0], [4.0]])],
                    axis=1,
                )
                self.assert_cat_matches(
                    result,
                    [[1.0, 3.0], [2.0, 4.0]],
                    shape=(2, 2),
                    stride=(2, 1),
                    case=name,
                )

    def test_concat_aliases_share_supported_1d_cat_path(self):
        for name in CAT_ALIASES:
            function = getattr(torch, name)
            with self.subTest(alias=name, case="list"):
                base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                offset_noncontiguous = base.transpose(0, 1)[1]
                tail = torch.tensor([-0.0, 7.5])
                result = function([offset_noncontiguous, torch.tensor([]), tail], dim=0)
                self.assert_cat_matches(result, [2.0, 5.0, -0.0, 7.5], case=name)
                self.assertFalse(result.is_set_to(offset_noncontiguous))
                self.assertFalse(result.is_set_to(tail))
                self.assertNotEqual(result.data_ptr(), offset_noncontiguous.data_ptr())
                self.assertNotEqual(result.data_ptr(), tail.data_ptr())

            with self.subTest(alias=name, case="tuple dim -1"):
                result = function((torch.tensor([]), torch.tensor([3.25])), dim=-1)
                self.assert_cat_matches(result, [3.25], case=name)

            with self.subTest(alias=name, case="keywords"):
                result = function(
                    tensors=[torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
                    out=None,
                )
                self.assert_cat_matches(result, [8.0, 9.0], case=name)

            with self.subTest(alias=name, case="axis"):
                result = function([torch.tensor([14.0]), torch.tensor([15.0])], axis=0)
                self.assert_cat_matches(result, [14.0, 15.0], case=name)

    def test_autograd_accumulates_repeated_inputs_and_no_grad_disables_recording(self):
        left = torch.tensor([1.0, 2.0], requires_grad=True)
        right = torch.tensor([3.0], requires_grad=True)
        result = torch.cat([left, right, left], dim=0)
        self.assert_cat_matches(
            result,
            [1.0, 2.0, 3.0, 1.0, 2.0],
            case="rank-1 autograd output",
            requires_grad=True,
            is_leaf=False,
        )
        (result * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(left.grad), np.asarray([5.0, 7.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(right.grad), np.asarray([3.0], dtype=np.float32)
        )

        matrix_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        matrix_right = torch.tensor([[5.0], [6.0]], requires_grad=True)
        matrix_result = torch.cat([matrix_left, matrix_right, matrix_left], dim=1)
        self.assert_cat_matches(
            matrix_result,
            [[1.0, 2.0, 5.0, 1.0, 2.0], [3.0, 4.0, 6.0, 3.0, 4.0]],
            shape=(2, 5),
            stride=(5, 1),
            case="rank-2 autograd output",
            requires_grad=True,
            is_leaf=False,
        )
        (
            matrix_result
            * torch.tensor(
                [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
            )
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(matrix_left.grad),
            np.asarray([[5.0, 7.0], [15.0, 17.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(matrix_right.grad),
            np.asarray([[3.0], [8.0]], dtype=np.float32),
        )

        row_top = torch.tensor([[7.0, 8.0]], requires_grad=True)
        row_bottom = torch.tensor([[9.0, 10.0], [11.0, 12.0]], requires_grad=True)
        row_result = torch.cat([row_top, row_bottom, row_top], dim=0)
        self.assert_cat_matches(
            row_result,
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0], [7.0, 8.0]],
            shape=(4, 2),
            stride=(2, 1),
            case="rank-2 row autograd output",
            requires_grad=True,
            is_leaf=False,
        )
        (
            row_result
            * torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        ).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(row_top.grad),
            np.asarray([[8.0, 10.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(row_bottom.grad),
            np.asarray([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        )

        no_grad_left = torch.tensor([1.0, 2.0], requires_grad=True)
        no_grad_right = torch.tensor([3.0], requires_grad=True)
        with torch.no_grad():
            no_grad_result = torch.cat([no_grad_left, no_grad_right], dim=-1)
        self.assert_cat_matches(no_grad_result, [1.0, 2.0, 3.0], case="no_grad")
        self.assertIsNone(no_grad_left.grad)
        self.assertIsNone(no_grad_right.grad)

    def test_torch_function_modes_intercept_forward_and_decline(self):
        left = torch.tensor([1.0])
        right = torch.tensor([2.0])
        inputs = [left, right]
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        accepting = RecordingMode()
        with accepting:
            result = torch.cat(inputs, dim=0)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [accepting],
            )
        self.assertIs(result, marker)
        self.assertEqual(accepting.calls, [(torch.cat, (), (inputs,), {"dim": 0})])

        empty_accepting = RecordingMode()
        with empty_accepting:
            self.assertIs(torch.cat([], dim=1), marker)
        self.assertEqual(empty_accepting.calls, [(torch.cat, (), ([],), {"dim": 1})])

        calls = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append(
                    (
                        self.label,
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = torch.cat(inputs, dim=0)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )
        self.assert_cat_matches(forwarded, [1.0, 2.0], case="forwarded modes")
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.cat for call in calls))
        self.assertTrue(all(call[2] == () for call in calls))
        self.assertTrue(all(call[3] == (inputs,) for call in calls))
        self.assertTrue(all(call[4] == {"dim": 0} for call in calls))
        self.assertEqual(calls[0][5], (lower,))
        self.assertEqual(calls[1][5], ())
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.cat'; all __torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                torch.cat(inputs, dim=0)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        invalid_mode = RecordingMode()
        with invalid_mode:
            with self.assertRaisesRegex(
                TypeError, r"^expected Tensor as element 1 in argument 0, but got int$"
            ):
                torch.cat([left, 1], dim=0)
        self.assertEqual(invalid_mode.calls, [])

    def test_torch_function_argument_overrides_dispatch_in_pytorch_order(self):
        marker = object()
        events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("left", func, types, args, kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("right", func, types, args, kwargs))
                return marker

        left = LeftOverride()
        right = RightOverride()
        inputs = [left, torch.tensor([]), right]
        self.assertIs(torch.cat(inputs, dim=0), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.cat)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(args, (inputs,))
            self.assertEqual(kwargs, {"dim": 0})

        events.clear()

        class DimensionOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("dim", func, types, args, kwargs))
                return marker

        dimension = DimensionOverride()
        self.assertIs(torch.cat([torch.tensor([1.0])], dim=dimension), marker)
        self.assertEqual(events[0][0], "dim")
        self.assertIs(events[0][1], torch.cat)
        self.assertEqual(events[0][2], (DimensionOverride,))
        self.assertEqual(events[0][4], {"dim": dimension})

        events.clear()

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("out", func, types, args, kwargs))
                return marker

        out = OutOverride()
        self.assertIs(torch.cat([torch.tensor([1.0])], out=out), marker)
        self.assertEqual(events[0][0], "out")
        self.assertIs(events[0][1], torch.cat)
        self.assertEqual(events[0][2], (OutOverride,))
        self.assertIs(events[0][4]["out"], out)

        events.clear()

        class TensorsOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("tensors", func, types, args, kwargs))
                return marker

        tensors = TensorsOverride()
        self.assertIs(torch.cat(tensors), marker)
        self.assertEqual(events[0][0], "tensors")
        self.assertIs(events[0][1], torch.cat)
        self.assertEqual(events[0][2], (TensorsOverride,))
        self.assertEqual(events[0][3], (tensors,))
        self.assertIsNone(events[0][4])

        class DecliningOverride:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.cat'; all __torch_function__ handlers returned NotImplemented:",
        ):
            torch.cat([DecliningOverride()], dim=0)
        self.assertEqual(DecliningOverride.calls, 1)

    def test_concat_alias_torch_function_dispatch_uses_alias_callable(self):
        for name in CAT_ALIASES:
            function = getattr(torch, name)
            left = torch.tensor([1.0])
            right = torch.tensor([2.0])
            inputs = [left, right]
            marker = object()

            class RecordingMode(torch.overrides.TorchFunctionMode):
                def __init__(self, result=marker):
                    self.calls = []
                    self.result = result

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    self.calls.append((func, types, args, kwargs))
                    return self.result

            with self.subTest(alias=name, case="mode accepts"):
                accepting = RecordingMode()
                with accepting:
                    self.assertIs(function(inputs, dim=0), marker)
                self.assertEqual(accepting.calls, [(function, (), (inputs,), {"dim": 0})])

            with self.subTest(alias=name, case="mode sees empty before native failure"):
                empty_accepting = RecordingMode()
                with empty_accepting:
                    self.assertIs(function([], dim=1), marker)
                self.assertEqual(
                    empty_accepting.calls, [(function, (), ([],), {"dim": 1})]
                )

            calls = []

            class ForwardingMode(torch.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    calls.append((self.label, func, types, args, kwargs))
                    return func(*args, **(kwargs or {}))

            with self.subTest(alias=name, case="mode forwards"):
                with ForwardingMode("lower"):
                    with ForwardingMode("upper"):
                        forwarded = function(inputs, dim=0)
                self.assert_cat_matches(forwarded, [1.0, 2.0], case=name)
                self.assertEqual([call[0] for call in calls], ["upper", "lower"])
                self.assertTrue(all(call[1] is function for call in calls))
                self.assertTrue(all(call[2] == () for call in calls))
                self.assertTrue(all(call[3] == (inputs,) for call in calls))
                self.assertTrue(all(call[4] == {"dim": 0} for call in calls))

            events = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return marker

            with self.subTest(alias=name, case="sequence element override"):
                override_inputs = [Override()]
                self.assertIs(function(override_inputs, axis=0), marker)
                func, dispatch_types, args, kwargs = events.pop()
                self.assertIs(func, function)
                self.assertEqual(dispatch_types, (Override,))
                self.assertEqual(args, (override_inputs,))
                self.assertEqual(kwargs, {"axis": 0})

            class DimensionOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return marker

            with self.subTest(alias=name, case="dimension override"):
                dimension = DimensionOverride()
                self.assertIs(function([left], dim=dimension), marker)
                func, dispatch_types, args, kwargs = events.pop()
                self.assertIs(func, function)
                self.assertEqual(dispatch_types, (DimensionOverride,))
                self.assertEqual(args, ([left],))
                self.assertEqual(kwargs, {"dim": dimension})

            class OutOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return marker

            with self.subTest(alias=name, case="out override"):
                out = OutOverride()
                self.assertIs(function([left], out=out), marker)
                func, dispatch_types, args, kwargs = events.pop()
                self.assertIs(func, function)
                self.assertEqual(dispatch_types, (OutOverride,))
                self.assertEqual(args, ([left],))
                self.assertEqual(kwargs, {"out": out})

            class DecliningOverride:
                calls = 0

                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    cls.calls += 1
                    return NotImplemented

            with self.subTest(alias=name, case="declining override"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^Multiple dispatch failed for 'torch\.{name}'; all __torch_function__ handlers returned NotImplemented:",
                ):
                    function([DecliningOverride()], dim=0)
                self.assertEqual(DecliningOverride.calls, 1)

    def test_rejects_unsupported_inputs_before_touching_out(self):
        destination = torch.tensor([9.0, 10.0])
        before = destination.tolist()
        with self.assertRaisesRegex(
            RuntimeError, r"^cat\(\): the 'out' argument is not supported$"
        ):
            torch.cat([torch.tensor([1.0]), torch.tensor([2.0])], out=destination)
        self.assertEqual(destination.tolist(), before)

        for sequence in ([], ()):
            with self.subTest(sequence=type(sequence).__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    r"^torch\.cat\(\): expected a non-empty list of Tensors$",
                ):
                    torch.cat(sequence)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^zero-dimensional tensor \(at position 0\) cannot be concatenated$",
        ):
            torch.cat([torch.tensor(1.0)])

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^cat\(\): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported$",
        ):
            torch.cat([torch.zeros((1, 1, 1))])

        with self.assertRaisesRegex(
            RuntimeError,
            r"^Tensors must have same number of dimensions: got 2 and 1$",
        ):
            torch.cat([torch.tensor([[1.0]]), torch.tensor([2.0])], dim=0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^Sizes of tensors must match except in dimension 0\. Expected size 2 but got size 3 for tensor number 1 in the list\.$",
        ):
            torch.cat([torch.ones((2, 2)), torch.ones((3, 3))], dim=0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^Sizes of tensors must match except in dimension 1\. Expected size 2 but got size 3 for tensor number 1 in the list\.$",
        ):
            torch.cat([torch.ones((2, 2)), torch.ones((3, 3))], dim=1)

        with self.assertRaisesRegex(
            TypeError,
            r"^expected Tensor as element 1 in argument 0, but got int$",
        ):
            torch.cat([torch.tensor([1.0]), 1])

        with self.assertRaisesRegex(
            TypeError,
            r"^cat\(\): argument 'tensors' \(position 1\) must be tuple of Tensors, not Tensor$",
        ):
            torch.cat(torch.tensor([1.0]))

    def test_rejects_unsupported_dimensions(self):
        tensor = torch.tensor([1.0])
        for dimension in (1, -2):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-1, 0\], but got {dimension}\)$",
                ):
                    torch.cat([tensor], dim=dimension)

        with self.assertRaisesRegex(
            IndexError,
            r"^Dimension out of range \(expected to be in range of \[-1, 0\], but got 1\)$",
        ):
            torch.cat([tensor], axis=1)

        matrix = torch.tensor([[1.0]])
        for dimension in (2, -3):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-2, 1\], but got {dimension}\)$",
                ):
                    torch.cat([matrix], dim=dimension)

        for dimension, type_name in ((True, "bool"), (None, "NoneType"), ("0", "str")):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^cat\(\): argument 'dim' must be int, not {type_name}$",
                ):
                    torch.cat([tensor], dim=dimension)

        with self.assertRaisesRegex(
            TypeError, r"^cat\(\): argument 'dim' must be int, not str$"
        ):
            torch.cat([tensor], axis="0")

        for call in (
            lambda: torch.cat([tensor], dim=0, axis=0),
            lambda: torch.cat([tensor], 0, axis=0),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    call()

    def test_public_callable_surface(self):
        self.assertTrue(hasattr(torch, "cat"))
        self.assertIn("cat", torch.__all__)
        self.assertEqual(torch.__all__.count("cat"), 1)
        self.assertEqual(torch.cat.__name__, "cat")
        self.assertEqual(torch.cat.__qualname__, "_VariableFunctionsClass.cat")
        self.assertIs(torch._C._VariableFunctionsClass.cat, torch.cat)
        for name in CAT_ALIASES:
            with self.subTest(alias=name):
                function = getattr(torch, name)
                self.assertTrue(hasattr(torch, name))
                self.assertIn(name, torch.__all__)
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
                self.assertIs(getattr(torch._C._VariableFunctionsClass, name), function)
                self.assertIsNot(function, torch.cat)
        self.assertIsNot(torch.concat, torch.concatenate)

    def test_concat_aliases_preserve_cat_fail_closed_boundaries(self):
        for name in CAT_ALIASES:
            function = getattr(torch, name)
            with self.subTest(alias=name, boundary="concrete out"):
                destination = torch.tensor([9.0, 10.0])
                before = destination.tolist()
                with self.assertRaisesRegex(
                    RuntimeError, r"^cat\(\): the 'out' argument is not supported$"
                ):
                    function([torch.tensor([1.0]), torch.tensor([2.0])], out=destination)
                self.assertEqual(destination.tolist(), before)

            for sequence in ([], ()):
                with self.subTest(alias=name, boundary=f"empty {type(sequence).__name__}"):
                    with self.assertRaisesRegex(
                        ValueError,
                        r"^torch\.cat\(\): expected a non-empty list of Tensors$",
                    ):
                        function(sequence)

            with self.subTest(alias=name, boundary="scalar"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^zero-dimensional tensor \(at position 0\) cannot be concatenated$",
                ):
                    function([torch.tensor(1.0)])

            with self.subTest(alias=name, boundary="rank greater than 2"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^cat\(\): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported$",
                ):
                    function([torch.zeros((1, 1, 1))])

            with self.subTest(alias=name, boundary="rank mismatch"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Tensors must have same number of dimensions: got 2 and 1$",
                ):
                    function([torch.tensor([[1.0]]), torch.tensor([2.0])], dim=0)

            with self.subTest(alias=name, boundary="shape mismatch"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Sizes of tensors must match except in dimension 1\. Expected size 2 but got size 3 for tensor number 1 in the list\.$",
                ):
                    function([torch.ones((2, 2)), torch.ones((3, 3))], dim=1)

            with self.subTest(alias=name, boundary="non-tensor element"):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^expected Tensor as element 1 in argument 0, but got int$",
                ):
                    function([torch.tensor([1.0]), 1])

            with self.subTest(alias=name, boundary="non-sequence tensors"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): argument 'tensors' \(position 1\) must be tuple of Tensors, not Tensor$",
                ):
                    function(torch.tensor([1.0]))

            with self.subTest(alias=name, boundary="unsupported dim"):
                with self.assertRaisesRegex(
                    IndexError,
                    r"^Dimension out of range \(expected to be in range of \[-1, 0\], but got 1\)$",
                ):
                    function([torch.tensor([1.0])], dim=1)

            with self.subTest(alias=name, boundary="bad dim type"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): argument 'dim' must be int, not str$",
                ):
                    function([torch.tensor([1.0])], axis="0")

            with self.subTest(alias=name, boundary="axis dim conflict"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\) got an unexpected keyword argument 'axis'$",
                ):
                    function([torch.tensor([1.0])], dim=0, axis=0)


if __name__ == "__main__":
    unittest.main()
