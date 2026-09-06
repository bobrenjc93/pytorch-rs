import unittest

import numpy as np
import torch_rs as torch


class TopLevelCatTests(unittest.TestCase):
    def assert_cat_matches(self, actual, expected_values, *, case):
        expected = np.asarray(expected_values, dtype=np.float32)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, (len(expected_values),))
            self.assertEqual(actual.stride(), (1,))
            self.assertEqual(actual.storage_offset(), 0)
            self.assertTrue(actual.is_contiguous())
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
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

    def test_no_grad_allows_grad_requiring_operands_without_recording(self):
        left = torch.tensor([1.0, 2.0], requires_grad=True)
        right = torch.tensor([3.0], requires_grad=True)

        with self.assertRaisesRegex(
            RuntimeError, r"^cat\(\): autograd recording is not supported$"
        ):
            torch.cat([left, right])
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

        with torch.no_grad():
            result = torch.cat([left, right], dim=-1)
        self.assert_cat_matches(result, [1.0, 2.0, 3.0], case="no_grad")
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

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
            r"^cat\(\): only exact native CPU float32 1-D Tensor inputs are supported$",
        ):
            torch.cat([torch.tensor([[1.0]])])

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


if __name__ == "__main__":
    unittest.main()
