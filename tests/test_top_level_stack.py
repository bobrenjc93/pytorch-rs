import unittest

import numpy as np
import torch_rs as torch


class TopLevelStackTests(unittest.TestCase):
    def assert_tensor_matches(
        self,
        actual,
        expected_values,
        *,
        shape,
        stride,
        case,
        requires_grad=False,
        is_leaf=True,
    ):
        expected = np.asarray(expected_values, dtype=np.float32)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(shape))
            self.assertEqual(actual.stride(), tuple(stride))
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

    def test_list_tuple_scalar_empty_views_and_axis_alias(self):
        base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        view = base.transpose(0, 1)
        fill = torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])

        result = torch.stack([view, fill], dim=1)
        self.assert_tensor_matches(
            result,
            [
                [[1.0, 4.0], [10.0, 11.0]],
                [[2.0, 5.0], [12.0, 13.0]],
                [[3.0, 6.0], [14.0, 15.0]],
            ],
            shape=(3, 2, 2),
            stride=(4, 2, 1),
            case="non-contiguous dim 1",
        )
        self.assertFalse(result.is_set_to(view))
        self.assertFalse(result.is_set_to(fill))
        self.assertNotEqual(result.data_ptr(), view.data_ptr())
        self.assertNotEqual(result.data_ptr(), fill.data_ptr())

        tuple_result = torch.stack(
            (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
            dim=-1,
        )
        self.assert_tensor_matches(
            tuple_result,
            [[1.0, 3.0], [2.0, 4.0]],
            shape=(2, 2),
            stride=(2, 1),
            case="tuple dim -1",
        )

        scalar_result = torch.stack([torch.tensor(2.0), torch.tensor(-0.0)])
        self.assert_tensor_matches(
            scalar_result,
            [2.0, -0.0],
            shape=(2,),
            stride=(1,),
            case="scalars",
        )

        empty_result = torch.stack(
            [torch.zeros((2, 0, 3)), torch.zeros((2, 0, 3))],
            dim=2,
        )
        self.assert_tensor_matches(
            empty_result,
            np.empty((2, 0, 2, 3), dtype=np.float32),
            shape=(2, 0, 2, 3),
            stride=(6, 6, 3, 1),
            case="empty middle",
        )

        keyword_result = torch.stack(
            tensors=[torch.tensor([8.0]), torch.tensor([9.0])],
            out=None,
        )
        self.assert_tensor_matches(
            keyword_result,
            [[8.0], [9.0]],
            shape=(2, 1),
            stride=(1, 1),
            case="keywords",
        )

        axis_result = torch.stack(
            [torch.tensor([14.0]), torch.tensor([15.0])],
            axis=1,
        )
        self.assert_tensor_matches(
            axis_result,
            [[14.0, 15.0]],
            shape=(1, 2),
            stride=(2, 1),
            case="axis alias",
        )

    def test_autograd_accumulates_repeated_inputs_and_no_grad_disables_recording(self):
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        result = torch.stack([left, right, left], dim=1)

        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        self.assert_tensor_matches(
            result,
            [
                [[1.0, 2.0], [5.0, 6.0], [1.0, 2.0]],
                [[3.0, 4.0], [7.0, 8.0], [3.0, 4.0]],
            ],
            shape=(2, 3, 2),
            stride=(6, 2, 1),
            case="autograd output",
            requires_grad=True,
            is_leaf=False,
        )

        weights = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )
        (result * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(left.grad), np.asarray([[6.0, 8.0], [18.0, 20.0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            np.asarray(right.grad), np.asarray([[3.0, 4.0], [9.0, 10.0]], dtype=np.float32)
        )

        no_grad_left = torch.tensor([1.0, 2.0], requires_grad=True)
        no_grad_right = torch.tensor([3.0, 4.0], requires_grad=True)
        with torch.no_grad():
            no_grad_result = torch.stack([no_grad_left, no_grad_right], dim=-1)
        self.assert_tensor_matches(
            no_grad_result,
            [[1.0, 3.0], [2.0, 4.0]],
            shape=(2, 2),
            stride=(2, 1),
            case="no_grad",
        )
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
            result = torch.stack(inputs, dim=1)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(),
                [accepting],
            )
        self.assertIs(result, marker)
        self.assertEqual(accepting.calls, [(torch.stack, (), (inputs,), {"dim": 1})])

        empty_accepting = RecordingMode()
        with empty_accepting:
            self.assertIs(torch.stack([], dim=1), marker)
        self.assertEqual(empty_accepting.calls, [(torch.stack, (), ([],), {"dim": 1})])

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
                forwarded = torch.stack(inputs, dim=1)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(),
                    [lower, upper],
                )
        self.assert_tensor_matches(
            forwarded,
            [[1.0, 2.0]],
            shape=(1, 2),
            stride=(2, 1),
            case="forwarded modes",
        )
        self.assertEqual([call[0] for call in calls], ["upper", "lower"])
        self.assertTrue(all(call[1] is torch.stack for call in calls))
        self.assertTrue(all(call[2] == () for call in calls))
        self.assertTrue(all(call[3] == (inputs,) for call in calls))
        self.assertTrue(all(call[4] == {"dim": 1} for call in calls))
        self.assertEqual(calls[0][5], (lower,))
        self.assertEqual(calls[1][5], ())
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.stack'; all __torch_function__ handlers returned NotImplemented:",
        ):
            with declining:
                torch.stack(inputs, dim=1)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        invalid_mode = RecordingMode()
        with invalid_mode:
            with self.assertRaisesRegex(
                TypeError, r"^expected Tensor as element 1 in argument 0, but got int$"
            ):
                torch.stack([left, 1], dim=0)
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
        inputs = [left, torch.tensor([0.0]), right]
        self.assertIs(torch.stack(inputs, dim=0), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.stack)
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
        self.assertIs(torch.stack([torch.tensor([1.0])], dim=dimension), marker)
        self.assertEqual(events[0][0], "dim")
        self.assertIs(events[0][1], torch.stack)
        self.assertEqual(events[0][2], (DimensionOverride,))
        self.assertEqual(events[0][4], {"dim": dimension})

        events.clear()

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("out", func, types, args, kwargs))
                return marker

        out = OutOverride()
        self.assertIs(torch.stack([torch.tensor([1.0])], out=out), marker)
        self.assertEqual(events[0][0], "out")
        self.assertIs(events[0][1], torch.stack)
        self.assertEqual(events[0][2], (OutOverride,))
        self.assertIs(events[0][4]["out"], out)

        events.clear()

        class TensorsOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("tensors", func, types, args, kwargs))
                return marker

        tensors = TensorsOverride()
        self.assertIs(torch.stack(tensors), marker)
        self.assertEqual(events[0][0], "tensors")
        self.assertIs(events[0][1], torch.stack)
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
            r"^Multiple dispatch failed for 'torch\.stack'; all __torch_function__ handlers returned NotImplemented:",
        ):
            torch.stack([DecliningOverride()], dim=0)
        self.assertEqual(DecliningOverride.calls, 1)

    def test_representative_validation_errors(self):
        destination = torch.tensor([[9.0]])
        before = destination.tolist()
        with self.assertRaisesRegex(
            RuntimeError, r"^stack\(\): the 'out' argument is not supported$"
        ):
            torch.stack([torch.tensor([1.0])], out=destination)
        self.assertEqual(destination.tolist(), before)

        for sequence in ([], ()):
            with self.subTest(sequence=type(sequence).__name__):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^stack expects a non-empty TensorList$",
                ):
                    torch.stack(sequence)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^stack expects each tensor to be equal size, but got \[1\] at entry 0 and \[2\] at entry 1$",
        ):
            torch.stack([torch.tensor([1.0]), torch.tensor([1.0, 2.0])])

        with self.assertRaisesRegex(
            TypeError,
            r"^expected Tensor as element 1 in argument 0, but got int$",
        ):
            torch.stack([torch.tensor([1.0]), 1])

        with self.assertRaisesRegex(
            TypeError,
            r"^stack\(\): argument 'tensors' \(position 1\) must be tuple of Tensors, not Tensor$",
        ):
            torch.stack(torch.tensor([1.0]))

        tensor = torch.tensor([1.0])
        for dimension in (2, -3):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-2, 1\], but got {dimension}\)$",
                ):
                    torch.stack([tensor], dim=dimension)

        with self.assertRaisesRegex(
            IndexError,
            r"^Dimension out of range \(expected to be in range of \[-2, 1\], but got 2\)$",
        ):
            torch.stack([tensor, torch.tensor([1.0, 2.0])], dim=2)

        for dimension, type_name in ((True, "bool"), (None, "NoneType"), ("0", "str")):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^stack\(\): argument 'dim' must be int, not {type_name}$",
                ):
                    torch.stack([tensor], dim=dimension)

        with self.assertRaisesRegex(
            TypeError, r"^stack\(\): argument 'dim' must be int, not str$"
        ):
            torch.stack([tensor], axis="0")

        for call in (
            lambda: torch.stack([tensor], dim=0, axis=0),
            lambda: torch.stack([tensor], 0, axis=0),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^stack\(\) got an unexpected keyword argument 'axis'$",
                ):
                    call()

        with self.assertRaisesRegex(
            TypeError,
            r"^stack\(\) missing 1 required positional arguments: \"tensors\"$",
        ):
            torch.stack()

        with self.assertRaisesRegex(
            TypeError,
            r"^stack\(\) takes from 1 to 2 positional arguments but 3 were given$",
        ):
            torch.stack([tensor], 0, 1)

    def test_public_callable_surface(self):
        self.assertTrue(hasattr(torch, "stack"))
        self.assertIn("stack", torch.__all__)
        self.assertEqual(torch.__all__.count("stack"), 1)
        self.assertEqual(torch.stack.__name__, "stack")
        self.assertEqual(torch.stack.__module__, "torch")
        self.assertEqual(torch.stack.__qualname__, "_VariableFunctionsClass.stack")
        self.assertIn("stack(tensors, dim=0, *, out=None)", torch.stack.__doc__)
        self.assertIs(torch._C._VariableFunctionsClass.stack, torch.stack)


if __name__ == "__main__":
    unittest.main()
