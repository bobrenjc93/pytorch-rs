import unittest

import numpy as np
import torch_rs as torch


VSTACK_ALIASES = ("vstack", "row_stack")


class TopLevelVstackTests(unittest.TestCase):
    @staticmethod
    def contiguous_stride(shape):
        strides = []
        stride = 1
        for dimension in reversed(shape):
            strides.append(stride)
            stride *= max(dimension, 1)
        return tuple(reversed(strides))

    def assert_tensor_matches(
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

    @staticmethod
    def assert_fresh_storage(test_case, result, inputs):
        for index, input in enumerate(inputs):
            if input.numel():
                with test_case.subTest(input_index=index, storage=True):
                    test_case.assertFalse(result.is_set_to(input))
                    test_case.assertNotEqual(result.data_ptr(), input.data_ptr())

    def test_values_shape_stride_and_fresh_storage_for_supported_inputs(self):
        for name in VSTACK_ALIASES:
            function = getattr(torch, name)
            with self.subTest(function=name, case="scalars"):
                inputs = [torch.tensor(-0.0), torch.tensor(2.5)]
                result = function(inputs)
                self.assert_tensor_matches(
                    result,
                    [[-0.0], [2.5]],
                    case=f"{name} scalars",
                )
                self.assert_fresh_storage(self, result, inputs)

            with self.subTest(function=name, case="vectors"):
                base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                view = base.transpose(0, 1)[1]
                tail = torch.tensor([-0.0, 7.5])
                self.assertEqual(view.tolist(), [2.0, 5.0])
                self.assertEqual(view.stride(), (3,))
                self.assertEqual(view.storage_offset(), 1)
                inputs = [view, tail]
                result = function(inputs)
                self.assert_tensor_matches(
                    result,
                    [[2.0, 5.0], [-0.0, 7.5]],
                    case=f"{name} vectors",
                )
                self.assert_fresh_storage(self, result, inputs)

            with self.subTest(function=name, case="matrices and empty rows"):
                base_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                base = torch.tensor(base_values.tolist())
                view = base[1].transpose(0, 1)
                expected_view = base_values[1].transpose(1, 0)
                tail = torch.tensor([[-0.0, 100.0, 101.0]])
                inputs = (view, torch.zeros((0, 3)), tail)
                result = function(inputs, out=None)
                self.assert_tensor_matches(
                    result,
                    np.concatenate(
                        [
                            expected_view,
                            np.zeros((0, 3), dtype=np.float32),
                            np.asarray(
                                [[-0.0, 100.0, 101.0]],
                                dtype=np.float32,
                            ),
                        ],
                        axis=0,
                    ),
                    shape=(5, 3),
                    stride=(3, 1),
                    case=f"{name} matrices",
                )
                self.assert_fresh_storage(self, result, inputs)

            with self.subTest(function=name, case="mixed scalar vector matrix"):
                inputs = [
                    torch.tensor(1.0),
                    torch.tensor([2.0]),
                    torch.tensor([[3.0], [4.0]]),
                ]
                result = function(tensors=inputs)
                self.assert_tensor_matches(
                    result,
                    [[1.0], [2.0], [3.0], [4.0]],
                    shape=(4, 1),
                    stride=(1, 1),
                    case=f"{name} mixed ranks",
                )
                self.assert_fresh_storage(self, result, inputs)

            with self.subTest(function=name, case="empty vectors"):
                inputs = [torch.tensor([]), torch.tensor([])]
                result = function(inputs)
                self.assert_tensor_matches(
                    result,
                    np.empty((2, 0), dtype=np.float32),
                    shape=(2, 0),
                    stride=(1, 1),
                    case=f"{name} empty vectors",
                )

            with self.subTest(function=name, case="single matrix"):
                single = torch.tensor([[11.0, 13.0], [17.0, 19.0]])
                result = function([single])
                self.assert_tensor_matches(
                    result,
                    [[11.0, 13.0], [17.0, 19.0]],
                    shape=(2, 2),
                    stride=(2, 1),
                    case=f"{name} single matrix",
                )
                self.assert_fresh_storage(self, result, [single])

    def test_autograd_accumulates_repeated_inputs_and_no_grad_disables_recording(self):
        for name in VSTACK_ALIASES:
            function = getattr(torch, name)
            with self.subTest(function=name, case="vectors"):
                left = torch.tensor([1.0, 2.0], requires_grad=True)
                right = torch.tensor([3.0, 4.0], requires_grad=True)
                result = function([left, right, left])
                self.assert_tensor_matches(
                    result,
                    [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]],
                    shape=(3, 2),
                    stride=(2, 1),
                    case=f"{name} vector autograd output",
                    requires_grad=True,
                    is_leaf=False,
                )
                (result * torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])).sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(left.grad),
                    np.asarray([6.0, 8.0], dtype=np.float32),
                )
                np.testing.assert_array_equal(
                    np.asarray(right.grad),
                    np.asarray([3.0, 4.0], dtype=np.float32),
                )

            with self.subTest(function=name, case="mixed ranks"):
                scalar = torch.tensor(1.5, requires_grad=True)
                vector = torch.tensor([2.5], requires_grad=True)
                matrix = torch.tensor([[3.5]], requires_grad=True)
                result = function([scalar, vector, matrix, scalar])
                self.assert_tensor_matches(
                    result,
                    [[1.5], [2.5], [3.5], [1.5]],
                    shape=(4, 1),
                    stride=(1, 1),
                    case=f"{name} mixed autograd output",
                    requires_grad=True,
                    is_leaf=False,
                )
                (result * torch.tensor([[1.0], [2.0], [3.0], [4.0]])).sum().backward()
                self.assertEqual(scalar.grad.item(), 5.0)
                self.assertEqual(vector.grad.tolist(), [2.0])
                self.assertEqual(matrix.grad.tolist(), [[3.0]])

            with self.subTest(function=name, case="no_grad"):
                left = torch.tensor([1.0, 2.0], requires_grad=True)
                right = torch.tensor([3.0, 4.0], requires_grad=True)
                with torch.no_grad():
                    result = function([left, right])
                self.assert_tensor_matches(
                    result,
                    [[1.0, 2.0], [3.0, 4.0]],
                    shape=(2, 2),
                    stride=(2, 1),
                    case=f"{name} no_grad",
                )
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)

    def test_torch_function_modes_intercept_forward_and_decline(self):
        for name in VSTACK_ALIASES:
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

            with self.subTest(function=name, case="mode accepts"):
                accepting = RecordingMode()
                with accepting:
                    self.assertIs(function(inputs, out=None), marker)
                self.assertEqual(
                    accepting.calls,
                    [(function, (), (inputs,), {"out": None})],
                )

            with self.subTest(function=name, case="empty before native failure"):
                empty_accepting = RecordingMode()
                with empty_accepting:
                    self.assertIs(function([]), marker)
                self.assertEqual(empty_accepting.calls, [(function, (), ([],), None)])

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

            with self.subTest(function=name, case="mode forwards"):
                lower = ForwardingMode("lower")
                upper = ForwardingMode("upper")
                with lower:
                    with upper:
                        forwarded = function(inputs)
                        self.assertEqual(
                            torch.overrides._get_current_function_mode_stack(),
                            [lower, upper],
                        )
                self.assert_tensor_matches(
                    forwarded,
                    [[1.0], [2.0]],
                    shape=(2, 1),
                    stride=(1, 1),
                    case=f"{name} forwarded modes",
                )
                self.assertEqual([call[0] for call in calls], ["upper", "lower"])
                self.assertTrue(all(call[1] is function for call in calls))
                self.assertTrue(all(call[2] == () for call in calls))
                self.assertTrue(all(call[3] == (inputs,) for call in calls))
                self.assertIsNone(calls[0][4])
                self.assertEqual(calls[1][4], {})
                self.assertEqual(calls[0][5], (lower,))
                self.assertEqual(calls[1][5], ())
                self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

            with self.subTest(function=name, case="mode declines"):
                declining = RecordingMode(NotImplemented)
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^Multiple dispatch failed for 'torch\.{name}'; all __torch_function__ handlers returned NotImplemented:",
                ):
                    with declining:
                        function(inputs)
                self.assertEqual(len(declining.calls), 1)
                self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

            with self.subTest(function=name, case="invalid input bypasses mode"):
                invalid_mode = RecordingMode()
                with invalid_mode:
                    with self.assertRaisesRegex(
                        TypeError,
                        r"^expected Tensor as element 1 in argument 0, but got int$",
                    ):
                        function([left, 1])
                self.assertEqual(invalid_mode.calls, [])

    def test_torch_function_argument_overrides_dispatch_in_pytorch_order(self):
        for name in VSTACK_ALIASES:
            function = getattr(torch, name)
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

            inputs = [LeftOverride(), torch.tensor([]), RightOverride()]
            self.assertIs(function(inputs), marker)
            self.assertEqual([event[0] for event in events], ["left", "right"])
            for _, dispatched_function, dispatch_types, args, kwargs in events:
                self.assertIs(dispatched_function, function)
                self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
                self.assertEqual(args, (inputs,))
                self.assertIsNone(kwargs)

            events.clear()

            class OutOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(("out", func, types, args, kwargs))
                    return marker

            out = OutOverride()
            self.assertIs(function([torch.tensor([1.0])], out=out), marker)
            self.assertEqual(events[0][0], "out")
            self.assertIs(events[0][1], function)
            self.assertEqual(events[0][2], (OutOverride,))
            self.assertIs(events[0][4]["out"], out)

            events.clear()

            class TensorsOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(("tensors", func, types, args, kwargs))
                    return marker

            tensors = TensorsOverride()
            self.assertIs(function(tensors), marker)
            self.assertEqual(events[0][0], "tensors")
            self.assertIs(events[0][1], function)
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
                rf"^Multiple dispatch failed for 'torch\.{name}'; all __torch_function__ handlers returned NotImplemented:",
            ):
                function([DecliningOverride()])
            self.assertEqual(DecliningOverride.calls, 1)

    def test_representative_validation_errors(self):
        for name in VSTACK_ALIASES:
            function = getattr(torch, name)
            tensor = torch.tensor([1.0])
            with self.subTest(function=name, boundary="concrete out"):
                destination = torch.tensor([[9.0]])
                before = destination.tolist()
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): the 'out' argument is not supported$",
                ):
                    function([tensor], out=destination)
                self.assertEqual(destination.tolist(), before)

            for sequence in ([], ()):
                with self.subTest(
                    function=name,
                    boundary=f"empty {type(sequence).__name__}",
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^vstack expects a non-empty TensorList$",
                    ):
                        function(sequence)

            with self.subTest(function=name, boundary="rank greater than 2"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): only exact native CPU float32 scalar, rank-1, or rank-2 Tensor inputs are supported$",
                ):
                    function([torch.zeros((1, 1, 1))])

            with self.subTest(function=name, boundary="shape mismatch"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Sizes of tensors must match except in dimension 0\. Expected size 2 but got size 1 for tensor number 1 in the list\.$",
                ):
                    function([torch.tensor([1.0, 2.0]), torch.tensor([3.0])])

            with self.subTest(function=name, boundary="non-tensor element"):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^expected Tensor as element 1 in argument 0, but got int$",
                ):
                    function([tensor, 1])

            with self.subTest(function=name, boundary="non-sequence tensors"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): argument 'tensors' \(position 1\) must be tuple of Tensors, not Tensor$",
                ):
                    function(tensor)

            with self.subTest(function=name, boundary="bad out type"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): argument 'out' must be Tensor, not int$",
                ):
                    function([tensor], out=1)

            with self.subTest(function=name, boundary="unexpected dim"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\) got an unexpected keyword argument 'dim'$",
                ):
                    function([tensor], dim=0)

            with self.subTest(function=name, boundary="missing tensors"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\) missing 1 required positional arguments: \"tensors\"$",
                ):
                    function()

            with self.subTest(function=name, boundary="too many positional"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\) takes 1 positional argument but 2 were given$",
                ):
                    function([tensor], 0)

    def test_public_callable_surface(self):
        for name in VSTACK_ALIASES:
            with self.subTest(function=name):
                function = getattr(torch, name)
                self.assertTrue(hasattr(torch, name))
                self.assertIn(name, torch.__all__)
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__module__, "torch")
                self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
                self.assertIn(f"{name}(tensors, *, out=None) -> Tensor", function.__doc__)
                self.assertIs(getattr(torch._C._VariableFunctionsClass, name), function)
                self.assertNotIn(name, torch.functional.__all__)
        self.assertIsNot(torch.vstack, torch.row_stack)


if __name__ == "__main__":
    unittest.main()
