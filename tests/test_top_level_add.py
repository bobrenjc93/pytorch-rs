import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TopLevelAddTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_tensor_tensor_values_broadcast_empty_offsets_and_special_bits(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left + right
        calls = (
            ("positional tensors", lambda: torch.add(left, right)),
            ("canonical keywords", lambda: torch.add(input=left, other=right)),
            ("x aliases", lambda: torch.add(x=left, x2=right)),
            ("x1 aliases", lambda: torch.add(x1=left, x2=right)),
            ("default alpha", lambda: torch.add(left, right, alpha=1)),
            ("float alpha", lambda: torch.add(left, right, alpha=1.0)),
            ("numpy alpha", lambda: torch.add(left, right, alpha=np.int64(1))),
            ("explicit out none", lambda: torch.add(left, right, out=None)),
        )
        for case, call in calls:
            self.assert_tensor_matches(call(), expected, case=case)

        offset_noncontiguous = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ).transpose(0, 1)[1]
        self.assertGreater(offset_noncontiguous.storage_offset(), 0)
        self.assertFalse(offset_noncontiguous.is_contiguous())
        offset_other = torch.tensor([10.0, 20.0])
        self.assert_tensor_matches(
            torch.add(offset_noncontiguous, offset_other),
            offset_noncontiguous + offset_other,
            case="offset noncontiguous",
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.add(empty, broadcast),
            empty + broadcast,
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        zeros = torch.zeros((5,))
        result = torch.add(special, zeros)
        self.assert_tensor_matches(
            result, special + zeros, case="signed zero nan infinity"
        )
        self.assertFalse(result.is_set_to(special))
        self.assertFalse(result.is_set_to(zeros))
        if result.numel():
            self.assertNotEqual(result.data_ptr(), special.data_ptr())
            self.assertNotEqual(result.data_ptr(), zeros.data_ptr())

    def test_tensor_scalar_right_values_empty_offsets_and_special_bits(self):
        left = torch.tensor(
            [[[1.0, -0.0], [2.0, float("inf")], [3.0, -6.0]]]
        ).transpose(0, 2)
        calls = (
            ("positional python bool", lambda: torch.add(left, True), left + True),
            ("positional python int", lambda: torch.add(left, -2), left + -2),
            ("positional python float", lambda: torch.add(left, 2.5), left + 2.5),
            (
                "canonical keywords",
                lambda: torch.add(input=left, other=np.float32(3.0)),
                left + np.float32(3.0),
            ),
            (
                "default alpha",
                lambda: torch.add(left, np.float64(-1.5), alpha=np.int64(1)),
                left + np.float64(-1.5),
            ),
            (
                "explicit out none",
                lambda: torch.add(left, np.bool_(True), out=None),
                left + np.bool_(True),
            ),
        )
        for case, call, expected in calls:
            self.assert_tensor_matches(call(), expected, case=case)

        offset_noncontiguous = left[1]
        self.assertGreater(offset_noncontiguous.storage_offset(), 0)
        self.assertFalse(offset_noncontiguous.is_contiguous())
        self.assert_tensor_matches(
            torch.add(offset_noncontiguous, np.float32(-0.0)),
            offset_noncontiguous + np.float32(-0.0),
            case="offset noncontiguous signed-zero scalar",
        )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_tensor_matches(
            torch.add(empty, -3.5),
            empty + -3.5,
            case="strided empty scalar",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        result = torch.add(special, np.float32(0.0))
        self.assert_tensor_matches(
            result, special + np.float32(0.0), case="signed zero nan infinity"
        )
        self.assertFalse(result.is_set_to(special))
        if result.numel():
            self.assertNotEqual(result.data_ptr(), special.data_ptr())

    def test_autograd_no_grad_and_full_sum_backward_reuse_tensor_add(self):
        function_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        function_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        function_output = torch.add(
            function_left.transpose(0, 1), function_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) + operator_right.transpose(
            0, 1
        )
        self.assert_tensor_matches(function_output, operator_output, case="views")
        function_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            function_left.grad, operator_left.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            function_right.grad, operator_right.grad, case="right gradient"
        )

        shared_function = torch.tensor([2.0, -3.0], requires_grad=True)
        shared_operator = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(shared_function, shared_function).sum().backward()
        (shared_operator + shared_operator).sum().backward()
        self.assert_tensor_matches(
            shared_function.grad,
            shared_operator.grad,
            case="shared operand gradient",
        )

        scalar_function = torch.tensor([2.0, -3.0], requires_grad=True)
        scalar_operator = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(scalar_function, 4.0).sum().backward()
        (scalar_operator + 4.0).sum().backward()
        self.assert_tensor_matches(
            scalar_function.grad,
            scalar_operator.grad,
            case="scalar right gradient",
        )

        empty_function = torch.zeros((2, 0, 3), requires_grad=True)
        empty_operator = torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(empty_function, torch.ones((1, 1, 3))).sum().backward()
        (empty_operator + torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            empty_function.grad, empty_operator.grad, case="empty gradient"
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = torch.add(left.transpose(0, 1), right.transpose(0, 1))
            scalar_output = torch.add(left, 2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(torch.add(left, right.transpose(0, 1)).requires_grad)
        self.assertTrue(torch.add(left, 2.0).requires_grad)

    def test_torch_function_modes_and_overrides_observe_original_calls(self):
        left = torch.tensor([2.0])
        right = torch.tensor([3.0])
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            (lambda: torch.add(left, right), (left, right), None),
            (lambda: torch.add(left, 4.0), (left, 4.0), None),
            (
                lambda: torch.add(input=left, other=right, alpha=2),
                (),
                ("input", "other", "alpha"),
            ),
            (
                lambda: torch.add(left, right, out=destination),
                (left, right),
                ("out",),
            ),
            (lambda: torch.add(x1=left, x2=right), (), ("x1", "x2")),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            if expected_keywords is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), expected_keywords)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = torch.add(input=left, other=right, alpha=1)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left + right, case="forwarded modes")

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                actual = torch.add(input=left, other=4.0, alpha=1)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left + 4.0, case="forwarded scalar")

        for call in (
            lambda: torch.add([], right),
            lambda: torch.add(left, []),
            lambda: torch.add(left, right, alpha=[]),
            lambda: torch.add(left, right, out=[]),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

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

        self.assertIs(torch.add(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.add(input=left, other=RightOverride(), alpha=2), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other", "alpha"))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.add(DecliningOverride(), left)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.add'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_unsupported_surface_errors_do_not_mutate_inputs(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([3.0])
        unsupported_operand = (
            r"^add\(\): only exact native CPU float32 Tensor/Tensor operands "
            r"or Tensor input with real-number other are supported$"
        )
        for call in (
            lambda: torch.add(2.0, tensor),
            lambda: torch.add(2, 3),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(NotImplementedError, unsupported_operand):
                    call()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            torch.add(tensor, other, alpha=2)
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): alpha values other than 1 are not supported$",
        ):
            torch.add(tensor, 2.0, alpha=2)
        for alpha in (True, np.bool_(True)):
            with self.subTest(alpha=type(alpha).__name__):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Boolean alpha only supported for Boolean results\.$",
                ):
                    torch.add(tensor, other, alpha=alpha)
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Boolean alpha only supported for Boolean results\.$",
                ):
                    torch.add(tensor, 2.0, alpha=alpha)

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, other, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, 2.0, out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        for call, message in (
            (
                lambda: torch.add(),
                "add() received an invalid combination of arguments - got (), "
                "but expected (Tensor input, Tensor or Number other, *, "
                "Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor),
                "add() received an invalid combination of arguments - got "
                "(Tensor), but expected (Tensor input, Tensor or Number other, "
                "*, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.add(tensor, other, other),
                "add() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.add(tensor, other, dtype=torch.float32),
                "add() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.add(tensor, other, device=torch.device("cpu")),
                "add() got an unexpected keyword argument 'device'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        self.assertFalse(hasattr(torch, "add_"))
        self.assertFalse(hasattr(torch.Tensor, "add_"))

    def test_callable_metadata_copy_pickle_reload_and_exports(self):
        function = torch.add
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "add")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.add")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertTrue(
            function.__doc__.startswith(
                "\nadd(input, other, *, alpha=1, out=None) -> Tensor\n\n"
            )
        )
        self.assertRegex(
            repr(function), r"^<built-in method add of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.add, function)
        for mutation in (
            lambda: setattr(owner, "add", None),
            lambda: delattr(owner, "add"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.add, function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertEqual(torch.__all__.count("add"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["add"], function)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.add, function)


if __name__ == "__main__":
    unittest.main()
