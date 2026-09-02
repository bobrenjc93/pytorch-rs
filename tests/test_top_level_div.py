import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TopLevelDivisionTests(unittest.TestCase):
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

    def test_supported_calls_reuse_true_division_values_layouts_and_edge_cases(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left / right

        for name in ("div", "divide"):
            function = getattr(torch, name)
            calls = (
                ("positional tensors", lambda function=function: function(left, right)),
                (
                    "canonical keywords",
                    lambda function=function: function(input=left, other=right),
                ),
                ("x aliases", lambda function=function: function(x=left, x2=right)),
                ("x1 aliases", lambda function=function: function(x1=left, x2=right)),
                (
                    "explicit true division",
                    lambda function=function: function(
                        left, right, rounding_mode=None
                    ),
                ),
                ("explicit out none", lambda function=function: function(left, right, out=None)),
            )
            for case, call in calls:
                self.assert_tensor_matches(call(), expected, case=(name, case))

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for name in ("div", "divide"):
                function = getattr(torch, name)
                for order, call, expected in (
                    (
                        "tensor/scalar",
                        lambda function=function, scalar=scalar: function(offset, scalar),
                        offset / scalar,
                    ),
                    (
                        "scalar/tensor",
                        lambda function=function, scalar=scalar: function(scalar, offset),
                        torch.tensor(float(scalar)) / offset,
                    ),
                    (
                        "keyword scalar/tensor",
                        lambda function=function, scalar=scalar: function(
                            input=scalar, other=offset
                        ),
                        torch.tensor(float(scalar)) / offset,
                    ),
                ):
                    self.assert_tensor_matches(
                        call(), expected, case=(name, order, type(scalar).__name__, scalar)
                    )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        for name in ("div", "divide"):
            self.assert_tensor_matches(
                getattr(torch, name)(empty, broadcast),
                empty / broadcast,
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        divisors = torch.tensor([1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0])
        for name in ("div", "divide"):
            result = getattr(torch, name)(special, divisors)
            self.assert_tensor_matches(
                result,
                special / divisors,
                case=(name, "signed zero nan infinity"),
            )
            self.assertFalse(result.is_set_to(special))
            self.assertFalse(result.is_set_to(divisors))
            if result.numel():
                self.assertNotEqual(result.data_ptr(), special.data_ptr())
                self.assertNotEqual(result.data_ptr(), divisors.data_ptr())

    def test_active_autograd_is_rejected_but_no_grad_uses_native_division(self):
        for name in ("div", "divide"):
            function = getattr(torch, name)
            left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            right = torch.tensor([[5.0], [7.0]], requires_grad=True)

            for case, call in (
                (
                    "tensor operands",
                    lambda function=function: function(
                        left.transpose(0, 1), right.transpose(0, 1)
                    ),
                ),
                ("tensor scalar", lambda function=function: function(left, 2.0)),
                ("scalar tensor", lambda function=function: function(2.0, left)),
            ):
                with self.subTest(name=name, case=case):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^{name}\(\): autograd recording is not supported$",
                    ):
                        call()
                    self.assertIsNone(left.grad)
                    self.assertIsNone(right.grad)

            with self.subTest(name=name, case="no_grad"):
                with torch.no_grad():
                    tensor_output = function(
                        left.transpose(0, 1), right.transpose(0, 1)
                    )
                    scalar_output = function(2.0, left)
                    expected_tensor_output = left.transpose(0, 1) / right.transpose(
                        0, 1
                    )
                    expected_scalar_output = torch.tensor(2.0) / left
                self.assertFalse(tensor_output.requires_grad)
                self.assertTrue(tensor_output.is_leaf)
                self.assert_tensor_matches(
                    tensor_output,
                    expected_tensor_output,
                    case=(name, "no_grad tensor"),
                )
                self.assertFalse(scalar_output.requires_grad)
                self.assertTrue(scalar_output.is_leaf)
                self.assert_tensor_matches(
                    scalar_output,
                    expected_scalar_output,
                    case=(name, "no_grad scalar"),
                )

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        left = torch.tensor([2.0])
        right = torch.tensor([4.0])
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("div", "divide"):
            function = getattr(torch, name)
            calls = (
                (lambda function=function: function(left, right), (left, right), None),
                (lambda function=function: function(left, 4.0), (left, 4.0), None),
                (lambda function=function: function(4.0, left), (4.0, left), None),
                (
                    lambda function=function: function(input=4.0, other=left),
                    (),
                    ("input", "other"),
                ),
                (
                    lambda function=function: function(
                        x1=left, x2=4.0, rounding_mode=None
                    ),
                    (),
                    ("x1", "x2", "rounding_mode"),
                ),
                (
                    lambda function=function: function(
                        left, right, rounding_mode="floor"
                    ),
                    (left, right),
                    ("rounding_mode",),
                ),
                (
                    lambda function=function: function(left, right, out=destination),
                    (left, right),
                    ("out",),
                ),
            )
            for call, expected_args, expected_keywords in calls:
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                dispatch_function, dispatch_types, args, kwargs = mode.calls[0]
                with self.subTest(name=name, keywords=expected_keywords):
                    self.assertIs(dispatch_function, function)
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
                    forwarded = function(input=4.0, other=left)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(forwarded, 4.0 / left, case=(name, "forwarded"))

            for call in (
                lambda function=function: function([], right),
                lambda function=function: function(left, []),
                lambda function=function: function(left, right, rounding_mode=1),
            ):
                mode = RecordingMode()
                with mode:
                    with self.assertRaises(TypeError):
                        call()
                self.assertEqual(mode.calls, [])

            events = []

            class Override:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return marker

            self.assertIs(function(Override(), right), marker)
            dispatch_function, dispatch_types, args, kwargs = events[0]
            self.assertIs(dispatch_function, function)
            self.assertEqual(dispatch_types, (Override,))
            self.assertIsInstance(args[0], Override)
            self.assertIs(args[1], right)
            self.assertIsNone(kwargs)

            events.clear()
            self.assertIs(function(left, right, rounding_mode=Override()), marker)
            _, dispatch_types, args, kwargs = events[0]
            self.assertEqual(dispatch_types, (Override,))
            self.assertIs(args[0], left)
            self.assertIs(args[1], right)
            self.assertEqual(tuple(kwargs), ("rounding_mode",))

            events.clear()
            self.assertIs(function(left, right, out=Override()), marker)
            _, dispatch_types, args, kwargs = events[0]
            self.assertEqual(dispatch_types, (Override,))
            self.assertIs(args[0], left)
            self.assertIs(args[1], right)
            self.assertEqual(tuple(kwargs), ("out",))

            class ScalarOverride(int):
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append((func, types, args, kwargs))
                    return marker

            events.clear()
            self.assertIs(function(ScalarOverride(4), left), marker)
            self.assertEqual(events[0][1], (ScalarOverride,))

            class DecliningOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    return NotImplemented

            with self.assertRaises(TypeError) as raised:
                function(DecliningOverride(), right)
            self.assertIn(
                f"Multiple dispatch failed for 'torch.{name}'",
                str(raised.exception),
            )

    def test_errors_callable_metadata_pickle_exports_and_unsupported_surface(self):
        tensor = torch.tensor([4.0])
        destination = torch.tensor([17.0])

        for name in ("div", "divide"):
            function = getattr(torch, name)
            with self.subTest(name=name, boundary="scalar only"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): scalar-scalar division is not supported; "
                    r"at least one operand must be Tensor$",
                ):
                    function(2, 3)
            with self.subTest(name=name, boundary="concrete out"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): the 'out' argument is not supported$",
                ):
                    function(tensor, tensor, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(name=name, boundary="rounding"):
                for rounding_mode in ("floor", "trunc", "bad"):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        rf"^{name}\(\): non-None rounding_mode is not supported$",
                    ):
                        function(tensor, 2, rounding_mode=rounding_mode)
            with self.subTest(name=name, boundary="rounding type"):
                with self.assertRaises(TypeError):
                    function(tensor, tensor, rounding_mode=1)
            with self.subTest(name=name, boundary="unsupported operands"):
                with self.assertRaises(TypeError):
                    function([], tensor)
                with self.assertRaises(TypeError):
                    function(tensor, [])

            self.assertIs(type(function), types.BuiltinFunctionType)
            self.assertEqual(function.__name__, name)
            self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
            self.assertEqual(function.__module__, "torch")
            self.assertIsNone(function.__text_signature__)
            self.assertIn("rounding_mode=None", function.__doc__)
            self.assertRegex(
                repr(function),
                rf"^<built-in method {name} of type object at 0x[0-9a-f]+>$",
            )
            with self.assertRaises(ValueError):
                inspect.signature(function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

            owner = function.__reduce__()[1][0]
            self.assertEqual(owner.__name__, "_VariableFunctionsClass")
            self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
            self.assertEqual(owner.__module__, "torch_rs._C")
            self.assertIs(owner, torch._C._VariableFunctionsClass)
            self.assertIs(getattr(owner, name), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

            self.assertEqual(torch.__all__.count(name), 1)
            self.assertNotIn("_VariableFunctionsClass", torch.__all__)
            self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
            self.assertFalse(hasattr(torch, f"{name}_"))

        self.assertIsNot(torch.div, torch.divide)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], torch.div)
        self.assertIs(wildcard_namespace["divide"], torch.divide)

        old_div = torch.div
        old_divide = torch.divide
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(reloaded.div, old_div)
        self.assertIs(reloaded.divide, old_divide)


if __name__ == "__main__":
    unittest.main()
