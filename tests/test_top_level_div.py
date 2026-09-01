import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


DIV_DOC = """
div(input, other, *, rounding_mode=None, out=None) -> Tensor

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.

.. math::
    \\text{out}_i = \\frac{\\text{input}_i}{\\text{other}_i}

The current native implementation supports exact CPU ``float32`` tensor/tensor,
tensor/real-scalar, and real-scalar/tensor true division with
``rounding_mode=None``. Scalar-only calls, concrete ``out`` tensors, non-``None``
``rounding_mode`` values, in-place variants, other dtypes/devices, and tensor
subclasses without a ``__torch_function__`` override remain unsupported.

Args:
    input (Tensor): the dividend tensor.
    other (Tensor or Number): the divisor tensor or number.

Keyword args:
    rounding_mode (str, optional): must be ``None``.
    out (Tensor, optional): unsupported except ``None``.
"""
DIVIDE_DOC = """
divide(input, other, *, rounding_mode=None, out=None) -> Tensor

Alias for :func:`torch.div`.
"""


def scalar_numerator(value):
    return torch.scalar_tensor(value)


def division_binding_message(name, summary):
    return (
        f"{name}() received an invalid combination of arguments - got "
        f"({summary}), but expected one of:\n"
        " * (Tensor input, Tensor other, *, str rounding_mode, Tensor out = None)\n"
        " * (Tensor input, Number other, *, str rounding_mode)\n"
        " * (Number input, Tensor other, *, str rounding_mode)\n"
    )


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

    def test_values_broadcast_scalars_empties_and_special_bits(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])

        for name in ("div", "divide"):
            function = getattr(torch, name)
            expected = left / right
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
                actual = call()
                self.assert_tensor_matches(actual, expected, case=(name, case))
                self.assertFalse(actual.is_set_to(left))
                self.assertFalse(actual.is_set_to(right))

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
                        scalar_numerator(scalar) / offset,
                    ),
                    (
                        "keyword tensor/scalar",
                        lambda function=function, scalar=scalar: function(
                            input=offset, other=scalar
                        ),
                        offset / scalar,
                    ),
                    (
                        "keyword scalar/tensor",
                        lambda function=function, scalar=scalar: function(
                            input=scalar, other=offset
                        ),
                        scalar_numerator(scalar) / offset,
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
            function = getattr(torch, name)
            self.assert_tensor_matches(
                function(special, divisors),
                special / divisors,
                case=(name, "signed zero nan infinity"),
            )
            self.assert_tensor_matches(
                function(-0.0, special),
                scalar_numerator(-0.0) / special,
                case=(name, "scalar first signed zero nan infinity"),
            )

    def test_active_autograd_is_rejected_but_no_grad_uses_native_division(self):
        for name in ("div", "divide"):
            function = getattr(torch, name)
            left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            right = torch.tensor([[5.0], [7.0]], requires_grad=True)
            with self.subTest(name=name, case="tensor operands"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(left.transpose(0, 1), right.transpose(0, 1))
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)

            with self.subTest(name=name, case="tensor scalar"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(left, 2.0)

            with self.subTest(name=name, case="scalar tensor"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(2.0, left)

            with self.subTest(name=name, case="no_grad"):
                with torch.no_grad():
                    tensor_output = function(
                        left.transpose(0, 1), right.transpose(0, 1)
                    )
                    scalar_output = function(2.0, left)
                    expected_tensor_output = left.transpose(0, 1) / right.transpose(
                        0, 1
                    )
                    expected_scalar_output = scalar_numerator(2.0) / left
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
                    case=(name, "no_grad scalar first"),
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
                    lambda function=function: function(x1=left, x2=4.0),
                    (),
                    ("x1", "x2"),
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
                function_arg, dispatch_types, args, kwargs = mode.calls[0]
                with self.subTest(name=name, keywords=expected_keywords):
                    self.assertIs(function_arg, function)
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
                    forwarded = function(input=4.0, other=left, out=None)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(forwarded, 4.0 / left, case=(name, "forwarded"))

            for call in (
                lambda function=function: function([], right),
                lambda function=function: function(left, []),
                lambda function=function: function(left, right, rounding_mode=[]),
                lambda function=function: function(left, right, out=[]),
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

            self.assertIs(function(LeftOverride(), RightOverride()), marker)
            self.assertEqual([event[0] for event in events], ["left", "right"])
            for _, function_arg, dispatch_types, args, kwargs in events:
                self.assertIs(function_arg, function)
                self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
                self.assertEqual(len(args), 2)
                self.assertIsNone(kwargs)

            events.clear()
            self.assertIs(function(input=left, other=RightOverride()), marker)
            _, function_arg, dispatch_types, args, kwargs = events[0]
            self.assertIs(function_arg, function)
            self.assertEqual(dispatch_types, (RightOverride,))
            self.assertEqual(args, ())
            self.assertEqual(tuple(kwargs), ("input", "other"))

            events.clear()

            class RoundingOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(("rounding", func, types, args, kwargs))
                    return marker

            self.assertIs(function(left, right, rounding_mode=RoundingOverride()), marker)
            _, function_arg, dispatch_types, args, kwargs = events[0]
            self.assertIs(function_arg, function)
            self.assertEqual(dispatch_types, (RoundingOverride,))
            self.assertEqual(args, (left, right))
            self.assertEqual(tuple(kwargs), ("rounding_mode",))

            events.clear()

            class OutOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    events.append(("out", func, types, args, kwargs))
                    return marker

            self.assertIs(function(left, right, out=OutOverride()), marker)
            _, function_arg, dispatch_types, args, kwargs = events[0]
            self.assertIs(function_arg, function)
            self.assertEqual(dispatch_types, (OutOverride,))
            self.assertEqual(args, (left, right))
            self.assertEqual(tuple(kwargs), ("out",))

            class DecliningOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    return NotImplemented

            with self.assertRaises(TypeError) as raised:
                function(DecliningOverride(), left)
            self.assertEqual(
                str(raised.exception),
                f"Multiple dispatch failed for 'torch.{name}'; all __torch_function__ "
                "handlers returned NotImplemented:\n\n"
                f"  - tensor subclass <class '{DecliningOverride.__module__}."
                f"{DecliningOverride.__qualname__}'>\n\n"
                "For more information, try re-running with TORCH_LOGS=not_implemented",
            )

    def test_errors_metadata_import_reload_and_unsupported_surface(self):
        tensor = torch.tensor([4.0])
        destination = torch.tensor([17.0])

        for function, name, doc in (
            (torch.div, "div", DIV_DOC),
            (torch.divide, "divide", DIVIDE_DOC),
        ):
            invalid_input_message = (
                f"{name}(): argument 'input' (position 1) must be Tensor, not list"
                if name == "div"
                else division_binding_message(name, "list, Tensor")
            )
            invalid_other_message = (
                f"{name}(): argument 'other' (position 2) must be Tensor, not list"
                if name == "div"
                else division_binding_message(name, "Tensor, list")
            )
            invalid_keyword_input_message = (
                f"{name}(): argument 'input' must be Tensor, not NoneType"
                if name == "div"
                else division_binding_message(name, "other=Tensor, input=NoneType, ")
            )
            cases = (
                (
                    lambda function=function: function(),
                    f'{name}() missing 2 required positional argument: "input", "other"',
                ),
                (
                    lambda function=function: function(tensor),
                    f'{name}() missing 1 required positional arguments: "other"',
                ),
                (
                    lambda function=function: function(tensor, tensor, tensor),
                    f"{name}() takes 2 positional arguments but 3 were given",
                ),
                (
                    lambda function=function: function([], tensor),
                    invalid_input_message,
                ),
                (
                    lambda function=function: function(tensor, []),
                    invalid_other_message,
                ),
                (
                    lambda function=function: function(input=None, other=tensor),
                    invalid_keyword_input_message,
                ),
                (
                    lambda function=function: function(tensor, tensor, input=tensor),
                    f"{name}() got multiple values for argument 'input'",
                ),
                (
                    lambda function=function: function(tensor, tensor, x2=tensor),
                    f"{name}() got an unexpected keyword argument 'x2'",
                ),
                (
                    lambda function=function: function(tensor, tensor, extra=True),
                    f"{name}() got an unexpected keyword argument 'extra'",
                ),
            )
            for call, message in cases:
                with self.subTest(function=name, message=message):
                    with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                        call()

            with self.assertRaisesRegex(
                RuntimeError, rf"^{name}\(\): the 'out' argument is not supported$"
            ):
                function(tensor, tensor, out=destination)
            self.assertEqual(destination.tolist(), [17.0])
            self.assert_tensor_matches(
                function(tensor, tensor, out=None),
                tensor / tensor,
                case=(name, "out none"),
            )
            for rounding_mode in ("floor", "trunc", "bad"):
                with self.subTest(function=name, rounding_mode=rounding_mode):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        rf"^{name}\(\): non-None rounding_mode is not supported$",
                    ):
                        function(tensor, tensor, rounding_mode=rounding_mode)
            with self.assertRaises(TypeError):
                function(tensor, tensor, rounding_mode=1)
            with self.assertRaises(TypeError):
                function(tensor, tensor, dtype=torch.float32)
            with self.assertRaises(TypeError):
                function(tensor, tensor, device=torch.device("cpu"))
            with self.assertRaisesRegex(
                TypeError,
                rf"^{name}\(\): scalar-scalar division is not supported; at least one operand must be Tensor$",
            ):
                function(2, 3)
            with self.assertRaisesRegex(TypeError, "^an integer is required$"):
                function(tensor, np.uint64(2**63))

            self.assertIs(function, getattr(torch._C, name))
            self.assertIs(type(function), types.BuiltinFunctionType)
            self.assertEqual(function.__name__, name)
            self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
            self.assertEqual(function.__module__, "torch")
            self.assertIsNone(function.__text_signature__)
            self.assertEqual(function.__doc__, doc)
            self.assertRegex(
                repr(function), r"^<built-in method .* of type object at 0x[0-9a-f]+>$"
            )
            with self.assertRaises(ValueError):
                inspect.signature(function)

            owner = function.__reduce__()[1][0]
            self.assertEqual(owner.__name__, "_VariableFunctionsClass")
            self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
            self.assertEqual(owner.__module__, "torch_rs._C")
            self.assertIs(owner, torch._C._VariableFunctionsClass)
            self.assertIs(getattr(owner, name), function)
            for mutation in (
                lambda name=name: setattr(owner, name, None),
                lambda name=name: delattr(owner, name),
            ):
                with self.assertRaises(TypeError):
                    mutation()
                self.assertIs(getattr(owner, name), function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

        self.assertIsNot(torch.div, torch.divide)
        self.assertEqual(torch.__all__.count("div"), 1)
        self.assertEqual(torch.__all__.count("divide"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], torch.div)
        self.assertIs(wildcard_namespace["divide"], torch.divide)
        self.assertFalse(hasattr(torch, "div_"))
        self.assertFalse(hasattr(torch, "divide_"))
        self.assertFalse(hasattr(torch.Tensor, "div_"))
        self.assertFalse(hasattr(torch.Tensor, "divide_"))
        self.assertFalse(hasattr(torch, "float64"))

        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})
        with self.assertRaisesRegex(
            TypeError,
            r"^tensor\(\): argument 'dtype' must be torch.dtype, not object$",
        ):
            torch.tensor([1.0], dtype=object())
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")

        function = torch.div
        alias = torch.divide
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.div, function)
        self.assertIs(torch.divide, alias)


if __name__ == "__main__":
    unittest.main()
