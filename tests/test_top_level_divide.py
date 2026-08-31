import copy
import importlib
import inspect
import math
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


DIV_DOC = r"""
div(input, other, *, rounding_mode=None, out=None) -> Tensor

Divides each element of the input ``input`` by the corresponding element of
:attr:`other`.


.. math::
    \text{out}_i = \frac{\text{input}_i}{\text{other}_i}


Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
Always promotes integer types to the default scalar type.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to divide by.

Keyword args:
    rounding_mode (str, optional): Type of rounding applied to the result:

        * ``None`` - default behavior. Performs no rounding and, if both ``input`` and
          ``other`` are integer types, promotes the inputs to the default scalar type.
          Equivalent to true division in Python (the ``/`` operator) and NumPy's
          ``np.true_divide``.
        * ``"trunc"`` - rounds the results of the division towards zero.
          Equivalent to C-style integer division.
        * ``"floor"`` - rounds the results of the division down.
          Equivalent to floor division in Python (the ``//`` operator).

    out (Tensor, optional): the output tensor.

Example::

    >>> x = torch.randn(5)
    >>> x
    tensor([ 0.3810,  1.2774, -0.2977, -0.3719,  0.4637])
    >>> torch.div(x, 0.5)
    tensor([ 0.7620,  2.5548, -0.5954, -0.7439,  0.9275])
"""

DIVIDE_DOC = """
divide(input, other, *, rounding_mode=None, out=None)

Alias for :func:`torch.div`.
"""


class TopLevelDivideTests(unittest.TestCase):
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

    def test_true_division_values_broadcast_layouts_and_scalars(self):
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        left = base.transpose(0, 2)[1]
        right = torch.tensor([[1.0], [2.0], [4.0]])
        tensor_expected = left / right

        for name in ("div", "divide"):
            function = getattr(torch, name)
            for case, call in (
                ("positional tensors", lambda function=function: function(left, right)),
                (
                    "canonical keywords",
                    lambda function=function: function(input=left, other=right),
                ),
                (
                    "legacy aliases",
                    lambda function=function: function(x1=left, x2=right),
                ),
                (
                    "explicit true rounding",
                    lambda function=function: function(
                        left, right, rounding_mode=None
                    ),
                ),
                (
                    "explicit out none",
                    lambda function=function: function(left, right, out=None),
                ),
            ):
                output = call()
                self.assert_tensor_matches(
                    output, tensor_expected, case=(name, case)
                )
                self.assertNotEqual(output.data_ptr(), left.data_ptr())
                self.assertNotEqual(output.data_ptr(), right.data_ptr())

            offset = left[1]
            for scalar in (True, -2, 2.5, np.bool_(True), np.int64(4), np.float32(-0.0)):
                for order, call, expected in (
                    (
                        "tensor/scalar",
                        lambda scalar=scalar, function=function: function(offset, scalar),
                        offset / scalar,
                    ),
                    (
                        "scalar/tensor",
                        lambda scalar=scalar, function=function: function(scalar, offset),
                        scalar / offset,
                    ),
                    (
                        "keyword tensor/scalar",
                        lambda scalar=scalar, function=function: function(
                            input=offset, other=scalar
                        ),
                        offset / scalar,
                    ),
                    (
                        "keyword scalar/tensor",
                        lambda scalar=scalar, function=function: function(
                            input=scalar, other=offset
                        ),
                        scalar / offset,
                    ),
                ):
                    self.assert_tensor_matches(
                        call(), expected, case=(name, order, type(scalar).__name__)
                    )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.div(empty, broadcast),
            empty / broadcast,
            case="strided broadcast empty",
        )

    def test_signed_zero_nan_and_infinity_match_operator_bits(self):
        left_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        right_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x3F80_0000,
            ),
            dtype=np.uint32,
        )
        left = torch.tensor(memoryview(left_bits.view(np.float32)))
        right = torch.tensor(memoryview(right_bits.view(np.float32)))

        for function in (torch.div, torch.divide):
            self.assert_tensor_matches(
                function(left, right), left / right, case=function.__name__
            )
            self.assert_tensor_matches(
                function(-0.0, right), -0.0 / right, case=(function.__name__, "left scalar")
            )
            self.assert_tensor_matches(
                function(left, -0.0), left / -0.0, case=(function.__name__, "right scalar")
            )

        self.assert_tensor_matches(
            torch.div(torch.tensor([math.nan, math.inf, -math.inf]), 2.0),
            torch.tensor([math.nan, math.inf, -math.inf]) / 2.0,
            case="named nonfinite",
        )

    def test_active_autograd_operands_are_rejected_and_no_grad_is_allowed(self):
        leaf = torch.tensor([[1.0, -0.0], [math.inf, math.nan]], requires_grad=True)
        detached_other = torch.tensor([[2.0], [-2.0]])

        for function in (torch.div, torch.divide):
            for case, call in (
                ("tensor left", lambda function=function: function(leaf, detached_other)),
                ("tensor right", lambda function=function: function(2.0, leaf)),
                (
                    "both tensors",
                    lambda function=function: function(leaf, leaf.detach()),
                ),
            ):
                with self.subTest(function=function.__name__, case=case):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        rf"^{function.__name__}\(\): autograd recording is not supported$",
                    ):
                        call()

            with torch.no_grad():
                actual = function(leaf.transpose(0, 1), detached_other.transpose(0, 1))
                expected = leaf.transpose(0, 1) / detached_other.transpose(0, 1)
            self.assertFalse(actual.requires_grad)
            self.assert_tensor_matches(
                actual, expected, case=(function.__name__, "no_grad")
            )

    def test_modes_and_overrides_use_public_dispatch(self):
        left = torch.tensor([4.0])
        right = torch.tensor([2.0])
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for function in (torch.div, torch.divide):
            calls = (
                ("tensor/tensor", lambda: function(left, right), (left, right), None),
                ("tensor/scalar", lambda: function(left, 4.0), (left, 4.0), None),
                ("scalar/tensor", lambda: function(4.0, left), (4.0, left), None),
                (
                    "keywords",
                    lambda: function(input=4.0, other=left, rounding_mode=None),
                    (),
                    ("input", "other", "rounding_mode"),
                ),
                (
                    "unsupported native options",
                    lambda: function(left, right, rounding_mode="floor", out=destination),
                    (left, right),
                    ("rounding_mode", "out"),
                ),
            )
            for case, call, expected_args, expected_keywords in calls:
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                self.assertEqual(len(mode.calls), 1)
                called_function, dispatch_types, args, kwargs = mode.calls[0]
                with self.subTest(function=function.__name__, case=case):
                    self.assertIs(called_function, function)
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
                    order.append((self.label, func))
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = function(input=4.0, other=left, rounding_mode=None)
            self.assertEqual(order, [("upper", function), ("lower", function)])
            self.assert_tensor_matches(
                forwarded, 4.0 / left, case=(function.__name__, "forwarded mode")
            )

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

        self.assertIs(torch.div(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.div)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        out_events = []

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_events.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.divide(left, right, out=OutOverride()), marker)
        function, dispatch_types, args, kwargs = out_events[0]
        self.assertIs(function, torch.divide)
        self.assertEqual(dispatch_types, (OutOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("out",))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError, "Multiple dispatch failed for 'torch.div'"
        ):
            torch.div(DecliningOverride(), left)

    def test_unsupported_arguments_and_surface_are_rejected(self):
        tensor = torch.tensor([1.0, 2.0])
        destination = torch.tensor([17.0, 19.0])

        for function in (torch.div, torch.divide):
            cases = (
                (
                    lambda function=function: function(),
                    TypeError,
                    f'{function.__name__}() missing 2 required positional argument: "input", "other"',
                ),
                (
                    lambda function=function: function(tensor),
                    TypeError,
                    f'{function.__name__}() missing 1 required positional arguments: "other"',
                ),
                (
                    lambda function=function: function(tensor, tensor, tensor),
                    TypeError,
                    f"{function.__name__}() takes 2 positional arguments but 3 were given",
                ),
                (
                    lambda function=function: function([], tensor),
                    TypeError,
                    f"{function.__name__}(): argument 'input' (position 1) must be Tensor, not list",
                ),
                (
                    lambda function=function: function(tensor, []),
                    TypeError,
                    f"{function.__name__}(): argument 'other' (position 2) must be Tensor, not list",
                ),
                (
                    lambda function=function: function(2.0, 3.0),
                    TypeError,
                    f"{function.__name__}(): scalar-scalar division is not supported; at least one operand must be Tensor",
                ),
                (
                    lambda function=function: function(tensor, tensor, input=tensor),
                    TypeError,
                    f"{function.__name__}() got multiple values for argument 'input'",
                ),
                (
                    lambda function=function: function(tensor, tensor, dtype=torch.float32),
                    TypeError,
                    f"{function.__name__}() got an unexpected keyword argument 'dtype'",
                ),
                (
                    lambda function=function: function(tensor, tensor, device="cpu"),
                    TypeError,
                    f"{function.__name__}() got an unexpected keyword argument 'device'",
                ),
                (
                    lambda function=function: function(tensor, tensor, x2=tensor),
                    TypeError,
                    f"{function.__name__}() got an unexpected keyword argument 'x2'",
                ),
                (
                    lambda function=function: function(tensor, np.uint64(2**63)),
                    TypeError,
                    "an integer is required",
                ),
                (
                    lambda function=function: function(2**64, tensor),
                    OverflowError,
                    "int too big to convert",
                ),
                (
                    lambda function=function: function(-(2**63) - 1, tensor),
                    OverflowError,
                    "can't convert negative int to unsigned",
                ),
            )
            for call, error_type, message in cases:
                with self.subTest(function=function.__name__, message=message):
                    with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                        call()

            for rounding_mode in ("trunc", "floor", 0):
                with self.subTest(
                    function=function.__name__, rounding_mode=rounding_mode
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        rf"^{function.__name__}\(\): non-None rounding_mode is not supported$",
                    ):
                        function(tensor, tensor, rounding_mode=rounding_mode)

            with self.subTest(function=function.__name__, out=True):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{function.__name__}\(\): the 'out' argument is not supported$",
                ):
                    function(tensor, tensor, out=destination)
                self.assertEqual(destination.tolist(), [17.0, 19.0])

        self.assertFalse(hasattr(torch.Tensor, "div"))
        self.assertFalse(hasattr(torch.Tensor, "divide"))
        self.assertFalse(hasattr(torch.Tensor, "div_"))
        self.assertFalse(hasattr(torch.Tensor, "divide_"))
        self.assertFalse(hasattr(torch, "div_"))
        self.assertFalse(hasattr(torch, "divide_"))
        with self.assertRaises(TypeError):
            class TensorSubclass(torch.Tensor):
                pass

    def test_callable_metadata_imports_wildcard_copy_pickle_and_reload(self):
        from torch_rs import div as imported_div
        from torch_rs import divide as imported_divide

        old_div = torch.div
        old_divide = torch.divide
        old_all = torch.__all__
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.div, old_div)
        self.assertIs(torch.divide, old_divide)
        self.assertIsNot(torch.__all__, old_all)
        self.assertIs(imported_div, torch.div)
        self.assertIs(imported_divide, torch.divide)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        for name, expected_doc in (("div", DIV_DOC), ("divide", DIVIDE_DOC)):
            function = getattr(torch, name)
            with self.subTest(name=name):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__name__, name)
                self.assertEqual(
                    function.__qualname__, f"_VariableFunctionsClass.{name}"
                )
                self.assertEqual(function.__module__, "torch")
                self.assertIsNone(function.__text_signature__)
                self.assertEqual(function.__doc__, expected_doc)
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
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertNotIn("_VariableFunctionsClass", torch.__all__)
                self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
                self.assertIs(wildcard_namespace[name], function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

        self.assertIsNot(torch.divide, torch.div)
        self.assertIsNot(
            torch._C._VariableFunctionsClass.divide,
            torch._C._VariableFunctionsClass.div,
        )


if __name__ == "__main__":
    unittest.main()
