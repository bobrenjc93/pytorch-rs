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

.. note::
    By default, this performs a "true" division like Python 3.
    See the :attr:`rounding_mode` argument for floor division.

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.
Always promotes integer types to the default scalar type.

Args:
    input (Tensor): the dividend
    other (Tensor or Number): the divisor

Keyword args:
    rounding_mode (str, optional): Type of rounding applied to the result:

        * None - default behavior. Performs no rounding and, if both :attr:`input` and
          :attr:`other` are integer types, promotes the inputs to the default scalar type.
          Equivalent to true division in Python (the ``/`` operator) and NumPy's ``np.true_divide``.
        * ``"trunc"`` - rounds the results of the division towards zero.
          Equivalent to C-style integer division.
        * ``"floor"`` - rounds the results of the division down.
          Equivalent to floor division in Python (the ``//`` operator) and NumPy's ``np.floor_divide``.

    out (Tensor, optional): the output tensor.

Examples::

    >>> x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])
    >>> torch.div(x, 0.5)
    tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])

    >>> a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
    ...                   [ 0.1815, -1.0111,  0.9805, -1.5923],
    ...                   [ 0.1062,  1.4581,  0.7759, -1.2344],
    ...                   [-0.1830, -0.0313,  1.1908, -1.4757]])
    >>> b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
    >>> torch.div(a, b)
    tensor([[-0.4620, -6.6051,  0.5676,  1.2639],
            [ 0.2260, -3.4509, -1.2086,  6.8990],
            [ 0.1322,  4.9764, -0.9564,  5.3484],
            [-0.2278, -0.1068, -1.4678,  6.3938]])

    >>> torch.div(a, b, rounding_mode='trunc')
    tensor([[-0., -6.,  0.,  1.],
            [ 0., -3., -1.,  6.],
            [ 0.,  4., -0.,  5.],
            [-0., -0., -1.,  6.]])

    >>> torch.div(a, b, rounding_mode='floor')
    tensor([[-1., -7.,  0.,  1.],
            [ 0., -4., -2.,  6.],
            [ 0.,  4., -1.,  5.],
            [-1., -1., -2.,  6.]])

"""

DIVIDE_DOC = """
divide(input, other, *, rounding_mode=None, out=None) -> Tensor

Alias for :func:`torch.div`.
"""


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

    def test_values_broadcast_scalars_empties_offsets_and_special_bits(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected = left / right
        for name in ("div", "divide"):
            calls = (
                ("positional tensors", lambda name=name: getattr(torch, name)(left, right)),
                (
                    "canonical keywords",
                    lambda name=name: getattr(torch, name)(input=left, other=right),
                ),
                ("x aliases", lambda name=name: getattr(torch, name)(x=left, x2=right)),
                ("x1 aliases", lambda name=name: getattr(torch, name)(x1=left, x2=right)),
                ("a alias", lambda name=name: getattr(torch, name)(a=left, other=right)),
                (
                    "explicit true division",
                    lambda name=name: getattr(torch, name)(
                        left, right, rounding_mode=None
                    ),
                ),
                ("explicit out none", lambda name=name: getattr(torch, name)(left, right, out=None)),
            )
            for case, call in calls:
                self.assert_tensor_matches(call(), expected, case=(name, case))

        offset_noncontiguous = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        ).transpose(0, 1)[1]
        self.assertGreater(offset_noncontiguous.storage_offset(), 0)
        self.assertFalse(offset_noncontiguous.is_contiguous())
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for name in ("div", "divide"):
                self.assert_tensor_matches(
                    getattr(torch, name)(offset_noncontiguous, scalar),
                    offset_noncontiguous / scalar,
                    case=(name, "tensor/scalar", type(scalar).__name__, scalar),
                )
                self.assert_tensor_matches(
                    getattr(torch, name)(input=scalar, other=offset_noncontiguous),
                    scalar / offset_noncontiguous,
                    case=(name, "scalar/tensor", type(scalar).__name__, scalar),
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
            with self.subTest(name=name, case="tensor operands"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(left.transpose(0, 1), right.transpose(0, 1))
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)

            with self.subTest(name=name, case="tensor/scalar"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(left, 2.0)

            with self.subTest(name=name, case="scalar/tensor"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): autograd recording is not supported$",
                ):
                    function(2.0, left)

            with self.subTest(name=name, case="no_grad"):
                with torch.no_grad():
                    tensor_output = function(left.transpose(0, 1), right.transpose(0, 1))
                    scalar_output = function(np.float32(2.0), left)
                    expected_tensor = left.transpose(0, 1) / right.transpose(0, 1)
                    expected_scalar = np.float32(2.0) / left
                self.assert_tensor_matches(
                    tensor_output, expected_tensor, case=(name, "no_grad tensor")
                )
                self.assert_tensor_matches(
                    scalar_output, expected_scalar, case=(name, "no_grad scalar")
                )

    def test_torch_function_modes_and_overrides_observe_original_calls(self):
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
                (lambda function=function: function(2, 3), (2, 3), None),
                (
                    lambda function=function: function(input=left, other=right, out=destination),
                    (),
                    ("input", "other", "out"),
                ),
                (
                    lambda function=function: function(
                        left, right, rounding_mode="floor"
                    ),
                    (left, right),
                    ("rounding_mode",),
                ),
            )
            for call, expected_args, expected_keywords in calls:
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                called_function, dispatch_types, args, kwargs = mode.calls[0]
                with self.subTest(name=name, keywords=expected_keywords):
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
                    order.append(self.label)
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = function(input=np.float32(8.0), other=left)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(
                forwarded, np.float32(8.0) / left, case=(name, "forwarded")
            )

            for call in (
                lambda function=function: function([], right),
                lambda function=function: function(left, []),
                lambda function=function: function(left, right, rounding_mode=1),
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
            for _, called_function, dispatch_types, args, kwargs in events:
                self.assertIs(called_function, function)
                self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
                self.assertEqual(len(args), 2)
                self.assertIsNone(kwargs)

            events.clear()
            self.assertIs(function(input=left, other=RightOverride(), out=destination), marker)
            _, called_function, dispatch_types, args, kwargs = events[0]
            self.assertIs(called_function, function)
            self.assertEqual(dispatch_types, (RightOverride,))
            self.assertEqual(args, ())
            self.assertEqual(tuple(kwargs), ("input", "other", "out"))

            events.clear()
            self.assertIs(function(left, right, rounding_mode=RightOverride()), marker)
            _, called_function, dispatch_types, args, kwargs = events[0]
            self.assertIs(called_function, function)
            self.assertEqual(dispatch_types, (RightOverride,))
            self.assertEqual(args, (left, right))
            self.assertEqual(tuple(kwargs), ("rounding_mode",))

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

    def test_errors_metadata_pickling_exports_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])

        for function, name, doc in (
            (torch.div, "div", DIV_DOC),
            (torch.divide, "divide", DIVIDE_DOC),
        ):
            with self.subTest(name=name, boundary="rounding"):
                for rounding_mode in ("floor", "trunc", "bad"):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        rf"^{name}\(\): non-None rounding_mode is not supported$",
                    ):
                        function(tensor, other, rounding_mode=rounding_mode)
            with self.subTest(name=name, boundary="out"):
                destination = torch.tensor([17.0])
                with self.assertRaisesRegex(
                    RuntimeError, rf"^{name}\(\): the 'out' argument is not supported$"
                ):
                    function(tensor, other, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(name=name, boundary="scalar-only"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): scalar-scalar division is not supported; "
                    r"at least one operand must be Tensor$",
                ):
                    function(2, 3)
            with self.subTest(name=name, boundary="wide integer"):
                with self.assertRaisesRegex(TypeError, "^an integer is required$"):
                    function(tensor, np.uint64(2**63))
            with self.subTest(name=name, boundary="overflow integer"):
                with self.assertRaisesRegex(OverflowError, "^int too big to convert$"):
                    function(tensor, 2**64)
                with self.assertRaisesRegex(
                    OverflowError, "^can't convert negative int to unsigned$"
                ):
                    function(-(2**63) - 1, tensor)
            with self.subTest(name=name, boundary="unsupported keywords"):
                for call in (
                    lambda function=function: function(tensor, other, dtype=torch.float32),
                    lambda function=function: function(tensor, other, device=torch.device("cpu")),
                ):
                    with self.assertRaises(TypeError):
                        call()
            with self.subTest(name=name, boundary="unsupported operands"):
                with self.assertRaises(TypeError):
                    function([], tensor)
                with self.assertRaises(TypeError):
                    function(tensor, [])
            with self.subTest(name=name, boundary="duplicate aliases"):
                for call in (
                    lambda function=function: function(input=tensor, x=other, other=other),
                    lambda function=function: function(input=tensor, other=other, x2=tensor),
                ):
                    with self.assertRaises(TypeError):
                        call()

            self.assertIs(type(function), types.BuiltinFunctionType)
            self.assertEqual(function.__name__, name)
            self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
            self.assertEqual(function.__module__, "torch")
            self.assertIsNone(function.__text_signature__)
            self.assertEqual(function.__doc__, doc)
            self.assertRegex(
                repr(function),
                rf"^<built-in method {name} of type object at 0x[0-9a-f]+>$",
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

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            self.assertEqual(torch.__all__.count(name), 1)

        self.assertIsNot(torch.div, torch.divide)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], torch.div)
        self.assertIs(wildcard_namespace["divide"], torch.divide)
        self.assertFalse(hasattr(torch, "div_"))
        self.assertFalse(hasattr(torch, "divide_"))
        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.div, wildcard_namespace["div"])
        self.assertIs(torch.divide, wildcard_namespace["divide"])


if __name__ == "__main__":
    unittest.main()
