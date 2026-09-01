import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


DIV_DOC = (
    "\ndiv(input, other, *, rounding_mode=None, out=None) -> Tensor\n\n"
    "Divides each element of the input ``input`` by the corresponding element of\n"
    ":attr:`other`.\n\n"
    ".. math::\n"
    "    \\text{out}_i = \\frac{\\text{input}_i}{\\text{other}_i}\n\n"
    ".. note::\n"
    '    By default, this performs a "true" division like Python 3.\n'
    "    See the :attr:`rounding_mode` argument for floor division.\n\n"
    "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.\n"
    "Always promotes integer types to the default scalar type.\n\n"
    "Args:\n"
    "    input (Tensor): the dividend\n"
    "    other (Tensor or Number): the divisor\n\n"
    "Keyword args:\n"
    "    rounding_mode (str, optional): Type of rounding applied to the result:\n\n"
    "        * None - default behavior. Performs no rounding and, if both :attr:`input` and\n"
    "          :attr:`other` are integer types, promotes the inputs to the default scalar type.\n"
    "          Equivalent to true division in Python (the ``/`` operator) and NumPy's ``np.true_divide``.\n"
    '        * ``"trunc"`` - rounds the results of the division towards zero.\n'
    "          Equivalent to C-style integer division.\n"
    '        * ``"floor"`` - rounds the results of the division down.\n'
    "          Equivalent to floor division in Python (the ``//`` operator) and NumPy's ``np.floor_divide``.\n\n"
    "    out (Tensor, optional): the output tensor.\n\n"
    "Examples::\n\n"
    "    >>> x = torch.tensor([ 0.3810,  1.2774, -0.2972, -0.3719,  0.4637])\n"
    "    >>> torch.div(x, 0.5)\n"
    "    tensor([ 0.7620,  2.5548, -0.5944, -0.7438,  0.9274])\n\n"
    "    >>> a = torch.tensor([[-0.3711, -1.9353, -0.4605, -0.2917],\n"
    "    ...                   [ 0.1815, -1.0111,  0.9805, -1.5923],\n"
    "    ...                   [ 0.1062,  1.4581,  0.7759, -1.2344],\n"
    "    ...                   [-0.1830, -0.0313,  1.1908, -1.4757]])\n"
    "    >>> b = torch.tensor([ 0.8032,  0.2930, -0.8113, -0.2308])\n"
    "    >>> torch.div(a, b)\n"
    "    tensor([[-0.4620, -6.6051,  0.5676,  1.2639],\n"
    "            [ 0.2260, -3.4509, -1.2086,  6.8990],\n"
    "            [ 0.1322,  4.9764, -0.9564,  5.3484],\n"
    "            [-0.2278, -0.1068, -1.4678,  6.3938]])\n\n"
    "    >>> torch.div(a, b, rounding_mode='trunc')\n"
    "    tensor([[-0., -6.,  0.,  1.],\n"
    "            [ 0., -3., -1.,  6.],\n"
    "            [ 0.,  4., -0.,  5.],\n"
    "            [-0., -0., -1.,  6.]])\n\n"
    "    >>> torch.div(a, b, rounding_mode='floor')\n"
    "    tensor([[-1., -7.,  0.,  1.],\n"
    "            [ 0., -4., -2.,  6.],\n"
    "            [ 0.,  4., -1.,  5.],\n"
    "            [-1., -1., -2.,  6.]])\n\n"
)
DIVIDE_DOC = (
    "\ndivide(input, other, *, rounding_mode=None, out=None) -> Tensor\n\n"
    "Alias for :func:`torch.div`.\n"
)


def division_binding_message(name, summary, mismatch=None):
    overloads = (
        f"{name}() received an invalid combination of arguments - got "
        f"({summary}), but expected one of:\n"
        " * (Tensor input, Tensor other, *, Tensor out = None)\n"
        " * (Tensor input, Tensor other, *, str rounding_mode, Tensor out = None)"
    )
    if name == "divide":
        overloads += "\n * (Tensor input, Number other)"
        if mismatch is not None:
            overloads += f"\n      didn't match because {mismatch}"
    return overloads + "\n * (Tensor input, Number other, *, str rounding_mode)\n"


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
        for function in (torch.div, torch.divide):
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
                self.assert_tensor_matches(call(), expected, case=(function.__name__, case))

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for function in (torch.div, torch.divide):
                for order, call, expected in (
                    (
                        "tensor/scalar",
                        lambda function=function, scalar=scalar: function(offset, scalar),
                        offset / scalar,
                    ),
                    (
                        "scalar/tensor",
                        lambda function=function, scalar=scalar: function(scalar, offset),
                        scalar / offset,
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
                        scalar / offset,
                    ),
                ):
                    self.assert_tensor_matches(
                        call(),
                        expected,
                        case=(function.__name__, order, type(scalar).__name__, scalar),
                    )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        for function in (torch.div, torch.divide):
            self.assert_tensor_matches(
                function(empty, broadcast),
                empty / broadcast,
                case=(function.__name__, "strided broadcast empty"),
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
        for function in (torch.div, torch.divide):
            result = function(special, divisors)
            self.assert_tensor_matches(
                result,
                special / divisors,
                case=(function.__name__, "signed zero nan infinity"),
            )
            self.assertFalse(result.is_set_to(special))
            self.assertFalse(result.is_set_to(divisors))
            if result.numel():
                self.assertNotEqual(result.data_ptr(), special.data_ptr())
                self.assertNotEqual(result.data_ptr(), divisors.data_ptr())

    def test_active_autograd_is_rejected_but_no_grad_uses_native_division(self):
        for function in (torch.div, torch.divide):
            left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            right = torch.tensor([[5.0], [7.0]], requires_grad=True)
            with self.subTest(function=function.__name__, case="tensor operands"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{function.__name__}\(\): autograd recording is not supported$",
                ):
                    function(left.transpose(0, 1), right.transpose(0, 1))
                self.assertIsNone(left.grad)
                self.assertIsNone(right.grad)

            with self.subTest(function=function.__name__, case="tensor scalar"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{function.__name__}\(\): autograd recording is not supported$",
                ):
                    function(left, 2.0)

            with self.subTest(function=function.__name__, case="scalar tensor"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{function.__name__}\(\): autograd recording is not supported$",
                ):
                    function(2.0, left)

            with self.subTest(function=function.__name__, case="no_grad"):
                with torch.no_grad():
                    tensor_output = function(
                        left.transpose(0, 1), right.transpose(0, 1)
                    )
                    scalar_output = function(2.0, left)
                    expected_tensor_output = left.transpose(0, 1) / right.transpose(
                        0, 1
                    )
                    expected_scalar_output = 2.0 / left
                self.assertFalse(tensor_output.requires_grad)
                self.assertTrue(tensor_output.is_leaf)
                self.assert_tensor_matches(
                    tensor_output,
                    expected_tensor_output,
                    case=(function.__name__, "no_grad tensor"),
                )
                self.assertFalse(scalar_output.requires_grad)
                self.assertTrue(scalar_output.is_leaf)
                self.assert_tensor_matches(
                    scalar_output,
                    expected_scalar_output,
                    case=(function.__name__, "no_grad scalar first"),
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

        calls = (
            (lambda: torch.div(left, right), torch.div, (left, right), None),
            (lambda: torch.div(left, 4.0), torch.div, (left, 4.0), None),
            (lambda: torch.div(4.0, left), torch.div, (4.0, left), None),
            (
                lambda: torch.div(input=4.0, other=left, rounding_mode="floor"),
                torch.div,
                (),
                ("input", "other", "rounding_mode"),
            ),
            (
                lambda: torch.divide(x1=left, x2=4.0, out=destination),
                torch.divide,
                (),
                ("x1", "x2", "out"),
            ),
        )
        for call, expected_function, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, expected_function)
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
                actual = torch.div(input=4.0, other=left, rounding_mode=None, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, 4.0 / left, case="forwarded modes")

        for call in (
            lambda: torch.div([], right),
            lambda: torch.div(left, []),
            lambda: torch.div(left, right, rounding_mode=1),
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

        self.assertIs(torch.div(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.div)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.div(input=left, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.div)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        events.clear()

        class OptionOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append((func, types, args, kwargs))
                return marker

        self.assertIs(
            torch.divide(left, right, rounding_mode=OptionOverride()), marker
        )
        function, dispatch_types, args, kwargs = events[-1]
        self.assertIs(function, torch.divide)
        self.assertEqual(dispatch_types, (OptionOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("rounding_mode",))

        self.assertIs(torch.divide(left, right, out=OptionOverride()), marker)
        function, dispatch_types, args, kwargs = events[-1]
        self.assertIs(function, torch.divide)
        self.assertEqual(dispatch_types, (OptionOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("out",))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.div(DecliningOverride(), left)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.div'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_import_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        for function in (torch.div, torch.divide):
            name = function.__name__
            cases = (
                (lambda function=function: function(), division_binding_message(name, "")),
                (
                    lambda function=function: function(tensor),
                    division_binding_message(name, "Tensor"),
                ),
                (
                    lambda function=function: function(2),
                    division_binding_message(name, "int"),
                ),
                (
                    lambda function=function: function(tensor, tensor, tensor),
                    division_binding_message(name, "Tensor, Tensor, Tensor"),
                ),
                (
                    lambda function=function: function([], tensor),
                    division_binding_message(
                        name,
                        "list, Tensor",
                        "some of the arguments have invalid types: "
                        "(!list of []!, !Tensor!)",
                    )
                    if name == "divide"
                    else f"{name}(): argument 'input' (position 1) must be Tensor, not list",
                ),
                (
                    lambda function=function: function(tensor, []),
                    division_binding_message(
                        name,
                        "Tensor, list",
                        "some of the arguments have invalid types: "
                        "(Tensor, !list of []!)",
                    )
                    if name == "divide"
                    else f"{name}(): argument 'other' (position 2) must be Tensor, not list",
                ),
                (
                    lambda function=function: function(input=None, other=tensor),
                    division_binding_message(
                        name,
                        "other=Tensor, input=NoneType, ",
                        "some of the arguments have invalid types: "
                        "(!input=NoneType!, !other=Tensor!, )",
                    )
                    if name == "divide"
                    else f"{name}(): argument 'input' must be Tensor, not NoneType",
                ),
                (
                    lambda function=function: function(tensor, tensor, input=tensor),
                    division_binding_message(name, "Tensor, Tensor, input=Tensor"),
                ),
                (
                    lambda function=function: function(tensor, tensor, x2=tensor),
                    division_binding_message(name, "Tensor, Tensor, x2=Tensor"),
                ),
                (
                    lambda function=function: function(tensor, tensor, extra=True),
                    division_binding_message(name, "Tensor, Tensor, extra=bool"),
                ),
                (
                    lambda function=function: function(tensor, tensor, dtype=torch.float32),
                    division_binding_message(name, "Tensor, Tensor, dtype=torch.dtype"),
                ),
                (
                    lambda function=function: function(tensor, tensor, device=torch.device("cpu")),
                    division_binding_message(name, "Tensor, Tensor, device=torch.device"),
                ),
                (
                    lambda function=function: function(tensor, tensor, rounding_mode=1),
                    division_binding_message(name, "Tensor, Tensor, rounding_mode=int"),
                ),
                (
                    lambda function=function: function(tensor, np.uint64(2**63)),
                    "an integer is required",
                ),
                (
                    lambda function=function: function(tensor, 2**64),
                    "int too big to convert",
                ),
                (
                    lambda function=function: function(-(2**63) - 1, tensor),
                    "can't convert negative int to unsigned",
                ),
                (
                    lambda function=function: function(2, 3),
                    f"{name}(): scalar-scalar division is not supported; at least "
                    "one operand must be Tensor",
                ),
            )
            for call, message in cases:
                with self.subTest(function=name, message=message):
                    with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                        call()

            destination = torch.tensor([17.0])
            with self.assertRaisesRegex(
                RuntimeError, rf"^{name}\(\): the 'out' argument is not supported$"
            ):
                function(tensor, tensor, out=destination)
            self.assertEqual(destination.tolist(), [17.0])

            with self.assertRaisesRegex(
                NotImplementedError,
                rf"^{name}\(\): non-None rounding_mode is not supported$",
            ):
                function(tensor, tensor, rounding_mode="floor")
            with self.assertRaisesRegex(
                NotImplementedError,
                rf"^{name}\(\): non-None rounding_mode is not supported$",
            ):
                function(tensor, 2, rounding_mode="trunc")

        function = torch.div
        alias = torch.divide
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertIs(type(alias), types.BuiltinFunctionType)
        self.assertIsNot(alias, function)
        self.assertEqual(function.__name__, "div")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.div")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, DIV_DOC)
        self.assertEqual(alias.__name__, "divide")
        self.assertEqual(alias.__qualname__, "_VariableFunctionsClass.divide")
        self.assertEqual(alias.__module__, "torch")
        self.assertIsNone(alias.__text_signature__)
        self.assertEqual(alias.__doc__, DIVIDE_DOC)
        for callable_object in (function, alias):
            self.assertRegex(
                repr(callable_object),
                rf"^<built-in method {callable_object.__name__} of type object at 0x[0-9a-f]+>$",
            )
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
            self.assertIs(copy.copy(callable_object), callable_object)
            self.assertIs(copy.deepcopy(callable_object), callable_object)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.div, function)
        self.assertIs(owner.divide, alias)
        for name in ("div", "divide"):
            original = getattr(owner, name)
            for mutation in (
                lambda name=name: setattr(owner, name, None),
                lambda name=name: delattr(owner, name),
            ):
                with self.assertRaises(TypeError):
                    mutation()
                self.assertIs(getattr(owner, name), original)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)
                self.assertIs(pickle.loads(pickle.dumps(alias, protocol)), alias)
        self.assertEqual(torch.__all__.count("div"), 1)
        self.assertEqual(torch.__all__.count("divide"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        from torch_rs import div as imported_div, divide as imported_divide

        self.assertIs(imported_div, function)
        self.assertIs(imported_divide, alias)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], function)
        self.assertIs(wildcard_namespace["divide"], alias)

        self.assertFalse(hasattr(torch, "div_"))
        self.assertFalse(hasattr(torch, "divide_"))

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.div, function)
        self.assertIs(torch.divide, alias)


if __name__ == "__main__":
    unittest.main()
