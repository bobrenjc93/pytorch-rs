import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
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


class TopLevelDivTests(unittest.TestCase):
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

    def test_positional_keyword_broadcast_scalar_layout_and_edge_values(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        tensor_expected = left / right
        for case, call in (
            ("positional tensors", lambda: torch.div(left, right)),
            ("canonical tensor keywords", lambda: torch.div(input=left, other=right)),
            (
                "rounding and out none",
                lambda: torch.div(
                    input=left, other=right, rounding_mode=None, out=None
                ),
            ),
        ):
            self.assert_tensor_matches(call(), tensor_expected, case=case)

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for order, call, expected in (
                (
                    "tensor/scalar",
                    lambda scalar=scalar: torch.div(offset, scalar),
                    offset / scalar,
                ),
                (
                    "keyword tensor/scalar",
                    lambda scalar=scalar: torch.div(
                        input=offset, other=scalar, rounding_mode=None
                    ),
                    offset / scalar,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(order, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.div(empty, broadcast), empty / broadcast, case="strided empty"
        )

        numerators = np.asarray(
            (0x3F80_0000, 0xBF80_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        denominators = np.asarray(
            (0x8000_0000, 0x0000_0000, 0x7F80_0000, 0xFF80_0000, 0x3F80_0000),
            dtype=np.uint32,
        )
        special_left = torch.tensor(memoryview(numerators.view(np.float32)))
        special_right = torch.tensor(memoryview(denominators.view(np.float32)))
        self.assert_tensor_matches(
            torch.div(special_left, special_right),
            special_left / special_right,
            case="IEEE special values",
        )

    def test_inference_only_autograd_boundary(self):
        tracked = torch.tensor([[2.0, 4.0]], requires_grad=True)
        other = torch.tensor([[1.0], [2.0]], requires_grad=True)
        for call in (
            lambda: torch.div(tracked, 2.0),
            lambda: torch.div(tracked.detach(), other),
            lambda: torch.div(tracked, other.detach()),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError, r"^div\(\): autograd recording is not supported$"
                ):
                    call()

        detached = torch.div(tracked.detach(), other.detach())
        self.assertFalse(detached.requires_grad)
        self.assertTrue(detached.is_leaf)

        with torch.no_grad():
            untracked = torch.div(tracked, other)
            scalar_untracked = torch.div(tracked, 2.0)
        self.assertFalse(untracked.requires_grad)
        self.assertFalse(scalar_untracked.requires_grad)
        self.assert_tensor_matches(untracked, tracked / other, case="no_grad tensor")
        self.assert_tensor_matches(scalar_untracked, tracked / 2.0, case="no_grad scalar")

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([6.0])
        right = torch.tensor([3.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            ("tensor/tensor", lambda: torch.div(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.div(left, 3.0), (left, 3.0), None),
            (
                "canonical keywords",
                lambda: torch.div(
                    input=left, other=right, rounding_mode=None, out=None
                ),
                (),
                ("input", "other", "rounding_mode", "out"),
            ),
        )
        for case, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.div)
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
                actual = torch.div(input=left, other=right, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left / right, case="forwarded modes")

        for call in (
            lambda: torch.div([], right),
            lambda: torch.div(4.0, right),
            lambda: torch.div(left, []),
            lambda: torch.div(left, right, out=torch.zeros((1,))),
            lambda: torch.div(left, right, rounding_mode="trunc"),
            lambda: torch.div(x1=left, x2=right),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(Exception):
                    call()
            self.assertEqual(mode.calls, [])

        wide_mode = RecordingMode()
        with wide_mode:
            self.assertIs(torch.div(left, np.uint64(2**63)), marker)
        self.assertEqual(len(wide_mode.calls), 1)

    def test_operand_overrides_order_types_and_declining_errors(self):
        native = torch.tensor([6.0])
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

        self.assertIs(torch.div(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.div)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.div(input=native, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.div)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        scalar_events = []

        class ScalarOverride(int):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                scalar_events.append((func, types))
                return marker

        self.assertIs(torch.div(native, ScalarOverride(3)), marker)
        self.assertEqual(scalar_events, [(torch.div, (ScalarOverride,))])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.div(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.div'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_pickling_exports_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        cases = (
            (
                lambda: torch.div(),
                'div() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.div(tensor),
                'div() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.div(tensor, tensor, tensor),
                "div() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.div([], tensor),
                "div(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.div(tensor, []),
                "div(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.div(input=None, other=tensor),
                "div(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.div(4.0, tensor),
                "div(): argument 'input' (position 1) must be Tensor, not float",
            ),
            (
                lambda: torch.div(4.0, 2.0),
                "div(): argument 'input' (position 1) must be Tensor, not float",
            ),
            (
                lambda: torch.div(x=tensor, other=tensor),
                "div() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.div(x1=tensor, x2=tensor),
                "div() got an unexpected keyword argument 'x1'",
            ),
            (
                lambda: torch.div(tensor, tensor, input=tensor),
                "div() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.div(tensor, tensor, x2=tensor),
                "div() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.div(tensor, tensor, extra=True),
                "div() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.div(tensor, tensor, out=destination),
                "div(): the 'out' argument is not supported",
            ),
            (
                lambda: torch.div(tensor, tensor, rounding_mode="trunc"),
                "div(): rounding_mode is not supported; only None is implemented",
            ),
            (
                lambda: torch.div(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.div(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.div(tensor, -(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()
        self.assertEqual(destination.tolist(), [17.0])

        function = torch.div
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "div")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.div")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method div of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.div, function)
        for mutation in (
            lambda: setattr(owner, "div", None),
            lambda: delattr(owner, "div"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.div, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("div"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertFalse(hasattr(torch, "divide"))
        self.assertFalse(hasattr(torch, "true_divide"))
        self.assertFalse(hasattr(torch.Tensor, "div"))
        self.assertFalse(hasattr(torch.Tensor, "divide"))
        self.assertFalse(hasattr(torch.Tensor, "true_divide"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["div"], function)


if __name__ == "__main__":
    unittest.main()
