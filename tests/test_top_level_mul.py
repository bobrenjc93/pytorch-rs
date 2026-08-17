import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nmul(input, other, *, out=None) -> Tensor\n\n"
    "Multiplies :attr:`input` by :attr:`other`.\n\n\n"
    ".. math::\n"
    "    \\text{out}_i = \\text{input}_i \\times \\text{other}_i\n\n\n"
    "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    other (Tensor or Number): the tensor or number to multiply input by.\n\n"
    "Keyword args:\n"
    "    out (Tensor, optional): the output tensor.\n\n"
    "Examples::\n\n"
    "    >>> a = torch.randn(3)\n"
    "    >>> a\n"
    "    tensor([ 0.2015, -0.4255,  2.6087])\n"
    "    >>> torch.mul(a, 100)\n"
    "    tensor([  20.1494,  -42.5491,  260.8663])\n\n"
    "    >>> b = torch.randn(4, 1)\n"
    "    >>> b\n"
    "    tensor([[ 1.1207],\n"
    "            [-0.3137],\n"
    "            [ 0.0700],\n"
    "            [ 0.8378]])\n"
    "    >>> c = torch.randn(1, 4)\n"
    "    >>> c\n"
    "    tensor([[ 0.5146,  0.1216, -0.5244,  2.2382]])\n"
    "    >>> torch.mul(b, c)\n"
    "    tensor([[ 0.5767,  0.1363, -0.5877,  2.5083],\n"
    "            [-0.1614, -0.0382,  0.1645, -0.7021],\n"
    "            [ 0.0360,  0.0085, -0.0367,  0.1567],\n"
    "            [ 0.4312,  0.1019, -0.4394,  1.8753]])\n"
)


class TopLevelMulTests(unittest.TestCase):
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
        tensor_expected = left * right
        for case, call in (
            ("positional tensors", lambda: torch.mul(left, right)),
            (
                "canonical tensor keywords",
                lambda: torch.mul(input=left, other=right),
            ),
            ("tensor aliases", lambda: torch.mul(x1=left, x2=right)),
        ):
            self.assert_tensor_matches(call(), tensor_expected, case=case)

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            for order, call, expected in (
                ("tensor/scalar", lambda scalar=scalar: torch.mul(offset, scalar), offset * scalar),
                ("scalar/tensor", lambda scalar=scalar: torch.mul(scalar, offset), scalar * offset),
                (
                    "keyword tensor/scalar",
                    lambda scalar=scalar: torch.mul(input=offset, other=scalar),
                    offset * scalar,
                ),
                (
                    "keyword scalar/tensor",
                    lambda scalar=scalar: torch.mul(input=scalar, other=offset),
                    scalar * offset,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(order, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.mul(empty, broadcast), empty * broadcast, case="strided empty"
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.mul(-0.0, special), -0.0 * special, case="IEEE special values"
        )

    def test_autograd_shared_operands_empties_and_no_grad_reuse_native_paths(self):
        function_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        function_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        function_output = torch.mul(
            function_left.transpose(0, 1), function_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) * operator_right.transpose(
            0, 1
        )
        self.assert_tensor_matches(function_output, operator_output, case="tracked views")
        function_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            function_left.grad, operator_left.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            function_right.grad, operator_right.grad, case="right gradient"
        )

        function_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.mul(function_shared, function_shared).sum().backward()
        (operator_shared * operator_shared).sum().backward()
        self.assert_tensor_matches(
            function_shared.grad, operator_shared.grad, case="shared operand gradient"
        )

        function_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.mul(4.0, function_scalar).sum().backward()
        (4.0 * operator_scalar).sum().backward()
        self.assert_tensor_matches(
            function_scalar.grad, operator_scalar.grad, case="scalar-first gradient"
        )

        function_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.mul(function_empty, torch.ones((1, 1, 3))).sum().backward()
        (operator_empty * torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            function_empty.grad, operator_empty.grad, case="empty gradient"
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = torch.mul(left.transpose(0, 1), right.transpose(0, 1))
            scalar_output = torch.mul(2.0, left)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(torch.mul(left, right.transpose(0, 1)).requires_grad)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([2.0])
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
            ("tensor/tensor", lambda: torch.mul(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.mul(left, 4.0), (left, 4.0), None),
            ("scalar/tensor", lambda: torch.mul(4.0, left), (4.0, left), None),
            (
                "canonical keywords",
                lambda: torch.mul(input=4.0, other=left),
                (),
                ("input", "other"),
            ),
            (
                "aliases",
                lambda: torch.mul(x1=left, x2=4.0),
                (),
                ("x1", "x2"),
            ),
        )
        for case, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.mul)
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
                actual = torch.mul(input=4.0, other=left)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, 4.0 * left, case="forwarded modes")

        for call in (
            lambda: torch.mul([], right),
            lambda: torch.mul(left, []),
            lambda: torch.mul(left, right, out=torch.zeros((1,))),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

        wide_mode = RecordingMode()
        with wide_mode:
            self.assertIs(torch.mul(left, np.uint64(2**63)), marker)
        self.assertEqual(len(wide_mode.calls), 1)

    def test_operand_overrides_order_types_and_declining_errors(self):
        native = torch.tensor([2.0])
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

        self.assertIs(torch.mul(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.mul)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.mul(input=native, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.mul)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        subclass_events = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_events.append(("derived", types))
                return marker

        self.assertIs(torch.mul(BaseOverride(), DerivedOverride()), marker)
        self.assertEqual(
            subclass_events, [("derived", (DerivedOverride, BaseOverride))]
        )

        scalar_events = []

        class ScalarOverride(int):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                scalar_events.append((func, types, args, kwargs))
                return marker

        scalar = ScalarOverride(4)
        self.assertIs(torch.mul(scalar, native), marker)
        self.assertEqual(scalar_events[0][1], (ScalarOverride,))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.mul(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.mul'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_callable_metadata_pickling_exports_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.mul(),
                'mul() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.mul(tensor),
                'mul() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.mul(2),
                'mul() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.mul(tensor, tensor, tensor),
                "mul() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.mul([], tensor),
                "mul(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.mul(tensor, []),
                "mul(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.mul(input=None, other=tensor),
                "mul(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.mul(tensor, tensor, input=tensor),
                "mul() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.mul(tensor, tensor, x2=tensor),
                "mul() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.mul(foo=tensor),
                'mul() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.mul(tensor, tensor, extra=True),
                "mul() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.mul(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.mul(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.mul(-(2**63) - 1, tensor),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            TypeError, r"^mul\(\) got an unexpected keyword argument 'out'$"
        ):
            torch.mul(tensor, tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaisesRegex(TypeError, "scalar-scalar multiplication"):
            torch.mul(2, 3)

        function = torch.mul
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "mul")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.mul")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method mul of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mul, function)
        for mutation in (
            lambda: setattr(owner, "mul", None),
            lambda: delattr(owner, "mul"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.mul, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("mul"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertFalse(hasattr(torch, "multiply"))
        self.assertNotIn("multiply", torch.__all__)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["mul"], function)
        self.assertNotIn("multiply", wildcard_namespace)


if __name__ == "__main__":
    unittest.main()
