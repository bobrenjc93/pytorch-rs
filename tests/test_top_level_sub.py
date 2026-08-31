import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


SUB_DOC = (
    "\nsub(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.\n\n"
    ".. math::\n"
    "    \\text{{out}}_i = \\text{{input}}_i - \\text{{alpha}} \\times "
    "\\text{{other}}_i\n\n\n"
    "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and complex "
    "inputs.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    other (Tensor or Number): the tensor or number to subtract from "
    ":attr:`input`.\n\n"
    "Keyword args:\n"
    "    alpha (Number): the multiplier for :attr:`other`.\n"
    "    out (Tensor, optional): the output tensor.\n\n"
    "Example::\n\n"
    "    >>> a = torch.tensor((1, 2))\n"
    "    >>> b = torch.tensor((0, 1))\n"
    "    >>> torch.sub(a, b, alpha=2)\n"
    "    tensor([1, 0])\n"
)

SUBTRACT_DOC = (
    "\nsubtract(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Alias for :func:`torch.sub`.\n"
)


class TopLevelSubTests(unittest.TestCase):
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

    def test_supported_calls_match_operator_values_layouts_and_edge_bits(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left - right
        for case, call in (
            ("positional tensors", lambda: torch.sub(left, right)),
            ("canonical keywords", lambda: torch.sub(input=left, other=right)),
            ("x aliases", lambda: torch.sub(x=left, x2=right)),
            ("x1 aliases", lambda: torch.sub(x1=left, x2=right)),
            ("out none", lambda: torch.sub(left, right, out=None)),
            ("alpha int one", lambda: torch.sub(left, right, alpha=1)),
            ("alpha float one", lambda: torch.sub(left, right, alpha=1.0)),
            ("alpha numpy int one", lambda: torch.sub(left, right, alpha=np.int64(1))),
            (
                "alpha numpy float one",
                lambda: torch.sub(left, right, alpha=np.float32(1.0)),
            ),
        ):
            self.assert_tensor_matches(call(), expected, case=case)

        offset = left[1]
        for scalar in (-2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for order, call, expected in (
                ("tensor/scalar", lambda scalar=scalar: torch.sub(offset, scalar), offset - scalar),
                ("scalar/tensor", lambda scalar=scalar: torch.sub(scalar, offset), scalar - offset),
                (
                    "keyword tensor/scalar",
                    lambda scalar=scalar: torch.sub(input=offset, other=scalar),
                    offset - scalar,
                ),
                (
                    "keyword scalar/tensor",
                    lambda scalar=scalar: torch.sub(input=scalar, other=offset),
                    scalar - offset,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(order, type(scalar).__name__)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.sub(empty, broadcast), empty - broadcast, case="strided empty"
        )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.sub(-0.0, special), -0.0 - special, case="IEEE special values"
        )

    def test_subtract_alias_reuses_sub_for_supported_surface(self):
        left = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        right = torch.tensor([[5.0], [7.0]], requires_grad=True)
        for case, alias_call, sub_call in (
            ("positional", lambda: torch.subtract(left, right), lambda: torch.sub(left, right)),
            (
                "canonical keywords",
                lambda: torch.subtract(input=left, other=right),
                lambda: torch.sub(input=left, other=right),
            ),
            (
                "aliases",
                lambda: torch.subtract(x1=left, x2=right),
                lambda: torch.sub(x1=left, x2=right),
            ),
            (
                "scalar first",
                lambda: torch.subtract(4.0, left),
                lambda: torch.sub(4.0, left),
            ),
            (
                "scalar second",
                lambda: torch.subtract(left, np.float32(-0.0)),
                lambda: torch.sub(left, np.float32(-0.0)),
            ),
        ):
            self.assert_tensor_matches(alias_call(), sub_call(), case=case)

    def test_autograd_shared_operands_empties_and_no_grad_reuse_operator_paths(self):
        function_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        function_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        function_output = torch.sub(
            function_left.transpose(0, 1), function_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) - operator_right.transpose(
            0, 1
        )
        self.assert_tensor_matches(function_output, operator_output, case="tracked views")

        function_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.sub(function_scalar, 4.0).sum().backward()
        (operator_scalar - 4.0).sum().backward()
        self.assert_tensor_matches(
            function_scalar.grad, operator_scalar.grad, case="scalar-second gradient"
        )

        function_reflected = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_reflected = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.sub(4.0, function_reflected).sum().backward()
        (4.0 - operator_reflected).sum().backward()
        self.assert_tensor_matches(
            function_reflected.grad,
            operator_reflected.grad,
            case="scalar-first gradient",
        )

        function_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        self.assert_tensor_matches(
            torch.sub(function_empty, torch.ones((1, 1, 3))),
            operator_empty - torch.ones((1, 1, 3)),
            case="empty tensor subtraction",
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = torch.sub(left.transpose(0, 1), right.transpose(0, 1))
            scalar_output = torch.sub(2.0, left)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(torch.sub(left, 2.0).requires_grad)
        self.assertTrue(torch.sub(2.0, left).requires_grad)

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
            ("sub positional", torch.sub, lambda: torch.sub(left, right), (left, right), None),
            (
                "sub keyword alpha",
                torch.sub,
                lambda: torch.sub(input=left, other=right, alpha=1),
                (),
                ("input", "other", "alpha"),
            ),
            (
                "subtract positional",
                torch.subtract,
                lambda: torch.subtract(left, 4.0),
                (left, 4.0),
                None,
            ),
            (
                "subtract keyword scalar",
                torch.subtract,
                lambda: torch.subtract(input=4.0, other=left, out=None),
                (),
                ("input", "other", "out"),
            ),
        )
        for case, expected_function, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
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
                self.asserted_function = func
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                result = torch.subtract(input=4.0, other=left, alpha=1)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(upper.asserted_function, torch.subtract)
        self.assertIs(lower.asserted_function, torch.subtract)
        self.assert_tensor_matches(result, torch.sub(4.0, left), case="forwarded modes")

        invalid_mode = RecordingMode()
        with invalid_mode:
            with self.assertRaises(TypeError):
                torch.sub(left, right, out=[])
        self.assertEqual(invalid_mode.calls, [])

    def test_operand_alpha_and_out_overrides_order_and_declining_errors(self):
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

        self.assertIs(torch.sub(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.sub)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.subtract(input=native, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.subtract)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("alpha", func, types, args, kwargs))
                return marker

        self.assertIs(torch.sub(native, native, alpha=AlphaOverride()), marker)
        self.assertEqual(events[-1][2], (AlphaOverride,))
        self.assertEqual(tuple(events[-1][4]), ("alpha",))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.subtract(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.subtract'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_pickling_exports_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        for call, message in (
            (
                lambda: torch.sub(),
                "sub() received an invalid combination of arguments - got (), "
                "but expected (Tensor input, Tensor other, *, Number alpha = 1, "
                "Tensor out = None)",
            ),
            (
                lambda: torch.sub(tensor, [], out=None),
                "sub(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.sub(input=None, other=tensor),
                "sub(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.sub(tensor, tensor, x2=tensor),
                "sub() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.sub(tensor, tensor, dtype=torch.float32),
                "sub() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.sub(tensor, tensor, device="cpu"),
                "sub() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.sub(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.sub(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.sub(-(2**63) - 1, tensor),
                "can't convert negative int to unsigned",
            ),
            (
                lambda: torch.sub(tensor, True),
                "Subtraction, the `-` operator, with a bool tensor is not supported. "
                "If you are trying to invert a mask, use the `~` or `logical_not()` "
                "operator instead.",
            ),
            (
                lambda: torch.sub(True, tensor),
                "Subtraction, the `-` operator, with a bool tensor is not supported. "
                "If you are trying to invert a mask, use the `~` or `logical_not()` "
                "operator instead.",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=True),
                "Boolean alpha only supported for Boolean results.",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=np.uint64(2**63)),
                "an integer is required",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        for call in (
            lambda: torch.sub(2, 3),
            lambda: torch.subtract(2, 3),
        ):
            with self.assertRaisesRegex(TypeError, "scalar-scalar subtraction"):
                call()

        destination = torch.tensor([17.0])
        for function in (torch.sub, torch.subtract):
            with self.subTest(function=function.__name__, option="out"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{function.__name__}\(\): the 'out' argument is not supported$",
                ):
                    function(tensor, tensor, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(function=function.__name__, option="alpha"):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{function.__name__}\(\): only default alpha=1 is supported$",
                ):
                    function(tensor, tensor, alpha=2)

        for function, doc in ((torch.sub, SUB_DOC), (torch.subtract, SUBTRACT_DOC)):
            with self.subTest(function=function.__name__):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(
                    function.__qualname__,
                    f"_VariableFunctionsClass.{function.__name__}",
                )
                self.assertEqual(function.__module__, "torch")
                self.assertIsNone(function.__text_signature__)
                self.assertEqual(function.__doc__, doc)
                self.assertRegex(
                    repr(function),
                    rf"^<built-in method {function.__name__} of type object at 0x[0-9a-f]+>$",
                )
                with self.assertRaises(ValueError):
                    inspect.signature(function)

                owner = function.__reduce__()[1][0]
                self.assertIs(owner, torch._C._VariableFunctionsClass)
                self.assertIs(getattr(owner, function.__name__), function)
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(function=function.__name__, protocol=protocol):
                        self.assertIs(
                            pickle.loads(pickle.dumps(function, protocol=protocol)),
                            function,
                        )
                self.assertEqual(torch.__all__.count(function.__name__), 1)
                wildcard_namespace = {}
                exec("from torch_rs import *", wildcard_namespace)
                self.assertIs(wildcard_namespace[function.__name__], function)

        self.assertIsNot(torch.sub, torch.subtract)
        self.assertFalse(hasattr(torch.Tensor, "sub"))
        self.assertFalse(hasattr(tensor, "sub"))
        self.assertFalse(hasattr(torch.Tensor, "sub_"))
        self.assertFalse(hasattr(tensor, "sub_"))
        self.assertFalse(hasattr(torch, "sub_"))
        self.assertNotIn("sub_", torch.__all__)

        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.sub, torch.sub)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch._C.sub, torch.sub)


if __name__ == "__main__":
    unittest.main()
