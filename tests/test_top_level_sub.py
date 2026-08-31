import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOCS = {
    "sub": (
        "\nsub(input, other, *, alpha=1, out=None) -> Tensor\n\n"
        "Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.\n\n"
        ".. math::\n"
        "    \\text{{out}}_i = \\text{{input}}_i - \\text{{alpha}} \\times \\text{{other}}_i\n\n\n"
        "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
        ":ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.\n\n"
        "Args:\n"
        "    input (Tensor): the input tensor.\n"
        "    other (Tensor or Number): the tensor or number to subtract from :attr:`input`.\n\n"
        "Keyword args:\n"
        "    alpha (Number): the multiplier for :attr:`other`.\n"
        "    out (Tensor, optional): the output tensor.\n\n"
        "Example::\n\n"
        "    >>> a = torch.tensor((1, 2))\n"
        "    >>> b = torch.tensor((0, 1))\n"
        "    >>> torch.sub(a, b, alpha=2)\n"
        "    tensor([1, 0])\n"
    ),
    "subtract": (
        "\nsubtract(input, other, *, alpha=1, out=None) -> Tensor\n\n"
        "Alias for :func:`torch.sub`.\n"
    ),
}


class TopLevelSubtractionTests(unittest.TestCase):
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

    def test_supported_calls_match_operator_values_layouts_and_defaults(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left - right
        calls = (
            ("sub positional tensors", lambda: torch.sub(left, right)),
            ("sub canonical keywords", lambda: torch.sub(input=left, other=right)),
            ("sub aliases", lambda: torch.sub(x1=left, x2=right)),
            (
                "sub default alpha and out",
                lambda: torch.sub(x=left, x2=right, alpha=np.float32(1.0), out=None),
            ),
            ("subtract positional tensors", lambda: torch.subtract(left, right)),
            (
                "subtract default alpha and out",
                lambda: torch.subtract(input=left, other=right, alpha=1.0, out=None),
            ),
        )
        for case, call in calls:
            actual = call()
            self.assert_tensor_matches(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(left))
            self.assertFalse(actual.is_set_to(right))

        offset = left[1]
        for scalar in (-2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for function in (torch.sub, torch.subtract):
                with self.subTest(function=function.__name__, scalar=repr(scalar)):
                    self.assert_tensor_matches(
                        function(offset, scalar),
                        offset - scalar,
                        case="tensor/scalar",
                    )
                    self.assert_tensor_matches(
                        function(scalar, offset),
                        scalar - offset,
                        case="scalar/tensor",
                    )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.sub(empty, broadcast), empty - broadcast, case="strided empty"
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.sub(special, -0.0), special - -0.0, case="special tensor/scalar"
        )
        self.assert_tensor_matches(
            torch.subtract(-0.0, special),
            -0.0 - special,
            case="special scalar/tensor",
        )

    def test_autograd_shared_operands_empties_and_no_grad_reuse_native_paths(self):
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
        torch.subtract(function_shared, function_shared).sum().backward()
        (operator_shared - operator_shared).sum().backward()
        self.assert_tensor_matches(
            function_shared.grad, operator_shared.grad, case="shared operand gradient"
        )

        function_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.sub(4.0, function_scalar).sum().backward()
        (4.0 - operator_scalar).sum().backward()
        self.assert_tensor_matches(
            function_scalar.grad, operator_scalar.grad, case="scalar-first gradient"
        )

        function_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.sub(function_empty, torch.ones((1, 1, 3))).sum().backward()
        (operator_empty - torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            function_empty.grad, operator_empty.grad, case="empty gradient"
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = torch.sub(left.transpose(0, 1), right.transpose(0, 1))
            scalar_output = torch.subtract(2.0, left)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(torch.sub(left, right.transpose(0, 1)).requires_grad)

    def test_torch_function_modes_and_overrides_observe_valid_calls(self):
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

        for function in (torch.sub, torch.subtract):
            calls = (
                (lambda function=function: function(left, right), (left, right), None),
                (lambda function=function: function(left, 4.0), (left, 4.0), None),
                (lambda function=function: function(4.0, left), (4.0, left), None),
                (lambda function=function: function(left, True), (left, True), None),
                (lambda function=function: function(False, left), (False, left), None),
                (
                    lambda function=function: function(
                        input=left, other=right, alpha=1, out=None
                    ),
                    (),
                    ("input", "other", "alpha", "out"),
                ),
                (
                    lambda function=function: function(
                        x1=left, x2=4.0, alpha=np.float32(1.0)
                    ),
                    (),
                    ("x1", "x2", "alpha"),
                ),
                (
                    lambda function=function: function(left, right, alpha=2),
                    (left, right),
                    ("alpha",),
                ),
                (
                    lambda function=function: function(left, right, alpha=True),
                    (left, right),
                    ("alpha",),
                ),
                (
                    lambda function=function: function(
                        left, right, out=torch.zeros((1,))
                    ),
                    (left, right),
                    ("out",),
                ),
            )
            for call, expected_args, expected_keywords in calls:
                mode = RecordingMode()
                with mode:
                    self.assertIs(call(), marker)
                func, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(func, function)
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
                    result = function(input=4.0, other=left, alpha=1, out=None)
            self.assertEqual(order, ["upper", "lower"])
            self.assert_tensor_matches(result, 4.0 - left, case=function.__name__)

            for call in (
                lambda function=function: function([], right),
                lambda function=function: function(left, []),
                lambda function=function: function(left, right, alpha=[]),
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

        self.assertIs(torch.sub(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.sub)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        alpha_events = []

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                alpha_events.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.subtract(left, right, alpha=AlphaOverride()), marker)
        function, dispatch_types, args, kwargs = alpha_events[0]
        self.assertIs(function, torch.subtract)
        self.assertEqual(dispatch_types, (AlphaOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("alpha",))

        out_events = []

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                out_events.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.sub(left, right, out=OutOverride()), marker)
        function, dispatch_types, args, kwargs = out_events[0]
        self.assertIs(function, torch.sub)
        self.assertEqual(dispatch_types, (OutOverride,))
        self.assertEqual(args, (left, right))
        self.assertEqual(tuple(kwargs), ("out",))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError, r"^Multiple dispatch failed for 'torch\.sub';"
        ):
            torch.sub(DecliningOverride(), left)

    def test_errors_callable_metadata_exports_copy_pickle_and_reload(self):
        tensor = torch.tensor([1.0])
        for function in (torch.sub, torch.subtract):
            name = function.__name__
            with self.subTest(function=name, surface="unsupported"):
                destination = torch.tensor([17.0])
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): the 'out' argument is not supported$",
                ):
                    function(tensor, tensor, out=destination)
                self.assertEqual(destination.tolist(), [17.0])

                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): non-default alpha is not supported; only alpha=1 is implemented$",
                ):
                    function(tensor, tensor, alpha=2)
                with self.assertRaisesRegex(
                    NotImplementedError,
                    rf"^{name}\(\): non-default alpha is not supported; only alpha=1 is implemented$",
                ):
                    function(tensor, tensor, alpha=1.00000001)
                with self.assertRaisesRegex(
                    RuntimeError, r"^Boolean alpha only supported for Boolean results\.$"
                ):
                    function(tensor, tensor, alpha=True)
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^{name}\(\): scalar-scalar subtraction is not supported; at least one operand must be Tensor$",
                ):
                    function(2, 3)
                for call in (
                    lambda function=function: function(tensor, True),
                    lambda function=function: function(False, tensor),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "bool tensor is not supported"
                    ):
                        call()
                with self.assertRaisesRegex(TypeError, "must be Number, not list"):
                    function(tensor, tensor, alpha=[])
                with self.assertRaises(TypeError):
                    function([], tensor)
                with self.assertRaises(TypeError):
                    function(tensor, [])
                with self.assertRaisesRegex(TypeError, "^an integer is required$"):
                    function(tensor, np.uint64(2**63))
                with self.assertRaisesRegex(OverflowError, "^int too big to convert$"):
                    function(tensor, 2**64)
                with self.assertRaisesRegex(
                    OverflowError, "^can't convert negative int to unsigned$"
                ):
                    function(-(2**63) - 1, tensor)

            with self.subTest(function=name, surface="callable"):
                self.assertIs(type(function), types.BuiltinFunctionType)
                self.assertEqual(function.__qualname__, f"_VariableFunctionsClass.{name}")
                self.assertEqual(function.__module__, "torch")
                self.assertIsNone(function.__text_signature__)
                self.assertEqual(function.__doc__, FUNCTION_DOCS[name])
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
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

                self.assertEqual(torch.__all__.count(name), 1)
                self.assertNotIn("_VariableFunctionsClass", torch.__all__)
                self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
                wildcard_namespace = {}
                exec("from torch_rs import *", wildcard_namespace)
                self.assertIs(wildcard_namespace[name], function)
                self.assertIs(getattr(importlib.reload(torch), name), function)

        self.assertIsNot(torch.sub, torch.subtract)
        self.assertFalse(hasattr(torch, "sub_"))
        self.assertFalse(hasattr(torch, "subtract_"))
        self.assertFalse(hasattr(torch.Tensor, "sub"))
        self.assertFalse(hasattr(torch.Tensor, "sub_"))
        self.assertFalse(hasattr(torch.Tensor, "subtract"))
        self.assertFalse(hasattr(torch.Tensor, "subtract_"))


if __name__ == "__main__":
    unittest.main()
