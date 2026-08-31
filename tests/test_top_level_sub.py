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
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and "
    "complex inputs.\n\n"
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


def subtract_binding_message(summary, mismatch=None):
    message = (
        "subtract() received an invalid combination of arguments - got "
        f"({summary}), but expected one of:\n"
        " * (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)\n"
        " * (Tensor input, Number other, Number alpha = 1)\n"
    )
    if mismatch is not None:
        message += f"      didn't match because {mismatch}\n"
    return message


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

    def test_sub_values_broadcast_scalars_empties_and_special_bits(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left - right
        calls = (
            ("positional tensors", lambda: torch.sub(left, right)),
            ("canonical keywords", lambda: torch.sub(input=left, other=right)),
            ("x aliases", lambda: torch.sub(x=left, x2=right)),
            ("x1 aliases", lambda: torch.sub(x1=left, x2=right)),
            ("default alpha", lambda: torch.sub(left, right, alpha=1)),
            ("default alpha numpy", lambda: torch.sub(left, right, alpha=np.int64(1))),
            ("explicit out none", lambda: torch.sub(left, right, out=None)),
        )
        for case, call in calls:
            self.assert_tensor_matches(call(), expected, case=case)

        offset = left[1]
        for scalar in (-2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for order, call, expected in (
                (
                    "tensor/scalar",
                    lambda scalar=scalar: torch.sub(offset, scalar),
                    offset - scalar,
                ),
                (
                    "scalar/tensor",
                    lambda scalar=scalar: torch.sub(scalar, offset),
                    scalar - offset,
                ),
                (
                    "keyword scalar/tensor",
                    lambda scalar=scalar: torch.sub(input=scalar, other=offset),
                    scalar - offset,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(order, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.sub(empty, broadcast),
            empty - broadcast,
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.sub(special, torch.zeros((5,))),
            special - torch.zeros((5,)),
            case="signed zero and non-finites",
        )

    def test_subtract_is_distinct_alias_with_same_native_semantics(self):
        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        alias_output = torch.subtract(left.transpose(0, 1), right.transpose(0, 1))
        sub_output = torch.sub(left.transpose(0, 1), right.transpose(0, 1))
        self.assertIsNot(torch.subtract, torch.sub)
        self.assert_tensor_matches(alias_output, sub_output, case="alias output")

        alias_scalar = torch.subtract(input=np.float32(4.0), other=left)
        sub_scalar = torch.sub(input=np.float32(4.0), other=left)
        self.assert_tensor_matches(alias_scalar, sub_scalar, case="alias scalar-first")

        alias_output.sum().backward()
        self.assertEqual(left.grad.tolist(), [[2.0, 2.0]])
        self.assertEqual(right.grad.tolist(), [[-2.0], [-2.0]])

    def test_autograd_no_grad_and_shared_operands_reuse_subtraction_path(self):
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
        torch.sub(function_shared, function_shared).sum().backward()
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
            scalar_output = torch.sub(2.0, left)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(torch.sub(left, right.transpose(0, 1)).requires_grad)

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
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
            (lambda: torch.sub(left, right), torch.sub, (left, right), None),
            (lambda: torch.sub(left, 4.0), torch.sub, (left, 4.0), None),
            (lambda: torch.sub(4.0, left), torch.sub, (4.0, left), None),
            (
                lambda: torch.sub(input=4.0, other=left, alpha=2),
                torch.sub,
                (),
                ("input", "other", "alpha"),
            ),
            (
                lambda: torch.sub(left, right, out=destination),
                torch.sub,
                (left, right),
                ("out",),
            ),
            (
                lambda: torch.subtract(x1=left, x2=4.0, alpha=True),
                torch.subtract,
                (),
                ("x1", "x2", "alpha"),
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
                actual = torch.sub(input=4.0, other=left, alpha=1)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, 4.0 - left, case="forwarded modes")

        for call in (
            lambda: torch.sub([], right),
            lambda: torch.sub(left, []),
            lambda: torch.sub(left, right, alpha=[]),
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

        events.clear()
        self.assertIs(torch.sub(input=left, other=RightOverride(), alpha=2), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.sub)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other", "alpha"))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.sub(DecliningOverride(), left)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.sub'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_import_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        for function, name in ((torch.sub, "sub"), (torch.subtract, "subtract")):
            common_cases = (
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
                    f"{name}(): scalar-scalar subtraction is not supported; at "
                    "least one operand must be Tensor",
                ),
            )
            if name == "sub":
                cases = (
                    (
                        lambda function=function: function(),
                        "sub() received an invalid combination of arguments - got "
                        "(), but expected (Tensor input, Tensor other, *, Number "
                        "alpha = 1, Tensor out = None)",
                    ),
                    (
                        lambda function=function: function(tensor),
                        "sub() received an invalid combination of arguments - got "
                        "(Tensor), but expected (Tensor input, Tensor other, *, "
                        "Number alpha = 1, Tensor out = None)",
                    ),
                    (
                        lambda function=function: function(2),
                        "sub() received an invalid combination of arguments - got "
                        "(int), but expected (Tensor input, Tensor other, *, "
                        "Number alpha = 1, Tensor out = None)",
                    ),
                    (
                        lambda function=function: function(tensor, tensor, tensor),
                        "sub() takes 2 positional arguments but 3 were given",
                    ),
                    (
                        lambda function=function: function([], tensor),
                        "sub(): argument 'input' (position 1) must be Tensor, "
                        "not list",
                    ),
                    (
                        lambda function=function: function(tensor, []),
                        "sub(): argument 'other' (position 2) must be Tensor, "
                        "not list",
                    ),
                    (
                        lambda function=function: function(input=None, other=tensor),
                        "sub(): argument 'input' must be Tensor, not NoneType",
                    ),
                    (
                        lambda function=function: function(tensor, tensor, input=tensor),
                        "sub() got multiple values for argument 'input'",
                    ),
                    (
                        lambda function=function: function(tensor, tensor, x2=tensor),
                        "sub() got an unexpected keyword argument 'x2'",
                    ),
                    (
                        lambda function=function: function(tensor, tensor, extra=True),
                        "sub() got an unexpected keyword argument 'extra'",
                    ),
                    (
                        lambda function=function: function(tensor, tensor, alpha=[]),
                        "sub(): argument 'alpha' must be Number, not list",
                    ),
                )
            else:
                cases = (
                    (
                        lambda function=function: function(),
                        subtract_binding_message(""),
                    ),
                    (
                        lambda function=function: function(tensor),
                        subtract_binding_message("Tensor"),
                    ),
                    (
                        lambda function=function: function(2),
                        subtract_binding_message("int"),
                    ),
                    (
                        lambda function=function: function(tensor, tensor, tensor),
                        subtract_binding_message(
                            "Tensor, Tensor, Tensor",
                            "some of the arguments have invalid types: "
                            "(Tensor, !Tensor!, !Tensor!)",
                        ),
                    ),
                    (
                        lambda function=function: function([], tensor),
                        subtract_binding_message("list, Tensor"),
                    ),
                    (
                        lambda function=function: function(tensor, []),
                        subtract_binding_message("Tensor, list"),
                    ),
                    (
                        lambda function=function: function(input=None, other=tensor),
                        subtract_binding_message("other=Tensor, input=NoneType, "),
                    ),
                    (
                        lambda function=function: function(tensor, tensor, input=tensor),
                        subtract_binding_message(
                            "Tensor, Tensor, input=Tensor",
                            "some of the keywords were incorrect: input",
                        ),
                    ),
                    (
                        lambda function=function: function(tensor, tensor, x2=tensor),
                        subtract_binding_message(
                            "Tensor, Tensor, x2=Tensor",
                            "some of the keywords were incorrect: x2",
                        ),
                    ),
                    (
                        lambda function=function: function(tensor, tensor, extra=True),
                        subtract_binding_message(
                            "Tensor, Tensor, extra=bool",
                            "some of the keywords were incorrect: extra",
                        ),
                    ),
                    (
                        lambda function=function: function(tensor, tensor, alpha=[]),
                        subtract_binding_message(
                            "Tensor, Tensor, alpha=list",
                            "some of the keywords were incorrect: alpha",
                        ),
                    ),
                )
            cases = cases + common_cases
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
                rf"^{name}\(\): alpha values other than 1 are not supported$",
            ):
                function(tensor, tensor, alpha=2)
            with self.assertRaisesRegex(
                RuntimeError, "^Boolean alpha only supported for Boolean results\\.$"
            ):
                function(tensor, tensor, alpha=True)
            with self.assertRaisesRegex(
                RuntimeError,
                r"^Subtraction, the `-` operator, with a bool tensor is not "
                r"supported\. If you are trying to invert a mask, use the `~` "
                r"or `logical_not\(\)` operator instead\.$",
            ):
                function(tensor, True)

        function = torch.sub
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sub")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sub")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, SUB_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method sub of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        alias = torch.subtract
        self.assertIs(type(alias), types.BuiltinFunctionType)
        self.assertIsNot(alias, function)
        self.assertEqual(alias.__name__, "subtract")
        self.assertEqual(alias.__qualname__, "_VariableFunctionsClass.subtract")
        self.assertEqual(alias.__module__, "torch")
        self.assertIsNone(alias.__text_signature__)
        self.assertEqual(alias.__doc__, SUBTRACT_DOC)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sub, function)
        self.assertIs(owner.subtract, alias)
        for name in ("sub", "subtract"):
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
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )
                self.assertIs(pickle.loads(pickle.dumps(alias, protocol=protocol)), alias)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(copy.copy(alias), alias)
        self.assertIs(copy.deepcopy(alias), alias)

        self.assertEqual(torch.__all__.count("sub"), 1)
        self.assertEqual(torch.__all__.count("subtract"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sub"], function)
        self.assertIs(wildcard_namespace["subtract"], alias)

        self.assertFalse(hasattr(torch.Tensor, "sub"))
        self.assertFalse(hasattr(torch.Tensor, "subtract"))
        self.assertFalse(hasattr(torch, "sub_"))
        self.assertFalse(hasattr(torch, "subtract_"))

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.sub, function)
        self.assertIs(torch.subtract, alias)


if __name__ == "__main__":
    unittest.main()
