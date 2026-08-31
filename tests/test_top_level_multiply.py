import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nmultiply(input, other, *, out=None)\n\n"
    "Alias for :func:`torch.mul`.\n"
)


class TopLevelMultiplyTests(unittest.TestCase):
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

    def test_supported_calls_reuse_mul_values_layouts_and_autograd(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        calls = (
            ("positional tensors", lambda: torch.multiply(left, right), lambda: torch.mul(left, right)),
            (
                "canonical keywords",
                lambda: torch.multiply(input=left, other=right),
                lambda: torch.mul(input=left, other=right),
            ),
            (
                "x aliases",
                lambda: torch.multiply(x=left, x2=right),
                lambda: torch.mul(x=left, x2=right),
            ),
            (
                "x1 aliases",
                lambda: torch.multiply(x1=left, x2=right),
                lambda: torch.mul(x1=left, x2=right),
            ),
        )
        for case, alias_call, mul_call in calls:
            self.assert_tensor_matches(alias_call(), mul_call(), case=case)

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            for order, alias_call, mul_call in (
                (
                    "tensor/scalar",
                    lambda scalar=scalar: torch.multiply(offset, scalar),
                    lambda scalar=scalar: torch.mul(offset, scalar),
                ),
                (
                    "scalar/tensor",
                    lambda scalar=scalar: torch.multiply(scalar, offset),
                    lambda scalar=scalar: torch.mul(scalar, offset),
                ),
                (
                    "keyword scalar/tensor",
                    lambda scalar=scalar: torch.multiply(input=scalar, other=offset),
                    lambda scalar=scalar: torch.mul(input=scalar, other=offset),
                ),
            ):
                self.assert_tensor_matches(
                    alias_call(), mul_call(), case=(order, type(scalar).__name__)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.multiply(empty, broadcast),
            torch.mul(empty, broadcast),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.multiply(-0.0, special),
            torch.mul(-0.0, special),
            case="signed zero and non-finites",
        )

        alias_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        alias_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        mul_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        mul_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        alias_output = torch.multiply(
            alias_left.transpose(0, 1), alias_right.transpose(0, 1)
        )
        mul_output = torch.mul(mul_left.transpose(0, 1), mul_right.transpose(0, 1))
        self.assert_tensor_matches(alias_output, mul_output, case="tracked views")
        alias_output.sum().backward()
        mul_output.sum().backward()
        self.assert_tensor_matches(alias_left.grad, mul_left.grad, case="left gradient")
        self.assert_tensor_matches(alias_right.grad, mul_right.grad, case="right gradient")

        alias_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        mul_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.multiply(alias_shared, alias_shared).sum().backward()
        torch.mul(mul_shared, mul_shared).sum().backward()
        self.assert_tensor_matches(
            alias_shared.grad, mul_shared.grad, case="shared operand gradient"
        )

        no_grad_input = torch.tensor([2.0, -3.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.multiply(4.0, no_grad_input)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(torch.multiply(no_grad_input, 4.0).requires_grad)

    def test_modes_receive_the_distinct_callable_and_can_forward(self):
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
            (lambda: torch.multiply(left, right), (left, right), None),
            (lambda: torch.multiply(left, 4.0), (left, 4.0), None),
            (lambda: torch.multiply(4.0, left), (4.0, left), None),
            (
                lambda: torch.multiply(input=4.0, other=left),
                (),
                ("input", "other"),
            ),
            (lambda: torch.multiply(x1=left, x2=4.0), (), ("x1", "x2")),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.multiply)
            self.assertIsNot(function, torch.mul)
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
                result = torch.multiply(input=4.0, other=left)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(upper.asserted_function, torch.multiply)
        self.assertIs(lower.asserted_function, torch.multiply)
        self.assert_tensor_matches(result, torch.mul(4.0, left), case="forwarded modes")

        out = torch.zeros((1,))
        for call in (
            lambda: torch.multiply([], right),
            lambda: torch.multiply(left, []),
            lambda: torch.multiply(left, right, out=out),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.multiply(left, np.uint64(2**63)), marker)
        self.assertEqual(len(mode.calls), 1)

    def test_overrides_receive_the_distinct_callable_and_declining_errors(self):
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

        self.assertIs(torch.multiply(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.multiply)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.multiply(input=native, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.multiply)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other"))

        scalar_events = []

        class ScalarOverride(int):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                scalar_events.append((func, types))
                return marker

        self.assertIs(torch.multiply(ScalarOverride(4), native), marker)
        self.assertEqual(scalar_events, [(torch.multiply, (ScalarOverride,))])

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.multiply(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.multiply'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_metadata_pickling_exports_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        overloads = (
            "but expected one of:\n"
            " * (Tensor input, Tensor other, *, Tensor out = None)\n"
            " * (Tensor input, Number other)\n"
        )
        cases = (
            (
                lambda: torch.multiply(),
                "multiply() received an invalid combination of arguments - got (), "
                f"{overloads}",
            ),
            (
                lambda: torch.multiply(tensor),
                "multiply() received an invalid combination of arguments - got "
                f"(Tensor), {overloads}",
            ),
            (
                lambda: torch.multiply([], tensor),
                "multiply() received an invalid combination of arguments - got "
                "(list, Tensor), but expected one of:\n"
                " * (Tensor input, Tensor other, *, Tensor out = None)\n"
                " * (Tensor input, Number other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!list of []!, !Tensor!)\n",
            ),
            (
                lambda: torch.multiply(tensor, []),
                "multiply() received an invalid combination of arguments - got "
                "(Tensor, list), but expected one of:\n"
                " * (Tensor input, Tensor other, *, Tensor out = None)\n"
                " * (Tensor input, Number other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (Tensor, !list of []!)\n",
            ),
            (
                lambda: torch.multiply(tensor, x2=[]),
                "multiply() received an invalid combination of arguments - got "
                "(Tensor, x2=list), but expected one of:\n"
                " * (Tensor input, Tensor other, *, Tensor out = None)\n"
                " * (Tensor input, Number other)\n"
                "      didn't match because some of the keywords were incorrect: x2\n",
            ),
            (
                lambda: torch.multiply(tensor, tensor, tensor),
                "multiply() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.multiply([], tensor, extra=True),
                "multiply(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.multiply(tensor, tensor, input=tensor),
                "multiply() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.multiply(tensor, tensor, x2=tensor),
                "multiply() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.multiply(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.multiply(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.multiply(-(2**63) - 1, tensor),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            TypeError, r"^multiply\(\) got an unexpected keyword argument 'out'$"
        ):
            torch.multiply(tensor, tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaisesRegex(
            TypeError,
            r"^multiply\(\): scalar-scalar multiplication is not supported; "
            r"at least one operand must be Tensor$",
        ):
            torch.multiply(2, 3)

        function = torch.multiply
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertIsNot(function, torch.mul)
        self.assertEqual(function.__name__, "multiply")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.multiply")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function),
            r"^<built-in method multiply of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.multiply, function)
        self.assertIsNot(owner.multiply, owner.mul)
        for mutation in (
            lambda: setattr(owner, "multiply", None),
            lambda: delattr(owner, "multiply"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.multiply, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("multiply"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["multiply"], function)


if __name__ == "__main__":
    unittest.main()
