import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nsub(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Subtracts :attr:`other`, scaled by :attr:`alpha`, from :attr:`input`.\n\n"
    ".. math::\n"
    "    \\text{{out}}_i = \\text{{input}}_i - \\text{{alpha}} "
    "\\times \\text{{other}}_i\n\n\n"
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

    def test_positional_keyword_broadcast_layout_empty_and_ieee_values(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left - right
        calls = (
            ("positional tensors", lambda: torch.sub(left, right)),
            ("canonical keywords", lambda: torch.sub(input=left, other=right)),
            ("x/x2 aliases", lambda: torch.sub(x=left, x2=right)),
            ("x1/x2 aliases", lambda: torch.sub(x1=left, x2=right)),
            ("alpha int one", lambda: torch.sub(left, right, alpha=1)),
            ("alpha float one", lambda: torch.sub(left, right, alpha=1.0)),
            ("alpha numpy int one", lambda: torch.sub(left, right, alpha=np.int64(1))),
            (
                "alpha numpy float one",
                lambda: torch.sub(left, right, alpha=np.float32(1.0)),
            ),
            (
                "alpha numpy bool true",
                lambda: torch.sub(left, right, alpha=np.bool_(True)),
            ),
            ("explicit out none", lambda: torch.sub(left, right, out=None)),
        )
        for case, call in calls:
            self.assert_tensor_matches(call(), expected, case=case)

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
        zero = torch.zeros((5,))
        self.assert_tensor_matches(
            torch.sub(special, zero), special - zero, case="IEEE special values"
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
        torch.sub(function_shared, function_shared).sum().backward()
        (operator_shared - operator_shared).sum().backward()
        self.assert_tensor_matches(
            function_shared.grad, operator_shared.grad, case="shared operand gradient"
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
            output = torch.sub(left.transpose(0, 1), right.transpose(0, 1))
        self.assertFalse(output.requires_grad)
        self.assertTrue(torch.sub(left, right.transpose(0, 1)).requires_grad)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([2.0])
        right = torch.tensor([3.0])
        destination = torch.tensor([17.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            ("tensor/tensor", lambda: torch.sub(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.sub(left, 4.0), (left, 4.0), None),
            ("scalar/tensor", lambda: torch.sub(4.0, left), (4.0, left), None),
            (
                "canonical keywords",
                lambda: torch.sub(input=left, other=right),
                (),
                ("input", "other"),
            ),
            (
                "aliases",
                lambda: torch.sub(x=left, x2=right, alpha=1),
                (),
                ("x", "x2", "alpha"),
            ),
            (
                "non-default alpha",
                lambda: torch.sub(left, right, alpha=2),
                (left, right),
                ("alpha",),
            ),
            (
                "concrete out",
                lambda: torch.sub(left, right, out=destination),
                (left, right),
                ("out",),
            ),
        )
        for case, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.sub)
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
                actual = torch.sub(input=left, other=right, alpha=1, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left - right, case="forwarded modes")

        for call in (
            lambda: torch.sub([], right),
            lambda: torch.sub(left, []),
            lambda: torch.sub(left, right, out=[]),
            lambda: torch.sub(left, right, dtype=torch.float32),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(Exception):
                    call()
            self.assertEqual(mode.calls, [])

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
                return NotImplemented

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("alpha", func, types, args, kwargs))
                return NotImplemented

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("out", func, types, args, kwargs))
                return marker

        self.assertIs(
            torch.sub(
                LeftOverride(),
                RightOverride(),
                alpha=AlphaOverride(),
                out=OutOverride(),
            ),
            marker,
        )
        self.assertEqual([event[0] for event in events], ["left", "right", "alpha", "out"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.sub)
            self.assertEqual(
                dispatch_types,
                (LeftOverride, RightOverride, AlphaOverride, OutOverride),
            )
            self.assertEqual(len(args), 2)
            self.assertEqual(tuple(kwargs), ("alpha", "out"))

        events.clear()
        class KeywordOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("keyword", func, types, args, kwargs))
                return marker

        self.assertIs(torch.sub(input=native, other=KeywordOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.sub)
        self.assertEqual(dispatch_types, (KeywordOverride,))
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

        self.assertIs(torch.sub(BaseOverride(), DerivedOverride()), marker)
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
        self.assertIs(torch.sub(scalar, native), marker)
        self.assertEqual(scalar_events[0][1], (ScalarOverride,))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.sub(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.sub'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_callable_metadata_pickling_exports_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        cases = (
            (
                lambda: torch.sub(),
                "sub() received an invalid combination of arguments - got (), but expected "
                "(Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.sub(tensor),
                "sub() received an invalid combination of arguments - got (Tensor), but "
                "expected (Tensor input, Tensor other, *, Number alpha = 1, Tensor out = None)",
            ),
            (
                lambda: torch.sub(tensor, tensor, tensor),
                "sub() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.sub([], tensor),
                "sub(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.sub(tensor, []),
                "sub(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.sub(input=None, other=tensor),
                "sub(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.sub(tensor, tensor, input=tensor),
                "sub() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.sub(tensor, tensor, x2=tensor),
                "sub() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.sub(foo=tensor),
                "sub() received an invalid combination of arguments - got unrecognized "
                "keyword arguments: foo",
            ),
            (
                lambda: torch.sub(tensor, tensor, dtype=torch.float32),
                "sub() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.sub(tensor, tensor, device=torch.device("cpu")),
                "sub() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.sub(tensor, 2.0),
                "sub(): argument 'other' (position 2) must be Tensor, not float",
            ),
            (
                lambda: torch.sub(2.0, tensor),
                "sub(): argument 'input' (position 1) must be Tensor, not float",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=2),
                "sub(): non-default alpha is not supported; only alpha=1 is implemented",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=0.0),
                "sub(): non-default alpha is not supported; only alpha=1 is implemented",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=np.bool_(False)),
                "sub(): non-default alpha is not supported; only alpha=1 is implemented",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=True),
                "Boolean alpha only supported for Boolean results.",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=1 + 0j),
                "For non-complex input tensors, argument alpha must not be a complex number.",
            ),
            (
                lambda: torch.sub(tensor, tensor, alpha=[]),
                "sub(): argument 'alpha' must be Number, not list",
            ),
            (
                lambda: torch.sub(tensor, tensor, out=[]),
                "sub(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sub(tensor, np.uint64(2**63)),
                "sub(): argument 'other' (position 2) must be Tensor, not numpy.uint64",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            RuntimeError, r"^sub\(\): the 'out' argument is not supported$"
        ):
            torch.sub(tensor, tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        self.assertFalse(hasattr(torch, "subtract"))
        self.assertNotIn("subtract", torch.__all__)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertNotIn("subtract", wildcard_namespace)

        function = torch.sub
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sub")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sub")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method sub of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sub, function)
        for mutation in (
            lambda: setattr(owner, "sub", None),
            lambda: delattr(owner, "sub"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.sub, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sub"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertIs(wildcard_namespace["sub"], function)


if __name__ == "__main__":
    unittest.main()
