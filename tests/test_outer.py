import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
outer(input, vec2, *, out=None) -> Tensor

Outer product of :attr:`input` and :attr:`vec2`.
If :attr:`input` is a vector of size :math:`n` and :attr:`vec2` is a vector of
size :math:`m`, then :attr:`out` must be a matrix of size :math:`(n \\times m)`.

.. note:: This function does not :ref:`broadcast <broadcasting-semantics>`.

Args:
    input (Tensor): 1-D input vector
    vec2 (Tensor): 1-D input vector

Keyword args:
    out (Tensor, optional): optional output matrix

Example::

    >>> v1 = torch.arange(1., 5.)
    >>> v2 = torch.arange(1., 4.)
    >>> torch.outer(v1, v2)
    tensor([[  1.,   2.,   3.],
            [  2.,   4.,   6.],
            [  3.,   6.,   9.],
            [  4.,   8.,  12.]])
"""


class OuterTests(unittest.TestCase):
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

    @staticmethod
    def expected_outer(left, right):
        return left.reshape((left.shape[0], 1)) * right.reshape(
            (1, right.shape[0])
        )

    @staticmethod
    def supported_calls(left, right):
        return (
            ("positional", lambda: torch.outer(left, right)),
            ("keywords", lambda: torch.outer(input=left, vec2=right)),
            ("x alias", lambda: torch.outer(x=left, vec2=right)),
            ("a alias", lambda: torch.outer(a=left, vec2=right)),
            ("x1 alias", lambda: torch.outer(x1=left, vec2=right)),
            ("out none", lambda: torch.outer(left, right, out=None)),
        )

    def test_values_contiguous_layouts_empties_and_non_finites(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(4, 6).tolist()
        )
        special_left_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special_right_bits = np.asarray(
            (0xBF80_0000, 0x0000_0000, 0x8000_0000, 0x3F00_0000),
            dtype=np.uint32,
        )
        cases = (
            ("contiguous", torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])),
            ("offset", base[2], base[1]),
            ("strided", base.transpose(0, 1)[1], base.transpose(0, 1)[4]),
            ("empty left", torch.zeros((3, 0))[1], torch.tensor([1.0, 2.0])),
            ("empty right", torch.tensor([1.0, 2.0]), torch.zeros((3, 0))[2]),
            ("both empty", torch.zeros((2, 0))[1], torch.zeros((3, 0))[2]),
            (
                "non-finite",
                torch.tensor(memoryview(special_left_bits.view(np.float32))),
                torch.tensor(memoryview(special_right_bits.view(np.float32))),
            ),
        )

        for name, left, right in cases:
            expected = self.expected_outer(left, right)
            for form, call in self.supported_calls(left, right):
                actual = call()
                self.assert_tensor_matches(actual, expected, case=(name, form))
                self.assertEqual(actual.storage_offset(), 0)
                self.assertTrue(actual.is_contiguous())

    def test_autograd_shared_operands_empty_vectors_and_no_grad(self):
        actual_left_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        actual_right_leaf = torch.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        expected_left_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        expected_right_leaf = torch.tensor(
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], requires_grad=True
        )
        actual = torch.outer(
            actual_left_leaf.transpose(0, 1)[1],
            actual_right_leaf.transpose(0, 1)[1],
        )
        expected = self.expected_outer(
            expected_left_leaf.transpose(0, 1)[1],
            expected_right_leaf.transpose(0, 1)[1],
        )
        self.assert_tensor_matches(actual, expected, case="tracked strided views")
        actual.sum().backward()
        expected.sum().backward()
        self.assert_tensor_matches(
            actual_left_leaf.grad, expected_left_leaf.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            actual_right_leaf.grad, expected_right_leaf.grad, case="right gradient"
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.outer(actual_shared, actual_shared).sum().backward()
        self.expected_outer(expected_shared, expected_shared).sum().backward()
        self.assert_tensor_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_empty = torch.zeros((0,), requires_grad=True)
        expected_empty = torch.zeros((0,), requires_grad=True)
        torch.outer(actual_empty, torch.tensor([2.0, 3.0])).sum().backward()
        self.expected_outer(expected_empty, torch.tensor([2.0, 3.0])).sum().backward()
        self.assert_tensor_matches(
            actual_empty.grad, expected_empty.grad, case="empty gradient"
        )

        left = torch.tensor([1.0, 2.0], requires_grad=True)
        right = torch.tensor([3.0, 4.0], requires_grad=True)
        with torch.no_grad():
            output = torch.outer(left, right)
        self.assertFalse(output.requires_grad)
        self.assertTrue(output.is_leaf)
        self.assertTrue(torch.outer(left, right).requires_grad)

    def test_modes_and_overrides_precede_native_limits_and_can_forward(self):
        left = torch.tensor([1.0, 2.0])
        right = torch.tensor([3.0])
        matrix = torch.ones((1, 1))
        destination = torch.zeros((2, 1))
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            (lambda: torch.outer(left, right), None),
            (lambda: torch.outer(input=left, vec2=right), ("input", "vec2")),
            (lambda: torch.outer(a=left, vec2=right, out=None), ("a", "vec2", "out")),
            (lambda: torch.outer(matrix, right), None),
            (lambda: torch.outer(left, right, out=destination), ("out",)),
        )
        for call, keyword_names in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.outer)
            self.assertEqual(dispatch_types, ())
            if keyword_names is None:
                self.assertEqual(len(args), 2)
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), keyword_names)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.outer(input=left, vec2=right, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(
            forwarded, self.expected_outer(left, right), case="forwarded modes"
        )

        events = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("base", func, types, args, kwargs))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(("derived", func, types, args, kwargs))
                return marker

        self.assertIs(torch.outer(BaseOverride(), right, out=DerivedOverride()), marker)
        label, function, dispatch_types, args, kwargs = events[0]
        self.assertEqual(label, "derived")
        self.assertIs(function, torch.outer)
        self.assertEqual(dispatch_types, (DerivedOverride, BaseOverride))
        self.assertEqual(len(args), 2)
        self.assertEqual(tuple(kwargs), ("out",))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.outer(DecliningOverride(), right)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.outer'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

        for call in (
            lambda: torch.outer([], right),
            lambda: torch.outer(left, []),
            lambda: torch.outer(left, right, out=[]),
            lambda: torch.outer(left, right, extra=True),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_rank_errors_and_concrete_out_rejection(self):
        vector = torch.tensor([1.0, 2.0])
        rank_cases = (
            (
                torch.tensor(1.0),
                vector,
                "outer: Expected 1-D argument self, but got 0-D",
            ),
            (
                torch.ones((1, 1)),
                vector,
                "outer: Expected 1-D argument self, but got 2-D",
            ),
            (
                vector,
                torch.tensor(1.0),
                "outer: Expected 1-D argument vec2, but got 0-D",
            ),
            (
                vector,
                torch.ones((1, 1)),
                "outer: Expected 1-D argument vec2, but got 2-D",
            ),
        )
        for left, right, message in rank_cases:
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    torch.outer(left, right)

        destination = torch.tensor([[17.0, 19.0], [23.0, 29.0]])
        before = destination.tolist()
        for call in (
            lambda: torch.outer(vector, vector, out=destination),
            lambda: torch.outer(input=vector, vec2=vector, out=destination),
            lambda: torch.outer(a=vector, vec2=vector, out=destination),
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"^outer\(\): the 'out' argument is not supported$"
            ):
                call()
            self.assertEqual(destination.tolist(), before)

        with self.assertRaisesRegex(
            RuntimeError, r"^outer: Expected 1-D argument self, but got 2-D$"
        ):
            torch.outer(torch.ones((1, 1)), vector, out=destination)
        self.assertEqual(destination.tolist(), before)

    def test_binding_metadata_exports_and_unsupported_aliases(self):
        vector = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.outer(),
                'outer() missing 2 required positional argument: "input", "vec2"',
            ),
            (
                lambda: torch.outer(vector),
                'outer() missing 1 required positional arguments: "vec2"',
            ),
            (
                lambda: torch.outer(vector, vector, vector),
                "outer() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.outer([], vector),
                "outer(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.outer(vector, []),
                "outer(): argument 'vec2' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.outer(input=None, vec2=vector),
                "outer(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.outer(vector, vector, input=vector),
                "outer() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.outer(vector, vector, vec2=vector),
                "outer() got multiple values for argument 'vec2'",
            ),
            (
                lambda: torch.outer(input=vector, x2=vector),
                'outer() missing 1 required positional arguments: "vec2"',
            ),
            (
                lambda: torch.outer(vector, vector, extra=True),
                "outer() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.outer(vector, vector, out=[]),
                "outer(): argument 'out' must be Tensor, not list",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        function = torch.outer
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "outer")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.outer")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method outer of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.outer, function)
        for mutation in (
            lambda: setattr(owner, "outer", None),
            lambda: delattr(owner, "outer"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.outer, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("outer"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["outer"], function)
        self.assertFalse(hasattr(torch.Tensor, "outer"))
        self.assertFalse(hasattr(torch, "ger"))


if __name__ == "__main__":
    unittest.main()
