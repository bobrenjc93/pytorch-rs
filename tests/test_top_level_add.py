import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


def tensor_from_bits(bits):
    values = np.asarray(bits, dtype=np.uint32)
    return torch.tensor(memoryview(values.view(np.float32)))


def tensor_bits(tensor):
    return np.asarray(tensor).reshape(-1).view(np.uint32)


FUNCTION_DOC = (
    "\nadd(input, other, *, alpha=1, out=None) -> Tensor\n\n"
    "Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.\n\n"
    ".. math::\n"
    "    \\text{{out}}_i = \\text{{input}}_i + \\text{{alpha}} \\times \\text{{other}}_i\n\n\n"
    "Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,\n"
    ":ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n"
    "    other (Tensor or Number): the tensor or number to add to :attr:`input`.\n\n"
    "Keyword arguments:\n"
    "    alpha (Number): the multiplier for :attr:`other`.\n"
    "    out (Tensor, optional): the output tensor.\n\n"
    "Examples::\n\n"
    "    >>> a = torch.randn(4)\n"
    "    >>> a\n"
    "    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])\n"
    "    >>> torch.add(a, 20)\n"
    "    tensor([ 20.0202,  21.0985,  21.3506,  19.3944])\n\n"
    "    >>> b = torch.randn(4)\n"
    "    >>> b\n"
    "    tensor([-0.9732, -0.3497,  0.6245,  0.4022])\n"
    "    >>> c = torch.randn(4, 1)\n"
    "    >>> c\n"
    "    tensor([[ 0.3743],\n"
    "            [-1.7724],\n"
    "            [-0.5811],\n"
    "            [-0.8017]])\n"
    "    >>> torch.add(b, c, alpha=10)\n"
    "    tensor([[  2.7695,   3.3930,   4.3672,   4.1450],\n"
    "            [-18.6971, -18.0736, -17.0994, -17.3216],\n"
    "            [ -6.7845,  -6.1610,  -5.1868,  -5.4090],\n"
    "            [ -8.9902,  -8.3667,  -7.3925,  -7.6147]])\n"
)


class TopLevelAddTests(unittest.TestCase):
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
        tensor_expected = left + right
        for case, call in (
            ("positional tensors", lambda: torch.add(left, right)),
            ("canonical tensor keywords", lambda: torch.add(input=left, other=right)),
            ("x1/x2 aliases", lambda: torch.add(x1=left, x2=right)),
            ("input/x2 aliases", lambda: torch.add(input=left, x2=right)),
            ("x1/other aliases", lambda: torch.add(x1=left, other=right)),
            ("explicit alpha one", lambda: torch.add(left, right, alpha=1)),
            ("tensor alpha one", lambda: torch.add(left, right, alpha=torch.tensor(1.0))),
            (
                "numpy alpha one and out none",
                lambda: torch.add(left, right, alpha=np.float32(1.0), out=None),
            ),
        ):
            self.assert_tensor_matches(call(), tensor_expected, case=case)

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            for order, call, expected in (
                ("tensor/scalar", lambda scalar=scalar: torch.add(offset, scalar), offset + scalar),
                ("scalar/tensor", lambda scalar=scalar: torch.add(scalar, offset), scalar + offset),
                (
                    "keyword tensor/scalar",
                    lambda scalar=scalar: torch.add(input=offset, other=scalar),
                    offset + scalar,
                ),
                (
                    "keyword scalar/tensor",
                    lambda scalar=scalar: torch.add(input=scalar, other=offset),
                    scalar + offset,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(order, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            torch.add(empty, broadcast), empty + broadcast, case="strided empty"
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.add(-0.0, special), -0.0 + special, case="IEEE special values"
        )

        nan_left = tensor_from_bits(
            (0x7FC1_1111, 0x7FC1_1111, 0x3F80_0000, 0x7F81_1111, 0xFF81_1111)
        )
        nan_right = tensor_from_bits(
            (0x7FC2_2222, 0x3F80_0000, 0x7FC2_2222, 0x3F80_0000, 0x7F82_2222)
        )
        scalar_nan = np.asarray([0x7FC3_3333], dtype=np.uint32).view(np.float32)[0]
        for case, actual, expected_bits in (
            (
                "tensor/tensor NaN payload precedence",
                torch.add(nan_left, nan_right),
                (0x7FC2_2222, 0x7FC1_1111, 0x7FC2_2222, 0x7FC1_1111, 0x7FC2_2222),
            ),
            (
                "tensor/scalar NaN payload precedence",
                torch.add(nan_left, scalar_nan),
                (0x7FC3_3333,) * 5,
            ),
            (
                "scalar/tensor NaN payload precedence",
                torch.add(scalar_nan, nan_left),
                (0x7FC1_1111, 0x7FC1_1111, 0x7FC3_3333, 0x7FC1_1111, 0xFFC1_1111),
            ),
        ):
            np.testing.assert_array_equal(
                tensor_bits(actual), np.asarray(expected_bits, dtype=np.uint32), err_msg=case
            )

    def test_scalar_autograd_tensor_tensor_no_grad_and_active_tensor_tensor_rejection(self):
        function_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        torch.add(4.0, function_scalar).sum().backward()
        (4.0 + operator_scalar).sum().backward()
        self.assert_tensor_matches(
            function_scalar.grad, operator_scalar.grad, case="scalar-first gradient"
        )

        function_view_leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        operator_view_leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        torch.add(function_view_leaf.transpose(0, 1), 3.0).sum().backward()
        (operator_view_leaf.transpose(0, 1) + 3.0).sum().backward()
        self.assert_tensor_matches(
            function_view_leaf.grad,
            operator_view_leaf.grad,
            case="strided tensor/scalar gradient",
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = torch.add(left.transpose(0, 1), right.transpose(0, 1))
            scalar_output = torch.add(left, 2.0)
        self.assert_tensor_matches(
            tensor_output,
            left.transpose(0, 1) + right.transpose(0, 1),
            case="no_grad tensor/tensor",
        )
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)

        with self.assertRaisesRegex(NotImplementedError, "tensor/tensor autograd"):
            torch.add(left, right)
        with self.assertRaisesRegex(NotImplementedError, "tensor/tensor autograd"):
            torch.add(left, torch.ones((1, 2)))

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
        left = torch.tensor([2.0])
        right = torch.tensor([3.0])
        out = torch.zeros((1,))
        complex_scalar = np.complex64(1j)
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        calls = (
            ("tensor/tensor", lambda: torch.add(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.add(left, 4.0), (left, 4.0), None),
            ("scalar/tensor", lambda: torch.add(4.0, left), (4.0, left), None),
            ("tensor/complex", lambda: torch.add(left, 1j), (left, 1j), None),
            ("complex/tensor", lambda: torch.add(1j, left), (1j, left), None),
            (
                "tensor/NumPy complex",
                lambda: torch.add(left, complex_scalar),
                (left, complex_scalar),
                None,
            ),
            (
                "canonical keywords",
                lambda: torch.add(input=4.0, other=left, alpha=1, out=None),
                (),
                ("input", "other", "alpha", "out"),
            ),
            (
                "nondefault alpha",
                lambda: torch.add(left, right, alpha=2),
                (left, right),
                ("alpha",),
            ),
            (
                "concrete out",
                lambda: torch.add(left, right, out=out),
                (left, right),
                ("out",),
            ),
            ("scalar/scalar", lambda: torch.add(2.0, 3.0), (2.0, 3.0), None),
        )
        for case, call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            with self.subTest(case=case):
                self.assertIs(function, torch.add)
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
                actual = torch.add(input=4.0, other=left, alpha=1, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, 4.0 + left, case="forwarded modes")

        for call in (
            lambda: torch.add([], right),
            lambda: torch.add(left, []),
            lambda: torch.add(left, right, alpha=[]),
            lambda: torch.add(left, right, out=[]),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(Exception):
                    call()
            self.assertEqual(mode.calls, [])

        wide_mode = RecordingMode()
        with wide_mode:
            self.assertIs(torch.add(left, np.uint64(2**63)), marker)
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

        self.assertIs(torch.add(LeftOverride(), RightOverride()), marker)
        self.assertEqual([event[0] for event in events], ["left", "right"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, torch.add)
            self.assertEqual(dispatch_types, (LeftOverride, RightOverride))
            self.assertEqual(len(args), 2)
            self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.add(input=native, other=RightOverride()), marker)
        _, function, dispatch_types, args, kwargs = events[0]
        self.assertIs(function, torch.add)
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

        self.assertIs(torch.add(BaseOverride(), DerivedOverride()), marker)
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
        self.assertIs(torch.add(scalar, native), marker)
        self.assertEqual(scalar_events[0][1], (ScalarOverride,))

        events.clear()
        self.assertIs(torch.add(RightOverride(), 1j), marker)
        tag, function, dispatch_types, args, kwargs = events[0]
        self.assertEqual(tag, "right")
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertIsInstance(args[0], RightOverride)
        self.assertEqual(args[1], 1j)
        self.assertIsNone(kwargs)

        events.clear()
        self.assertIs(torch.add(np.complex64(1j), RightOverride()), marker)
        tag, function, dispatch_types, args, kwargs = events[0]
        self.assertEqual(tag, "right")
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (RightOverride,))
        self.assertEqual(args[0], np.complex64(1j))
        self.assertIsInstance(args[1], RightOverride)
        self.assertIsNone(kwargs)

        keyword_events = []

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                keyword_events.append(("alpha", func, types, args, kwargs))
                return marker

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                keyword_events.append(("out", func, types, args, kwargs))
                return marker

        self.assertIs(torch.add(native, native, alpha=AlphaOverride()), marker)
        tag, function, dispatch_types, args, kwargs = keyword_events[-1]
        self.assertEqual(tag, "alpha")
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (AlphaOverride,))
        self.assertEqual(args, (native, native))
        self.assertEqual(tuple(kwargs), ("alpha",))

        self.assertIs(torch.add(native, native, out=OutOverride()), marker)
        tag, function, dispatch_types, args, kwargs = keyword_events[-1]
        self.assertEqual(tag, "out")
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (OutOverride,))
        self.assertEqual(args, (native, native))
        self.assertEqual(tuple(kwargs), ("out",))

        keyword_events.clear()
        self.assertIs(torch.add(RightOverride(), native, alpha=AlphaOverride()), marker)
        self.assertEqual([event[0] for event in events[-1:]], ["right"])
        self.assertEqual(events[-1][2], (RightOverride, AlphaOverride))

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaises(TypeError) as raised:
            torch.add(DecliningOverride(), native)
        self.assertEqual(
            str(raised.exception),
            "Multiple dispatch failed for 'torch.add'; all __torch_function__ "
            "handlers returned NotImplemented:\n\n"
            f"  - tensor subclass <class '{DecliningOverride.__module__}."
            f"{DecliningOverride.__qualname__}'>\n\n"
            "For more information, try re-running with TORCH_LOGS=not_implemented",
        )

    def test_errors_callable_metadata_pickling_exports_reload_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.add(),
                'add\\(\\) missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.add(tensor),
                'add\\(\\) missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.add(2),
                'add\\(\\) missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.add(tensor, tensor, tensor),
                "add\\(\\) takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.add([], tensor),
                "add\\(\\): argument 'input' \\(position 1\\) must be Tensor, not list",
            ),
            (
                lambda: torch.add(tensor, []),
                "add\\(\\): argument 'other' \\(position 2\\) must be Tensor, not list",
            ),
            (
                lambda: torch.add(input=None, other=tensor),
                "add\\(\\): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.add(tensor, tensor, input=tensor),
                "add\\(\\) got multiple values for argument 'input'",
            ),
            (
                lambda: torch.add(tensor, tensor, x2=tensor),
                "add\\(\\) got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.add(foo=tensor),
                "add\\(\\) got an unexpected keyword argument 'foo'",
            ),
            (
                lambda: torch.add(tensor, tensor, extra=True),
                "add\\(\\) got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.add(tensor, tensor, dtype=torch.float32),
                "add\\(\\) got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.add(tensor, tensor, device=torch.device("cpu")),
                "add\\(\\) got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.add(tensor, tensor, alpha=2),
                "add\\(\\): nondefault alpha is not supported; only alpha=1 is implemented",
            ),
            (
                lambda: torch.add(tensor, 1j),
                "add\\(\\): only exact native CPU float32 Tensor operands and real scalars are supported",
            ),
            (
                lambda: torch.add(1j, tensor),
                "add\\(\\): only exact native CPU float32 Tensor operands and real scalars are supported",
            ),
            (
                lambda: torch.add(np.complex64(1j), tensor),
                "add\\(\\): only exact native CPU float32 Tensor operands and real scalars are supported",
            ),
            (
                lambda: torch.add(tensor, tensor, alpha=torch.tensor(2.0)),
                "add\\(\\): nondefault alpha is not supported; only alpha=1 is implemented",
            ),
            (
                lambda: torch.add(tensor, tensor, alpha=True),
                "Boolean alpha only supported for Boolean results\\.",
            ),
            (
                lambda: torch.add(tensor, tensor, alpha=None),
                "add\\(\\): argument 'alpha' must be Number, not NoneType",
            ),
            (
                lambda: torch.add(tensor, tensor, out=[]),
                "add\\(\\): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.add(2, 3),
                "add\\(\\): scalar-scalar addition is not supported; at least one operand must be Tensor",
            ),
            (lambda: torch.add(tensor, np.uint64(2**63)), "an integer is required"),
            (lambda: torch.add(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.add(-(2**63) - 1, tensor),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{message}$"):
                    call()

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(NotImplementedError, "out.*not supported"):
            torch.add(tensor, tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        function = torch.add
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "add")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.add")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertRegex(
            repr(function), r"^<built-in method add of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.add, function)
        for mutation in (
            lambda: setattr(owner, "add", None),
            lambda: delattr(owner, "add"),
        ):
            with self.assertRaises(TypeError):
                mutation()
            self.assertIs(owner.add, function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("add"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["add"], function)
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded.add, function)

        self.assertFalse(hasattr(torch.Tensor, "add"))
        self.assertFalse(hasattr(torch.Tensor, "add_"))
        self.assertFalse(hasattr(torch, "add_"))
        self.assertFalse(hasattr(torch, "sub"))
        self.assertNotIn("add_", torch.__all__)
        self.assertNotIn("sub", torch.__all__)


if __name__ == "__main__":
    unittest.main()
