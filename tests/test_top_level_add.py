import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


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

    def test_supported_tensor_and_scalar_values_layouts_and_defaults(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        left = base.transpose(0, 2)
        right = torch.tensor([[[10.0], [20.0], [30.0]]])
        tensor_expected = left + right
        for case, call in (
            ("positional tensors", lambda: torch.add(left, right)),
            ("canonical keywords", lambda: torch.add(input=left, other=right)),
            ("legacy x alias", lambda: torch.add(x=left, x2=right)),
            ("legacy a alias", lambda: torch.add(a=left, other=right)),
            ("legacy x1 alias", lambda: torch.add(x1=left, x2=right)),
            ("alpha int default", lambda: torch.add(left, right, alpha=1)),
            ("alpha float default", lambda: torch.add(left, right, alpha=1.0)),
            ("alpha numpy default", lambda: torch.add(left, right, alpha=np.int64(1))),
            ("out none", lambda: torch.add(left, right, out=None)),
            (
                "all keyword defaults",
                lambda: torch.add(input=left, other=right, alpha=np.float32(1.0), out=None),
            ),
        ):
            self.assert_tensor_matches(call(), tensor_expected, case=case)

        offset = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            for order, call, expected in (
                ("tensor/scalar", lambda scalar=scalar: torch.add(offset, scalar), offset + scalar),
                ("scalar/tensor", lambda scalar=scalar: torch.add(scalar, offset), scalar + offset),
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
            torch.add(special, -0.0), special + -0.0, case="IEEE special values"
        )
        nan_scalar = np.asarray([0x7FA1_2345], dtype=np.uint32).view(np.float32)[0]
        nan_tensor_bits = np.asarray(
            [0x7FC5_4321, 0x0000_0000, 0x8000_0000], dtype=np.uint32
        )
        nan_tensor = torch.tensor(memoryview(nan_tensor_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.add(nan_scalar, nan_tensor),
            torch.tensor(
                memoryview(
                    np.asarray(
                        [0x7FC5_4321, 0x7FE1_2345, 0x7FE1_2345], dtype=np.uint32
                    ).view(np.float32)
                )
            ),
            case="scalar/tensor NaN payloads",
        )
        self.assert_tensor_matches(
            torch.add(nan_tensor, nan_scalar),
            torch.tensor(
                memoryview(
                    np.asarray(
                        [0x7FE1_2345, 0x7FE1_2345, 0x7FE1_2345], dtype=np.uint32
                    ).view(np.float32)
                )
            ),
            case="tensor/scalar NaN payloads",
        )

    def test_scalar_autograd_no_grad_and_tensor_tensor_autograd_boundary(self):
        function_leaf = torch.tensor([[2.0, -3.0], [5.0, -7.0]], requires_grad=True)
        operator_leaf = torch.tensor([[2.0, -3.0], [5.0, -7.0]], requires_grad=True)

        torch.add(function_leaf.transpose(0, 1), 4.0).sum().backward()
        (operator_leaf.transpose(0, 1) + 4.0).sum().backward()
        self.assert_tensor_matches(
            function_leaf.grad, operator_leaf.grad, case="scalar gradient"
        )

        left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^add\(\): tensor/tensor autograd recording is not supported$",
        ):
            torch.add(left.transpose(0, 1), right.transpose(0, 1))

        with torch.no_grad():
            untracked = torch.add(left.transpose(0, 1), right.transpose(0, 1))
            expected_untracked = left.transpose(0, 1) + right.transpose(0, 1)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assert_tensor_matches(
            untracked,
            expected_untracked,
            case="tensor/tensor no_grad",
        )
        self.assertTrue(torch.add(left, 2.0).requires_grad)

    def test_torch_function_modes_receive_original_calls_and_can_forward(self):
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
            ("tensor/tensor", lambda: torch.add(left, right), (left, right), None),
            ("tensor/scalar", lambda: torch.add(left, 4.0), (left, 4.0), None),
            ("scalar/tensor", lambda: torch.add(4.0, left), (4.0, left), None),
            (
                "keyword defaults",
                lambda: torch.add(input=left, other=right, alpha=1, out=None),
                (),
                ("input", "other", "alpha", "out"),
            ),
            (
                "unsupported alpha",
                lambda: torch.add(left, right, alpha=2),
                (left, right),
                ("alpha",),
            ),
            (
                "unsupported out",
                lambda: torch.add(left, right, out=destination),
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
                actual = torch.add(input=left, other=4.0, alpha=1)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, left + 4.0, case="forwarded modes")

        for call in (
            lambda: torch.add([], right),
            lambda: torch.add(left, []),
            lambda: torch.add(left, right, dtype=torch.float32),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

    def test_operand_alpha_and_out_overrides_order_types_and_declining_errors(self):
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

        scalar_events = []

        class ScalarOverride(int):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                scalar_events.append((func, types))
                return marker

        self.assertIs(torch.add(native, ScalarOverride(4)), marker)
        self.assertEqual(scalar_events, [(torch.add, (ScalarOverride,))])

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

        self.assertIs(
            torch.add(input=native, other=native, alpha=AlphaOverride(), out=OutOverride()),
            marker,
        )
        label, function, dispatch_types, args, kwargs = keyword_events[0]
        self.assertEqual(label, "alpha")
        self.assertIs(function, torch.add)
        self.assertEqual(dispatch_types, (AlphaOverride, OutOverride))
        self.assertEqual(args, ())
        self.assertEqual(tuple(kwargs), ("input", "other", "alpha", "out"))

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
                'add() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.add(tensor),
                'add() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.add(tensor, tensor, tensor),
                "add() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.add(2, 3),
                "add(): scalar-scalar addition is not supported; at least one operand must be Tensor",
            ),
            (
                lambda: torch.add([], tensor),
                "add(): argument 'input' (position 1) must be Tensor, not list",
            ),
            (
                lambda: torch.add(tensor, []),
                "add(): argument 'other' (position 2) must be Tensor, not list",
            ),
            (
                lambda: torch.add(input=None, other=tensor),
                "add(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.add(tensor, tensor, input=tensor),
                "add() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.add(tensor, tensor, x2=tensor),
                "add() got an unexpected keyword argument 'x2'",
            ),
            (
                lambda: torch.add(tensor, tensor, dtype=torch.float32),
                "add() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.add(tensor, tensor, device=torch.device("cpu")),
                "add() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.add(tensor, tensor, alpha=[]),
                "add(): argument 'alpha' must be Number, not list",
            ),
            (
                lambda: torch.add(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.add(tensor, 2**64), "int too big to convert"),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        for alpha in (0, 2, 1.25, np.bool_(False), np.float32(2.0)):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^add\(\): non-default alpha is not supported$",
                ):
                    torch.add(tensor, tensor, alpha=alpha)
        for alpha in (False, True):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Boolean alpha only supported for Boolean results\.$",
                ):
                    torch.add(tensor, tensor, alpha=alpha)

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
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
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

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
        from torch_rs import add

        self.assertIs(add, function)
        self.assertIs(importlib.reload(torch._C), torch._C)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.add, function)

        for name in ("add_", "sub", "subtract"):
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
        for name in ("add", "add_"):
            self.assertFalse(hasattr(torch.Tensor, name))
            self.assertFalse(hasattr(tensor, name))


if __name__ == "__main__":
    unittest.main()
