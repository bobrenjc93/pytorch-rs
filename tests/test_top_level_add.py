import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
add(input, other, *, alpha=1, out=None) -> Tensor

Adds :attr:`other`, scaled by :attr:`alpha`, to :attr:`input`.

.. math::
    \\text{out}_i = \\text{input}_i + \\text{alpha} \\times \\text{other}_i

Supports :ref:`broadcasting to a common shape <broadcasting-semantics>`,
:ref:`type promotion <type-promotion-doc>`, and integer, float, and complex inputs.

Args:
    input (Tensor): the input tensor.
    other (Tensor or Number): the tensor or number to add to input.

Keyword args:
    alpha (Number): the multiplier for :attr:`other`.
    out (Tensor, optional): the output tensor.

Examples::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.0202,  1.0985,  1.3506, -0.6056])
    >>> torch.add(a, 20)
    tensor([20.0202, 21.0985, 21.3506, 19.3944])

    >>> b = torch.randn(4)
    >>> b
    tensor([-0.9732, -0.3497,  0.6245,  0.4022])
    >>> c = torch.randn(4, 1)
    >>> c
    tensor([[ 0.3743],
            [-1.7724],
            [-0.5811],
            [-0.8017]])
    >>> torch.add(b, c, alpha=10)
    tensor([[  2.7695,   3.3930,   4.3675,   4.1448],
            [-18.6971, -18.0736, -17.0991, -17.3214],
            [ -6.7845,  -6.1610,  -5.1865,  -5.4088],
            [ -8.9902,  -8.3667,  -7.3922,  -7.6145]])
"""


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

    def test_supported_scalar_calls_reuse_operator_values_layouts_and_edges(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        offset = strided[1]
        scalars = (
            True,
            -2,
            2**64 - 1,
            2.5,
            np.bool_(False),
            np.int64(3),
            np.uint64(2**63 - 1),
            np.float32(-0.0),
        )
        for scalar in scalars:
            for case, call, expected in (
                (
                    "tensor/scalar",
                    lambda scalar=scalar: torch.add(offset, scalar),
                    offset + scalar,
                ),
                (
                    "scalar/tensor",
                    lambda scalar=scalar: torch.add(scalar, offset),
                    scalar + offset,
                ),
                (
                    "canonical keywords",
                    lambda scalar=scalar: torch.add(input=offset, other=scalar),
                    offset + scalar,
                ),
                (
                    "legacy aliases",
                    lambda scalar=scalar: torch.add(x=offset, x2=scalar),
                    offset + scalar,
                ),
                (
                    "explicit defaults",
                    lambda scalar=scalar: torch.add(
                        x1=scalar, other=offset, alpha=1.0, out=None
                    ),
                    scalar + offset,
                ),
            ):
                self.assert_tensor_matches(
                    call(), expected, case=(case, type(scalar).__name__, scalar)
                )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_tensor_matches(
            torch.add(empty, -7.0), empty + -7.0, case="strided empty"
        )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            torch.add(-0.0, special), -0.0 + special, case="IEEE special values"
        )

    def test_autograd_no_grad_and_empty_gradients_reuse_scalar_addition(self):
        function_leaf = torch.tensor(
            np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist(),
            requires_grad=True,
        )
        operator_leaf = torch.tensor(
            np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist(),
            requires_grad=True,
        )
        function_output = torch.add(4.0, function_leaf.transpose(0, 1)[1])
        operator_output = 4.0 + operator_leaf.transpose(0, 1)[1]
        self.assert_tensor_matches(function_output, operator_output, case="tracked view")
        function_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            function_leaf.grad, operator_leaf.grad, case="view gradient"
        )

        function_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.add(function_empty.transpose(0, 2), -3.0).sum().backward()
        (operator_empty.transpose(0, 2) + -3.0).sum().backward()
        self.assert_tensor_matches(
            function_empty.grad, operator_empty.grad, case="empty gradient"
        )

        no_grad_input = torch.tensor([1.0, 2.0], requires_grad=True)
        with torch.no_grad():
            untracked = torch.add(no_grad_input, 5.0)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(torch.add(no_grad_input, 5.0).requires_grad)

    def test_modes_observe_valid_generated_calls_before_native_limits(self):
        tensor = torch.tensor([2.0])
        other = torch.tensor([3.0])
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
            (lambda: torch.add(tensor, 4.0), (tensor, 4.0), None),
            (lambda: torch.add(4.0, tensor), (4.0, tensor), None),
            (lambda: torch.add(tensor, other), (tensor, other), None),
            (
                lambda: torch.add(input=tensor, other=4.0, alpha=1, out=None),
                (),
                ("input", "other", "alpha", "out"),
            ),
            (
                lambda: torch.add(tensor, 4.0, alpha=2, out=destination),
                (tensor, 4.0),
                ("alpha", "out"),
            ),
        )
        for call, expected_args, expected_keywords in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            function, dispatch_types, args, kwargs = mode.calls[0]
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
                actual = torch.add(input=4.0, other=tensor, alpha=1, out=None)
        self.assertEqual(order, ["upper", "lower"])
        self.assert_tensor_matches(actual, 4.0 + tensor, case="forwarded modes")

        for call in (
            lambda: torch.add([], tensor),
            lambda: torch.add(tensor, []),
            lambda: torch.add(tensor, 4.0, out=[]),
            lambda: torch.add(tensor, 4.0, alpha=[]),
        ):
            mode = RecordingMode()
            with mode:
                with self.assertRaises(TypeError):
                    call()
            self.assertEqual(mode.calls, [])

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

        scalar_events = []

        class ScalarOverride(int):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                scalar_events.append((func, types))
                return marker

        self.assertIs(torch.add(ScalarOverride(4), native), marker)
        self.assertEqual(scalar_events, [(torch.add, (ScalarOverride,))])

        option_events = []

        class AlphaOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                option_events.append(("alpha", func, types, args, kwargs))
                return marker

        class OutOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                option_events.append(("out", func, types, args, kwargs))
                return marker

        self.assertIs(torch.add(native, 1.0, alpha=AlphaOverride()), marker)
        self.assertIs(torch.add(native, 1.0, out=OutOverride()), marker)
        self.assertEqual(option_events[0][2], (AlphaOverride,))
        self.assertEqual(option_events[1][2], (OutOverride,))

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(torch.add(BaseOverride(), DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

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

    def test_errors_metadata_pickling_exports_and_unsupported_surface(self):
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
                lambda: torch.add(2),
                'add() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.add(tensor, 1.0, 2.0),
                "add() takes 2 positional arguments but 3 were given",
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
                lambda: torch.add(tensor, 1.0, input=tensor),
                "add() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.add(tensor, 1.0, x2=tensor),
                "add() got multiple values for argument 'other'",
            ),
            (
                lambda: torch.add(tensor, 1.0, dtype=torch.float32),
                "add() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.add(tensor, 1.0, device=torch.device("cpu")),
                "add() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.add(foo=tensor),
                'add() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.add(tensor, 1.0, extra=True),
                "add() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.add(tensor, 1 + 2j),
                "add(): argument 'other' (position 2) must be Tensor, not complex",
            ),
            (
                lambda: torch.add(tensor, np.uint64(2**63)),
                "an integer is required",
            ),
            (lambda: torch.add(tensor, 2**64), "int too big to convert"),
            (
                lambda: torch.add(-(2**63) - 1, tensor),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            NotImplementedError,
            r"^add\(\): Tensor/Tensor operands are not supported; exactly one "
            r"operand must be a Tensor and the other a Number$",
        ):
            torch.add(tensor, tensor)
        with self.assertRaisesRegex(
            TypeError,
            r"^add\(\): scalar-scalar addition is not supported; at least one "
            r"operand must be Tensor$",
        ):
            torch.add(2, 3)
        with self.assertRaisesRegex(
            NotImplementedError, r"^add\(\): alpha values other than 1 are not supported$"
        ):
            torch.add(tensor, 2.0, alpha=2)
        for alpha in (True, False, np.bool_(True), np.bool_(False)):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Boolean alpha only supported for Boolean results\.$",
                ):
                    torch.add(tensor, 2.0, alpha=alpha)

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(tensor, 2.0, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaisesRegex(
            TypeError, r"^add\(\): argument 'out' must be Tensor, not list$"
        ):
            torch.add(tensor, 2.0, out=[])
        with self.assertRaisesRegex(
            TypeError, r"^add\(\): argument 'alpha' must be Number, not list$"
        ):
            torch.add(tensor, 2.0, alpha=[])

        self.assertFalse(hasattr(torch.Tensor, "add"))

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


if __name__ == "__main__":
    unittest.main()
