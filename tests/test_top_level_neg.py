import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
neg(input, *, out=None) -> Tensor

Returns a new tensor with the negative of the elements of :attr:`input`.

.. math::
    \\text{out} = -1 \\times \\text{input}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(5)
    >>> a
    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])
    >>> torch.neg(a)
    tensor([-0.0090,  0.2262,  0.0682,  0.2866, -0.3940])
"""


class TopLevelNegTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.neg(source)),
            ("input", lambda: torch.neg(input=source)),
            ("x", lambda: torch.neg(x=source)),
            ("a", lambda: torch.neg(a=source)),
            ("x1", lambda: torch.neg(x1=source)),
            ("out none", lambda: torch.neg(source, out=None)),
            ("alias and out none", lambda: torch.neg(x=source, out=None)),
        )

    @staticmethod
    def make_autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-0.0, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        if case == "offset":
            return leaf, strided[1]
        if case == "strided":
            return leaf, strided
        raise AssertionError(f"unknown autograd case: {case}")

    def test_supported_calls_reuse_tensor_neg_values_bits_and_layouts(self):
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF85_4321,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("strided", strided),
            ("float bits", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

        for case, source in cases:
            expected = source.neg()
            for form, call in self.supported_calls(source):
                self.assert_tensor_matches(call(), expected, case=(case, form))

    def test_extreme_empty_metadata_errors_reuse_tensor_neg_path(self):
        source = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            source.neg()
        for form, call in self.supported_calls(source):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError, "Stride calculation overflowed"
                ):
                    call()

    def test_autograd_no_grad_and_repeated_backward_reuse_tensor_negation(self):
        forms = tuple(form for form, _ in self.supported_calls(torch.tensor(0.0)))
        for case in ("scalar", "empty", "offset", "strided"):
            for form in forms:
                function_leaf, function_source = self.make_autograd_case(case)
                method_leaf, method_source = self.make_autograd_case(case)
                output = dict(self.supported_calls(function_source))[form]()
                method_output = method_source.neg()

                self.assert_tensor_matches(
                    output, method_output, case=(case, form, "output")
                )
                output.sum().backward()
                method_output.sum().backward()
                self.assert_tensor_matches(
                    function_leaf.grad,
                    method_leaf.grad,
                    case=(case, form, "gradient"),
                )

        function_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        method_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        function_output = torch.neg(function_leaf.transpose(0, 1), out=None)
        method_output = method_leaf.transpose(0, 1).neg()
        function_loss = function_output.sum()
        method_loss = method_output.sum()
        function_loss.backward()
        method_loss.backward()
        function_loss.backward()
        method_loss.backward()
        self.assert_tensor_matches(
            function_leaf.grad, method_leaf.grad, case="repeated backward"
        )

        no_grad_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            no_grad_output = torch.neg(no_grad_leaf.transpose(0, 1), out=None)
        self.assertFalse(no_grad_output.requires_grad)
        self.assertTrue(no_grad_output.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)
        self.assertTrue(torch.neg(no_grad_leaf).requires_grad)

    def test_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([1.0, -2.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.neg(source, out=destination)),
            ("keyword", lambda: torch.neg(input=source, out=destination)),
            ("alias", lambda: torch.neg(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^neg\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])

        self.assert_tensor_matches(
            torch.neg(source, out=None), source.neg(), case="explicit out none"
        )

    def test_modes_and_overrides_observe_calls_before_native_limits(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.neg(input=tensor, out=destination), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.neg)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.neg(Override()), marker)
        self.assertIs(torch.neg(tensor, out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.neg)
            self.assertEqual(dispatch_types, (Override,))

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

        self.assertIs(torch.neg(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.neg(input=tensor, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [-1.0])
        forwarded.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [-1.0])

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.neg(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_pickling_exports_and_unsupported_aliases(self):
        function = torch.neg
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "neg")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.neg")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method neg of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.neg, function)
        for action in (
            lambda: setattr(owner, "neg", None),
            lambda: delattr(owner, "neg"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.neg, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("neg"), 1)
        self.assertFalse(hasattr(torch, "negative"))
        self.assertFalse(hasattr(torch.Tensor, "neg_"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["neg"], function)
        self.assertNotIn("negative", wildcard_namespace)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (lambda: torch.neg(), 'neg() missing 1 required positional arguments: "input"'),
            (
                lambda: torch.neg(tensor, tensor),
                "neg() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.neg(tensor, input=tensor),
                "neg() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.neg(out=tensor),
                'neg() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.neg(1, extra=True),
                "neg(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.neg(input=[]),
                "neg(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.neg(tensor, out=[]),
                "neg(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.neg(tensor, extra=True, out=[]),
                "neg(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.neg(tensor, extra=True),
                "neg() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.neg(input=tensor, a=tensor),
                "neg() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.neg(a=tensor, x=tensor, out=None),
                "neg() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.neg(x=tensor, a=tensor, out=None),
                "neg() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.neg(np.zeros((2, 3), dtype=np.float32)),
                "neg(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
