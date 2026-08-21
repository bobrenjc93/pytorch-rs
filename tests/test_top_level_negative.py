import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
negative(input, *, out=None) -> Tensor

Alias for :func:`torch.neg`
"""


class TopLevelNegativeTests(unittest.TestCase):
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
            ("positional", lambda: torch.negative(source)),
            ("input", lambda: torch.negative(input=source)),
            ("x", lambda: torch.negative(x=source)),
            ("a", lambda: torch.negative(a=source)),
            ("x1", lambda: torch.negative(x1=source)),
            ("out none", lambda: torch.negative(source, out=None)),
            ("alias and out none", lambda: torch.negative(x=source, out=None)),
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

    def test_supported_calls_reuse_tensor_negative_values_bits_and_layouts(self):
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
            expected = source.negative()
            for form, call in self.supported_calls(source):
                self.assert_tensor_matches(call(), expected, case=(case, form))

    def test_extreme_empty_metadata_errors_reuse_tensor_negative_path(self):
        source = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            source.negative()
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
                method_output = method_source.negative()

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
        function_output = torch.negative(function_leaf.transpose(0, 1), out=None)
        method_output = method_leaf.transpose(0, 1).negative()
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
            no_grad_output = torch.negative(no_grad_leaf.transpose(0, 1), out=None)
        self.assertFalse(no_grad_output.requires_grad)
        self.assertTrue(no_grad_output.is_leaf)
        self.assertIsNone(no_grad_leaf.grad)
        self.assertTrue(torch.negative(no_grad_leaf).requires_grad)

    def test_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([1.0, -2.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.negative(source, out=destination)),
            ("keyword", lambda: torch.negative(input=source, out=destination)),
            ("alias", lambda: torch.negative(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^negative\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])

        self.assert_tensor_matches(
            torch.negative(source, out=None), source.negative(), case="explicit out none"
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
            self.assertIs(torch.negative(input=tensor, out=destination), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.negative)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.negative(Override()), marker)
        self.assertIs(torch.negative(tensor, out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.negative)
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

        self.assertIs(torch.negative(BaseOverride(), out=DerivedOverride()), marker)
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
                forwarded = torch.negative(input=tensor, out=None)
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
            self.assertIs(torch.negative(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_pickling_exports_and_unsupported_inplace_aliases(self):
        function = torch.negative
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "negative")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.negative")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method negative of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.negative, function)
        for action in (
            lambda: setattr(owner, "negative", None),
            lambda: delattr(owner, "negative"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.negative, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("negative"), 1)
        self.assertIsNot(function, torch.neg)
        self.assertIsNot(owner.negative, owner.neg)
        self.assertFalse(hasattr(torch.Tensor, "neg_"))
        self.assertFalse(hasattr(torch.Tensor, "negative_"))
        self.assertFalse(hasattr(torch, "neg_"))
        self.assertFalse(hasattr(torch, "negative_"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["negative"], function)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.negative(),
                'negative() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.negative(tensor, tensor),
                "negative() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.negative(tensor, input=tensor),
                "negative() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.negative(out=tensor),
                'negative() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.negative(1, extra=True),
                "negative(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.negative(input=[]),
                "negative(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.negative(tensor, out=[]),
                "negative(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.negative(tensor, extra=True, out=[]),
                "negative(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.negative(tensor, extra=True),
                "negative() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.negative(input=tensor, a=tensor),
                "negative() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.negative(a=tensor, x=tensor, out=None),
                "negative() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.negative(x=tensor, a=tensor, out=None),
                "negative() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.negative(np.zeros((2, 3), dtype=np.float32)),
                "negative(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
