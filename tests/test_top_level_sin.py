import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
sin(input, *, out=None) -> Tensor

Returns a new tensor with the sine of the elements in the :attr:`input` tensor,
where each value in this input tensor is in radians.

.. math::
    \\text{out}_{i} = \\sin(\\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> torch.sin(a)
    tensor([-0.5194,  0.1343, -0.4032, -0.2711])
"""


class TopLevelSinTests(unittest.TestCase):
    def assert_matches_method(self, actual, expected, *, case):
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

    def supported_calls(self, source):
        return (
            ("positional", lambda: torch.sin(source)),
            ("input", lambda: torch.sin(input=source)),
            ("x", lambda: torch.sin(x=source)),
            ("a", lambda: torch.sin(a=source)),
            ("x1", lambda: torch.sin(x1=source)),
            ("out none", lambda: torch.sin(source, out=None)),
            ("alias and out none", lambda: torch.sin(x=source, out=None)),
        )

    @staticmethod
    def make_autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(1.5, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1]

        leaf = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf.transpose(0, 2)[1]
        if case == "strided":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown autograd case: {case}")

    def test_supported_calls_reuse_tensor_sin_values_and_layouts(self):
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x3F00_0000,
                0xBF00_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0xD015_02F9,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
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
            ("numerical edges", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

        for case, source in cases:
            expected = source.sin()
            for form, call in self.supported_calls(source):
                self.assert_matches_method(call(), expected, case=(case, form))

    def test_extreme_empty_metadata_error_matches_tensor_sin(self):
        source = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            source.sin()
        for form, call in self.supported_calls(source):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError, "Stride calculation overflowed"
                ):
                    call()

    def test_autograd_scalar_empty_offset_and_strided_reuse_tensor_sine_vjp(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case in ("scalar", "empty", "offset", "strided"):
            for form in forms:
                function_leaf, function_source = self.make_autograd_case(case)
                method_leaf, method_source = self.make_autograd_case(case)
                output = dict(self.supported_calls(function_source))[form]()
                method_output = method_source.sin()

                self.assert_matches_method(
                    output, method_output, case=(case, form, "output")
                )
                output.sum().backward()
                method_output.sum().backward()
                self.assert_matches_method(
                    function_leaf.grad,
                    method_leaf.grad,
                    case=(case, form, "gradient"),
                )

    def test_gradient_accumulation_no_grad_and_freed_graph_reuse_method_path(self):
        values = [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]]
        weights = [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]
        function_leaf = torch.tensor(values, requires_grad=True)
        method_leaf = torch.tensor(values, requires_grad=True)
        function_source = function_leaf.transpose(0, 1)
        method_source = method_leaf.transpose(0, 1)

        function_loss = (
            torch.sin(function_source, out=None) * torch.tensor(weights)
        ).sum()
        method_loss = (method_source.sin() * torch.tensor(weights)).sum()
        function_loss.backward()
        method_loss.backward()
        self.assert_matches_method(
            function_leaf.grad, method_leaf.grad, case="first gradient"
        )

        torch.sin(input=function_source).sum().backward()
        method_source.sin().sum().backward()
        self.assert_matches_method(
            function_leaf.grad, method_leaf.grad, case="accumulated gradient"
        )

        with self.assertRaises(RuntimeError) as function_raised:
            function_loss.backward()
        with self.assertRaises(RuntimeError) as method_raised:
            method_loss.backward()
        self.assertEqual(str(function_raised.exception), str(method_raised.exception))

        no_grad_function_leaf = torch.tensor(values, requires_grad=True)
        no_grad_method_leaf = torch.tensor(values, requires_grad=True)
        with torch.no_grad():
            function_output = torch.sin(
                no_grad_function_leaf.transpose(0, 1), out=None
            )
            method_output = no_grad_method_leaf.transpose(0, 1).sin()
        self.assert_matches_method(
            function_output, method_output, case="no_grad output"
        )
        self.assertIsNone(no_grad_function_leaf.grad)
        self.assertTrue(torch.sin(no_grad_function_leaf).requires_grad)

        detached = no_grad_function_leaf.detach().transpose(0, 1)
        self.assert_matches_method(
            torch.sin(detached), detached.sin(), case="detached input"
        )

    def test_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([0.0, 1.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.sin(source, out=destination)),
            ("keyword", lambda: torch.sin(input=source, out=destination)),
            ("alias", lambda: torch.sin(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^sin\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])

        self.assert_matches_method(
            torch.sin(source, out=None), source.sin(), case="explicit out none"
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
            self.assertIs(torch.sin(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.sin)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        input_override = Override()
        self.assertIs(torch.sin(input_override), marker)
        out_override = Override()
        self.assertIs(torch.sin(torch.tensor([1.0]), out=out_override), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.sin)
            self.assertEqual(dispatch_types, (Override,))

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
            self.assertIs(torch.sin(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.sin
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sin")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sin")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method sin of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sin, function)
        for action in (
            lambda: setattr(owner, "sin", None),
            lambda: delattr(owner, "sin"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.sin, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sin"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sin"], function)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.sin(),
                'sin() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sin(tensor, tensor),
                "sin() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.sin(tensor, input=tensor),
                "sin() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.sin(out=tensor),
                'sin() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sin(extra=tensor),
                'sin() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sin(1, extra=True),
                "sin(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sin(input=[]),
                "sin(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.sin(tensor, out=[]),
                "sin(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sin(tensor, extra=True, out=[]),
                "sin(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sin(tensor, extra=True),
                "sin() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.sin(input=tensor, a=tensor),
                "sin() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sin(a=tensor, x=tensor, out=None),
                "sin() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sin(x=tensor, a=tensor, out=None),
                "sin() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.sin(np.zeros((2, 3), dtype=np.float32)),
                "sin(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
