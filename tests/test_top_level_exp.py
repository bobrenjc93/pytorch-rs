import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """
exp(input, *, out=None) -> Tensor

Returns a new tensor with the exponential of the elements
of the input tensor :attr:`input`.

.. math::
    y_{i} = e^{x_{i}}

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> torch.exp(torch.tensor([0, math.log(2.)]))
    tensor([ 1.,  2.])
"""


class TopLevelExpTests(unittest.TestCase):
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
            ("positional", lambda: torch.exp(source)),
            ("input", lambda: torch.exp(input=source)),
            ("x", lambda: torch.exp(x=source)),
            ("a", lambda: torch.exp(a=source)),
            ("x1", lambda: torch.exp(x1=source)),
            ("out none", lambda: torch.exp(source, out=None)),
            ("alias and out none", lambda: torch.exp(x=source, out=None)),
        )

    def test_supported_calls_reuse_tensor_exp_values_and_layouts(self):
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
                0xC2C8_0000,
                0xC2D0_0000,
                0x42B0_0000,
                0x42B2_0000,
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
            expected = source.exp()
            for form, call in self.supported_calls(source):
                self.assert_matches_method(call(), expected, case=(case, form))

    def test_extreme_empty_metadata_error_matches_tensor_exp(self):
        source = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            source.exp()
        for form, call in self.supported_calls(source):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError, "Stride calculation overflowed"
                ):
                    call()

    def test_no_grad_is_supported_but_recording_remains_explicitly_unsupported(self):
        leaf = torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        # Tensor.exp retains its existing behavior; only the new top-level
        # wrapper guards the missing autograd edge.
        method_output = source.exp()
        self.assertFalse(method_output.requires_grad)
        self.assertTrue(method_output.is_leaf)

        for form, call in self.supported_calls(source):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^exp\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    output = call()
                self.assert_matches_method(
                    output, method_output, case=(form, "no_grad")
                )

        detached = source.detach()
        self.assert_matches_method(
            torch.exp(detached), detached.exp(), case="detached input"
        )

    def test_concrete_out_tensor_is_rejected_without_mutation(self):
        source = torch.tensor([0.0, 1.0])
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.exp(source, out=destination)),
            ("keyword", lambda: torch.exp(input=source, out=destination)),
            ("alias", lambda: torch.exp(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^exp\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])

        self.assert_matches_method(
            torch.exp(source, out=None), source.exp(), case="explicit out none"
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
            self.assertIs(torch.exp(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.exp)
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
        self.assertIs(torch.exp(input_override), marker)
        out_override = Override()
        self.assertIs(torch.exp(torch.tensor([1.0]), out=out_override), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.exp)
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
            self.assertIs(torch.exp(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.exp
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "exp")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.exp")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method exp of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.exp, function)
        for action in (
            lambda: setattr(owner, "exp", None),
            lambda: delattr(owner, "exp"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.exp, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("exp"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["exp"], function)

    def test_binding_and_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.exp(),
                'exp() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.exp(tensor, tensor),
                "exp() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.exp(tensor, input=tensor),
                "exp() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.exp(out=tensor),
                'exp() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.exp(extra=tensor),
                'exp() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.exp(1, extra=True),
                "exp(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.exp(input=[]),
                "exp(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.exp(tensor, out=[]),
                "exp(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.exp(tensor, extra=True, out=[]),
                "exp(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.exp(tensor, extra=True),
                "exp() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.exp(input=tensor, a=tensor),
                "exp() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.exp(a=tensor, x=tensor, out=None),
                "exp() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.exp(x=tensor, a=tensor, out=None),
                "exp() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.exp(np.zeros((2, 3), dtype=np.float32)),
                "exp(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
