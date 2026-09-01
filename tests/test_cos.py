import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = """
cos() -> Tensor

See :func:`torch.cos`
"""

FUNCTION_DOC = """
cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine of the elements in the :attr:`input` tensor.

.. math::
    \\text{out}_{i} = \\cos(\\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([-0.5461,  0.1347, -2.7266, -0.2746])
    >>> torch.cos(a)
    tensor([ 0.8548,  0.9909, -0.9146,  0.9626])
"""


class CosTests(unittest.TestCase):
    def assert_tensor_bits_match(self, actual, expected, *, case):
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
            ("positional", lambda: torch.cos(source)),
            ("input", lambda: torch.cos(input=source)),
            ("x", lambda: torch.cos(x=source)),
            ("a", lambda: torch.cos(a=source)),
            ("x1", lambda: torch.cos(x1=source)),
            ("out none", lambda: torch.cos(source, out=None)),
            ("alias and out none", lambda: torch.cos(x=source, out=None)),
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

    def test_method_and_top_level_preserve_values_layouts_and_fresh_storage(self):
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
            ("contiguous", base),
            ("offset", strided[1]),
            ("strided", strided),
            (
                "numerical edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

        for case, source in cases:
            method_output = source.cos()
            self.assertFalse(method_output.is_set_to(source))
            for form, call in self.supported_calls(source):
                output = call()
                self.assert_tensor_bits_match(output, method_output, case=(case, form))
                self.assertFalse(output.is_set_to(source))

    def test_autograd_vjp_accumulation_no_grad_and_higher_order_boundary(self):
        scalar = torch.tensor(1.5, requires_grad=True)
        scalar.cos().backward()
        np.testing.assert_allclose(
            np.asarray(scalar.grad),
            np.asarray(-np.sin(np.float32(1.5)), dtype=np.float32),
            rtol=2.0e-6,
            atol=0.0,
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.cos().sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), empty.tolist())

        values = np.asarray([[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], dtype=np.float32)
        weights = np.asarray([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]], dtype=np.float32)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        loss = (leaf.transpose(0, 1).cos() * torch.tensor(weights.tolist())).sum()
        loss.backward()
        np.testing.assert_allclose(
            np.asarray(leaf.grad),
            -np.sin(values) * weights.T,
            rtol=2.0e-6,
            atol=0.0,
        )

        accumulated = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        accumulated.cos().sum().backward()
        first = np.asarray(accumulated.grad).copy()
        accumulated.cos().sum().backward()
        np.testing.assert_array_equal(np.asarray(accumulated.grad), first * 2.0)

        tracked = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        with torch.no_grad():
            untracked = tracked.cos()
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertFalse(untracked.is_set_to(tracked))
        self.assertTrue(tracked.cos().requires_grad)

        higher_order = torch.tensor([0.5], requires_grad=True)
        higher_order_loss = higher_order.cos().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        np.testing.assert_allclose(
            np.asarray(higher_order.grad),
            -np.sin(np.asarray([0.5], dtype=np.float32)),
            rtol=2.0e-6,
            atol=0.0,
        )

    def test_grad_fn_name_and_freed_graph_errors_are_specific_to_cos(self):
        probability = torch.tensor([4.0], requires_grad=True).cos()
        with self.assertRaisesRegex(ValueError, r"grad_fn=<CosBackward0>"):
            torch.nn.functional.dropout(
                torch.tensor([1.0]), p=probability, training=False
            )

        freed = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
        loss = freed.cos().sum()
        loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([0.5])
        descriptor = inspect.getattr_static(torch.Tensor, "cos")
        bound = tensor.cos

        self.assertIs(torch.Tensor.cos, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'cos' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "cos")
        self.assertEqual(descriptor.__qualname__, "TensorBase.cos")
        self.assertEqual(bound.__name__, "cos")
        self.assertEqual(bound.__qualname__, "Tensor.cos")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)

        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        cases = (
            (lambda: tensor.cos(1), "TensorBase.cos() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.cos() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.cos() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.cos(1, 2),
                "TensorBase.cos() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.cos(input=tensor),
                (
                    "Tensor.cos() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.cos() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.cos() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.cos() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.cos() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'cos' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.cos() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_tensorbase_torch_function_modes_dispatch_before_native_limits(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "cos")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tensor.cos()
        self.assertIs(result, marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            extreme.cos()
        bypass = RecordingMode(marker)
        with bypass:
            self.assertIs(extreme.cos(), marker)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.cos()
        self.assertEqual(order, ["upper", "lower"])
        forwarded.sum().backward()
        np.testing.assert_allclose(
            np.asarray(tensor.grad),
            -np.sin(np.asarray([0.5], dtype=np.float32)),
            rtol=2.0e-6,
            atol=0.0,
        )

    def test_top_level_torch_function_modes_and_overrides_dispatch_first(self):
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
            self.assertIs(torch.cos(input=tensor, out=destination), marker)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.cos)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.cos(Override()), marker)
        self.assertIs(torch.cos(torch.tensor([1.0]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for function, dispatch_types, _, _ in override_calls:
            self.assertIs(function, torch.cos)
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
            self.assertIs(torch.cos(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

    def test_callable_metadata_documentation_pickling_and_exports(self):
        function = torch.cos
        self.assertIs(function, torch._C.cos)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "cos")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.cos")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method cos of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.cos, function)
        for action in (
            lambda: setattr(owner, "cos", None),
            lambda: delattr(owner, "cos"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.cos, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("cos"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["cos"], function)

    def test_unsupported_arguments_out_dtype_device_and_inplace_forms(self):
        tensor = torch.tensor([0.5])
        destination = torch.tensor([17.0])

        with self.assertRaisesRegex(
            RuntimeError, r"^cos\(\): the 'out' argument is not supported$"
        ):
            torch.cos(tensor, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        self.assert_tensor_bits_match(
            torch.cos(tensor, out=None), tensor.cos(), case="explicit out none"
        )

        cases = (
            (
                lambda: torch.cos(),
                'cos() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.cos(tensor, tensor),
                "cos() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.cos(tensor, input=tensor),
                "cos() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.cos(out=tensor),
                'cos() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.cos(1, extra=True),
                "cos(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.cos(input=[]),
                "cos(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.cos(tensor, out=[]),
                "cos(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.cos(tensor, dtype=torch.float32),
                "cos() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.cos(tensor, device=torch.device("cpu")),
                "cos() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch.Tensor, "cos_"))
        self.assertFalse(hasattr(tensor, "cos_"))
        self.assertFalse(hasattr(torch, "cos_"))
        self.assertNotIn("cos_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.cos(out=None)


if __name__ == "__main__":
    unittest.main()
