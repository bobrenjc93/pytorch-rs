import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


TENSOR_DOC = """
cos() -> Tensor

See :func:`torch.cos`
"""

FUNCTION_DOC = """
cos(input, *, out=None) -> Tensor

Returns a new tensor with the cosine of the elements of :attr:`input` given in radians.

.. math::
    \\text{out}_{i} = \\cos(\\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 1.4309,  1.2706, -0.8562,  0.9796])
    >>> torch.cos(a)
    tensor([ 0.1395,  0.2957,  0.6553,  0.5574])
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

    @staticmethod
    def supported_calls(source):
        return (
            ("positional", lambda: torch.cos(source)),
            ("input", lambda: torch.cos(input=source)),
            ("x", lambda: torch.cos(x=source)),
            ("a", lambda: torch.cos(a=source)),
            ("x1", lambda: torch.cos(x1=source)),
            ("out none", lambda: torch.cos(source, out=None)),
            ("alias and out none", lambda: torch.cos(x=source, out=None)),
        )

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
                0x0080_0000,
                0x8080_0000,
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
            ("noncontiguous", strided),
            ("numerical edges", torch.tensor(memoryview(special_bits.view(np.float32)))),
        )

        for case, source in cases:
            method_output = source.cos()
            top_level_output = torch.cos(input=source, out=None)
            self.assert_tensor_bits_match(
                top_level_output, method_output, case=(case, "top-level")
            )
            self.assertFalse(method_output.is_set_to(source))
            self.assertFalse(top_level_output.is_set_to(source))
            self.assertFalse(method_output.is_set_to(top_level_output))

        edge_values = np.asarray(cases[-1][1].cos(), dtype=np.float32).reshape(-1)
        np.testing.assert_array_equal(
            edge_values[:6].view(np.uint32),
            np.asarray([1.0] * 6, dtype=np.float32).view(np.uint32),
        )
        self.assertTrue(np.isnan(edge_values[6:]).all())

    def test_autograd_through_full_sum_uses_negative_sine_vjp(self):
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xC000_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        leaf = torch.tensor(memoryview(input_bits.view(np.float32)), requires_grad=True)
        leaf.cos().sum().backward()
        self.assert_tensor_bits_match(
            leaf.grad,
            -leaf.detach().sin(),
            case="full sum gradient",
        )

        base_values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        function_leaf = torch.tensor(base_values.tolist(), requires_grad=True)
        method_leaf = torch.tensor(base_values.tolist(), requires_grad=True)
        torch.cos(function_leaf.transpose(0, 2), out=None).sum().backward()
        method_leaf.transpose(0, 2).cos().sum().backward()
        self.assert_tensor_bits_match(
            function_leaf.grad,
            method_leaf.grad,
            case="noncontiguous top-level gradient",
        )

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        torch.cos(empty.transpose(0, 2)[1], out=None).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.stride(), (3, 3, 1))

    def test_graph_boundaries_and_higher_order_gradients_remain_unsupported(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        with torch.no_grad():
            untracked = leaf.transpose(0, 1).cos()
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(leaf.cos().requires_grad)

        detached = leaf.detach().transpose(0, 1)
        self.assert_tensor_bits_match(
            torch.cos(detached), detached.cos(), case="detached input"
        )

        loss = leaf.cos().sum()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            loss.backward(create_graph=True)
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertIsNotNone(leaf.grad)
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

    def test_tensorbase_descriptor_metadata_and_method_unsupported_forms(self):
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
        self.assertEqual(descriptor.__doc__, TENSOR_DOC)
        self.assertEqual(bound.__doc__, TENSOR_DOC)
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
                lambda: tensor.cos(input=tensor),
                (
                    "Tensor.cos() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.cos() takes no keyword arguments"
                ),
            ),
            (lambda: descriptor(), "unbound method TensorBase.cos() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'cos' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertFalse(hasattr(torch.Tensor, "cos_"))
        self.assertFalse(hasattr(tensor, "cos_"))
        with self.assertRaises(TypeError):
            tensor.cos(out=None)

    def test_top_level_modes_overrides_and_concrete_out_boundaries(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        destination = torch.tensor([17.0])
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
        self.assertEqual(len(mode.calls), 1)
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

        self.assertIs(torch.cos(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        for form, call in (
            ("positional", lambda: torch.cos(tensor, out=destination)),
            ("keyword", lambda: torch.cos(input=tensor, out=destination)),
            ("alias", lambda: torch.cos(x=tensor, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^cos\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0])

    def test_callable_metadata_imports_exports_reload_copy_and_pickle(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.cos
        wildcard_namespace = {}
        explicit_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        exec("from torch_rs import cos", explicit_namespace)

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
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.cos, function)
        self.assertIs(native.cos, function)
        self.assertEqual(package.__all__.count("cos"), 1)
        self.assertNotIn("_VariableFunctionsClass", package.__all__)
        self.assertFalse(hasattr(package, "_VariableFunctionsClass"))
        self.assertIs(wildcard_namespace["cos"], function)
        self.assertIs(explicit_namespace["cos"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.cos, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.cos, function)
        self.assertEqual(package.__all__.count("cos"), 1)

    def test_binding_and_unsupported_scope_are_documented(self):
        tensor = torch.tensor([1.0])
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
                lambda: torch.cos(extra=tensor),
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
                lambda: torch.cos(tensor, extra=True, out=[]),
                "cos(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.cos(tensor, extra=True),
                "cos() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.cos(input=tensor, a=tensor),
                "cos() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.cos(a=tensor, x=tensor, out=None),
                "cos() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.cos(x=tensor, a=tensor, out=None),
                "cos() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.cos(np.zeros((2, 3), dtype=np.float32)),
                "cos(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            rf"tensor subclass <class '{DecliningOverride.__module__}\..*DecliningOverride'>",
        ):
            torch.cos(DecliningOverride())

        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^tensor\(\): argument 'dtype' must be torch.dtype, not object$",
        ):
            torch.tensor([1.0], dtype=object())
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")
        with self.assertRaisesRegex(
            TypeError, r"^type 'torch_rs\.Tensor' is not an acceptable base type$"
        ):
            type("TensorSubclass", (torch.Tensor,), {})


if __name__ == "__main__":
    unittest.main()
