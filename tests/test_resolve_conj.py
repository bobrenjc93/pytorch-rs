import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


METHOD_DOC = "\nresolve_conj() -> Tensor\n\nSee :func:`torch.resolve_conj`\n"
FUNCTION_DOC = (
    "\nresolve_conj(input) -> Tensor\n\n"
    "Returns a new tensor with materialized conjugation if :attr:`input`'s "
    "conjugate bit is set to `True`,\n"
    "else returns :attr:`input`. The output tensor will always have its conjugate "
    "bit set to `False`.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n"
    "    >>> y = x.conj()\n"
    "    >>> y.is_conj()\n"
    "    True\n"
    "    >>> z = y.resolve_conj()\n"
    "    >>> z\n"
    "    tensor([-1 - 1j, -2 - 2j, 3 + 3j])\n"
    "    >>> z.is_conj()\n"
    "    False\n"
)


class TensorResolveConjTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 2.0).transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )

        self.assertFalse(strided_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        return (
            leaf,
            tracked,
            (
                ("scalar", torch.tensor(-3.5)),
                ("empty", torch.zeros((2, 0, 3))),
                ("eager negative", source.neg()),
                ("strided view", strided_view),
                ("offset strided view", offset_view),
                ("extreme empty view", extreme_empty),
                (
                    "signed zeros and non-finites",
                    torch.tensor(memoryview(special_bits.view(np.float32))),
                ),
                ("autograd leaf", leaf),
                ("autograd non-leaf view", tracked),
                ("detached autograd view", tracked.detach()),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

    def top_level_calls(self, tensor):
        return (
            ("positional", torch.resolve_conj(tensor)),
            ("input", torch.resolve_conj(input=tensor)),
            ("x", torch.resolve_conj(x=tensor)),
            ("a", torch.resolve_conj(a=tensor)),
            ("x1", torch.resolve_conj(x1=tensor)),
        )

    def test_clear_conjugate_bit_resolves_to_exact_receiver_without_changes(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                bits = self.value_bits(tensor)

                result = tensor.resolve_conj()

                self.assertIs(result, tensor)
                self.assertIs(result.is_conj(), False)
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr(),
                        result.dtype,
                        result.device,
                        result.requires_grad,
                        result.is_leaf,
                    ),
                    metadata,
                )
                if bits is not None:
                    np.testing.assert_array_equal(self.value_bits(result), bits)

        tracked.resolve_conj().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        gradient = leaf.grad
        self.assertIs(leaf.resolve_conj(), leaf)
        self.assertIs(leaf.grad, gradient)

    def test_top_level_clear_conjugate_bit_is_the_same_exact_identity(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            metadata = (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.dtype,
                tensor.device,
                tensor.layout,
                tensor.requires_grad,
                tensor.is_leaf,
            )
            bits = self.value_bits(tensor)

            for form, result in self.top_level_calls(tensor):
                with self.subTest(
                    case=case,
                    form=form,
                    shape=tensor.shape,
                    stride=tensor.stride(),
                ):
                    self.assertIs(result, tensor)
                    self.assertIs(result.is_conj(), False)
                    self.assertEqual(
                        (
                            result.shape,
                            result.stride(),
                            result.storage_offset(),
                            result.data_ptr(),
                            result.dtype,
                            result.device,
                            result.layout,
                            result.requires_grad,
                            result.is_leaf,
                        ),
                        metadata,
                    )
                    if bits is not None:
                        np.testing.assert_array_equal(self.value_bits(result), bits)

        torch.resolve_conj(a=tracked).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        gradient = leaf.grad
        for _, result in self.top_level_calls(leaf):
            self.assertIs(result, leaf)
            self.assertIs(result.grad, gradient)

    def test_descriptor_documentation_and_signature_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "resolve_conj")
        bound = tensor.resolve_conj

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'resolve_conj' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "resolve_conj")
        self.assertEqual(descriptor.__qualname__, "TensorBase.resolve_conj")
        self.assertEqual(bound.__name__, "resolve_conj")
        self.assertEqual(bound.__qualname__, "Tensor.resolve_conj")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), tensor)

    def test_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "resolve_conj")
        bound = tensor.resolve_conj
        cases = (
            (
                lambda: tensor.resolve_conj(1),
                "TensorBase.resolve_conj() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.resolve_conj() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.resolve_conj() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.resolve_conj(1, 2),
                "TensorBase.resolve_conj() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.resolve_conj(input=tensor),
                "TensorBase.resolve_conj() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.resolve_conj() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.resolve_conj() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.resolve_conj() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'resolve_conj' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.resolve_conj() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "resolve_conj")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.resolve_conj()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.resolve_conj()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

    def test_not_implemented_reenters_the_declining_top_mode(self):
        tensor = torch.tensor([1.0])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return NotImplemented

        class AcceptingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        lower = AcceptingMode()
        upper = DecliningMode()
        with self.assertRaisesRegex(
            RecursionError, r"^maximum recursion depth exceeded$"
        ):
            with lower:
                with upper:
                    tensor.resolve_conj()
        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, 0)
        self.assertIs(tensor.resolve_conj(), tensor)

    def test_top_level_callable_metadata_documentation_and_exports(self):
        function = torch.resolve_conj
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "resolve_conj")
        self.assertEqual(
            function.__qualname__, "_VariableFunctionsClass.resolve_conj"
        )
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method resolve_conj of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.resolve_conj, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("resolve_conj"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["resolve_conj"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.resolve_conj(),
                'resolve_conj() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.resolve_conj(tensor, tensor),
                "resolve_conj() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.resolve_conj(tensor, input=tensor),
                "resolve_conj() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.resolve_conj(tensor, extra=True, input=tensor),
                "resolve_conj() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.resolve_conj(tensor, input=tensor, extra=True),
                "resolve_conj() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.resolve_conj(extra=tensor),
                'resolve_conj() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.resolve_conj(1, extra=True),
                "resolve_conj(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.resolve_conj(input=[]),
                "resolve_conj(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.resolve_conj(a=1),
                "resolve_conj(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.resolve_conj(x=[]),
                "resolve_conj(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.resolve_conj(x1=None),
                "resolve_conj(): argument 'input' must be Tensor, not NoneType",
            ),
            (
                lambda: torch.resolve_conj(a=tensor, x=tensor),
                "resolve_conj() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.resolve_conj(x=tensor, a=tensor),
                "resolve_conj() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.resolve_conj(input=tensor, x1=tensor),
                "resolve_conj() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_top_level_torch_function_modes_and_overrides(self):
        tensor = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        calls = (
            (None, lambda: torch.resolve_conj(tensor)),
            ("input", lambda: torch.resolve_conj(input=tensor)),
            ("x", lambda: torch.resolve_conj(x=tensor)),
            ("a", lambda: torch.resolve_conj(a=tensor)),
            ("x1", lambda: torch.resolve_conj(x1=tensor)),
        )
        for keyword, call in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.resolve_conj)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,) if keyword is None else ())
            self.assertEqual(
                kwargs, None if keyword is None else {keyword: tensor}
            )

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.resolve_conj(a=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.resolve_conj(x=value), marker)
        function, dispatch_types, args, kwargs = override_calls[0]
        self.assertIs(function, torch.resolve_conj)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"x": value})

    def test_top_level_declining_mode_reports_variable_function_error(self):
        tensor = torch.tensor([1.0])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        mode = DecliningMode()
        message = (
            "Multiple dispatch failed for 'torch.resolve_conj'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - mode object {mode!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            with mode:
                torch.resolve_conj(tensor)
        self.assertIs(torch.resolve_conj(tensor), tensor)

    def test_top_level_scope_does_not_add_complex_dtypes(self):
        self.assertTrue(hasattr(torch, "resolve_conj"))
        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
