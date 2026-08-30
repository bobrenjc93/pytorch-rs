import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


METHOD_DOC = "\nconj() -> Tensor\n\nSee :func:`torch.conj`\n"
FUNCTION_DOC = (
    "\nconj(input) -> Tensor\n\n"
    "Returns a view of :attr:`input` with a flipped conjugate bit. If "
    ":attr:`input` has a non-complex dtype,\n"
    "this function just returns :attr:`input`.\n\n"
    ".. note::\n"
    "    :func:`torch.conj` performs a lazy conjugation, but the actual "
    "conjugated tensor can be materialized\n"
    "    at any time using :func:`torch.resolve_conj`.\n\n"
    ".. warning:: In the future, :func:`torch.conj` may return a "
    "non-writeable view for an :attr:`input` of\n"
    "             non-complex dtype. It's recommended that programs not "
    "modify the tensor returned by :func:`torch.conj_physical`\n"
    "             when :attr:`input` is of non-complex dtype to be "
    "compatible with this change.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> x = torch.tensor([-1 + 1j, -2 + 2j, 3 - 3j])\n"
    "    >>> x.is_conj()\n"
    "    False\n"
    "    >>> y = torch.conj(x)\n"
    "    >>> y.is_conj()\n"
    "    True\n"
)


class TensorConjTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        source = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        strided = source.transpose(0, 2)
        offset = strided[1]
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
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
        special = torch.tensor(memoryview(special_bits.view(np.float32)))

        self.assertEqual(leaf.shape, (2, 3))
        self.assertFalse(non_leaf.is_leaf)
        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertGreater(empty.storage_offset(), 0)

        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", empty),
            ("offset", offset),
            ("noncontiguous", strided),
            ("special values", special),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
        )

    @staticmethod
    def value_bits(tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

    def assert_identity_result(self, source, result):
        detached = source.detach()
        metadata = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.dtype,
            source.device,
            source.layout,
            source.requires_grad,
            source.is_leaf,
            source.data_ptr(),
            source.is_conj(),
        )
        bits = self.value_bits(source)

        self.assertIs(result, source)
        self.assertTrue(result.is_set_to(detached))
        self.assertEqual(
            (
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.dtype,
                result.device,
                result.layout,
                result.requires_grad,
                result.is_leaf,
                result.data_ptr(),
                result.is_conj(),
            ),
            metadata,
        )
        self.assertIs(result.is_conj(), False)
        if bits is not None:
            np.testing.assert_array_equal(self.value_bits(result), bits)

    def top_level_calls(self, source):
        return (
            ("positional", torch.conj(source)),
            ("input", torch.conj(input=source)),
            ("x", torch.conj(x=source)),
            ("a", torch.conj(a=source)),
            ("x1", torch.conj(x1=source)),
        )

    def test_method_is_exact_identity_for_supported_real_tensors(self):
        for case, source in self.tensor_cases():
            with self.subTest(case=case):
                self.assert_identity_result(source, source.conj())

    def test_top_level_is_exact_identity_for_supported_real_tensors(self):
        for case, source in self.tensor_cases():
            for form, result in self.top_level_calls(source):
                with self.subTest(case=case, form=form):
                    self.assert_identity_result(source, result)

    def test_leaf_non_leaf_no_grad_and_repeated_backward(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        non_leaf = (leaf * 2.0).transpose(0, 1)

        self.assertIs(leaf.conj(), leaf)
        self.assertTrue(leaf.conj().requires_grad)
        self.assertTrue(leaf.conj().is_leaf)
        self.assertIs(non_leaf.conj(), non_leaf)
        self.assertTrue(non_leaf.conj().requires_grad)
        self.assertFalse(non_leaf.conj().is_leaf)

        with torch.no_grad():
            no_grad_leaf = leaf.conj()
            no_grad_non_leaf = torch.conj(non_leaf)
        self.assertIs(no_grad_leaf, leaf)
        self.assertTrue(no_grad_leaf.requires_grad)
        self.assertTrue(no_grad_leaf.is_leaf)
        self.assertIs(no_grad_non_leaf, non_leaf)
        self.assertTrue(no_grad_non_leaf.requires_grad)
        self.assertFalse(no_grad_non_leaf.is_leaf)

        reusable_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        reusable_loss = reusable_leaf.transpose(0, 0).conj().sum()
        reusable_loss.backward()
        reusable_loss.backward()
        self.assertEqual(reusable_leaf.grad.tolist(), [2.0, 2.0])

        torch.conj(non_leaf).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        gradient = leaf.grad
        self.assertIs(torch.conj(leaf).grad, gradient)

    def test_descriptor_documentation_and_signature_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "conj")
        bound = tensor.conj

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'conj' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "conj")
        self.assertEqual(descriptor.__qualname__, "TensorBase.conj")
        self.assertEqual(bound.__name__, "conj")
        self.assertEqual(bound.__qualname__, "Tensor.conj")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), tensor)

    def test_method_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "conj")
        bound = tensor.conj
        cases = (
            (
                lambda: tensor.conj(1),
                "TensorBase.conj() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.conj() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.conj() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.conj(1, 2),
                "TensorBase.conj() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.conj(input=tensor),
                re.compile(r"Tensor(Base)?\.conj\(\) takes no keyword arguments"),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.conj() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.conj() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.conj() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'conj' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.conj() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                if isinstance(message, str):
                    self.assertEqual(str(raised.exception), message)
                else:
                    self.assertRegex(str(raised.exception), message)

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "conj")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.conj()
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
                forwarded = tensor.conj()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)


class TopLevelConjTests(unittest.TestCase):
    def test_callable_metadata_documentation_and_exports_match_pytorch(self):
        function = torch.conj
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "conj")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.conj")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method conj of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.conj, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("conj"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["conj"], function)

    def test_binding_errors_and_unsupported_extensions(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.conj(),
                'conj() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.conj(tensor, tensor),
                "conj() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.conj(tensor, input=tensor),
                "conj() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.conj(tensor, out=None),
                "conj() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: torch.conj(tensor, dtype=torch.float32),
                "conj() got an unexpected keyword argument 'dtype'",
            ),
            (
                lambda: torch.conj(tensor, device="cpu"),
                "conj() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.conj(1, extra=True),
                "conj(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.conj(input=[]),
                "conj(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.conj(a=1),
                "conj(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.conj(x=[]),
                "conj(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.conj(x1=None),
                "conj(): argument 'input' must be Tensor, not NoneType",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
            "conj_physical",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
        self.assertFalse(hasattr(torch.Tensor, "conj_"))
        self.assertFalse(hasattr(tensor, "conj_"))
        self.assertFalse(hasattr(tensor, "conj_physical"))

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
            (None, lambda: torch.conj(tensor)),
            ("input", lambda: torch.conj(input=tensor)),
            ("x", lambda: torch.conj(x=tensor)),
            ("a", lambda: torch.conj(a=tensor)),
            ("x1", lambda: torch.conj(x1=tensor)),
        )
        for keyword, call in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, torch.conj)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (tensor,) if keyword is None else ())
            self.assertEqual(kwargs, None if keyword is None else {keyword: tensor})

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.conj(a=tensor)
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)
        self.assertIs(forwarded.is_conj(), False)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        self.assertIs(torch.conj(x=value), marker)
        function, dispatch_types, args, kwargs = override_calls[0]
        self.assertIs(function, torch.conj)
        self.assertEqual(dispatch_types, (Override,))
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"x": value})

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        message = (
            "Multiple dispatch failed for 'torch.conj'; all "
            "__torch_function__ handlers returned NotImplemented:\n\n"
            f"  - tensor subclass {DecliningOverride!r}\n\n"
            "For more information, try re-running with "
            "TORCH_LOGS=not_implemented"
        )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            torch.conj(DecliningOverride())
        self.assertIs(torch.conj(tensor), tensor)


if __name__ == "__main__":
    unittest.main()
