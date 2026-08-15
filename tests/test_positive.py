import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch
from tests.signature_utils import assert_no_argument_signature


METHOD_DOC = "\npositive() -> Tensor\n\nSee :func:`torch.positive`\n"
FUNCTION_DOC = (
    "\npositive(input) -> Tensor\n\n"
    "Returns :attr:`input`.\n"
    "Throws a runtime error if :attr:`input` is a bool tensor.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> t = torch.randn(5)\n"
    "    >>> t\n"
    "    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n"
    "    >>> torch.positive(t)\n"
    "    tensor([ 0.0090, -0.2262, -0.0682, -0.2866,  0.3940])\n"
)


class TensorPositiveTests(unittest.TestCase):
    def assert_identity_call(self, source):
        metadata = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.dtype,
            source.device,
            source.requires_grad,
            source.is_leaf,
            source.data_ptr(),
        )
        bits = np.asarray(source).reshape(-1).view(np.uint32).copy()
        detached = source.detach()

        result = source.positive()
        operator_result = +source

        self.assertIs(result, source)
        self.assertIs(operator_result, source)
        self.assertTrue(result.is_set_to(detached))
        self.assertEqual(
            (
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.dtype,
                result.device,
                result.requires_grad,
                result.is_leaf,
                result.data_ptr(),
            ),
            metadata,
        )
        np.testing.assert_array_equal(
            np.asarray(result).reshape(-1).view(np.uint32), bits
        )

    def test_scalar_empty_offset_strided_and_special_values_are_exact_identities(self):
        base = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        strided = base.transpose(0, 2)
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

        self.assertEqual(strided.storage_offset(), 0)
        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(offset.is_contiguous())
        self.assertEqual(empty.shape, (0, 2))
        self.assertGreater(empty.storage_offset(), 0)

        for case, source in (
            ("scalar", torch.tensor(-0.0)),
            ("empty", empty),
            ("offset", offset),
            ("strided", strided),
            ("special values", special),
        ):
            with self.subTest(case=case):
                self.assert_identity_call(source)

    def test_leaf_and_non_leaf_graph_state_is_unchanged(self):
        leaf = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        leaf_result = leaf.positive()
        self.assertIs(leaf_result, leaf)
        self.assertIs(+leaf, leaf)
        self.assertTrue(leaf_result.requires_grad)
        self.assertTrue(leaf_result.is_leaf)

        source = (leaf_result * 3.0).transpose(0, 1)[1]
        graph_before = (
            source.requires_grad,
            source.is_leaf,
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
        )

        result = source.positive()
        operator_result = +source

        self.assertIs(result, source)
        self.assertIs(operator_result, source)
        self.assertEqual(
            (
                operator_result.requires_grad,
                operator_result.is_leaf,
                operator_result.shape,
                operator_result.stride(),
                operator_result.storage_offset(),
                operator_result.data_ptr(),
            ),
            graph_before,
        )
        operator_result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(leaf.positive(), leaf)
        self.assertIs(+leaf, leaf)
        self.assertIs(leaf.grad, gradient)

    def test_descriptor_documentation_and_signature_behavior(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "positive")
        operator_descriptor = inspect.getattr_static(torch.Tensor, "__pos__")
        bound = tensor.positive
        operator_bound = tensor.__pos__

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertIs(type(operator_bound), types.BuiltinMethodType)
        self.assertIs(operator_descriptor, descriptor)
        self.assertIs(torch.Tensor.__dict__["__pos__"], descriptor)
        with self.assertRaises(AttributeError):
            inspect.getattr_static(descriptor.__objclass__, "__pos__")
        self.assertEqual(
            repr(descriptor),
            "<method 'positive' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__qualname__, "TensorBase.positive")
        self.assertEqual(bound.__qualname__, "Tensor.positive")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        for callable_object in (descriptor, bound, operator_bound):
            self.assertEqual(callable_object.__name__, "positive")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        assert_no_argument_signature(self, operator_bound, "()")

        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(**{}), tensor)
        self.assertIs(operator_bound(**{}), tensor)

    def test_invalid_call_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "positive")
        bound = tensor.positive
        operator_bound = tensor.__pos__
        cases = (
            (
                lambda: tensor.positive(1),
                "TensorBase.positive() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.positive(1, 2),
                "TensorBase.positive() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.positive(dim=0),
                "TensorBase.positive() takes no keyword arguments",
            ),
            (
                lambda: bound(1),
                "Tensor.positive() takes no arguments (1 given)",
            ),
            (
                lambda: bound(dim=0),
                "Tensor.positive() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.positive() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.positive() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'positive' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.positive() needs an argument",
            ),
            (
                lambda: tensor.__pos__(1),
                "TensorBase.positive() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.__pos__(1, 2),
                "TensorBase.positive() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.__pos__(dim=0),
                "TensorBase.positive() takes no keyword arguments",
            ),
            (
                lambda: operator_bound(1),
                "Tensor.positive() takes no arguments (1 given)",
            ),
            (
                lambda: operator_bound(dim=0),
                "Tensor.positive() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


class TopLevelPositiveTests(unittest.TestCase):
    def positive_calls(self, source):
        return (
            ("positional", torch.positive(source)),
            ("input", torch.positive(input=source)),
            ("x", torch.positive(x=source)),
            ("a", torch.positive(a=source)),
            ("x1", torch.positive(x1=source)),
        )

    def assert_identity_calls(self, source):
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
        )
        bits = np.asarray(detached).reshape(-1).view(np.uint32).copy()

        for form, result in self.positive_calls(source):
            with self.subTest(form=form):
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
                    ),
                    metadata,
                )
                np.testing.assert_array_equal(
                    np.asarray(result.detach()).reshape(-1).view(np.uint32), bits
                )

    def test_all_call_forms_are_exact_identities_for_supported_layouts(self):
        base = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        strided = base.transpose(0, 2)
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

        for case, source in (
            ("scalar", torch.tensor(-0.0)),
            ("empty", empty),
            ("offset", offset),
            ("strided", strided),
            ("special values", special),
        ):
            with self.subTest(case=case):
                self.assert_identity_calls(source)

    def test_all_call_forms_preserve_leaf_and_non_leaf_autograd_state(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]

        for case, source in (("leaf", leaf), ("non-leaf", non_leaf)):
            with self.subTest(case=case):
                self.assert_identity_calls(source)

        torch.positive(a=non_leaf).sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        for _, result in self.positive_calls(leaf):
            self.assertIs(result.grad, gradient)

    def test_callable_metadata_documentation_and_exports_match_pytorch(self):
        function = torch.positive
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "positive")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.positive")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method positive of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.positive, function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("positive"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["positive"], function)

    def test_binding_and_tensor_type_error_precedence_matches_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.positive(),
                'positive() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.positive(tensor, tensor),
                "positive() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.positive(tensor, input=tensor),
                "positive() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.positive(tensor, extra=True, input=tensor),
                "positive() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.positive(tensor, input=tensor, extra=True),
                "positive() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.positive(extra=tensor),
                'positive() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.positive(1, extra=True),
                "positive(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.positive(input=[]),
                "positive(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.positive(a=1),
                "positive(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.positive(x=[]),
                "positive(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.positive(a=tensor, x=tensor),
                "positive() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.positive(x=tensor, a=tensor),
                "positive() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.positive(input=tensor, a=tensor),
                "positive() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.positive(input=tensor, x1=tensor),
                "positive() got an unexpected keyword argument 'x1'",
            ),
            (
                lambda: torch.positive(x=tensor, x1=tensor),
                "positive() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.positive(x1=tensor, x=tensor),
                "positive() got an unexpected keyword argument 'x1'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
