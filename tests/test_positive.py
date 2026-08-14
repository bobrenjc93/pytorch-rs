import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\npositive() -> Tensor\n\nSee :func:`torch.positive`\n"


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

        self.assertIs(result, source)
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

        self.assertIs(result, source)
        self.assertEqual(
            (
                result.requires_grad,
                result.is_leaf,
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.data_ptr(),
            ),
            graph_before,
        )
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assertIs(leaf.positive(), leaf)
        self.assertIs(leaf.grad, gradient)

    def test_descriptor_documentation_and_signature_behavior(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "positive")
        bound = tensor.positive

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'positive' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__qualname__, "TensorBase.positive")
        self.assertEqual(bound.__qualname__, "Tensor.positive")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "positive")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(**{}), tensor)

    def test_invalid_call_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "positive")
        bound = tensor.positive
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
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
