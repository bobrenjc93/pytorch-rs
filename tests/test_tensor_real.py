import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


PROPERTY_DOC = (
    "\nReturns a new tensor containing real values of the :attr:`self` tensor "
    "for a complex-valued input tensor.\n"
    "The returned tensor and :attr:`self` share the same underlying storage.\n\n"
    "Returns :attr:`self` if :attr:`self` is a real-valued tensor.\n\n"
    "Example::\n\n"
    "    >>> x=torch.randn(4, dtype=torch.cfloat)\n"
    "    >>> x\n"
    "    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), "
    "(-1.6492-0.0633j), (-0.0638-0.8119j)])\n"
    "    >>> x.real\n"
    "    tensor([ 0.3100, -0.5445, -1.6492, -0.0638])\n\n"
)


class TensorRealTests(unittest.TestCase):
    def tensor_cases(self):
        scalar_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = torch.tensor(memoryview(scalar_bits.view(np.float32)))
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        offset = strided[1]
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertEqual(empty.shape, (0, 2))
        self.assertGreater(empty.storage_offset(), 0)
        self.assertTrue(leaf.is_leaf)
        self.assertFalse(non_leaf.is_leaf)
        return (
            *(
                (f"float32 bits 0x{bits:08x}", scalar_storage[index])
                for index, bits in enumerate(scalar_bits)
            ),
            ("empty offset view", empty),
            ("offset strided view", offset),
            ("strided view", strided),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
        )

    def test_supported_tensors_return_the_exact_receiver_without_side_effects(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                    tensor.data_ptr(),
                )
                detached = tensor.detach()
                bits = np.asarray(detached).reshape(-1).view(np.uint32).copy()

                result = tensor.real

                self.assertIs(result, tensor)
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
                    np.asarray(result.detach()).reshape(-1).view(np.uint32), bits
                )

    def test_leaf_and_non_leaf_graphs_are_not_changed(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        self.assertIs(leaf.real, leaf)
        self.assertTrue(leaf.is_leaf)

        non_leaf = (leaf.real * 3.0).transpose(0, 1)[1]
        graph_before = (
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            non_leaf.shape,
            non_leaf.stride(),
            non_leaf.storage_offset(),
            non_leaf.data_ptr(),
        )

        result = non_leaf.real

        self.assertIs(result, non_leaf)
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
        self.assertIs(leaf.real, leaf)
        self.assertIs(leaf.grad, gradient)

    def test_tensorbase_descriptor_is_documented_and_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "real")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "real")
        self.assertEqual(descriptor.__qualname__, "TensorBase.real")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'real' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.real, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), tensor)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'real' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "real", torch.tensor([2.0])),
            lambda: delattr(tensor, "real"),
            lambda: descriptor.__set__(tensor, torch.tensor([2.0])),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'real' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )


if __name__ == "__main__":
    unittest.main()
