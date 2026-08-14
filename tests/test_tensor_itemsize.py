import inspect
import types
import unittest

import torch_rs as torch


PROPERTY_DOC = "\nAlias for :meth:`~Tensor.element_size()`\n"


class TensorItemsizeTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        base = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        offset = base[1]
        strided = base.transpose(0, 1)

        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(strided.is_contiguous())
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset view", offset),
            ("strided view", strided),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", leaf.grad),
        )

    def test_width_uses_dtype_metadata_without_accessing_storage(self):
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
                )

                self.assertIs(type(tensor.itemsize), int)
                self.assertEqual(tensor.itemsize, 4)
                self.assertEqual(tensor.itemsize, tensor.dtype.itemsize)
                self.assertEqual(tensor.itemsize, tensor.element_size())
                self.assertEqual(tensor.nbytes, tensor.numel() * tensor.itemsize)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_tensorbase_descriptor_is_documented_and_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "itemsize")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "itemsize")
        self.assertEqual(descriptor.__qualname__, "TensorBase.itemsize")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'itemsize' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.itemsize, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(descriptor.__get__(tensor, torch.Tensor), 4)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'itemsize' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "itemsize", 8),
            lambda: delattr(tensor, "itemsize"),
            lambda: descriptor.__set__(tensor, 8),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'itemsize' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )


if __name__ == "__main__":
    unittest.main()
