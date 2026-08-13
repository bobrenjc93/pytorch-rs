import inspect
import types
import unittest

import torch_rs as torch


class DTypeItemsizeTests(unittest.TestCase):
    def test_float32_aliases_expose_the_native_element_size(self):
        aliases = (
            torch.float32,
            torch.float,
            torch.tensor(3.5).dtype,
            torch.get_default_dtype(),
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertIs(alias, torch.float32)
                self.assertIs(type(alias.itemsize), int)
                self.assertEqual(alias.itemsize, 4)

    def test_descriptor_is_owned_by_dtype_and_is_read_only(self):
        descriptor = inspect.getattr_static(torch.dtype, "itemsize")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertEqual(descriptor.__name__, "itemsize")
        self.assertEqual(descriptor.__qualname__, "dtype.itemsize")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        self.assertEqual(descriptor.__get__(torch.float32, torch.dtype), 4)

        actions = (
            lambda: setattr(torch.float32, "itemsize", 8),
            lambda: delattr(torch.float32, "itemsize"),
            lambda: descriptor.__set__(torch.float32, 8),
            lambda: descriptor.__delete__(torch.float32),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'itemsize' of 'torch_rs.dtype' objects is not writable",
                )

    def test_scalar_empty_and_strided_tensor_widths_agree(self):
        tensors = (
            torch.tensor(3.5),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2),
        )
        self.assertFalse(tensors[-1].is_contiguous())
        for tensor in tensors:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assertEqual(tensor.dtype.itemsize, tensor.element_size())
                self.assertEqual(tensor.dtype.itemsize, 4)


if __name__ == "__main__":
    unittest.main()
