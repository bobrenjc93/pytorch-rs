import inspect
import types
import unittest

import torch_rs as torch


class DTypeIsSignedTests(unittest.TestCase):
    def test_float32_aliases_expose_the_native_dtype_predicate(self):
        aliases = (
            torch.float32,
            torch.float,
            torch.tensor(3.5).dtype,
            torch.get_default_dtype(),
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertIs(alias, torch.float32)
                self.assertIs(type(alias.is_signed), bool)
                self.assertIs(alias.is_signed, True)

    def test_descriptor_is_owned_by_dtype_and_is_read_only(self):
        descriptor = inspect.getattr_static(torch.dtype, "is_signed")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "is_signed")
        self.assertEqual(descriptor.__qualname__, "dtype.is_signed")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(torch.dtype.is_signed, descriptor)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        self.assertIs(descriptor.__get__(torch.float32, torch.dtype), True)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'is_signed' for 'torch_rs.dtype' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(torch.float32, "is_signed", False),
            lambda: delattr(torch.float32, "is_signed"),
            lambda: descriptor.__set__(torch.float32, False),
            lambda: descriptor.__delete__(torch.float32),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'is_signed' of 'torch_rs.dtype' objects "
                    "is not writable",
                )

    def test_dtype_property_agrees_with_tensor_method(self):
        tensors = (
            torch.tensor(3.5),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2),
            torch.tensor([1.0], requires_grad=True),
        )
        self.assertFalse(tensors[2].is_contiguous())
        for tensor in tensors:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assertIs(type(tensor.dtype.is_signed), bool)
                self.assertIs(tensor.dtype.is_signed, tensor.is_signed())
                self.assertIs(tensor.dtype.is_signed, True)


if __name__ == "__main__":
    unittest.main()
