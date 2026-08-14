import inspect
import types
import unittest

import torch_rs as torch


class DTypeIsComplexTests(unittest.TestCase):
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
                self.assertIs(type(alias.is_complex), bool)
                self.assertIs(alias.is_complex, False)

    def test_descriptor_is_owned_by_dtype_and_is_read_only(self):
        descriptor = inspect.getattr_static(torch.dtype, "is_complex")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "is_complex")
        self.assertEqual(descriptor.__qualname__, "dtype.is_complex")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(torch.dtype.is_complex, descriptor)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        self.assertIs(descriptor.__get__(torch.float32, torch.dtype), False)

        actions = (
            lambda: setattr(torch.float32, "is_complex", True),
            lambda: delattr(torch.float32, "is_complex"),
            lambda: descriptor.__set__(torch.float32, True),
            lambda: descriptor.__delete__(torch.float32),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'is_complex' of 'torch_rs.dtype' objects "
                    "is not writable",
                )


if __name__ == "__main__":
    unittest.main()
