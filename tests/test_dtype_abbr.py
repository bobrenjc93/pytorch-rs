import importlib
import inspect
import types
import unittest

import torch_rs as torch


class DTypeAbbrTests(unittest.TestCase):
    def float32_sources(self):
        return (
            torch.float32,
            torch.float,
            torch.tensor(-0.0, dtype=torch.float32).dtype,
            torch.zeros((2, 0, 3), dtype=torch.float32).dtype,
            torch.zeros((2, 3, 4), dtype=torch.float32).transpose(0, 2).dtype,
            torch.tensor([1.0], dtype=torch.float32, requires_grad=True).dtype,
            torch.get_default_dtype(),
        )

    def test_every_float32_source_exposes_the_native_abbreviation(self):
        for dtype in self.float32_sources():
            with self.subTest(dtype=dtype):
                self.assertIs(dtype, torch.float32)
                self.assertIs(type(dtype.abbr), str)
                self.assertEqual(dtype.abbr, "f32")

    def test_descriptor_is_owned_by_dtype_and_is_read_only(self):
        descriptor = inspect.getattr_static(torch.dtype, "abbr")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "abbr")
        self.assertEqual(descriptor.__qualname__, "dtype.abbr")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(torch.dtype.abbr, descriptor)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        value = descriptor.__get__(torch.float32, torch.dtype)
        self.assertIs(type(value), str)
        self.assertEqual(value, "f32")

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'abbr' for 'torch_rs.dtype' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(torch.float32, "abbr", "f64"),
            lambda: delattr(torch.float32, "abbr"),
            lambda: descriptor.__set__(torch.float32, "f64"),
            lambda: descriptor.__delete__(torch.float32),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'abbr' of 'torch_rs.dtype' objects is not writable",
                )

    def test_native_module_reload_preserves_the_descriptor_and_singleton(self):
        dtype_type = torch.dtype
        float32 = torch.float32
        descriptor = inspect.getattr_static(torch.dtype, "abbr")

        self.assertIs(importlib.reload(torch._C), torch._C)
        self.assertIs(torch.dtype, dtype_type)
        self.assertIs(torch.float32, float32)
        self.assertIs(torch.float, float32)
        self.assertIs(inspect.getattr_static(torch.dtype, "abbr"), descriptor)
        self.assertEqual(torch.float32.abbr, "f32")

    def test_alternate_dtypes_and_to_complex_remain_unsupported(self):
        for name in ("float16", "float64", "complex64"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
        self.assertFalse(hasattr(torch.dtype, "to_complex"))


if __name__ == "__main__":
    unittest.main()
