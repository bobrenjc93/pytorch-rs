import importlib
import inspect
import types
import unittest

import torch_rs as torch


class DTypeAbbrTests(unittest.TestCase):
    def test_every_float32_source_exposes_the_native_abbreviation(self):
        aliases = (
            torch.float32,
            torch.float,
            torch.tensor(-0.0, dtype=torch.float32).dtype,
            torch.zeros((2, 0, 3), dtype=torch.float32)
            .transpose(0, 2)
            .dtype,
            torch.tensor(
                [1.0], dtype=torch.float32, requires_grad=True
            ).dtype,
            torch.get_default_dtype(),
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertIs(alias, torch.float32)
                self.assertIs(type(alias.abbr), str)
                self.assertEqual(alias.abbr, "f32")

    def test_descriptor_metadata_and_errors_match_the_native_type(self):
        descriptor = inspect.getattr_static(torch.dtype, "abbr")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "abbr")
        self.assertEqual(descriptor.__qualname__, "dtype.abbr")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(torch.dtype.abbr, descriptor)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        self.assertEqual(descriptor.__get__(torch.float32, torch.dtype), "f32")

        cases = (
            (
                lambda: setattr(torch.float32, "abbr", "f64"),
                AttributeError,
                "attribute 'abbr' of 'torch_rs.dtype' objects is not writable",
            ),
            (
                lambda: delattr(torch.float32, "abbr"),
                AttributeError,
                "attribute 'abbr' of 'torch_rs.dtype' objects is not writable",
            ),
            (
                lambda: descriptor.__set__(torch.float32, "f64"),
                AttributeError,
                "attribute 'abbr' of 'torch_rs.dtype' objects is not writable",
            ),
            (
                lambda: descriptor.__delete__(torch.float32),
                AttributeError,
                "attribute 'abbr' of 'torch_rs.dtype' objects is not writable",
            ),
            (
                lambda: descriptor.__get__(1, int),
                TypeError,
                "descriptor 'abbr' for 'torch_rs.dtype' objects doesn't apply "
                "to a 'int' object",
            ),
        )
        for action, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    action()
                self.assertEqual(str(raised.exception), message)

    def test_package_and_native_reload_preserve_dtype_objects(self):
        package = torch
        native = torch._C
        dtype_type = torch.dtype
        canonical = torch.float32
        descriptor = inspect.getattr_static(dtype_type, "abbr")

        self.assertIs(importlib.reload(package), package)
        self.assertIs(torch.dtype, dtype_type)
        self.assertIs(torch.float32, canonical)
        self.assertIs(inspect.getattr_static(torch.dtype, "abbr"), descriptor)
        self.assertEqual(torch.float32.abbr, "f32")

        self.assertIs(importlib.reload(native), native)
        self.assertIs(torch.dtype, dtype_type)
        self.assertIs(torch.float32, canonical)
        self.assertIs(inspect.getattr_static(torch.dtype, "abbr"), descriptor)
        self.assertIs(type(torch.float32.abbr), str)
        self.assertEqual(torch.float32.abbr, "f32")

    def test_other_dtype_singletons_and_to_complex_remain_unsupported(self):
        for name in (
            "float16",
            "half",
            "bfloat16",
            "float64",
            "double",
            "complex64",
            "cfloat",
            "int32",
            "bool",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))

        self.assertFalse(hasattr(torch.dtype, "to_complex"))
        self.assertFalse(hasattr(torch.float32, "to_complex"))


if __name__ == "__main__":
    unittest.main()
