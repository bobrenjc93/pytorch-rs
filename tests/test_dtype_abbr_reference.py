import importlib
import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DTypeAbbrReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dtype.abbr differentials require pinned PyTorch 2.13.0"
            )

    def normalized_error(self, module, action):
        try:
            action()
        except Exception as error:
            message = str(error).replace(module.dtype.__module__, "torch")
            return type(error).__name__, message
        self.fail(f"{module.__name__} allowed an invalid dtype.abbr operation")

    def alias_contract(self, module):
        canonical = module.float32
        aliases = (
            module.float32,
            module.float,
            module.tensor(-0.0, dtype=module.float32).dtype,
            module.zeros((2, 0, 3), dtype=module.float32)
            .transpose(0, 2)
            .dtype,
            module.tensor(
                [1.0], dtype=module.float32, requires_grad=True
            ).dtype,
            module.get_default_dtype(),
        )
        return tuple(
            (alias is canonical, type(alias.abbr).__name__, alias.abbr)
            for alias in aliases
        )

    def test_float32_sources_match_pytorch_2_13(self):
        self.assertEqual(
            self.alias_contract(torch),
            self.alias_contract(reference_torch),
        )

    def test_descriptor_metadata_and_errors_match_pytorch_2_13(self):
        actual = inspect.getattr_static(torch.dtype, "abbr")
        expected = inspect.getattr_static(reference_torch.dtype, "abbr")

        for descriptor, module in (
            (actual, torch),
            (expected, reference_torch),
        ):
            self.assertIs(type(descriptor), types.GetSetDescriptorType)
            self.assertFalse(callable(descriptor))
            self.assertIs(descriptor.__objclass__, module.dtype)
            self.assertIs(module.dtype.abbr, descriptor)
            self.assertIs(descriptor.__get__(None, module.dtype), descriptor)

        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__get__(torch.float32, torch.dtype),
            expected.__get__(reference_torch.float32, reference_torch.dtype),
        )

        actual_actions = (
            lambda: setattr(torch.float32, "abbr", "f64"),
            lambda: delattr(torch.float32, "abbr"),
            lambda: actual.__set__(torch.float32, "f64"),
            lambda: actual.__delete__(torch.float32),
            lambda: actual.__get__(1, int),
        )
        expected_actions = (
            lambda: setattr(reference_torch.float32, "abbr", "f64"),
            lambda: delattr(reference_torch.float32, "abbr"),
            lambda: expected.__set__(reference_torch.float32, "f64"),
            lambda: expected.__delete__(reference_torch.float32),
            lambda: expected.__get__(1, int),
        )
        for actual_action, expected_action in zip(
            actual_actions, expected_actions, strict=True
        ):
            self.assertEqual(
                self.normalized_error(torch, actual_action),
                self.normalized_error(reference_torch, expected_action),
            )

    def reload_contract(self, module):
        native = module._C
        dtype_type = module.dtype
        canonical = module.float32
        descriptor = inspect.getattr_static(dtype_type, "abbr")

        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module.dtype is dtype_type,
            module.float32 is canonical,
            inspect.getattr_static(module.dtype, "abbr") is descriptor,
            type(module.float32.abbr).__name__,
            module.float32.abbr,
        )

    def test_native_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_broader_dtype_surface_remains_deliberately_unsupported(self):
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
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))

        self.assertTrue(hasattr(reference_torch.dtype, "to_complex"))
        self.assertIs(
            reference_torch.float32.to_complex(), reference_torch.complex64
        )
        self.assertFalse(hasattr(torch.dtype, "to_complex"))
        self.assertFalse(hasattr(torch.float32, "to_complex"))


if __name__ == "__main__":
    unittest.main()
