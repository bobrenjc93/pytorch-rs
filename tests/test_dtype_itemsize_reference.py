import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DTypeItemsizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dtype.itemsize differentials require pinned PyTorch 2.13.0"
            )

    def make_tensors(self, module):
        return (
            module.tensor(3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            module.zeros((2, 3, 4), dtype=module.float32).transpose(0, 2),
        )

    def normalized_error(self, module, action):
        try:
            action()
        except Exception as error:
            message = str(error).replace(module.dtype.__module__, "torch")
            return type(error).__name__, message
        self.fail(f"{module.__name__} allowed dtype.itemsize assignment")

    def test_float32_aliases_match_pytorch_2_13(self):
        for module in (torch, reference_torch):
            aliases = (
                module.float32,
                module.float,
                module.tensor(3.5, dtype=module.float32).dtype,
                module.get_default_dtype(),
            )
            for alias in aliases:
                self.assertIs(alias, module.float32)
                self.assertIs(type(alias.itemsize), int)

        actual = tuple(
            value.itemsize
            for value in (
                torch.float32,
                torch.float,
                torch.tensor(3.5).dtype,
                torch.get_default_dtype(),
            )
        )
        expected = tuple(
            value.itemsize
            for value in (
                reference_torch.float32,
                reference_torch.float,
                reference_torch.tensor(3.5, dtype=reference_torch.float32).dtype,
                reference_torch.get_default_dtype(),
            )
        )
        self.assertEqual(actual, expected)

    def test_descriptor_ownership_and_assignment_errors_match_pytorch_2_13(self):
        actual = inspect.getattr_static(torch.dtype, "itemsize")
        expected = inspect.getattr_static(reference_torch.dtype, "itemsize")

        for descriptor, module in (
            (actual, torch),
            (expected, reference_torch),
        ):
            self.assertIs(type(descriptor), types.GetSetDescriptorType)
            self.assertIs(descriptor.__objclass__, module.dtype)
            self.assertIs(descriptor.__get__(None, module.dtype), descriptor)

        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__get__(torch.float32, torch.dtype),
            expected.__get__(reference_torch.float32, reference_torch.dtype),
        )

        actual_actions = (
            lambda: setattr(torch.float32, "itemsize", 8),
            lambda: delattr(torch.float32, "itemsize"),
            lambda: actual.__set__(torch.float32, 8),
            lambda: actual.__delete__(torch.float32),
        )
        expected_actions = (
            lambda: setattr(reference_torch.float32, "itemsize", 8),
            lambda: delattr(reference_torch.float32, "itemsize"),
            lambda: expected.__set__(reference_torch.float32, 8),
            lambda: expected.__delete__(reference_torch.float32),
        )
        for actual_action, expected_action in zip(
            actual_actions, expected_actions, strict=True
        ):
            self.assertEqual(
                self.normalized_error(torch, actual_action),
                self.normalized_error(reference_torch, expected_action),
            )

    def test_tensor_dtype_itemsize_matches_element_size_for_all_layouts(self):
        actual_tensors = self.make_tensors(torch)
        expected_tensors = self.make_tensors(reference_torch)
        self.assertFalse(actual_tensors[-1].is_contiguous())
        self.assertFalse(expected_tensors[-1].is_contiguous())

        for actual, expected in zip(
            actual_tensors, expected_tensors, strict=True
        ):
            with self.subTest(shape=actual.shape, stride=actual.stride()):
                actual_values = (actual.dtype.itemsize, actual.element_size())
                expected_values = (expected.dtype.itemsize, expected.element_size())
                self.assertEqual(actual_values, expected_values)
                self.assertEqual(actual_values[0], actual_values[1])


if __name__ == "__main__":
    unittest.main()
