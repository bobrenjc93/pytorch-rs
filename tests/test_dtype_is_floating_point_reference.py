import inspect
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DTypeIsFloatingPointReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dtype.is_floating_point differentials require pinned PyTorch 2.13.0"
            )

    def make_tensors(self, module):
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            module.tensor(3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            module.zeros((2, 3, 4), dtype=module.float32).transpose(0, 2),
            extreme_empty,
        )

    def normalized_error(self, module, action):
        try:
            action()
        except Exception as error:
            message = str(error).replace(module.dtype.__module__, "torch")
            return type(error).__name__, message
        self.fail(f"{module.__name__} allowed dtype.is_floating_point assignment")

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
                self.assertIs(type(alias.is_floating_point), bool)

        actual = tuple(
            value.is_floating_point
            for value in (
                torch.float32,
                torch.float,
                torch.tensor(3.5).dtype,
                torch.get_default_dtype(),
            )
        )
        expected = tuple(
            value.is_floating_point
            for value in (
                reference_torch.float32,
                reference_torch.float,
                reference_torch.tensor(3.5).dtype,
                reference_torch.get_default_dtype(),
            )
        )
        self.assertEqual(actual, expected)

    def test_descriptor_ownership_and_assignment_errors_match_pytorch_2_13(self):
        actual = inspect.getattr_static(torch.dtype, "is_floating_point")
        expected = inspect.getattr_static(
            reference_torch.dtype, "is_floating_point"
        )

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
            lambda: setattr(torch.float32, "is_floating_point", False),
            lambda: delattr(torch.float32, "is_floating_point"),
            lambda: actual.__set__(torch.float32, False),
            lambda: actual.__delete__(torch.float32),
        )
        expected_actions = (
            lambda: setattr(
                reference_torch.float32, "is_floating_point", False
            ),
            lambda: delattr(reference_torch.float32, "is_floating_point"),
            lambda: expected.__set__(reference_torch.float32, False),
            lambda: expected.__delete__(reference_torch.float32),
        )
        for actual_action, expected_action in zip(
            actual_actions, expected_actions, strict=True
        ):
            self.assertEqual(
                self.normalized_error(torch, actual_action),
                self.normalized_error(reference_torch, expected_action),
            )

    def test_dtype_tensor_and_top_level_results_match_pytorch_2_13(self):
        actual_tensors = self.make_tensors(torch)
        expected_tensors = self.make_tensors(reference_torch)
        self.assertFalse(actual_tensors[2].is_contiguous())
        self.assertFalse(expected_tensors[2].is_contiguous())

        for actual, expected in zip(
            actual_tensors, expected_tensors, strict=True
        ):
            with self.subTest(shape=actual.shape, stride=actual.stride()):
                actual_values = (
                    actual.dtype.is_floating_point,
                    actual.is_floating_point(),
                    torch.is_floating_point(actual),
                )
                expected_values = (
                    expected.dtype.is_floating_point,
                    expected.is_floating_point(),
                    reference_torch.is_floating_point(expected),
                )
                self.assertEqual(actual_values, expected_values)
                self.assertTrue(
                    all(type(value) is bool for value in actual_values)
                )


if __name__ == "__main__":
    unittest.main()
