import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DTypeIsSignedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dtype.is_signed differentials require pinned PyTorch 2.13.0"
            )

    def normalized_error(self, module, action):
        try:
            action()
        except Exception as error:
            message = str(error).replace(module.dtype.__module__, "torch")
            return type(error).__name__, message
        self.fail(f"{module.__name__} allowed dtype.is_signed operation")

    def alias_contract(self, module):
        canonical = module.float32
        aliases = (
            module.float32,
            module.float,
            module.tensor(3.5).dtype,
            module.get_default_dtype(),
        )
        return {
            "canonical_identity": tuple(alias is canonical for alias in aliases),
            "value_types": tuple(type(alias.is_signed).__name__ for alias in aliases),
            "values": tuple(alias.is_signed for alias in aliases),
        }

    def test_float32_alias_identity_and_values_match_pytorch_2_13(self):
        self.assertEqual(
            self.alias_contract(torch),
            self.alias_contract(reference_torch),
        )

    def test_descriptor_ownership_and_mutation_errors_match_pytorch_2_13(self):
        actual = inspect.getattr_static(torch.dtype, "is_signed")
        expected = inspect.getattr_static(reference_torch.dtype, "is_signed")

        for descriptor, module in (
            (actual, torch),
            (expected, reference_torch),
        ):
            self.assertIs(type(descriptor), types.GetSetDescriptorType)
            self.assertFalse(callable(descriptor))
            self.assertIs(descriptor.__objclass__, module.dtype)
            self.assertIs(module.dtype.is_signed, descriptor)
            self.assertIs(descriptor.__get__(None, module.dtype), descriptor)

        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertIs(
            actual.__get__(torch.float32, torch.dtype),
            expected.__get__(reference_torch.float32, reference_torch.dtype),
        )

        actual_actions = (
            lambda: setattr(torch.float32, "is_signed", False),
            lambda: delattr(torch.float32, "is_signed"),
            lambda: actual.__set__(torch.float32, False),
            lambda: actual.__delete__(torch.float32),
            lambda: actual.__get__(1, int),
        )
        expected_actions = (
            lambda: setattr(reference_torch.float32, "is_signed", False),
            lambda: delattr(reference_torch.float32, "is_signed"),
            lambda: expected.__set__(reference_torch.float32, False),
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

    def tensor_contract(self, module):
        tensors = (
            module.tensor(3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            module.zeros((2, 3, 4), dtype=module.float32).transpose(0, 2),
            module.tensor([1.0], dtype=module.float32, requires_grad=True),
        )
        return tuple(
            (
                tensor.dtype.is_signed,
                type(tensor.dtype.is_signed).__name__,
                tensor.is_signed(),
                type(tensor.is_signed()).__name__,
            )
            for tensor in tensors
        )

    def test_dtype_property_and_tensor_method_match_pytorch_2_13(self):
        self.assertEqual(
            self.tensor_contract(torch),
            self.tensor_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
