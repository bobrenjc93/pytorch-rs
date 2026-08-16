import inspect
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class DTypeToRealReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dtype.to_real differentials require pinned PyTorch 2.13.0"
            )

    def normalized_error(self, module, action):
        try:
            action()
        except Exception as error:
            message = str(error).replace(module.dtype.__module__, "torch")
            return type(error).__name__, message
        self.fail(f"{module.__name__} allowed an invalid dtype.to_real operation")

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
            (alias is canonical, alias.to_real() is canonical) for alias in aliases
        )

    def test_float32_alias_identity_matches_pytorch_2_13(self):
        self.assertEqual(
            self.alias_contract(torch),
            self.alias_contract(reference_torch),
        )

    def test_descriptor_documentation_signatures_and_unbound_calls_match(self):
        actual_descriptor = inspect.getattr_static(torch.dtype, "to_real")
        expected_descriptor = inspect.getattr_static(
            reference_torch.dtype, "to_real"
        )
        actual_bound = torch.float32.to_real
        expected_bound = reference_torch.float32.to_real

        for descriptor, bound, module in (
            (actual_descriptor, actual_bound, torch),
            (expected_descriptor, expected_bound, reference_torch),
        ):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertIs(descriptor.__objclass__, module.dtype)
            self.assertIs(module.dtype.to_real, descriptor)
            self.assertIs(descriptor.__get__(None, module.dtype), descriptor)
            self.assertIs(bound.__self__, module.float32)
            assert_no_argument_signature(self, descriptor, "(self, /)")
            assert_no_argument_signature(self, bound, "()")
            self.assertIs(descriptor(module.float32), module.float32)

        self.assertEqual(actual_descriptor.__name__, expected_descriptor.__name__)
        self.assertEqual(
            actual_descriptor.__qualname__, expected_descriptor.__qualname__
        )
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertEqual(actual_bound.__doc__, expected_bound.__doc__)
        self.assertEqual(
            actual_descriptor.__text_signature__,
            expected_descriptor.__text_signature__,
        )
        self.assertEqual(
            actual_bound.__text_signature__, expected_bound.__text_signature__
        )

    def test_argument_receiver_and_attribute_errors_match_pytorch_2_13(self):
        actual_descriptor = inspect.getattr_static(torch.dtype, "to_real")
        expected_descriptor = inspect.getattr_static(
            reference_torch.dtype, "to_real"
        )
        actual_bound = torch.float32.to_real
        expected_bound = reference_torch.float32.to_real
        cases = (
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (
                lambda: actual_bound(value=True),
                lambda: expected_bound(value=True),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(torch.float32, 1),
                lambda: expected_descriptor(reference_torch.float32, 1),
            ),
            (
                lambda: actual_descriptor(torch.float32, value=True),
                lambda: expected_descriptor(
                    reference_torch.float32, value=True
                ),
            ),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
            (
                lambda: actual_descriptor.__get__(1, int),
                lambda: expected_descriptor.__get__(1, int),
            ),
            (
                lambda: setattr(torch.float32, "to_real", None),
                lambda: setattr(reference_torch.float32, "to_real", None),
            ),
            (
                lambda: delattr(torch.float32, "to_real"),
                lambda: delattr(reference_torch.float32, "to_real"),
            ),
        )
        for actual_call, expected_call in cases:
            self.assertEqual(
                self.normalized_error(torch, actual_call),
                self.normalized_error(reference_torch, expected_call),
            )

    def test_complex_to_real_mapping_remains_outside_supported_dtypes(self):
        self.assertIs(
            reference_torch.complex64.to_real(), reference_torch.float32
        )
        self.assertFalse(hasattr(torch, "complex64"))


if __name__ == "__main__":
    unittest.main()
