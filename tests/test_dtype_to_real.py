import inspect
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


class DTypeToRealTests(unittest.TestCase):
    def test_every_float32_alias_returns_the_canonical_singleton(self):
        aliases = (
            torch.float32,
            torch.float,
            torch.tensor(-0.0).dtype,
            torch.zeros((2, 0, 3)).transpose(0, 2).dtype,
            torch.tensor([1.0], requires_grad=True).dtype,
            torch.get_default_dtype(),
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertIs(alias, torch.float32)
                self.assertIs(alias.to_real(), torch.float32)

    def test_descriptor_metadata_signatures_and_unbound_call(self):
        descriptor = inspect.getattr_static(torch.dtype, "to_real")
        bound = torch.float32.to_real

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "to_real")
        self.assertEqual(descriptor.__qualname__, "dtype.to_real")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIsNone(bound.__doc__)
        self.assertIs(torch.dtype.to_real, descriptor)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        self.assertIs(bound.__self__, torch.float32)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertIs(descriptor(torch.float32), torch.float32)
        self.assertIs(
            descriptor.__get__(torch.float32, torch.dtype)(), torch.float32
        )

    def test_argument_receiver_and_attribute_errors(self):
        descriptor = inspect.getattr_static(torch.dtype, "to_real")
        bound = torch.float32.to_real
        cases = (
            (
                lambda: bound(1),
                TypeError,
                "dtype.to_real() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1, 2),
                TypeError,
                "dtype.to_real() takes no arguments (2 given)",
            ),
            (
                lambda: bound(value=True),
                TypeError,
                "dtype.to_real() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                TypeError,
                "unbound method dtype.to_real() needs an argument",
            ),
            (
                lambda: descriptor(torch.float32, 1),
                TypeError,
                "dtype.to_real() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(torch.float32, value=True),
                TypeError,
                "dtype.to_real() takes no keyword arguments",
            ),
            (
                lambda: descriptor(1),
                TypeError,
                "descriptor 'to_real' for 'torch_rs.dtype' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor.__get__(1, int),
                TypeError,
                "descriptor 'to_real' for 'torch_rs.dtype' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: setattr(torch.float32, "to_real", None),
                AttributeError,
                "'torch_rs.dtype' object attribute 'to_real' is read-only",
            ),
            (
                lambda: delattr(torch.float32, "to_real"),
                AttributeError,
                "'torch_rs.dtype' object attribute 'to_real' is read-only",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
