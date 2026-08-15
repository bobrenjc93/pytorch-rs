import copy
import math
import pickle
import struct
import unittest

import torch_rs as torch


CONSTANT_BITS = {
    "e": 0x4005BF0A8B145769,
    "pi": 0x400921FB54442D18,
    "nan": 0x7FF8000000000000,
    "inf": 0x7FF0000000000000,
}


def float_bits(value):
    return int.from_bytes(struct.pack(">d", value), "big")


def as_float32(value):
    return struct.unpack(">f", struct.pack(">f", value))[0]


class NumericConstantTests(unittest.TestCase):
    def test_constants_are_exact_python_floats_with_ieee_classifications(self):
        for name, expected_bits in CONSTANT_BITS.items():
            with self.subTest(name=name):
                value = getattr(torch, name)
                self.assertIs(type(value), float)
                self.assertIs(value, getattr(math, name))
                self.assertEqual(float_bits(value), expected_bits)

        self.assertTrue(math.isfinite(torch.e))
        self.assertTrue(math.isfinite(torch.pi))
        self.assertGreater(torch.inf, 0.0)
        self.assertTrue(math.isinf(torch.inf))
        self.assertTrue(math.isnan(torch.nan))

    def test_direct_and_wildcard_exports_are_stable(self):
        self.assertEqual(
            [name for name in torch.__all__ if name in CONSTANT_BITS],
            ["e", "pi", "nan", "inf"],
        )
        for name in CONSTANT_BITS:
            self.assertEqual(torch.__all__.count(name), 1)
            self.assertFalse(hasattr(torch._C, name))

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        for name in CONSTANT_BITS:
            with self.subTest(name=name):
                self.assertIs(wildcard_namespace[name], getattr(torch, name))

    def test_copy_and_pickle_preserve_plain_float_contracts(self):
        for name, expected_bits in CONSTANT_BITS.items():
            value = getattr(torch, name)
            with self.subTest(name=name, operation="copy"):
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(value, protocol=protocol))
                    self.assertIs(type(restored), float)
                    self.assertIsNot(restored, value)
                    self.assertEqual(float_bits(restored), expected_bits)

    def test_tensor_defaults_and_scalar_arithmetic_are_unchanged(self):
        default_dtype = torch.get_default_dtype()
        base = torch.tensor([1.0, -2.0, 0.0])

        self.assertIs(default_dtype, torch.float32)
        self.assertIs(base.dtype, default_dtype)
        for name in CONSTANT_BITS:
            value = getattr(torch, name)
            with self.subTest(name=name):
                self.assertIs(torch.tensor(value).dtype, default_dtype)
                self.assertIs(torch.scalar_tensor(value).dtype, default_dtype)
                for result in (
                    base + value,
                    value + base,
                    base * value,
                    value * base,
                ):
                    self.assertIs(result.dtype, default_dtype)

        self.assertEqual(base.tolist(), [1.0, -2.0, 0.0])
        self.assertIs(torch.get_default_dtype(), default_dtype)
        pi32 = as_float32(torch.pi)
        e32 = as_float32(torch.e)
        self.assertEqual(
            (base + torch.pi).tolist(),
            [as_float32(1.0 + pi32), as_float32(-2.0 + pi32), pi32],
        )
        self.assertEqual(
            (base * torch.e).tolist(),
            [e32, as_float32(-2.0 * e32), 0.0],
        )
        self.assertTrue(all(math.isinf(value) for value in (base + torch.inf).tolist()))
        self.assertTrue(all(math.isnan(value) for value in (base + torch.nan).tolist()))


if __name__ == "__main__":
    unittest.main()
