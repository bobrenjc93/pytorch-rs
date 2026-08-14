import inspect
import sys
import types
import unittest

import torch_rs as torch


class DTypeIsFloatingPointTests(unittest.TestCase):
    def test_float32_aliases_expose_the_native_dtype_category(self):
        aliases = (
            torch.float32,
            torch.float,
            torch.tensor(3.5).dtype,
            torch.get_default_dtype(),
        )
        for alias in aliases:
            with self.subTest(alias=alias):
                self.assertIs(alias, torch.float32)
                self.assertIs(alias.is_floating_point, True)

    def test_descriptor_is_owned_by_dtype_and_is_read_only(self):
        descriptor = inspect.getattr_static(torch.dtype, "is_floating_point")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertEqual(descriptor.__name__, "is_floating_point")
        self.assertEqual(descriptor.__qualname__, "dtype.is_floating_point")
        self.assertIs(descriptor.__objclass__, torch.dtype)
        self.assertIsNone(descriptor.__doc__)
        self.assertIs(descriptor.__get__(None, torch.dtype), descriptor)
        self.assertIs(descriptor.__get__(torch.float32, torch.dtype), True)

        actions = (
            lambda: setattr(torch.float32, "is_floating_point", False),
            lambda: delattr(torch.float32, "is_floating_point"),
            lambda: descriptor.__set__(torch.float32, False),
            lambda: descriptor.__delete__(torch.float32),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'is_floating_point' of 'torch_rs.dtype' objects is not writable",
                )

    def test_dtype_tensor_and_top_level_queries_agree_without_data_access(self):
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        tensors = (
            torch.tensor(3.5),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2),
            extreme_empty,
        )
        self.assertFalse(tensors[2].is_contiguous())
        for tensor in tensors:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                results = (
                    tensor.dtype.is_floating_point,
                    tensor.is_floating_point(),
                    torch.is_floating_point(tensor),
                )
                self.assertEqual(results, (True, True, True))
                self.assertTrue(all(type(result) is bool for result in results))


if __name__ == "__main__":
    unittest.main()
