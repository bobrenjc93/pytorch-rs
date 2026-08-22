import struct
import unittest

import numpy as np
import torch_rs as torch


def float32_from_bits(bits):
    return float(np.array(bits, dtype=np.uint32).view(np.float32)[()])


def float64_from_bits(bits):
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def float32_bits(tensor):
    value = tensor.detach() if tensor.requires_grad else tensor
    return np.asarray(value).view(np.uint32).reshape(-1).tolist()


class TensorFloatListTests(unittest.TestCase):
    def test_exact_float_list_preserves_values_metadata_and_leaf_autograd(self):
        values = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            float32_from_bits(0x00000001),
            float32_from_bits(0x007FFFFF),
            float32_from_bits(0x00800000),
            float32_from_bits(0x7F7FFFFF),
            float("inf"),
            float("-inf"),
            float32_from_bits(0x7FC12345),
            float32_from_bits(0xFFC12345),
            float32_from_bits(0x7F812345),
            float32_from_bits(0xFF812345),
            float64_from_bits(0x7FF8000000000001),
            float64_from_bits(0xFFF8000000000001),
        ]
        expected_bits = [
            0x00000000,
            0x80000000,
            0x3F800000,
            0xC0200000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC12345,
            0xFFC12345,
            0x7FC12345,
            0xFFC12345,
            0x7FC00000,
            0xFFC00000,
        ]

        tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=torch.device("cpu"),
            requires_grad=True,
        )
        values[0] = 99.0

        self.assertEqual(tensor.shape, (len(expected_bits),))
        self.assertEqual(tensor.stride(), (1,))
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), len(expected_bits))
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertIsNone(tensor.grad)
        self.assertEqual(float32_bits(tensor), expected_bits)

        tensor.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [1.0] * len(expected_bits))

    def test_empty_exact_list_keeps_vector_metadata(self):
        tensor = torch.tensor([])
        self.assertEqual(tensor.shape, (0,))
        self.assertEqual(tensor.stride(), (1,))
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), 0)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertTrue(tensor.is_leaf)
        self.assertFalse(tensor.requires_grad)
        self.assertEqual(tensor.tolist(), [])

    def test_nested_mixed_and_subclassed_inputs_keep_fallback_behavior(self):
        class ListSubclass(list):
            calls = []

            def __len__(self):
                type(self).calls.append("len")
                return super().__len__()

            def __getitem__(self, index):
                type(self).calls.append(("getitem", index))
                return super().__getitem__(index)

        class FloatSubclass(float):
            calls = 0

            def __float__(self):
                type(self).calls += 1
                return 99.0

        class CustomNumeric:
            calls = 0

            def __float__(self):
                type(self).calls += 1
                return 3.5

        self.assertEqual(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]).tolist(),
            [[1.0, 2.0], [3.0, 4.0]],
        )
        self.assertEqual(torch.tensor([1.0, 2, 3.0]).tolist(), [1.0, 2.0, 3.0])

        subclassed_list = ListSubclass([1.0, 2.0])
        self.assertEqual(torch.tensor(subclassed_list).tolist(), [1.0, 2.0])
        self.assertEqual(
            ListSubclass.calls,
            ["len", ("getitem", 0), ("getitem", 1)],
        )

        self.assertEqual(
            torch.tensor([1.0, FloatSubclass(2.5)]).tolist(),
            [1.0, 2.5],
        )
        self.assertEqual(FloatSubclass.calls, 0)

        self.assertEqual(
            torch.tensor(
                [1.0, CustomNumeric()], dtype=torch.float32
            ).tolist(),
            [1.0, 3.5],
        )
        self.assertEqual(CustomNumeric.calls, 1)

    def test_late_fallback_errors_are_not_reparsed_or_rewritten(self):
        class RaisingNumeric:
            calls = 0

            def __float__(self):
                type(self).calls += 1
                raise RuntimeError("custom conversion failed")

        with self.assertRaisesRegex(
            TypeError,
            "^tensor data must contain real numbers in a rectangular sequence$",
        ):
            torch.tensor([1.0, 2.0, RaisingNumeric()], dtype=torch.float32)
        self.assertEqual(RaisingNumeric.calls, 1)

        with self.assertRaisesRegex(
            ValueError,
            "^expected a rectangular sequence, but nested shapes differ$",
        ):
            torch.tensor([1.0, [2.0]])
        with self.assertRaisesRegex(
            TypeError,
            "^tensor data must contain real numbers in a rectangular sequence$",
        ):
            torch.tensor([1.0, object()])


if __name__ == "__main__":
    unittest.main()
