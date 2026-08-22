import math
import struct
import unittest

import numpy as np
import torch_rs as torch


class TensorFlatFloatListTests(unittest.TestCase):
    def test_exact_float_list_preserves_values_metadata_and_autograd_leaf(self):
        positive_nan = struct.unpack("=d", struct.pack("=Q", 0x7FF8_1234_5678_9ABC))[0]
        negative_nan = struct.unpack("=d", struct.pack("=Q", 0xFFF8_7654_3210_ABCD))[0]
        values = [
            0.0,
            -0.0,
            1.0 + 2.0**-24,
            1.0 + 3.0 * 2.0**-24,
            float(np.finfo(np.float32).max),
            math.inf,
            -math.inf,
            positive_nan,
            negative_nan,
        ]

        tensor = torch.tensor(
            values,
            dtype=torch.float32,
            device=torch.device("cpu"),
            requires_grad=True,
        )
        self.assertEqual(tensor.shape, (len(values),))
        self.assertEqual(tensor.stride(), (1,))
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(tensor).view(np.uint32),
            np.asarray(values, dtype=np.float32).view(np.uint32),
        )

        tensor.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(tensor.grad),
            np.ones(len(values), dtype=np.float32),
        )

    def test_empty_exact_list_remains_a_contiguous_float32_vector(self):
        tensor = torch.tensor([], requires_grad=True)
        self.assertEqual(tensor.shape, (0,))
        self.assertEqual(tensor.stride(), (1,))
        self.assertEqual(tensor.tolist(), [])
        self.assertIs(tensor.dtype, torch.float32)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)

    def test_non_exact_inputs_keep_recursive_parser_behavior(self):
        events = []

        class TrackingList(list):
            def __len__(self):
                events.append("len")
                return super().__len__()

            def __getitem__(self, index):
                events.append(("getitem", index))
                return super().__getitem__(index)

        tracked = TrackingList([1.0, -2.0, 3.0])
        self.assertEqual(torch.tensor(tracked).tolist(), [1.0, -2.0, 3.0])
        self.assertEqual(events, ["len", ("getitem", 0), ("getitem", 1), ("getitem", 2)])

        float_events = []

        class FloatSubclass(float):
            def __float__(self):
                float_events.append("float subclass")
                return 99.0

        subclassed_values = [FloatSubclass(1.25), FloatSubclass(-2.5)]
        self.assertEqual(torch.tensor(subclassed_values).tolist(), [1.25, -2.5])
        self.assertEqual(float_events, [])

        class FloatLike:
            def __init__(self, value):
                self.value = value

            def __float__(self):
                float_events.append(self.value)
                return self.value

        mixed_values = [1.0, FloatLike(-2.5), 3]
        self.assertEqual(torch.tensor(mixed_values).tolist(), [1.0, -2.5, 3.0])
        self.assertEqual(float_events, [-2.5])

        self.assertEqual(torch.tensor((1.0, -2.0, 3.0)).tolist(), [1.0, -2.0, 3.0])
        self.assertEqual(
            torch.tensor([[1.0, -2.0], [3.0, 4.0]]).tolist(),
            [[1.0, -2.0], [3.0, 4.0]],
        )

    def test_fallback_errors_and_list_subclass_scalar_coercion_are_unchanged(self):
        events = []

        class RaisingFloat:
            def __float__(self):
                events.append("float")
                raise RuntimeError("float conversion failed")

        with self.assertRaisesRegex(TypeError, "rectangular sequence"):
            torch.tensor([1.0, RaisingFloat(), 3.0])
        self.assertEqual(events, ["float"])

        with self.assertRaisesRegex(ValueError, "nested shapes differ"):
            torch.tensor([[1.0], [2.0, 3.0]])

        events.clear()

        class ScalarList(list):
            def __float__(self):
                events.append("float")
                return 4.5

        scalar = torch.tensor(ScalarList([1.0, 2.0]))
        self.assertEqual(scalar.shape, ())
        self.assertEqual(scalar.item(), 4.5)
        self.assertEqual(events, ["float"])


if __name__ == "__main__":
    unittest.main()
