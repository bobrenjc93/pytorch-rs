import math
import struct
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFlatFloatListReferenceTests(unittest.TestCase):
    def assert_matches(self, values, *, requires_grad=False):
        actual = torch.tensor(
            values,
            dtype=torch.float32,
            device="cpu",
            requires_grad=requires_grad,
        )
        expected = reference_torch.tensor(
            values,
            dtype=reference_torch.float32,
            device="cpu",
            requires_grad=requires_grad,
        )
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            expected.detach().numpy().reshape(-1).view(np.uint32),
        )
        return actual, expected

    def test_exact_float_list_values_and_autograd_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [
            0.0,
            -0.0,
            1.0 + 2.0**-24,
            1.0 + 3.0 * 2.0**-24,
            float(np.finfo(np.float32).tiny),
            float(np.nextafter(np.float32(0), np.float32(1))),
            float(np.finfo(np.float32).max),
            math.inf,
            -math.inf,
            struct.unpack("=d", struct.pack("=Q", 0x7FF8_1234_5678_9ABC))[0],
            struct.unpack("=d", struct.pack("=Q", 0xFFF8_7654_3210_ABCD))[0],
        ]
        actual, expected = self.assert_matches(values, requires_grad=True)

        actual.sum().backward()
        expected.sum().backward()
        np.testing.assert_array_equal(np.asarray(actual.grad), expected.grad.numpy())

    def test_supported_fallback_inputs_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        class FloatSubclass(float):
            pass

        class ListSubclass(list):
            pass

        cases = (
            (),
            (1.0, -2.0, 3.0),
            [1.0, 2, -3.5],
            [[1.0, -2.0], [3.0, 4.0]],
            [FloatSubclass(1.25), FloatSubclass(-2.5)],
            ListSubclass([1.0, -2.0, 3.0]),
        )
        for values in cases:
            with self.subTest(type=type(values).__name__, values=values):
                self.assert_matches(values)

    def test_fallback_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for values in ([[1.0], [2.0, 3.0]],):
            with self.subTest(values=values):
                with self.assertRaises(BaseException) as actual_raised:
                    torch.tensor(values)
                with self.assertRaises(BaseException) as expected_raised:
                    reference_torch.tensor(values)
                self.assertEqual(
                    type(actual_raised.exception).__name__,
                    type(expected_raised.exception).__name__,
                )


if __name__ == "__main__":
    unittest.main()
