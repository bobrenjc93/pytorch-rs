import struct
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def float64_from_bits(bits):
    return struct.unpack("=d", struct.pack("=Q", bits))[0]


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFloatListReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError(f"expected PyTorch 2.13.0, found {version}")

    def assert_matches(self, actual_source, expected_source):
        actual = torch.tensor(
            actual_source,
            dtype=torch.float32,
            device="cpu",
        )
        expected = reference_torch.tensor(
            expected_source,
            dtype=reference_torch.float32,
            device="cpu",
        )
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            expected.numpy().reshape(-1).view(np.uint32),
        )

    def assert_error_type_matches(self, actual_source, expected_source):
        with self.assertRaises(BaseException) as actual_raised:
            torch.tensor(actual_source, dtype=torch.float32)
        with self.assertRaises(BaseException) as expected_raised:
            reference_torch.tensor(expected_source, dtype=reference_torch.float32)
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )

    def test_exact_float_values_and_special_bits_match_pytorch_2_13(self):
        values = [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float64_from_bits(0x7FF8_0000_0000_0001),
            float64_from_bits(0xFFF8_1234_5678_9ABC),
            float64_from_bits(0x7FF0_0000_0000_0001),
            float64_from_bits(0xFFF0_0000_0000_0001),
            1.0000000596046448,
            1.0000001788139343,
            3.4028235677973366e38,
            1.0e-50,
        ]
        self.assert_matches(values, list(values))

    def test_metadata_copy_and_autograd_leaf_match_pytorch_2_13(self):
        actual_source = [1.25, -2.5, 4.0]
        expected_source = list(actual_source)
        actual = torch.tensor(
            actual_source,
            dtype=torch.float32,
            device="cpu",
            requires_grad=True,
        )
        expected = reference_torch.tensor(
            expected_source,
            dtype=reference_torch.float32,
            device="cpu",
            requires_grad=True,
        )

        actual_source[0] = 99.0
        expected_source[0] = 99.0
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint32),
            expected.detach().numpy().view(np.uint32),
        )

        actual.sum().backward()
        expected.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual.grad).view(np.uint32),
            expected.grad.numpy().view(np.uint32),
        )

    def test_nested_mixed_and_subclassed_inputs_match_pytorch_2_13(self):
        class ListSubclass(list):
            pass

        class FloatSubclass(float):
            def __float__(self):
                return 99.0

        cases = (
            (
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ),
            ([1.25, 2, 3.75], [1.25, 2, 3.75]),
            (ListSubclass([5.0, 6.0]), ListSubclass([5.0, 6.0])),
            (
                [FloatSubclass(1.25), FloatSubclass(-2.5)],
                [FloatSubclass(1.25), FloatSubclass(-2.5)],
            ),
        )
        for actual_source, expected_source in cases:
            with self.subTest(type=type(actual_source).__name__):
                self.assert_matches(actual_source, expected_source)

    def test_custom_numeric_values_and_fallback_errors_match_pytorch_2_13(self):
        class CustomNumeric:
            def __init__(self, value):
                self.value = value

            def __float__(self):
                return self.value

        self.assert_matches(
            [CustomNumeric(1.5), CustomNumeric(-3.25)],
            [CustomNumeric(1.5), CustomNumeric(-3.25)],
        )

        for actual_source, expected_source in (
            ([[1.0], [2.0, 3.0]], [[1.0], [2.0, 3.0]]),
            ([1.0, object()], [1.0, object()]),
        ):
            with self.subTest(source=actual_source):
                self.assert_error_type_matches(actual_source, expected_source)


if __name__ == "__main__":
    unittest.main()
