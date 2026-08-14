import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ReluReferenceTests(unittest.TestCase):
    def assert_matches(self, actual_input, expected_input, expected_bits, *, case):
        actual_output = actual_input.relu()
        expected_output = expected_input.relu()

        with self.subTest(case=case):
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(
                actual_input.storage_offset(), expected_input.storage_offset()
            )
            self.assertEqual(actual_output.shape, tuple(expected_output.shape))
            self.assertEqual(actual_output.stride(), expected_output.stride())
            self.assertEqual(
                actual_output.storage_offset(), expected_output.storage_offset()
            )
            self.assertEqual(
                actual_output.is_contiguous(), expected_output.is_contiguous()
            )
            self.assertIs(actual_output.dtype, torch.float32)
            self.assertEqual(actual_output.device, torch.device("cpu"))
            self.assertFalse(actual_output.is_set_to(actual_input))
            self.assertFalse(expected_output.is_set_to(expected_input))

            actual_bits = (
                np.asarray(actual_output, dtype=np.float32)
                .reshape(-1)
                .view(np.uint32)
            )
            reference_bits = (
                expected_output.cpu().numpy().reshape(-1).view(np.uint32)
            )
            np.testing.assert_array_equal(actual_bits, reference_bits)
            np.testing.assert_array_equal(
                actual_bits, np.asarray(expected_bits, dtype=np.uint32).reshape(-1)
            )

    def test_signed_zero_values_layouts_and_copies_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        for zero_bits in (0x8000_0000, 0x0000_0000):
            zero = np.asarray(zero_bits, dtype=np.uint32).view(np.float32).item()
            self.assert_matches(
                torch.tensor(zero),
                reference_torch.tensor(zero),
                [zero_bits],
                case=("scalar", hex(zero_bits)),
            )

        input_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x8000_0001,
                0x0000_0001,
                0xFF7F_FFFF,
                0x7F7F_FFFF,
                0xBF00_0000,
                0x3F00_0000,
            ),
            dtype=np.uint32,
        )
        expected_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0x0000_0000,
                0x3F80_0000,
                0x0000_0000,
                0x7F80_0000,
                0x0000_0000,
                0x0000_0001,
                0x0000_0000,
                0x7F7F_FFFF,
                0x0000_0000,
                0x3F00_0000,
            ),
            dtype=np.uint32,
        )
        storage_bits = np.concatenate(
            (np.full(input_bits.size, 0x3F80_0000, dtype=np.uint32), input_bits)
        )
        storage_values = memoryview(storage_bits.view(np.float32))
        actual_offset = torch.tensor(storage_values).reshape(2, 3, 4)[1]
        expected_offset = reference_torch.tensor(storage_values).reshape(2, 3, 4)[1]

        self.assert_matches(
            actual_offset,
            expected_offset,
            expected_bits,
            case="offset contiguous",
        )
        self.assert_matches(
            actual_offset.transpose(0, 1),
            expected_offset.transpose(0, 1),
            expected_bits.reshape(3, 4).transpose(1, 0).reshape(-1),
            case="offset strided",
        )


if __name__ == "__main__":
    unittest.main()
