import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ContiguousReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.cpu().numpy(),
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_seeded_views_formats_identity_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0xC0171_213)
        shapes = [(), (0,), (1,), (2, 0, 3), (2, 3, 4), (2, 3, 2, 4)]
        for _ in range(28):
            rank = int(rng.integers(0, 7))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            if elements:
                actual = torch.tensor(values.item() if not shape else values.tolist())
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)
            else:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)

            if len(shape) >= 2:
                actual = actual.transpose(0, -1)
                expected = expected.transpose(0, -1)
            if actual.shape and actual.shape[0] > 0 and case % 3 == 1:
                actual = actual[-1]
                expected = expected[-1]
            if case % 4 == 0:
                actual = actual.squeeze()
                expected = expected.squeeze()

            formats = [(torch.contiguous_format, reference_torch.contiguous_format)]
            if len(actual.shape) == 4:
                formats.append((torch.channels_last, reference_torch.channels_last))
            if len(actual.shape) == 5:
                formats.append((torch.channels_last_3d, reference_torch.channels_last_3d))

            for actual_format, expected_format in formats:
                actual_output = actual.contiguous(memory_format=actual_format)
                expected_output = expected.contiguous(memory_format=expected_format)
                self.assertEqual(actual_output is actual, expected_output is expected)
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=(case, str(actual_format), "output"),
                )
                self.assertIs(
                    actual_output.contiguous(memory_format=actual_format), actual_output
                )

                for operation, actual_downstream, expected_downstream in (
                    ("reshape", actual_output.reshape(-1), expected_output.reshape(-1)),
                    ("clone", actual_output.clone(), expected_output.clone()),
                    ("arithmetic", actual_output * 1.5 + 0.25, expected_output * 1.5 + 0.25),
                    ("reduction", actual_output.sum(), expected_output.sum()),
                ):
                    self.assert_matches(
                        actual_downstream,
                        expected_downstream,
                        case=(case, str(actual_format), operation),
                    )
                self.assertEqual(actual_output.tolist(), expected_output.tolist())
                self.assertEqual(np.asarray(actual_output).shape, expected_output.cpu().numpy().shape)
                self.assertIn("tensor(", repr(actual_output))

    def test_special_values_zero_sizes_and_canonical_channel_strides(self):
        bits = np.array(
            [0x00000000, 0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000, 0x40A00000],
            dtype=np.uint32,
        )
        values = bits.view(np.float32).reshape(2, 3)
        actual = torch.tensor(values.tolist()).transpose(0, 1).contiguous()
        expected = reference_torch.tensor(values).transpose(0, 1).contiguous()
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            expected.numpy().reshape(-1).view(np.uint32),
        )

        for shape, actual_format, expected_format in (
            ((2, 0, 4, 5), torch.channels_last, reference_torch.channels_last),
            ((2, 3, 0, 5), torch.channels_last, reference_torch.channels_last),
            ((0, 3, 4, 5), torch.channels_last, reference_torch.channels_last),
            ((2, 3, 4, 0, 6), torch.channels_last_3d, reference_torch.channels_last_3d),
        ):
            actual = torch.zeros(shape).contiguous(memory_format=actual_format)
            expected = reference_torch.zeros(shape).contiguous(memory_format=expected_format)
            self.assert_matches(actual, expected, case=(shape, str(actual_format)))

    def test_argument_rank_preserve_and_overflow_errors_match_exactly(self):
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        self.assertIs(
            actual.contiguous(memory_format=torch.preserve_format), actual
        )
        self.assertIs(
            expected.contiguous(memory_format=reference_torch.preserve_format),
            expected,
        )
        calls = (
            (lambda: actual.contiguous(torch.contiguous_format), lambda: expected.contiguous(reference_torch.contiguous_format)),
            (lambda: actual.contiguous(memory_format=None), lambda: expected.contiguous(memory_format=None)),
            (lambda: actual.contiguous(memory_format=1), lambda: expected.contiguous(memory_format=1)),
            (
                lambda: actual.contiguous(unexpected=None, memory_format=1),
                lambda: expected.contiguous(unexpected=None, memory_format=1),
            ),
            (lambda: actual.contiguous(unexpected=None), lambda: expected.contiguous(unexpected=None)),
            (lambda: actual.contiguous(memory_format=torch.channels_last), lambda: expected.contiguous(memory_format=reference_torch.channels_last)),
            (lambda: actual.contiguous(memory_format=torch.channels_last_3d), lambda: expected.contiguous(memory_format=reference_torch.channels_last_3d)),
            (
                lambda: actual.transpose(0, 1).contiguous(memory_format=torch.preserve_format),
                lambda: expected.transpose(0, 1).contiguous(memory_format=reference_torch.preserve_format),
            ),
        )
        for actual_call, expected_call in calls:
            self.assert_error_matches(actual_call, expected_call)

        for rank in range(7):
            actual_ranked = torch.zeros((2,) * rank)
            expected_ranked = reference_torch.zeros((2,) * rank)
            if rank != 4:
                self.assert_error_matches(
                    lambda tensor=actual_ranked: tensor.contiguous(
                        memory_format=torch.channels_last
                    ),
                    lambda tensor=expected_ranked: tensor.contiguous(
                        memory_format=reference_torch.channels_last
                    ),
                )
            if rank != 5:
                self.assert_error_matches(
                    lambda tensor=actual_ranked: tensor.contiguous(
                        memory_format=torch.channels_last_3d
                    ),
                    lambda tensor=expected_ranked: tensor.contiguous(
                        memory_format=reference_torch.channels_last_3d
                    ),
                )

        maximum = sys.maxsize
        actual_extreme = torch.zeros((0,)).reshape((2, 0, maximum, maximum))
        expected_extreme = reference_torch.zeros((0,)).reshape((2, 0, maximum, maximum))
        self.assert_error_matches(
            lambda: actual_extreme.contiguous(memory_format=torch.channels_last),
            lambda: expected_extreme.contiguous(memory_format=reference_torch.channels_last),
        )


if __name__ == "__main__":
    unittest.main()
