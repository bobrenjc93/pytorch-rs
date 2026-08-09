import sys
import unittest

import numpy as np
import torch_rs as torch


class FlattenTests(unittest.TestCase):
    def test_method_and_top_level_forms_identity_and_scalar(self):
        source = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        )
        for output in (
            source.flatten(),
            source.flatten(0),
            source.flatten(start_dim=0, end_dim=-1),
            torch.flatten(source),
            torch.flatten(source, 0, -1),
            torch.flatten(input=source, start_dim=0, end_dim=-1),
        ):
            self.assertEqual(output.shape, (8,))
            self.assertEqual(output.stride(), (1,))
            self.assertEqual(output.tolist(), list(np.arange(1, 9, dtype=np.float32)))

        self.assertIs(source.flatten(1, 1), source)
        self.assertIs(source.flatten(start_dim=-1, end_dim=-1), source)
        self.assertIs(torch.flatten(source, 0, 0), source)
        self.assertIs(torch.flatten(input=source, start_dim=-1, end_dim=-1), source)

        scalar = torch.tensor(-0.0)
        scalar_output = scalar.flatten()
        self.assertIsNot(scalar_output, scalar)
        self.assertEqual(scalar_output.shape, (1,))
        self.assertEqual(scalar_output.stride(), (1,))
        self.assertEqual(
            np.asarray(scalar_output).view(np.uint32).item(),
            np.asarray([-0.0], dtype=np.float32).view(np.uint32).item(),
        )

    def test_stride_compatible_ranges_preserve_offsets_and_incompatible_ranges_copy(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        source = torch.tensor(values.tolist()).transpose(0, 1)[1]
        compatible = source.flatten(1, -1)
        self.assertEqual(source.shape, (2, 4, 5))
        self.assertEqual(compatible.shape, (2, 20))
        self.assertEqual(compatible.stride(), (60, 1))
        self.assertEqual(compatible.storage_offset(), 20)
        self.assertEqual(
            compatible.tolist(), values.transpose(1, 0, 2, 3)[1].reshape(2, 20).tolist()
        )

        non_contiguous = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()).transpose(0, 1)
        copied = non_contiguous.flatten(0, 1)
        self.assertEqual(copied.shape, (6, 4))
        self.assertEqual(copied.stride(), (4, 1))
        self.assertEqual(copied.storage_offset(), 0)
        self.assertTrue(copied.is_contiguous())
        self.assertEqual(
            copied.tolist(), np.arange(24, dtype=np.float32).reshape(2, 3, 4).transpose(1, 0, 2).reshape(6, 4).tolist()
        )

        del non_contiguous
        self.assertEqual(copied.clone().tolist(), copied.tolist())
        self.assertEqual(copied.reshape(2, 12).stride(), (12, 1))
        self.assertEqual((copied + 1).sum().item(), sum(range(24)) + 24)
        self.assertEqual(np.asarray(copied).shape, (6, 4))
        self.assertIn("tensor(", repr(copied))

    def test_empty_singleton_high_rank_and_extreme_metadata(self):
        cases = (
            ((0,), (0,), (1,)),
            ((2, 0, 3), (0,), (1,)),
            ((1, 2, 1, 3), (6,), (1,)),
            ((1,) * 65, (1,), (1,)),
        )
        for shape, expected_shape, expected_stride in cases:
            with self.subTest(shape=shape):
                output = torch.zeros(shape).flatten()
                self.assertEqual(output.shape, expected_shape)
                self.assertEqual(output.stride(), expected_stride)
                self.assertEqual(output.numel(), int(np.prod(shape)) if shape else 1)

        partial = torch.zeros((2, 0, 3)).transpose(0, 2).flatten(1, 2)
        self.assertEqual(partial.shape, (3, 0))
        self.assertEqual(partial.stride(), (1, 1))
        self.assertEqual(partial.tolist(), [[], [], []])

        maximum = sys.maxsize
        extreme = torch.zeros((0,)).reshape((0, maximum, maximum))
        wrapped = extreme.flatten(1, 2)
        self.assertEqual(wrapped.shape, (0, 1))
        self.assertEqual(wrapped.stride(), (1, 1))

        wrapping_negative_one = torch.zeros(
            (3, 0, 6_148_914_691_236_517_205)
        ).transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "unspecified dimension size -1"):
            wrapping_negative_one.flatten(1, 2)

    def test_binding_types_dimension_order_and_diagnostics(self):
        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(tensor.flatten(np.int64(1), np.int32(2)).shape, (2, 12))
        self.assertEqual(torch.flatten(tensor, start_dim=-2, end_dim=-1).shape, (2, 12))

        cases = (
            (lambda: tensor.flatten(2, 1), RuntimeError, "flatten() has invalid args: start_dim cannot come after end_dim"),
            (lambda: tensor.flatten(-4), IndexError, "Dimension out of range (expected to be in range of [-3, 2], but got -4)"),
            (lambda: tensor.flatten(end_dim=3), IndexError, "Dimension out of range (expected to be in range of [-3, 2], but got 3)"),
            (lambda: tensor.flatten(None), TypeError, "flatten(): argument 'start_dim' (position 1) must be int, not NoneType"),
            (lambda: tensor.flatten(start_dim=torch.float32), TypeError, "flatten(): argument 'start_dim' must be int, not torch.dtype"),
            (lambda: tensor.flatten(True), TypeError, "flatten(): argument 'start_dim' (position 1) must be int, not bool"),
            (lambda: tensor.flatten(0, 1, 2), TypeError, "flatten() takes from 0 to 2 positional arguments but 3 were given"),
            (lambda: tensor.flatten(0, start_dim=1), TypeError, "flatten() got multiple values for argument 'start_dim'"),
            (lambda: tensor.flatten(dim=1), TypeError, "flatten() got an unexpected keyword argument 'dim'"),
            (lambda: torch.flatten(), TypeError, 'flatten() missing 1 required positional arguments: "input"'),
            (lambda: torch.flatten([1.0]), TypeError, "flatten(): argument 'input' (position 1) must be Tensor, not list"),
            (lambda: torch.flatten(input=1), TypeError, "flatten(): argument 'input' must be Tensor, not int"),
            (lambda: torch.flatten(tensor, None), TypeError, "flatten(): argument 'start_dim' (position 2) must be int, not NoneType"),
            (lambda: torch.flatten(tensor, input=tensor), TypeError, "flatten() got multiple values for argument 'input'"),
            (lambda: torch.flatten(tensor, 0, -1, 1), TypeError, "flatten() takes from 1 to 3 positional arguments but 4 were given"),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, "^" + __import__("re").escape(message) + "$"):
                    call()

        for call in (lambda: tensor.flatten(2**100), lambda: torch.flatten(tensor, 2**100)):
            with self.assertRaisesRegex(ValueError, "^Overflow when unpacking long long$"):
                call()


if __name__ == "__main__":
    unittest.main()
