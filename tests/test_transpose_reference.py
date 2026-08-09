import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TransposeReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case, operation):
        with self.subTest(case=case, operation=operation):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
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

    def test_seeded_transposes_views_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x7A6A_213)
        shapes = [(), (0,), (2, 0, 3), (1, 3, 2), (2, 3, 4)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-3.0, 3.0, size=elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)
            else:
                actual = torch.tensor(values.item() if shape == () else values.tolist())
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)

            # Exercise both indexed bases and transpose-then-index views.
            if len(shape) >= 2 and shape[0] > 0 and case % 4 == 1:
                actual = actual[-1]
                expected = expected[-1]

            rank = len(actual.shape)
            dimensions = [0, -1] if rank == 0 else list(range(-rank, rank))
            for chain in range(2):
                dim0 = dimensions[int(rng.integers(0, len(dimensions)))]
                dim1 = dimensions[int(rng.integers(0, len(dimensions)))]
                if (case + chain) % 2:
                    actual = torch.transpose(actual, dim0, dim1)
                else:
                    actual = actual.transpose(dim0, dim1)
                expected = expected.transpose(dim0, dim1)

            if actual.shape and actual.shape[0] > 0 and case % 3 == 0:
                index = int(rng.integers(-actual.shape[0], actual.shape[0]))
                actual = actual[index]
                expected = expected[index]

            self.assert_matches(actual, expected, case=case, operation="view")
            for operation, actual_output, expected_output in (
                ("clone", actual.clone(), expected.clone()),
                ("relu", actual.relu(), expected.relu()),
                ("sin", actual.sin(), expected.sin()),
                ("exp", actual.exp(), expected.exp()),
                ("scalar", actual + 1.25, expected + 1.25),
                ("binary", actual + actual, expected + expected),
                ("reshape", actual.reshape(-1), expected.reshape(-1)),
                ("sum", actual.sum(), expected.sum()),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=case,
                    operation=operation,
                )

            if actual.shape:
                trailing = actual.shape[-1]
                actual_row = torch.zeros((trailing,))
                expected_row = reference_torch.zeros(
                    (trailing,), dtype=reference_torch.float32
                )
                self.assert_matches(
                    actual + actual_row,
                    expected + expected_row,
                    case=case,
                    operation="broadcast",
                )

    def test_seeded_rank_two_transposed_matmul_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x6A76_213)
        for case in range(20):
            rows, inner, columns = (
                int(value) for value in rng.integers(0, 7, size=3)
            )
            left_values = rng.normal(size=(inner, rows)).astype(np.float32)
            right_values = rng.normal(size=(columns, inner)).astype(np.float32)
            if left_values.size:
                actual_left = torch.tensor(left_values.tolist()).transpose(0, 1)
            else:
                actual_left = torch.zeros((inner, rows)).transpose(0, 1)
            if right_values.size:
                actual_right = torch.tensor(right_values.tolist()).transpose(0, 1)
            else:
                actual_right = torch.zeros((columns, inner)).transpose(0, 1)
            expected_left = reference_torch.tensor(left_values).transpose(0, 1)
            expected_right = reference_torch.tensor(right_values).transpose(0, 1)

            self.assert_matches(
                actual_left @ actual_right,
                expected_left @ expected_right,
                case=case,
                operation="matmul",
            )

    def test_python_layout_queries_and_transpose_diagnostics_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_channels_3d = (
            torch.zeros((2, 4, 5, 6, 3))
            .transpose(1, 4)
            .transpose(2, 4)
            .transpose(3, 4)
        )
        expected_channels_3d = (
            reference_torch.zeros((2, 4, 5, 6, 3))
            .transpose(1, 4)
            .transpose(2, 4)
            .transpose(3, 4)
        )
        tensor_cases = (
            (torch.zeros((2, 3, 4, 5)), reference_torch.zeros((2, 3, 4, 5))),
            (
                torch.zeros((1, 1, 2, 2)).transpose(1, 3),
                reference_torch.zeros((1, 1, 2, 2)).transpose(1, 3),
            ),
            (actual_channels_3d, expected_channels_3d),
            (
                torch.zeros((2, 4, 5, 0)).transpose(1, 3).transpose(2, 3),
                reference_torch.zeros((2, 4, 5, 0)).transpose(1, 3).transpose(2, 3),
            ),
        )
        formats = (
            (torch.preserve_format, reference_torch.preserve_format),
            (torch.contiguous_format, reference_torch.contiguous_format),
            (torch.channels_last, reference_torch.channels_last),
            (torch.channels_last_3d, reference_torch.channels_last_3d),
        )
        for case, (actual, expected) in enumerate(tensor_cases):
            with self.subTest(case=case, memory_format="default"):
                self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            for actual_format, expected_format in formats:
                with self.subTest(case=case, memory_format=actual_format):
                    self.assertEqual(
                        actual.is_contiguous(memory_format=actual_format),
                        expected.is_contiguous(memory_format=expected_format),
                    )

        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        error_cases = (
            (
                lambda: actual.is_contiguous(torch.contiguous_format),
                lambda: expected.is_contiguous(reference_torch.contiguous_format),
            ),
            (
                lambda: actual.is_contiguous(memory_format=None),
                lambda: expected.is_contiguous(memory_format=None),
            ),
            (
                lambda: actual.is_contiguous(unexpected=None),
                lambda: expected.is_contiguous(unexpected=None),
            ),
            (lambda: actual.transpose(1.5, 0), lambda: expected.transpose(1.5, 0)),
            (
                lambda: actual.transpose(dim0=1.5, dim1=0),
                lambda: expected.transpose(dim0=1.5, dim1=0),
            ),
            (
                lambda: torch.transpose(actual, 0, 1.5),
                lambda: reference_torch.transpose(expected, 0, 1.5),
            ),
            (
                lambda: torch.transpose(input=actual, dim0=0, dim1=1.5),
                lambda: reference_torch.transpose(input=expected, dim0=0, dim1=1.5),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_singleton_pointwise_and_reflected_division_layouts_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
        actual = torch.tensor(values.tolist()).transpose(1, 3)
        expected = reference_torch.tensor(values).transpose(1, 3)

        for operation, actual_output, expected_output in (
            ("relu", actual.relu(), expected.relu()),
            ("sin", actual.sin(), expected.sin()),
            ("binary", actual + actual, expected + expected),
            ("reflected_division", 1.0 / actual, 1.0 / expected),
        ):
            self.assert_matches(
                actual_output,
                expected_output,
                case="singleton_channels_last",
                operation=operation,
            )
            self.assertEqual(
                actual_output.reshape(actual_output.shape).stride(),
                expected_output.reshape(expected_output.shape).stride(),
            )

        actual_vector = torch.tensor([[1.0, 2.0]]).transpose(0, 1)
        expected_vector = reference_torch.tensor([[1.0, 2.0]]).transpose(0, 1)
        self.assert_matches(
            1.0 / actual_vector,
            1.0 / expected_vector,
            case="reflected_vector",
            operation="reflected_division",
        )

        empty_cases = (
            (
                torch.zeros((1, 0)).transpose(0, 1),
                reference_torch.zeros((1, 0)).transpose(0, 1),
            ),
            (torch.zeros((1, 0, 1)), reference_torch.zeros((1, 0, 1))),
            (
                torch.zeros((2, 0, 3)).transpose(0, 2),
                reference_torch.zeros((2, 0, 3)).transpose(0, 2),
            ),
        )
        for case, (actual_empty, expected_empty) in enumerate(empty_cases):
            self.assert_matches(
                1.0 / actual_empty,
                1.0 / expected_empty,
                case=f"reflected_empty_{case}",
                operation="reflected_division",
            )

        actual_large = torch.zeros((2, 0, sys.maxsize))
        expected_large = reference_torch.zeros((2, 0, sys.maxsize))
        self.assertEqual((1.0 / actual_large).stride(), (1.0 / expected_large).stride())

        actual_extreme = actual_large.transpose(0, 1)
        expected_extreme = expected_large.transpose(0, 1)
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            1.0 / actual_extreme
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            1.0 / expected_extreme


if __name__ == "__main__":
    unittest.main()
