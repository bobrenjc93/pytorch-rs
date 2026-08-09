import ast
import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class PermuteReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case, operation):
        with self.subTest(case=case, operation=operation):
            self.assertEqual(actual.shape, tuple(expected.shape))
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

    def test_seeded_permutations_and_stride_aware_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x0E213)
        shapes = [
            (),
            (0,),
            (1,),
            (2, 0, 3),
            (0, 2, 3),
            (2, 1, 3),
            (1, 0, 1, 2),
            (1, 1, 1, 1, 1, 1, 1, 2),
        ]
        for _ in range(28):
            rank = int(rng.integers(0, 8))
            shapes.append(tuple(int(value) for value in rng.integers(0, 4, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-2.0, 2.0, size=elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)
            else:
                actual = torch.tensor(values.item() if shape == () else values.tolist())
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)

            if len(shape) >= 2 and case % 3 == 1:
                base_dimensions = tuple(range(1, len(shape))) + (0,)
                actual = actual.permute(base_dimensions)
                expected = expected.permute(base_dimensions)
            if actual.shape and actual.shape[0] > 0 and case % 5 == 2:
                actual = actual[-1]
                expected = expected[-1]

            rank = len(actual.shape)
            dimensions = [int(value) for value in rng.permutation(rank)]
            signed_dimensions = [
                dimension - rank if (case + index) % 2 else dimension
                for index, dimension in enumerate(dimensions)
            ]
            if rank == 0 or case % 4 == 0:
                actual_view = actual.permute(tuple(signed_dimensions))
                expected_view = expected.permute(tuple(signed_dimensions))
            elif case % 4 == 1:
                actual_view = actual.permute(*signed_dimensions)
                expected_view = expected.permute(*signed_dimensions)
            elif case % 4 == 2:
                actual_view = actual.permute(list(signed_dimensions))
                expected_view = expected.permute(list(signed_dimensions))
            else:
                actual_view = actual.permute(dims=tuple(signed_dimensions))
                expected_view = expected.permute(dims=tuple(signed_dimensions))

            self.assert_matches(actual_view, expected_view, case=case, operation="view")

            inverse = [0] * rank
            for output_axis, input_axis in enumerate(dimensions):
                inverse[input_axis] = output_axis
            self.assert_matches(
                actual_view.permute(inverse),
                expected_view.permute(inverse),
                case=case,
                operation="inverse",
            )

            for operation, actual_output, expected_output in (
                ("clone", actual_view.clone(), expected_view.clone()),
                ("relu", actual_view.relu(), expected_view.relu()),
                ("sin", actual_view.sin(), expected_view.sin()),
                ("exp", actual_view.exp(), expected_view.exp()),
                ("scalar", actual_view + 1.25, expected_view + 1.25),
                ("binary", actual_view + actual_view, expected_view + expected_view),
                ("subtract", 1.25 - actual_view, 1.25 - expected_view),
                ("multiply", actual_view * 1.25, expected_view * 1.25),
                ("divide", actual_view / 1.25, expected_view / 1.25),
                ("reshape", actual_view.reshape(-1), expected_view.reshape(-1)),
                ("sum", actual_view.sum(), expected_view.sum()),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=case,
                    operation=operation,
                )

            self.assertEqual(actual_view.tolist(), expected_view.tolist())
            expected_flat = expected_view.reshape(-1).tolist()
            representation = repr(actual_view)
            shape_suffix = f", shape={list(expected_view.shape)!r})"
            self.assertTrue(representation.startswith("tensor(["))
            self.assertTrue(representation.endswith(shape_suffix))
            represented_values = ast.literal_eval(
                representation[len("tensor(") : -len(shape_suffix)]
            )
            np.testing.assert_allclose(
                represented_values,
                expected_flat,
                rtol=2.0e-6,
                atol=1.0e-6,
            )

    def test_composed_permute_and_transpose_equivalence_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        actual = torch.tensor(values.tolist())
        expected = reference_torch.tensor(values)

        actual_composed = actual.permute(2, 0, 3, 1).permute(1, 3, 0, 2)
        expected_composed = expected.permute(2, 0, 3, 1).permute(1, 3, 0, 2)
        self.assert_matches(
            actual_composed,
            expected_composed,
            case="composed",
            operation="view",
        )

        actual_transpose = actual.transpose(0, 3)
        expected_transpose = expected.transpose(0, 3)
        self.assert_matches(
            actual.permute(3, 1, 2, 0),
            expected.permute(3, 1, 2, 0),
            case="transpose",
            operation="permute",
        )
        self.assertEqual(actual_transpose.stride(), expected_transpose.stride())
        np.testing.assert_array_equal(np.asarray(actual_transpose), expected_transpose.numpy())

        left_values = np.arange(6, dtype=np.float32).reshape(2, 3)
        right_values = np.arange(8, dtype=np.float32).reshape(4, 2)
        actual_left = torch.tensor(left_values.tolist()).permute(1, 0)
        actual_right = torch.tensor(right_values.tolist()).permute(1, 0)
        expected_left = reference_torch.tensor(left_values).permute(1, 0)
        expected_right = reference_torch.tensor(right_values).permute(1, 0)
        self.assert_matches(
            actual_left @ actual_right,
            expected_left @ expected_right,
            case="matmul",
            operation="permuted_inputs",
        )

    def test_errors_and_argument_binding_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        error_cases = (
            (lambda: actual.permute(), lambda: expected.permute()),
            (lambda: actual.permute(0, 1), lambda: expected.permute(0, 1)),
            (lambda: actual.permute(()), lambda: expected.permute(())),
            (lambda: actual.permute(0, -3, 2), lambda: expected.permute(0, -3, 2)),
            (lambda: actual.permute(0, 0, 3), lambda: expected.permute(0, 0, 3)),
            (lambda: actual.permute(3, 0, 0), lambda: expected.permute(3, 0, 0)),
            (lambda: actual.permute(0, 1, 3), lambda: expected.permute(0, 1, 3)),
            (lambda: actual.permute(-4, 1, 2), lambda: expected.permute(-4, 1, 2)),
            (
                lambda: actual.permute((0, 1, 2), 0),
                lambda: expected.permute((0, 1, 2), 0),
            ),
            (
                lambda: actual.permute(dims=(0, 1, 2), unexpected=True),
                lambda: expected.permute(dims=(0, 1, 2), unexpected=True),
            ),
            (
                lambda: actual.permute(0, 1, 2, dims=(0, 1, 2)),
                lambda: expected.permute(0, 1, 2, dims=(0, 1, 2)),
            ),
            (lambda: actual.permute(0.0, 1, 2), lambda: expected.permute(0.0, 1, 2)),
            (lambda: actual.permute(0, 1.0, 2), lambda: expected.permute(0, 1.0, 2)),
            (
                lambda: actual.permute((0.0, 1, 2)),
                lambda: expected.permute((0.0, 1, 2)),
            ),
            (
                lambda: actual.permute(dims=(0.0, 1, 2)),
                lambda: expected.permute(dims=(0.0, 1, 2)),
            ),
            (lambda: actual.permute(range(3)), lambda: expected.permute(range(3))),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_scalar = torch.tensor(1.0)
        expected_scalar = reference_torch.tensor(1.0)
        self.assert_error_matches(
            lambda: actual_scalar.permute(0),
            lambda: expected_scalar.permute(0),
        )

        actual_extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        expected_extreme = reference_torch.zeros((sys.maxsize, 0, 2, 2))
        self.assert_error_matches(
            lambda: actual_extreme.permute(0, 2, 3, 1),
            lambda: expected_extreme.permute(0, 2, 3, 1),
        )

    def test_integer_protocol_dimension_forms_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        dimension_cases = (
            (IntSubclass(0), np.int64(1), np.uint32(2)),
            (0, IndexOnly(), 2),
            (0, True, 2),
        )
        for case, dimensions in enumerate(dimension_cases):
            self.assert_matches(
                actual.permute(dimensions),
                expected.permute(dimensions),
                case=case,
                operation="integer_protocol",
            )


if __name__ == "__main__":
    unittest.main()
