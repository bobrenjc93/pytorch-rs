import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SqueezeReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case, operation):
        with self.subTest(case=case, operation=operation):
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

    def test_seeded_values_metadata_and_composed_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x5AEE2E_213)
        shapes = [(), (1,), (0,), (1, 0, 1), (1, 2, 1, 3, 1)]
        for _ in range(32):
            rank = int(rng.integers(0, 9))
            shape = tuple(int(value) for value in rng.integers(0, 5, size=rank))
            if rank and not any(dimension == 1 for dimension in shape):
                axis = int(rng.integers(0, rank))
                shape = shape[:axis] + (1,) + shape[axis + 1 :]
            shapes.append(shape)

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

            rank = len(actual.shape)
            if case % 5 == 0:
                actual = torch.squeeze(actual)
                expected = reference_torch.squeeze(expected)
            elif case % 5 == 1:
                actual = actual.squeeze()
                expected = expected.squeeze()
            elif case % 5 == 2:
                dimension = -1 if rank == 0 else int(rng.integers(-rank, rank))
                actual = actual.squeeze(dimension)
                expected = expected.squeeze(dimension)
            else:
                dimensions = [0] if rank == 0 else list(range(rank))[::2]
                if case % 5 == 3:
                    actual = actual.squeeze(tuple(dimensions))
                    expected = expected.squeeze(tuple(dimensions))
                else:
                    actual = torch.squeeze(actual, dim=dimensions)
                    expected = reference_torch.squeeze(expected, dim=dimensions)

            self.assert_matches(actual, expected, case=case, operation="view")
            for operation, actual_output, expected_output in (
                ("clone", actual.clone(), expected.clone()),
                ("relu", actual.relu(), expected.relu()),
                ("arithmetic", actual * 1.5 + 0.25, expected * 1.5 + 0.25),
                ("sum", actual.sum(), expected.sum()),
                ("reshape", actual.reshape(-1), expected.reshape(-1)),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=case,
                    operation=operation,
                )

            self.assertEqual(actual.tolist(), expected.tolist())
            np.testing.assert_allclose(
                np.asarray(actual), expected.numpy(), rtol=2.0e-6, atol=1.0e-6
            )
            self.assertIn("tensor(", repr(actual))
            if len(actual.shape) >= 2:
                self.assert_matches(
                    actual.transpose(0, -1),
                    expected.transpose(0, -1),
                    case=case,
                    operation="transpose",
                )

    def test_non_contiguous_offsets_forms_and_top_level_method_parity(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(48, dtype=np.float32).reshape(2, 1, 3, 2, 4)
        actual_source = torch.tensor(values.tolist()).transpose(0, 4)[1]
        expected_source = reference_torch.tensor(values).transpose(0, 4)[1]

        calls = (
            (lambda value: value.squeeze(), lambda value: value.squeeze()),
            (lambda value: value.squeeze(0), lambda value: value.squeeze(0)),
            (lambda value: value.squeeze((0, 2)), lambda value: value.squeeze((0, 2))),
            (lambda value: value.squeeze([0, -3]), lambda value: value.squeeze([0, -3])),
            (lambda value: torch.squeeze(value), lambda value: reference_torch.squeeze(value)),
            (
                lambda value: torch.squeeze(value, dim=(0, -3)),
                lambda value: reference_torch.squeeze(value, dim=(0, -3)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(calls):
            self.assert_matches(
                actual_call(actual_source),
                expected_call(expected_source),
                case=case,
                operation="form",
            )

        actual_method = actual_source.squeeze(0, 2)
        expected_method = expected_source.squeeze(0, 2)
        self.assert_matches(actual_method, expected_method, case="variadic", operation="method")

    def test_scalar_zero_sized_high_rank_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for shape in ((), (1,), (0, 1), (1, 0), (1, 0, 1, 2)):
            actual = torch.zeros(shape)
            expected = reference_torch.zeros(shape)
            for case, (actual_call, expected_call) in enumerate(
                (
                    (lambda: actual.squeeze(), lambda: expected.squeeze()),
                    (lambda: actual.squeeze(()), lambda: expected.squeeze(())),
                    (lambda: torch.squeeze(actual), lambda: reference_torch.squeeze(expected)),
                )
            ):
                self.assert_matches(
                    actual_call(), expected_call(), case=(shape, case), operation="edge"
                )

        actual_high = torch.zeros((1,) * 65)
        expected_high = reference_torch.zeros((1,) * 65)
        self.assert_matches(
            actual_high.squeeze(), expected_high.squeeze(), case="rank65", operation="all"
        )
        self.assert_matches(
            actual_high.squeeze(0),
            expected_high.squeeze(0),
            case="rank65",
            operation="single",
        )
        self.assert_error_matches(
            lambda: actual_high.squeeze(()), lambda: expected_high.squeeze(())
        )

        actual = torch.zeros((1, 2, 1))
        expected = reference_torch.zeros((1, 2, 1))
        self.assert_matches(
            actual.squeeze([0, True]),
            expected.squeeze([0, True]),
            case="mixed-bool-list",
            operation="binding",
        )
        error_cases = (
            (lambda: actual.squeeze(None), lambda: expected.squeeze(None)),
            (lambda: torch.squeeze(actual, None), lambda: reference_torch.squeeze(expected, None)),
            (
                lambda: torch.squeeze(input=actual, dim=None),
                lambda: reference_torch.squeeze(input=expected, dim=None),
            ),
            (lambda: actual.squeeze(True), lambda: expected.squeeze(True)),
            (lambda: actual.squeeze(dim=np.float64(1)), lambda: expected.squeeze(dim=np.float64(1))),
            (lambda: actual.squeeze([1.0]), lambda: expected.squeeze([1.0])),
            (lambda: actual.squeeze(((0,),)), lambda: expected.squeeze(((0,),))),
            (lambda: actual.squeeze((0, -3)), lambda: expected.squeeze((0, -3))),
            (lambda: torch.squeeze(actual, [0, 0]), lambda: reference_torch.squeeze(expected, [0, 0])),
            (lambda: actual.squeeze(3), lambda: expected.squeeze(3)),
            (lambda: actual.squeeze(2**100), lambda: expected.squeeze(2**100)),
            (
                lambda: torch.squeeze(np.zeros((1,), dtype=np.float32)),
                lambda: reference_torch.squeeze(np.zeros((1,), dtype=np.float32)),
            ),
            (
                lambda: torch.squeeze(np.zeros((1,), dtype=np.float32), 0),
                lambda: reference_torch.squeeze(np.zeros((1,), dtype=np.float32), 0),
            ),
            (lambda: actual.squeeze(torch.float32), lambda: expected.squeeze(reference_torch.float32)),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        with self.assertRaises(TypeError) as actual_overflow:
            actual.squeeze([2**100])
        with self.assertRaises(TypeError) as expected_overflow:
            expected.squeeze([2**100])
        for error in (actual_overflow.exception, expected_overflow.exception):
            self.assertIn("failed to unpack", str(error))
            self.assertIn("Overflow when unpacking long long", str(error))

        binding_cases = (
            (lambda: torch.squeeze(), lambda: reference_torch.squeeze()),
            (lambda: torch.squeeze(actual, 0, 2), lambda: reference_torch.squeeze(expected, 0, 2)),
            (lambda: actual.squeeze(0, dim=2), lambda: expected.squeeze(0, dim=2)),
        )
        for actual_call, expected_call in binding_cases:
            self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
