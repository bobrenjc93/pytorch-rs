import sys
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FlattenReferenceTests(unittest.TestCase):
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

    def test_seeded_layout_compositions_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0xF1A77E_213)
        shapes = [(), (0,), (1,), (2, 0, 3), (1, 2, 1, 3), (2, 3, 4)]
        for _ in range(36):
            rank = int(rng.integers(0, 8))
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

            rank = len(actual.shape)
            if rank == 0:
                start_dim = end_dim = 0
            else:
                start_dim = int(rng.integers(0, rank))
                end_dim = int(rng.integers(start_dim, rank))
                if case % 2:
                    start_dim -= rank
                    end_dim -= rank

            if case % 2:
                actual_output = torch.flatten(actual, start_dim, end_dim)
                expected_output = reference_torch.flatten(expected, start_dim, end_dim)
            else:
                actual_output = actual.flatten(start_dim, end_dim)
                expected_output = expected.flatten(start_dim, end_dim)

            self.assertEqual(actual_output is actual, expected_output is expected)
            self.assert_matches(actual_output, expected_output, case=case, operation="flatten")
            for operation, actual_result, expected_result in (
                ("squeeze", actual_output.squeeze(), expected_output.squeeze()),
                ("contiguous", actual_output.contiguous(), expected_output.contiguous()),
                ("reshape", actual_output.reshape(-1), expected_output.reshape(-1)),
                ("clone", actual_output.clone(), expected_output.clone()),
                ("arithmetic", actual_output * 1.5 + 0.25, expected_output * 1.5 + 0.25),
            ):
                self.assert_matches(
                    actual_result,
                    expected_result,
                    case=case,
                    operation=operation,
                )
            np.testing.assert_allclose(
                np.asarray(actual_output.sum()),
                expected_output.sum().numpy(),
                rtol=1.0e-4,
                atol=1.0e-5,
            )
            if len(actual_output.shape) >= 2:
                self.assert_matches(
                    actual_output.transpose(0, -1),
                    expected_output.transpose(0, -1),
                    case=case,
                    operation="transpose",
                )
            self.assertEqual(actual_output.tolist(), expected_output.tolist())
            self.assertEqual(np.asarray(actual_output).shape, expected_output.numpy().shape)
            self.assertIn("tensor(", repr(actual_output))

    def test_view_copy_offsets_lifetimes_and_special_values_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        actual_source = torch.tensor(values.tolist()).transpose(0, 1)[1]
        expected_source = reference_torch.tensor(values).transpose(0, 1)[1]
        actual_view = actual_source.flatten(1, -1)
        expected_view = expected_source.flatten(1, -1)
        self.assert_matches(actual_view, expected_view, case="compatible", operation="view")
        self.assertEqual(
            expected_view.untyped_storage().data_ptr(),
            expected_source.untyped_storage().data_ptr(),
        )

        bits = np.array(
            [0x00000000, 0x80000000, 0x7FC12345, 0x7F800000, 0xFF800000, 0x40A00000],
            dtype=np.uint32,
        )
        special = bits.view(np.float32).reshape(2, 3)
        actual_copy = torch.tensor(special.tolist()).transpose(0, 1).flatten()
        expected_base = reference_torch.tensor(special).transpose(0, 1)
        expected_copy = expected_base.flatten()
        self.assert_matches(actual_copy, expected_copy, case="incompatible", operation="copy")
        self.assertNotEqual(
            expected_copy.untyped_storage().data_ptr(),
            expected_base.untyped_storage().data_ptr(),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_copy).view(np.uint32), expected_copy.numpy().view(np.uint32)
        )

        del actual_source, expected_source, expected_base
        self.assert_matches(actual_view, expected_view, case="lifetime-view", operation="list")
        self.assert_matches(actual_copy, expected_copy, case="lifetime-copy", operation="numpy")

    def test_scalar_empty_high_rank_wrapping_and_argument_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for shape in ((), (0,), (1,), (2, 0, 3), (1,) * 65):
            actual = torch.zeros(shape)
            expected = reference_torch.zeros(shape)
            for actual_call, expected_call in (
                (lambda: actual.flatten(), lambda: expected.flatten()),
                (lambda: torch.flatten(actual), lambda: reference_torch.flatten(expected)),
                (lambda: actual.flatten(0, -1), lambda: expected.flatten(0, -1)),
            ):
                self.assert_matches(
                    actual_call(), expected_call(), case=shape, operation="edge"
                )

        maximum = sys.maxsize
        actual_extreme = torch.zeros((0,)).reshape((0, maximum, maximum))
        expected_extreme = reference_torch.zeros((0,)).reshape((0, maximum, maximum))
        self.assert_matches(
            actual_extreme.flatten(1, 2),
            expected_extreme.flatten(1, 2),
            case="wrapping-product",
            operation="edge",
        )
        wrapping_factor = 6_148_914_691_236_517_205
        actual_wrapping_error = torch.zeros((3, 0, wrapping_factor)).transpose(0, 1)
        expected_wrapping_error = reference_torch.zeros(
            (3, 0, wrapping_factor)
        ).transpose(0, 1)
        self.assert_error_matches(
            lambda: actual_wrapping_error.flatten(1, 2),
            lambda: expected_wrapping_error.flatten(1, 2),
        )

        actual_symbolic = torch.zeros((0,)).reshape(
            (2**31, 2**32, 0)
        ).transpose(0, 2)
        expected_symbolic = reference_torch.zeros((0,)).reshape(
            (2**31, 2**32, 0)
        ).transpose(0, 2)
        with self.assertRaises(RuntimeError) as actual_symbolic_error:
            actual_symbolic.flatten(1, 2)
        with self.assertRaises(RuntimeError) as expected_symbolic_error:
            expected_symbolic.flatten(1, 2)
        message = "SymIntArrayRef expected to contain only concrete integers"
        self.assertIn(message, str(actual_symbolic_error.exception))
        self.assertIn(message, str(expected_symbolic_error.exception))

        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        error_cases = (
            (lambda: actual.flatten(2, 1), lambda: expected.flatten(2, 1)),
            (lambda: actual.flatten(-4), lambda: expected.flatten(-4)),
            (lambda: actual.flatten(end_dim=3), lambda: expected.flatten(end_dim=3)),
            (lambda: actual.flatten(None), lambda: expected.flatten(None)),
            (lambda: actual.flatten(start_dim=torch.float32), lambda: expected.flatten(start_dim=reference_torch.float32)),
            (lambda: actual.flatten(True), lambda: expected.flatten(True)),
            (lambda: actual.flatten(0, 1, 2), lambda: expected.flatten(0, 1, 2)),
            (lambda: actual.flatten(0, start_dim=1), lambda: expected.flatten(0, start_dim=1)),
            (lambda: actual.flatten(dim=1), lambda: expected.flatten(dim=1)),
            (lambda: torch.flatten(), lambda: reference_torch.flatten()),
            (lambda: torch.flatten([1.0]), lambda: reference_torch.flatten([1.0])),
            (lambda: torch.flatten(input=1), lambda: reference_torch.flatten(input=1)),
            (lambda: torch.flatten(actual, None), lambda: reference_torch.flatten(expected, None)),
            (lambda: torch.flatten(actual, input=actual), lambda: reference_torch.flatten(expected, input=expected)),
            (lambda: torch.flatten(actual, None, input=actual), lambda: reference_torch.flatten(expected, None, input=expected)),
            (lambda: torch.flatten(actual, 0, None, input=actual), lambda: reference_torch.flatten(expected, 0, None, input=expected)),
            (lambda: torch.flatten(actual, 0, None, start_dim=1), lambda: reference_torch.flatten(expected, 0, None, start_dim=1)),
            (lambda: actual.flatten(0, None, start_dim=1), lambda: expected.flatten(0, None, start_dim=1)),
            (lambda: actual.flatten(0, start_dim=1, end_dim=None), lambda: expected.flatten(0, start_dim=1, end_dim=None)),
            (lambda: torch.flatten(actual, 0, -1, 1), lambda: reference_torch.flatten(expected, 0, -1, 1)),
            (lambda: actual.flatten(2**100), lambda: expected.flatten(2**100)),
            (lambda: torch.flatten(actual, end_dim=None), lambda: reference_torch.flatten(expected, end_dim=None)),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertEqual(actual.flatten(np.int64(1)).shape, expected.flatten(np.int64(1)).shape)


if __name__ == "__main__":
    unittest.main()
