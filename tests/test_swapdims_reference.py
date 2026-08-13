import inspect
import sys
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSwapdimsReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.detach().cpu().numpy(),
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

    def test_seeded_views_metadata_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x5A9D_213)
        shapes = [(), (0,), (2, 0, 3), (1, 3, 2), (2, 3, 4)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)
            else:
                actual = torch.tensor(values.item() if shape == () else values.tolist())
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)

            if len(shape) >= 2 and shape[0] > 0 and case % 4 == 1:
                actual = actual.transpose(0, -1)[-1]
                expected = expected.transpose(0, -1)[-1]

            rank = len(actual.shape)
            dimensions = [0, -1] if rank == 0 else list(range(-rank, rank))
            for chain in range(2):
                dim0 = dimensions[int(rng.integers(0, len(dimensions)))]
                dim1 = dimensions[int(rng.integers(0, len(dimensions)))]
                if (case + chain) % 2:
                    actual = actual.swapdims(dim0, dim1)
                    expected = expected.swapdims(dim0, dim1)
                else:
                    actual = actual.swapdims(dim0=dim0, dim1=dim1)
                    expected = expected.swapdims(dim0=dim0, dim1=dim1)

            self.assert_matches(actual, expected, case=f"view-{case}")
            for operation, actual_output, expected_output in (
                ("clone", actual.clone(), expected.clone()),
                ("sin", actual.sin(), expected.sin()),
                ("scalar", actual + 1.25, expected + 1.25),
                ("reshape", actual.reshape(-1), expected.reshape(-1)),
                ("sum", actual.sum(), expected.sum()),
            ):
                self.assert_matches(
                    actual_output, expected_output, case=f"{operation}-{case}"
                )

    def test_autograd_and_storage_view_metadata_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 3, 2)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = actual_leaf.swapdims(0, -1)
        expected = expected_leaf.swapdims(0, -1)

        self.assert_matches(actual, expected, case="tracked-view")
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertEqual(expected.data_ptr(), expected_leaf.data_ptr())
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="gradient")

        actual_scalar = torch.tensor(2.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(2.5, requires_grad=True)
        actual_scalar_view = actual_scalar.swapdims(0, -1)
        expected_scalar_view = expected_scalar.swapdims(0, -1)
        (actual_scalar_view * 7.0).backward()
        (expected_scalar_view * 7.0).backward()
        self.assert_matches(
            actual_scalar.grad, expected_scalar.grad, case="scalar-gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.swapdims(0, -1).sum().backward()
        expected_empty.swapdims(0, -1).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty-gradient")

        with torch.no_grad():
            actual_untracked = actual_leaf.swapdims(0, 1)
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.swapdims(0, 1)
        self.assertEqual(actual_untracked.requires_grad, expected_untracked.requires_grad)
        self.assertEqual(actual_untracked.is_leaf, expected_untracked.is_leaf)

        actual_offset = torch.tensor(values.tolist()).transpose(0, 2)[1].swapdims(0, 1)
        expected_offset = (
            reference_torch.tensor(values).transpose(0, 2)[1].swapdims(0, 1)
        )
        self.assert_matches(actual_offset, expected_offset, case="offset-view")

    def test_binding_dimension_and_overflow_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            timedelta = np.timedelta64(1)
        cases = (
            (lambda: actual.swapdims(), lambda: expected.swapdims()),
            (lambda: actual.swapdims(0), lambda: expected.swapdims(0)),
            (lambda: actual.swapdims(0, 1, 2), lambda: expected.swapdims(0, 1, 2)),
            (
                lambda: actual.swapdims(dim1=0, unexpected=None),
                lambda: expected.swapdims(dim1=0, unexpected=None),
            ),
            (
                lambda: actual.swapdims(0, 1, unexpected=None),
                lambda: expected.swapdims(0, 1, unexpected=None),
            ),
            (
                lambda: actual.swapdims(0, dim0=1, dim1=0),
                lambda: expected.swapdims(0, dim0=1, dim1=0),
            ),
            (lambda: actual.swapdims(1.5), lambda: expected.swapdims(1.5)),
            (
                lambda: actual.swapdims(dim0=np.bool_(True)),
                lambda: expected.swapdims(dim0=np.bool_(True)),
            ),
            (
                lambda: actual.swapdims(np.float64(1.5), 0),
                lambda: expected.swapdims(np.float64(1.5), 0),
            ),
            (
                lambda: actual.swapdims(1.5, 0, unexpected=None),
                lambda: expected.swapdims(1.5, 0, unexpected=None),
            ),
            (
                lambda: actual.swapdims(2**100, unexpected=None),
                lambda: expected.swapdims(2**100, unexpected=None),
            ),
            (
                lambda: actual.swapdims(2**100, 0, unexpected=None),
                lambda: expected.swapdims(2**100, 0, unexpected=None),
            ),
            (
                lambda: actual.swapdims(2**100, 1.5),
                lambda: expected.swapdims(2**100, 1.5),
            ),
            (
                lambda: actual.swapdims(timedelta, 0),
                lambda: expected.swapdims(timedelta, 0),
            ),
            (
                lambda: actual.swapdims(np.uint64(2**63), 0),
                lambda: expected.swapdims(np.uint64(2**63), 0),
            ),
            (lambda: actual.swapdims(2**100, 0), lambda: expected.swapdims(2**100, 0)),
            (lambda: actual.swapdims(-3, 0), lambda: expected.swapdims(-3, 0)),
            (lambda: actual.swapdims(0, 2), lambda: expected.swapdims(0, 2)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_scalar = torch.tensor(1.0)
        expected_scalar = reference_torch.tensor(1.0)
        self.assert_error_matches(
            lambda: actual_scalar.swapdims(-2, 0),
            lambda: expected_scalar.swapdims(-2, 0),
        )

        actual_extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        expected_extreme = reference_torch.zeros((sys.maxsize, 0, 2, 2))
        self.assert_error_matches(
            lambda: actual_extreme.swapdims(1, 3),
            lambda: expected_extreme.swapdims(1, 3),
        )

    def test_descriptor_matches_pytorch_2_13(self):
        actual_descriptor = torch.Tensor.swapdims
        expected_descriptor = reference_torch.Tensor.swapdims
        self.assertEqual(
            actual_descriptor.__text_signature__, expected_descriptor.__text_signature__
        )
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        for descriptor in (actual_descriptor, expected_descriptor):
            with self.subTest(descriptor=descriptor):
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)


if __name__ == "__main__":
    unittest.main()
