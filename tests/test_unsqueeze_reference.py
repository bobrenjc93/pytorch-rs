import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class UnsqueezeReferenceTests(unittest.TestCase):
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

    def test_shapes_strides_offsets_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x213_0A65)
        shapes = [(), (0,), (1,), (2, 0, 3), (2, 3, 4)]
        for _ in range(24):
            rank = int(rng.integers(0, 8))
            shape = tuple(int(value) for value in rng.integers(0, 5, size=rank))
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
            dim = int(rng.integers(-(rank + 1), rank + 1))
            if case % 3 == 0:
                actual_view = actual.unsqueeze(dim)
                expected_view = expected.unsqueeze(dim)
            elif case % 3 == 1:
                actual_view = torch.unsqueeze(actual, dim)
                expected_view = reference_torch.unsqueeze(expected, dim)
            else:
                actual_view = torch.unsqueeze(input=actual, axis=dim)
                expected_view = reference_torch.unsqueeze(input=expected, axis=dim)

            self.assert_matches(actual_view, expected_view, case=case, operation="view")
            for operation, actual_output, expected_output in (
                ("clone", actual_view.clone(), expected_view.clone()),
                ("relu", actual_view.relu(), expected_view.relu()),
                ("arithmetic", actual_view * 1.5 + 0.25, expected_view * 1.5 + 0.25),
                ("sum", actual_view.sum(), expected_view.sum()),
                ("reshape", actual_view.reshape(-1), expected_view.reshape(-1)),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=case,
                    operation=operation,
                )
            self.assertEqual(actual_view.tolist(), expected_view.tolist())

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)

        actual_view = actual_leaf.transpose(0, 3)[1].unsqueeze(2)
        expected_view = expected_leaf.transpose(0, 3)[1].unsqueeze(2)
        self.assert_matches(actual_view, expected_view, case="autograd", operation="view")
        self.assertEqual(actual_view.requires_grad, expected_view.requires_grad)
        self.assertEqual(actual_view.is_leaf, expected_view.is_leaf)

        weights = torch.ones(tuple(actual_view.shape))
        expected_weights = reference_torch.ones(tuple(expected_view.shape))
        (actual_view * weights).sum().backward()
        (expected_view * expected_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().numpy()
        )

        actual_no_grad_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_no_grad_leaf = reference_torch.tensor(values, requires_grad=True)
        with torch.no_grad():
            actual_no_grad = torch.unsqueeze(
                actual_no_grad_leaf.transpose(0, 3)[1], dim=2
            )
        with reference_torch.no_grad():
            expected_no_grad = reference_torch.unsqueeze(
                expected_no_grad_leaf.transpose(0, 3)[1], dim=2
            )
        self.assertEqual(actual_no_grad.requires_grad, expected_no_grad.requires_grad)
        self.assertEqual(actual_no_grad.is_leaf, expected_no_grad.is_leaf)
        self.assertIsNone(actual_no_grad_leaf.grad)
        self.assertIsNone(expected_no_grad_leaf.grad)

    def test_errors_and_bindings_match_pytorch_2_13_for_exact_native_surface(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        error_cases = (
            (lambda: actual.unsqueeze(), lambda: expected.unsqueeze()),
            (lambda: actual.unsqueeze(None), lambda: expected.unsqueeze(None)),
            (lambda: actual.unsqueeze(True), lambda: expected.unsqueeze(True)),
            (lambda: actual.unsqueeze(1.5), lambda: expected.unsqueeze(1.5)),
            (lambda: actual.unsqueeze(dim=np.float64(1)), lambda: expected.unsqueeze(dim=np.float64(1))),
            (lambda: actual.unsqueeze(-4), lambda: expected.unsqueeze(-4)),
            (lambda: actual.unsqueeze(3), lambda: expected.unsqueeze(3)),
            (lambda: actual.unsqueeze(2**100), lambda: expected.unsqueeze(2**100)),
            (lambda: actual.unsqueeze(0, dim=1), lambda: expected.unsqueeze(0, dim=1)),
            (lambda: actual.unsqueeze(dim=0, axis=1), lambda: expected.unsqueeze(dim=0, axis=1)),
            (lambda: torch.unsqueeze(), lambda: reference_torch.unsqueeze()),
            (lambda: torch.unsqueeze(actual), lambda: reference_torch.unsqueeze(expected)),
            (lambda: torch.unsqueeze(actual, None), lambda: reference_torch.unsqueeze(expected, None)),
            (lambda: torch.unsqueeze(actual, True), lambda: reference_torch.unsqueeze(expected, True)),
            (lambda: torch.unsqueeze(actual, 1.5), lambda: reference_torch.unsqueeze(expected, 1.5)),
            (lambda: torch.unsqueeze(actual, -4), lambda: reference_torch.unsqueeze(expected, -4)),
            (lambda: torch.unsqueeze(actual, 3), lambda: reference_torch.unsqueeze(expected, 3)),
            (lambda: torch.unsqueeze(actual, 2**100), lambda: reference_torch.unsqueeze(expected, 2**100)),
            (lambda: torch.unsqueeze(actual, 0, 1), lambda: reference_torch.unsqueeze(expected, 0, 1)),
            (lambda: torch.unsqueeze(actual, 0, dim=1), lambda: reference_torch.unsqueeze(expected, 0, dim=1)),
            (lambda: torch.unsqueeze(actual, 0, out=None), lambda: reference_torch.unsqueeze(expected, 0, out=None)),
            (
                lambda: torch.unsqueeze(np.zeros((2, 3), dtype=np.float32), 0),
                lambda: reference_torch.unsqueeze(np.zeros((2, 3), dtype=np.float32), 0),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(error_cases):
            with self.subTest(error_case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertFalse(hasattr(torch.Tensor, "unsqueeze_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "unsqueeze_"))


if __name__ == "__main__":
    unittest.main()
