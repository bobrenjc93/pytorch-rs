import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SinReferenceTests(unittest.TestCase):
    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def test_autograd_preserves_scalar_empty_and_strided_history(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_scalar = torch.tensor(1.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(1.5, requires_grad=True)
        actual_scalar_output = actual_scalar.sin()
        expected_scalar_output = expected_scalar.sin()
        self.assert_metadata_matches(
            actual_scalar_output,
            expected_scalar_output,
            case="scalar output",
        )
        actual_scalar_output.backward()
        expected_scalar_output.backward()
        self.assert_metadata_matches(
            actual_scalar.grad,
            expected_scalar.grad,
            case="scalar gradient",
        )
        self.assertEqual(
            np.asarray(actual_scalar.grad).view(np.uint32).item(),
            expected_scalar.grad.detach().numpy().view(np.uint32).item(),
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty_output = actual_empty.sin()
        expected_empty_output = expected_empty.sin()
        self.assert_metadata_matches(
            actual_empty_output,
            expected_empty_output,
            case="empty output",
        )
        actual_empty_output.sum().backward()
        expected_empty_output.sum().backward()
        self.assert_metadata_matches(
            actual_empty.grad,
            expected_empty.grad,
            case="empty gradient",
        )
        self.assertEqual(actual_empty.grad.tolist(), expected_empty.grad.tolist())

        values = [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]]
        weights = [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_output = actual_leaf.transpose(0, 1).sin()
        expected_output = expected_leaf.transpose(0, 1).sin()
        self.assert_metadata_matches(
            actual_output,
            expected_output,
            case="strided output",
        )
        (actual_output * torch.tensor(weights)).sum().backward()
        (expected_output * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.detach().numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )

    def test_vjp_matches_grad_output_times_cosine_of_saved_input(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xC000_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x0000_0000,
                0x7F80_0000,
                0x3F80_0000,
                0xBF80_0000,
            ),
            dtype=np.uint32,
        )
        input_values = input_bits.view(np.float32)
        weight_values = weight_bits.view(np.float32)
        actual_leaf = torch.tensor(memoryview(input_values), requires_grad=True)
        expected_leaf = reference_torch.tensor(input_values, requires_grad=True)
        actual_weights = torch.tensor(memoryview(weight_values))
        expected_weights = reference_torch.tensor(weight_values)

        (actual_leaf.sin() * actual_weights).sum().backward()
        (expected_leaf.sin() * expected_weights).sum().backward()
        expected_formula = expected_weights * expected_leaf.detach().cos()
        np.testing.assert_array_equal(
            expected_leaf.grad.detach().numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )

    def test_detach_no_grad_and_freed_graph_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)

        actual_detached_input = actual_leaf.detach().transpose(0, 1).sin()
        expected_detached_input = expected_leaf.detach().transpose(0, 1).sin()
        self.assert_metadata_matches(
            actual_detached_input,
            expected_detached_input,
            case="detached input",
        )

        actual_tracked = actual_leaf.transpose(0, 1).sin()
        expected_tracked = expected_leaf.transpose(0, 1).sin()
        actual_detached_output = actual_tracked.detach()
        expected_detached_output = expected_tracked.detach()
        self.assert_metadata_matches(
            actual_detached_output,
            expected_detached_output,
            case="detached output",
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.transpose(0, 1).sin()
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.transpose(0, 1).sin()
        self.assert_metadata_matches(
            actual_untracked,
            expected_untracked,
            case="no_grad output",
        )

        with torch.no_grad():
            actual_no_grad_view = actual_leaf.transpose(0, 1)
        with reference_torch.no_grad():
            expected_no_grad_view = expected_leaf.transpose(0, 1)
        actual_boundary_output = actual_no_grad_view.sin()
        expected_boundary_output = expected_no_grad_view.sin()
        self.assert_metadata_matches(
            actual_boundary_output,
            expected_boundary_output,
            case="operation after no_grad view",
        )
        actual_boundary_loss = actual_boundary_output.sum()
        expected_boundary_loss = expected_boundary_output.sum()
        actual_boundary_loss.backward()
        expected_boundary_loss.backward()
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)
        with self.assertRaises(RuntimeError) as expected_boundary_raised:
            expected_boundary_loss.backward()
        with self.assertRaises(RuntimeError) as actual_boundary_raised:
            actual_boundary_loss.backward()
        self.assertEqual(
            str(actual_boundary_raised.exception),
            str(expected_boundary_raised.exception),
        )

        actual_loss = actual_tracked.sum()
        expected_loss = expected_tracked.sum()
        actual_loss.backward()
        expected_loss.backward()
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.detach().numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )
        with self.assertRaises(RuntimeError) as expected_raised:
            expected_loss.backward()
        with self.assertRaises(RuntimeError) as actual_raised:
            actual_loss.backward()
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))


if __name__ == "__main__":
    unittest.main()
