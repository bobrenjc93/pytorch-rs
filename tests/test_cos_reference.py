import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CosReferenceTests(unittest.TestCase):
    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_values_close(self, actual, expected, *, case):
        self.assert_metadata_matches(actual, expected, case=case)
        with self.subTest(case=case):
            np.testing.assert_allclose(
                np.asarray(actual, dtype=np.float32),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_scalar_empty_strided_and_non_finite_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_cases = [("scalar", torch.tensor(0.5))]
        expected_cases = [("scalar", reference_torch.tensor(0.5))]
        for shape in ((0,), (1, 0), (0, 1), (1, 0, 1), (2, 0, 3)):
            actual_cases.append((shape, torch.zeros(shape)))
            expected_cases.append((shape, reference_torch.zeros(shape)))

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist())
        expected_base = reference_torch.tensor(values)
        actual_transposed = actual_base.transpose(0, 2)
        expected_transposed = expected_base.transpose(0, 2)
        actual_cases.extend(
            (
                ("transposed dense view", actual_transposed),
                ("offset non-dense view", actual_transposed[1]),
                (
                    "twice-transposed offset view",
                    actual_transposed[1].transpose(0, 1),
                ),
            )
        )
        expected_cases.extend(
            (
                ("transposed dense view", expected_transposed),
                ("offset non-dense view", expected_transposed[1]),
                (
                    "twice-transposed offset view",
                    expected_transposed[1].transpose(0, 1),
                ),
            )
        )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7FA1_2345,
                0xFFA5_4321,
            ),
            dtype=np.uint32,
        )
        special_values = memoryview(special_bits.view(np.float32))
        actual_cases.append(("signed zero and non-finites", torch.tensor(special_values)))
        expected_cases.append(
            (
                "signed zero and non-finites",
                reference_torch.tensor(special_values),
            )
        )

        for (actual_case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases
        ):
            self.assertEqual(actual_case, expected_case)
            actual_output = actual.cos()
            expected_output = expected.cos()
            self.assert_values_close(
                actual_output,
                expected_output,
                case=actual_case,
            )
            if actual_case == "signed zero and non-finites":
                np.testing.assert_array_equal(
                    np.asarray(actual_output).view(np.uint32),
                    expected_output.numpy().view(np.uint32),
                )

    def test_vjp_matches_negative_sine_through_composed_views(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.linspace(-2.75, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.asarray(
            [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32
        )
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_view = actual_leaf.transpose(0, 2)[1].transpose(0, 1)
        expected_view = expected_leaf.transpose(0, 2)[1].transpose(0, 1)
        expected_view.retain_grad()
        actual_weights = torch.tensor(weights.tolist())
        expected_weights = reference_torch.tensor(weights)

        actual_output = actual_view.cos()
        expected_output = expected_view.cos()
        self.assert_values_close(actual_output, expected_output, case="composed view")
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()

        expected_formula = -expected_view.detach().sin() * expected_weights
        np.testing.assert_array_equal(
            expected_view.grad.numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )

        actual_scalar = torch.tensor(1.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(1.5, requires_grad=True)
        actual_scalar.cos().backward()
        expected_scalar.cos().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_scalar.grad).view(np.uint32),
            expected_scalar.grad.numpy().view(np.uint32),
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.cos().sum().backward()
        expected_empty.cos().sum().backward()
        self.assert_values_close(actual_empty.grad, expected_empty.grad, case="empty VJP")

    def test_vjp_signed_zero_and_non_finites_matches_explicit_formula(self):
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
        actual_leaf = torch.tensor(
            memoryview(input_bits.view(np.float32)), requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            input_bits.view(np.float32), requires_grad=True
        )
        actual_view = actual_leaf.reshape(3, 4).transpose(0, 1)
        expected_view = expected_leaf.reshape(3, 4).transpose(0, 1)
        expected_view.retain_grad()
        actual_weights = torch.tensor(memoryview(weight_bits.view(np.float32))).reshape(
            4, 3
        )
        expected_weights = reference_torch.tensor(weight_bits.view(np.float32)).reshape(
            4, 3
        )

        (actual_view.cos() * actual_weights).sum().backward()
        (expected_view.cos() * expected_weights).sum().backward()
        expected_formula = -expected_view.detach().sin() * expected_weights
        np.testing.assert_array_equal(
            expected_view.grad.numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_leaf.grad.numpy().view(np.uint32),
        )

    def test_detach_no_grad_freed_graph_and_no_argument_errors_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)

        actual_detached_input = actual_leaf.detach().transpose(0, 1).cos()
        expected_detached_input = expected_leaf.detach().transpose(0, 1).cos()
        self.assert_values_close(
            actual_detached_input,
            expected_detached_input,
            case="detached input",
        )

        actual_tracked = actual_leaf.transpose(0, 1).cos()
        expected_tracked = expected_leaf.transpose(0, 1).cos()
        self.assert_values_close(
            actual_tracked.detach(),
            expected_tracked.detach(),
            case="detached output",
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.transpose(0, 1).cos()
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.transpose(0, 1).cos()
        self.assert_values_close(actual_untracked, expected_untracked, case="no_grad")

        with torch.no_grad():
            actual_no_grad_view = actual_leaf.transpose(0, 1)
        with reference_torch.no_grad():
            expected_no_grad_view = expected_leaf.transpose(0, 1)
        actual_boundary_loss = actual_no_grad_view.cos().sum()
        expected_boundary_loss = expected_no_grad_view.cos().sum()
        actual_boundary_loss.backward()
        expected_boundary_loss.backward()
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)
        self.assert_error_matches(
            actual_boundary_loss.backward,
            expected_boundary_loss.backward,
        )

        actual_loss = actual_tracked.sum()
        expected_loss = expected_tracked.sum()
        actual_loss.backward()
        expected_loss.backward()
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )
        self.assert_error_matches(actual_loss.backward, expected_loss.backward)

        self.assertEqual(torch.Tensor.cos.__doc__, reference_torch.Tensor.cos.__doc__)
        self.assertIsNone(torch.Tensor.cos.__text_signature__)
        self.assertIsNone(reference_torch.Tensor.cos.__text_signature__)
        actual = torch.tensor([0.0])
        expected = reference_torch.tensor([0.0])
        for actual_call, expected_call in (
            (lambda: actual.cos(1), lambda: expected.cos(1)),
            (lambda: actual.cos(1, 2), lambda: expected.cos(1, 2)),
            (lambda: actual.cos(unexpected=True), lambda: expected.cos(unexpected=True)),
            (
                lambda: actual.cos(1, unexpected=True),
                lambda: expected.cos(1, unexpected=True),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
