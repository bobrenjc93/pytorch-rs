import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorGradAssignmentReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.grad differentials require pinned PyTorch 2.13.0"
            )

    def reset_contract(self, module, shape):
        leaf = module.ones(shape, requires_grad=True)
        leaf.grad = None
        loss = leaf.sum()
        loss.backward()
        first = leaf.grad
        first_pointer = first.data_ptr()
        first_values = np.asarray(first).copy()

        leaf.grad = None
        cleared = leaf.grad is None
        loss.backward()
        fresh = leaf.grad
        fresh_values = np.asarray(fresh).copy()
        fresh_identity = fresh is not first
        fresh_storage = leaf.numel() == 0 or fresh.data_ptr() != first_pointer

        loss.backward()
        return {
            "shape": tuple(leaf.shape),
            "stride": leaf.stride(),
            "cleared": cleared,
            "first_values": first_values,
            "fresh_values": fresh_values,
            "fresh_identity": fresh_identity,
            "fresh_storage": fresh_storage,
            "fresh_cached": leaf.grad is fresh,
            "accumulated_values": np.asarray(fresh).copy(),
            "old_values_unchanged": np.asarray(first).copy(),
        }

    def test_reset_cycle_matches_pytorch_for_scalar_empty_and_multi_element_leaves(self):
        for shape in ((), (2, 0, 3), (2, 3)):
            with self.subTest(shape=shape):
                actual = self.reset_contract(torch, shape)
                expected = self.reset_contract(reference_torch, shape)
                for key in (
                    "shape",
                    "stride",
                    "cleared",
                    "fresh_identity",
                    "fresh_storage",
                    "fresh_cached",
                ):
                    self.assertEqual(actual[key], expected[key], key)
                for key in (
                    "first_values",
                    "fresh_values",
                    "accumulated_values",
                    "old_values_unchanged",
                ):
                    np.testing.assert_array_equal(actual[key], expected[key])

    def independent_leaf_contract(self, module):
        left = module.tensor([1.0, 2.0], requires_grad=True)
        right = module.tensor([3.0, 4.0], requires_grad=True)
        left_loss = left.sum()
        right_loss = (-right).sum()
        left_loss.backward()
        right_loss.backward()
        old_left = left.grad
        right_gradient = right.grad

        left.grad = None
        after_clear = (
            left.grad is None,
            right.grad is right_gradient,
            np.asarray(right_gradient).copy(),
        )
        left_loss.backward()
        after_left_backward = (
            left.grad is not old_left,
            np.asarray(left.grad).copy(),
            np.asarray(old_left).copy(),
            right.grad is right_gradient,
            np.asarray(right_gradient).copy(),
        )
        right_loss.backward()
        return after_clear, after_left_backward, (
            right.grad is right_gradient,
            np.asarray(right_gradient).copy(),
        )

    def test_independent_leaf_accumulators_match_pytorch(self):
        actual = self.independent_leaf_contract(torch)
        expected = self.independent_leaf_contract(reference_torch)
        for actual_stage, expected_stage in zip(actual, expected, strict=True):
            for actual_value, expected_value in zip(
                actual_stage, expected_stage, strict=True
            ):
                if isinstance(actual_value, np.ndarray):
                    np.testing.assert_array_equal(actual_value, expected_value)
                else:
                    self.assertEqual(actual_value, expected_value)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.grad unexpectedly accepted the operation")

    def test_invalid_non_tensor_assignment_errors_match_pytorch(self):
        for value in (1, object()):
            actual = torch.tensor([1.0], requires_grad=True)
            expected = reference_torch.tensor([1.0], requires_grad=True)
            with self.subTest(value=type(value).__name__):
                self.assertEqual(
                    self.error(lambda: setattr(actual, "grad", value)),
                    self.error(lambda: setattr(expected, "grad", value)),
                )
                self.assertIsNone(actual.grad)
                self.assertIsNone(expected.grad)

    def test_deliberately_unsupported_mutations_leave_native_state_intact(self):
        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        loss = (-leaf).sum()
        loss.backward()
        gradient = leaf.grad

        self.assertEqual(
            self.error(
                lambda: setattr(
                    leaf, "grad", torch.tensor([7.0, 8.0])
                )
            ),
            (
                "NotImplementedError",
                "torch_rs only supports assigning None to Tensor.grad",
            ),
        )
        self.assertEqual(
            self.error(lambda: delattr(leaf, "grad")),
            (
                "AttributeError",
                "Tensor.grad cannot be deleted; assign None to clear it",
            ),
        )
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(gradient.tolist(), [-1.0, -1.0])
        loss.backward()
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(gradient.tolist(), [-2.0, -2.0])

        expected = reference_torch.tensor([2.0, 3.0], requires_grad=True)
        expected.grad = reference_torch.tensor([7.0, 8.0])
        self.assertEqual(expected.grad.tolist(), [7.0, 8.0])
        del expected.grad
        self.assertIsNone(expected.grad)


if __name__ == "__main__":
    unittest.main()
