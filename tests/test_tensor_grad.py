import inspect
import unittest

import numpy as np
import torch_rs as torch


GRAD_DOC = """
This attribute is ``None`` by default and becomes a Tensor the first time a call to
:func:`backward` computes gradients for ``self``.
The attribute will then contain the gradients computed and future calls to
:func:`backward` will accumulate (add) gradients into it.
"""


class TensorGradAssignmentTests(unittest.TestCase):
    def assert_reset_cycle(self, shape):
        leaf = torch.ones(shape, requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "grad")
        metadata = (
            leaf.shape,
            leaf.stride(),
            leaf.storage_offset(),
            leaf.data_ptr(),
            leaf.requires_grad,
            leaf.is_leaf,
        )

        self.assertIsNone(descriptor.__set__(leaf, None))
        self.assertIsNone(leaf.grad)

        loss = leaf.sum()
        loss.backward()
        first = leaf.grad
        first_pointer = first.data_ptr()
        expected_once = np.ones(shape, dtype=np.float32)
        np.testing.assert_array_equal(np.asarray(first), expected_once)
        self.assertIs(leaf.grad, first)

        leaf.grad = None
        self.assertIsNone(leaf.grad)
        np.testing.assert_array_equal(np.asarray(first), expected_once)

        loss.backward()
        fresh = leaf.grad
        self.assertIsNot(fresh, first)
        np.testing.assert_array_equal(np.asarray(fresh), expected_once)
        np.testing.assert_array_equal(np.asarray(first), expected_once)
        if leaf.numel() != 0:
            self.assertNotEqual(fresh.data_ptr(), first_pointer)

        loss.backward()
        self.assertIs(leaf.grad, fresh)
        np.testing.assert_array_equal(np.asarray(fresh), expected_once * 2.0)
        np.testing.assert_array_equal(np.asarray(first), expected_once)
        self.assertEqual(
            (
                leaf.shape,
                leaf.stride(),
                leaf.storage_offset(),
                leaf.data_ptr(),
                leaf.requires_grad,
                leaf.is_leaf,
            ),
            metadata,
        )

    def test_none_clears_scalar_empty_and_multi_element_leaf_gradients(self):
        for shape in ((), (2, 0, 3), (2, 3)):
            with self.subTest(shape=shape):
                self.assert_reset_cycle(shape)

    def test_clearing_one_leaf_does_not_change_other_leaf_accumulators(self):
        left = torch.tensor([1.0, 2.0], requires_grad=True)
        right = torch.tensor([3.0, 4.0], requires_grad=True)
        left_loss = left.sum()
        right_loss = (-right).sum()

        left_loss.backward()
        right_loss.backward()
        old_left = left.grad
        right_gradient = right.grad

        left.grad = None
        self.assertIsNone(left.grad)
        self.assertIs(right.grad, right_gradient)
        self.assertEqual(right_gradient.tolist(), [-1.0, -1.0])

        left_loss.backward()
        self.assertIsNot(left.grad, old_left)
        self.assertEqual(left.grad.tolist(), [1.0, 1.0])
        self.assertEqual(old_left.tolist(), [1.0, 1.0])
        self.assertIs(right.grad, right_gradient)
        self.assertEqual(right_gradient.tolist(), [-1.0, -1.0])

        right_loss.backward()
        self.assertIs(right.grad, right_gradient)
        self.assertEqual(right_gradient.tolist(), [-2.0, -2.0])

    def test_none_assignment_is_supported_for_every_native_leaf_state(self):
        source = torch.tensor([[1.0, 2.0]], requires_grad=True)
        tracked = -source
        with torch.no_grad():
            no_grad_view = source.transpose(0, 1)

        leaves = (
            torch.tensor([1.0]),
            source,
            tracked.detach(),
            no_grad_view,
            torch.zeros((2, 0, 3), requires_grad=True),
        )
        for leaf in leaves:
            with self.subTest(shape=leaf.shape, requires_grad=leaf.requires_grad):
                self.assertTrue(leaf.is_leaf)
                self.assertIsNone(setattr(leaf, "grad", None))
                self.assertIsNone(leaf.grad)

    def test_replacement_deletion_and_non_leaf_clear_are_non_mutating(self):
        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        reusable_loss = (-leaf).sum()
        reusable_loss.backward()
        gradient = leaf.grad
        descriptor = inspect.getattr_static(torch.Tensor, "grad")
        metadata = (
            leaf.shape,
            leaf.stride(),
            leaf.storage_offset(),
            leaf.data_ptr(),
            gradient.data_ptr(),
            gradient.tolist(),
        )

        replacement = torch.tensor([7.0, 8.0])
        for mutation in (
            lambda: setattr(leaf, "grad", replacement),
            lambda: descriptor.__set__(leaf, replacement),
        ):
            with self.subTest(kind="tensor replacement", mutation=mutation):
                with self.assertRaises(NotImplementedError) as raised:
                    mutation()
                self.assertEqual(
                    str(raised.exception),
                    "torch_rs only supports assigning None to Tensor.grad",
                )
                self.assertIs(leaf.grad, gradient)
                self.assertEqual(gradient.tolist(), [-1.0, -1.0])

        for value, type_name in ((1, "int"), (object(), "object")):
            with self.subTest(kind="invalid value", value=value):
                with self.assertRaises(TypeError) as raised:
                    leaf.grad = value
                self.assertEqual(
                    str(raised.exception),
                    "assigned grad expected to be a Tensor or None but got "
                    f"grad of type {type_name}",
                )
                self.assertIs(leaf.grad, gradient)

        for deletion in (
            lambda: delattr(leaf, "grad"),
            lambda: descriptor.__delete__(leaf),
        ):
            with self.subTest(kind="deletion", deletion=deletion):
                with self.assertRaises(AttributeError) as raised:
                    deletion()
                self.assertEqual(
                    str(raised.exception),
                    "Tensor.grad cannot be deleted; assign None to clear it",
                )
                self.assertIs(leaf.grad, gradient)

        self.assertEqual(
            (
                leaf.shape,
                leaf.stride(),
                leaf.storage_offset(),
                leaf.data_ptr(),
                gradient.data_ptr(),
                gradient.tolist(),
            ),
            metadata,
        )
        reusable_loss.backward()
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(gradient.tolist(), [-2.0, -2.0])

        other_leaf = torch.tensor([4.0, 5.0], requires_grad=True)
        non_leaf = -other_leaf
        with self.assertRaises(RuntimeError) as raised:
            non_leaf.grad = None
        self.assertEqual(
            str(raised.exception), "grad can only be cleared on leaf tensors"
        )
        non_leaf.sum().backward()
        self.assertEqual(other_leaf.grad.tolist(), [-1.0, -1.0])

    def test_descriptor_documents_the_accumulating_gradient_contract(self):
        descriptor = inspect.getattr_static(torch.Tensor, "grad")
        self.assertEqual(descriptor.__doc__, GRAD_DOC)
        self.assertEqual(descriptor.__name__, "grad")
        self.assertTrue(hasattr(descriptor, "__set__"))
        self.assertTrue(hasattr(descriptor, "__delete__"))


if __name__ == "__main__":
    unittest.main()
