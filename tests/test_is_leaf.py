import inspect
import types
import unittest

import torch_rs as torch


class TensorIsLeafTests(unittest.TestCase):
    def test_leaf_status_tracks_recorded_autograd_history(self):
        ordinary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        gradient_leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        recorded_operation = gradient_leaf * 2.0
        recorded_view = gradient_leaf.transpose(0, 1)

        cases = (
            ("ordinary tensor", ordinary, True),
            ("ordinary operation", ordinary + 1.0, True),
            ("ordinary view", ordinary.transpose(0, 1), True),
            ("gradient leaf", gradient_leaf, True),
            ("recorded operation", recorded_operation, False),
            ("recorded view", recorded_view, False),
            ("detached operation", recorded_operation.detach(), True),
            ("detached view", recorded_view.detach(), True),
        )
        for case, tensor, expected in cases:
            with self.subTest(case=case):
                self.assertIs(type(tensor.is_leaf), bool)
                self.assertIs(tensor.is_leaf, expected)

        recorded_operation.sum().backward()
        self.assertFalse(recorded_operation.is_leaf)
        self.assertTrue(gradient_leaf.grad.is_leaf)

    def test_no_grad_results_are_leaves_but_later_recorded_results_are_not(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked_operation = leaf * 2.0
        with torch.no_grad():
            operation = leaf * 2.0
            views = (
                leaf.transpose(0, 1),
                tracked_operation.transpose(0, 1),
                leaf.reshape(4),
                leaf[0],
            )

        self.assertFalse(operation.requires_grad)
        self.assertTrue(operation.is_leaf)
        for view in views:
            with self.subTest(shape=view.shape, stride=view.stride()):
                self.assertTrue(view.requires_grad)
                self.assertTrue(view.is_leaf)
                self.assertFalse((view + 1.0).is_leaf)

    def test_is_leaf_is_a_read_only_tensor_descriptor(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_leaf")
        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "is_leaf")
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), True)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)

        with self.assertRaisesRegex(
            AttributeError, r"attribute 'is_leaf'.*not writable"
        ):
            tensor.is_leaf = False
        with self.assertRaisesRegex(
            AttributeError, r"attribute 'is_leaf'.*not writable"
        ):
            del tensor.is_leaf


if __name__ == "__main__":
    unittest.main()
