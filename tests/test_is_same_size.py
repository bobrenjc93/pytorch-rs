import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorIsSameSizeTests(unittest.TestCase):
    def assert_same_size(self, left, right, expected):
        result = left.is_same_size(right)
        self.assertIs(type(result), bool)
        self.assertIs(result, expected)

    def test_compares_only_shape_across_storage_and_layouts(self):
        source = torch.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ]
        )
        same_shape = (
            source,
            source.detach(),
            source.clone(),
            torch.zeros((2, 3, 2)),
            source.transpose(0, 2).transpose(0, 2),
        )
        for case, other in enumerate(same_shape):
            with self.subTest(kind="same", case=case):
                self.assert_same_size(source, other, True)

        different_shape = (
            source.reshape((6, 2)),
            source.reshape((2, 6)),
            source.transpose(1, 2),
            torch.zeros((12,)),
        )
        for case, other in enumerate(different_shape):
            with self.subTest(kind="different", case=case):
                self.assert_same_size(source, other, False)

        square = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        transposed = square.transpose(0, 1)
        self.assertNotEqual(square.stride(), transposed.stride())
        self.assert_same_size(square, transposed, True)

        first_row = source[0]
        second_row = source[1]
        self.assertNotEqual(first_row.storage_offset(), second_row.storage_offset())
        self.assert_same_size(first_row, second_row, True)

    def test_scalars_and_empty_tensors(self):
        self.assert_same_size(torch.tensor(3.0), torch.tensor(-8.0), True)
        self.assert_same_size(torch.tensor(3.0), torch.tensor([3.0]), False)

        empty = torch.zeros((2, 0, 3))
        self.assert_same_size(empty, torch.ones((2, 0, 3)), True)
        self.assert_same_size(empty, torch.zeros((6, 0)), False)
        self.assert_same_size(torch.zeros((0,)), torch.zeros((1, 0)), False)

        extreme_empty = torch.zeros((0,)).reshape((2, 0, sys.maxsize))
        independent = torch.ones((0,)).reshape((2, 0, sys.maxsize))
        self.assert_same_size(extreme_empty, extreme_empty.detach(), True)
        self.assert_same_size(extreme_empty, independent, True)
        self.assert_same_size(
            extreme_empty,
            independent.transpose(0, 2),
            False,
        )

    def test_comparison_does_not_change_storage_or_autograd_state(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        independent = torch.zeros((2, 2), requires_grad=True)
        graph_before = (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            tracked.requires_grad,
            tracked.is_leaf,
            independent.requires_grad,
            independent.is_leaf,
            independent.grad,
            tracked.is_set_to(tracked.detach()),
            tracked.is_set_to(independent),
        )

        self.assert_same_size(tracked, independent, True)

        graph_after = (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            tracked.requires_grad,
            tracked.is_leaf,
            independent.requires_grad,
            independent.is_leaf,
            independent.grad,
            tracked.is_set_to(tracked.detach()),
            tracked.is_set_to(independent),
        )
        self.assertEqual(graph_after, graph_before)

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertIsNone(independent.grad)

    def test_descriptor_metadata_and_other_keyword(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_same_size")
        bound = tensor.is_same_size

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertTrue(callable(callable_object))
            self.assertEqual(callable_object.__name__, "is_same_size")
            self.assertIsNone(callable_object.__doc__)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor, tensor), True)
        self.assertIs(tensor.is_same_size(other=tensor), True)
        self.assertIs(tensor.is_same_size(other=torch.tensor([[1.0]])), False)

    def test_binding_and_non_tensor_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: tensor.is_same_size(),
                'is_same_size() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.is_same_size(tensor, tensor),
                "is_same_size() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.is_same_size(tensor, tensor, tensor),
                "is_same_size() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: tensor.is_same_size(tensor, other=tensor),
                "is_same_size() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.is_same_size(tensor=tensor),
                'is_same_size() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.is_same_size(tensor, extra=True),
                "is_same_size() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.is_same_size(1),
                "is_same_size(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.is_same_size(None),
                "is_same_size(): argument 'other' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.is_same_size([]),
                "is_same_size(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.is_same_size(
                    np.zeros((2, 3), dtype=np.float32)
                ),
                "is_same_size(): argument 'other' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: tensor.is_same_size(other=1),
                "is_same_size(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.is_same_size(other=None),
                "is_same_size(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.is_same_size(other=[]),
                "is_same_size(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: tensor.is_same_size(**{"other": 1, "extra": True}),
                "is_same_size(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.is_same_size(**{"extra": True, "other": 1}),
                "is_same_size(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.is_same_size(1, other=tensor),
                "is_same_size(): argument 'other' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
