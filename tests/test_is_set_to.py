import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nis_set_to(tensor) -> bool\n\n"
    "Returns True if both tensors are pointing to the exact same memory (same\n"
    "storage, offset, size and stride).\n"
)


class TensorIsSetToTests(unittest.TestCase):
    def assert_is_set_to(self, left, right, expected):
        result = left.is_set_to(right)
        self.assertIs(type(result), bool)
        self.assertIs(result, expected)

    def test_storage_offset_shape_and_stride_must_all_match(self):
        source = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ]
        )
        identical_view = source.transpose(1, 1)
        restored_view = source.transpose(0, 2).transpose(0, 2)

        true_cases = (
            source,
            source.detach(),
            source.reshape((3, 2, 4)),
            identical_view,
            restored_view,
        )
        for case, other in enumerate(true_cases):
            with self.subTest(kind="identical", case=case):
                self.assert_is_set_to(source, other, True)

        false_cases = (
            source.clone(),
            torch.tensor(source.tolist()),
            source.reshape((6, 4)),
            source.transpose(0, 2),
        )
        for case, other in enumerate(false_cases):
            with self.subTest(kind="different", case=case):
                self.assert_is_set_to(source, other, False)

        square = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(square.shape, square.transpose(0, 1).shape)
        self.assertNotEqual(square.stride(), square.transpose(0, 1).stride())
        self.assert_is_set_to(square, square.transpose(0, 1), False)

        first_row = source[0]
        same_first_row = source[0]
        second_row = source[1]
        self.assertEqual(first_row.shape, second_row.shape)
        self.assertEqual(first_row.stride(), second_row.stride())
        self.assertNotEqual(first_row.storage_offset(), second_row.storage_offset())
        self.assert_is_set_to(first_row, same_first_row, True)
        self.assert_is_set_to(first_row, first_row.detach(), True)
        self.assert_is_set_to(first_row, second_row, False)

    def test_scalar_empty_offset_and_autograd_tensors_are_metadata_only(self):
        scalar = torch.tensor(3.0, requires_grad=True)
        self.assert_is_set_to(scalar, scalar.detach(), True)
        self.assert_is_set_to(scalar, scalar.clone(), False)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        self.assert_is_set_to(empty, empty.detach(), True)
        self.assert_is_set_to(empty, empty.clone(), False)
        self.assert_is_set_to(empty, empty.transpose(0, 2), False)

        empty_base = torch.zeros((2, 0, 3)).transpose(0, 2)
        offset_empty = empty_base[1]
        same_offset_empty = empty_base[1]
        self.assertGreater(offset_empty.storage_offset(), 0)
        self.assert_is_set_to(offset_empty, same_offset_empty, True)
        self.assert_is_set_to(offset_empty, offset_empty.detach(), True)

        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        self.assert_is_set_to(extreme_empty, extreme_empty.detach(), True)
        self.assert_is_set_to(extreme_empty, extreme_empty.clone(), False)

        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        detached = tracked.detach()
        graph_before = (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            tracked.requires_grad,
            tracked.is_leaf,
            detached.requires_grad,
            detached.is_leaf,
        )
        self.assert_is_set_to(tracked, detached, True)
        self.assertEqual(
            (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                tracked.requires_grad,
                tracked.is_leaf,
                detached.requires_grad,
                detached.is_leaf,
            ),
            graph_before,
        )
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def test_descriptor_metadata_and_tensor_keyword(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_set_to")
        bound = tensor.is_set_to

        self.assertFalse(hasattr(torch, "is_set_to"))
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertTrue(callable(callable_object))
            self.assertEqual(callable_object.__name__, "is_set_to")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor, tensor), True)
        self.assertIs(tensor.is_set_to(tensor=tensor), True)
        self.assertIs(tensor.is_set_to(tensor=tensor.clone()), False)

    def test_binding_and_non_tensor_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: tensor.is_set_to(),
                'is_set_to() missing 1 required positional arguments: "tensor"',
            ),
            (
                lambda: tensor.is_set_to(tensor, tensor),
                "is_set_to() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.is_set_to(tensor, tensor=tensor),
                "is_set_to() got multiple values for argument 'tensor'",
            ),
            (
                lambda: tensor.is_set_to(other=tensor),
                'is_set_to() missing 1 required positional arguments: "tensor"',
            ),
            (
                lambda: tensor.is_set_to(tensor, extra=True),
                "is_set_to() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.is_set_to(1),
                "is_set_to(): argument 'tensor' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.is_set_to(None),
                "is_set_to(): argument 'tensor' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.is_set_to(tensor=[]),
                "is_set_to(): argument 'tensor' must be Tensor, not list",
            ),
            (
                lambda: tensor.is_set_to(np.zeros((2, 3), dtype=np.float32)),
                "is_set_to(): argument 'tensor' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: tensor.is_set_to(**{"tensor": 1, "extra": True}),
                "is_set_to(): argument 'tensor' must be Tensor, not int",
            ),
            (
                lambda: tensor.is_set_to(**{"extra": True, "tensor": 1}),
                "is_set_to(): argument 'tensor' must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
