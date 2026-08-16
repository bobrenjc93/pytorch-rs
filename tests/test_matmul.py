import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\nmatmul(tensor2) -> Tensor\n\nSee :func:`torch.matmul`\n"


class TensorMatmulTests(unittest.TestCase):
    def assert_delegates_to_operator(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, expected.dtype)
            self.assertEqual(actual.device, expected.device)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def layout_cases(self):
        offset_left = torch.tensor(
            np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist()
        )[1]
        offset_right = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist()
        )[1]

        strided_left = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).transpose(0, 1)
        strided_right = torch.tensor([[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]]).transpose(
            0, 1
        )

        special_left = torch.tensor(
            [[float("inf"), 1.0], [float("-inf"), -1.0], [float("nan"), 2.0]]
        )
        special_right = torch.tensor([[1.0, -1.0], [0.5, 1.0]])

        return (
            (
                "contiguous",
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
            ),
            ("offset contiguous", offset_left, offset_right),
            ("strided", strided_left, strided_right),
            (
                "offset empty rows",
                torch.zeros((2, 0, 2)).transpose(0, 2)[1],
                torch.ones((2, 4)),
            ),
            ("empty inner", torch.ones((2, 0)), torch.zeros((0, 3))),
            ("non-finite", special_left, special_right),
        )

    def test_positional_and_keyword_calls_delegate_to_matrix_operator(self):
        for case, left, right in self.layout_cases():
            expected = left @ right
            self.assert_delegates_to_operator(
                left.matmul(right), expected, case=(case, "positional")
            )
            self.assert_delegates_to_operator(
                left.matmul(other=right), expected, case=(case, "keyword")
            )

    def test_existing_operator_autograd_behavior_is_preserved(self):
        method_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        method_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        operator_left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

        method_output = method_left.matmul(other=method_right)
        operator_output = operator_left @ operator_right
        self.assert_delegates_to_operator(
            method_output, operator_output, case="requires-grad operands"
        )
        self.assertFalse(method_output.requires_grad)
        self.assertTrue(method_output.is_leaf)

        for output in (method_output, operator_output):
            with self.assertRaises(RuntimeError) as raised:
                output.sum().backward()
            self.assertEqual(
                str(raised.exception),
                "element 0 of tensors does not require grad and does not have a grad_fn",
            )
        for operand in (method_left, method_right, operator_left, operator_right):
            self.assertIsNone(operand.grad)

    def test_rank_two_shape_errors_reuse_operator_and_other_ranks_stay_unsupported(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        ):
            left = torch.zeros(left_shape)
            right = torch.zeros(right_shape)
            message = (
                "mat1 and mat2 shapes cannot be multiplied "
                f"({left_shape[0]}x{left_shape[1]} and "
                f"{right_shape[0]}x{right_shape[1]})"
            )
            for call in (
                lambda left=left, right=right: left.matmul(right),
                lambda left=left, right=right: left.matmul(other=right),
                lambda left=left, right=right: left @ right,
            ):
                with self.subTest(left=left_shape, right=right_shape):
                    with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                        call()

        rank_cases = (
            (torch.tensor(1.0), torch.ones((1, 1))),
            (torch.ones((2,)), torch.ones((2, 2))),
            (torch.ones((1, 2, 2)), torch.ones((2, 2))),
        )
        for left, right in rank_cases:
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaises(RuntimeError) as method_error:
                    left.matmul(right)
                with self.assertRaises(RuntimeError) as operator_error:
                    left @ right
                self.assertEqual(str(method_error.exception), str(operator_error.exception))
                self.assertIn("requires two rank-2 tensors", str(method_error.exception))

        self.assertFalse(hasattr(torch, "matmul"))

    def test_descriptor_metadata_unbound_call_and_binding_errors(self):
        tensor = torch.tensor([[1.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "matmul")
        bound = tensor.matmul

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor.__qualname__, "TensorBase.matmul")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(bound.__qualname__, "Tensor.matmul")
        self.assertIsNone(bound.__module__)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "matmul")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assert_delegates_to_operator(
            descriptor(tensor, other=tensor), tensor @ tensor, case="unbound keyword"
        )

        cases = (
            (lambda: tensor.matmul(), 'matmul() missing 1 required positional arguments: "other"'),
            (
                lambda: tensor.matmul(tensor, tensor),
                "matmul() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.matmul(tensor, other=tensor),
                "matmul() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.matmul(tensor, out=tensor),
                "matmul() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.matmul(wat=tensor),
                'matmul() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.matmul([]),
                "matmul(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.matmul(other=None),
                "matmul(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.matmul([], out=tensor),
                "matmul(): argument 'other' (position 1) must be Tensor, not list",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            TypeError, r"^unbound method TensorBase\.matmul\(\) needs an argument$"
        ):
            descriptor()
        with self.assertRaisesRegex(
            TypeError,
            r"^descriptor 'matmul' for 'torch\._C\.TensorBase' objects "
            r"doesn't apply to a 'int' object$",
        ):
            descriptor(1, tensor)


if __name__ == "__main__":
    unittest.main()
