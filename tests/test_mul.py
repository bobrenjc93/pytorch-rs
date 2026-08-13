import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorMulTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_tensor_and_real_scalar_calls_reuse_operator_semantics(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])
        expected = left * right
        self.assert_tensor_matches(left.mul(right), expected, case="tensor positional")
        self.assert_tensor_matches(
            left.mul(other=right), expected, case="tensor keyword"
        )

        offset_view = left[1]
        for scalar in (True, -2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            expected = offset_view * scalar
            self.assert_tensor_matches(
                offset_view.mul(scalar), expected, case=("scalar positional", scalar)
            )
            self.assert_tensor_matches(
                offset_view.mul(other=scalar),
                expected,
                case=("scalar keyword", scalar),
            )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            empty.mul(other=broadcast), empty * broadcast, case="strided empty"
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        self.assert_tensor_matches(
            special.mul(-0.0), special * -0.0, case="signed zero and non-finites"
        )

    def test_autograd_shared_operands_empties_and_no_grad_match_operator(self):
        method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        method_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        method_output = method_left.transpose(0, 1).mul(
            other=method_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) * operator_right.transpose(
            0, 1
        )
        self.assert_tensor_matches(method_output, operator_output, case="tracked views")
        method_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            method_left.grad, operator_left.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            method_right.grad, operator_right.grad, case="right gradient"
        )

        method_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        method_shared.mul(method_shared).sum().backward()
        (operator_shared * operator_shared).sum().backward()
        self.assert_tensor_matches(
            method_shared.grad, operator_shared.grad, case="shared operand gradient"
        )

        method_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        method_empty.mul(other=torch.ones((1, 1, 3))).sum().backward()
        (operator_empty * torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            method_empty.grad, operator_empty.grad, case="empty gradient"
        )

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = no_grad_left.transpose(0, 1).mul(no_grad_right.transpose(0, 1))
            scalar_output = no_grad_left.mul(other=2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(no_grad_left.mul(no_grad_right.transpose(0, 1)).requires_grad)

    def test_descriptor_metadata_unbound_call_and_argument_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "mul")
        bound = tensor.mul

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "mul")
        self.assertEqual(bound.__name__, "mul")
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            descriptor.__doc__,
            "\nmul(value) -> Tensor\n\nSee :func:`torch.mul`.\n",
        )
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assert_tensor_matches(
            descriptor(tensor, other=tensor), tensor * tensor, case="unbound call"
        )

        cases = (
            (lambda: tensor.mul(), 'mul() missing 1 required positional arguments: "other"'),
            (
                lambda: tensor.mul(tensor, tensor),
                "mul() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.mul(tensor, other=tensor),
                "mul() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.mul(tensor, out=tensor),
                "mul() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.mul(wat=tensor),
                'mul() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.mul([]),
                "mul(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.mul(other=None),
                "mul(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.mul([], out=tensor),
                "mul(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (lambda: tensor.mul(np.uint64(2**63)), "an integer is required"),
            (lambda: tensor.mul(2**64), "int too big to convert"),
            (
                lambda: tensor.mul(-(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaises(TypeError):
            descriptor()
        with self.assertRaises(TypeError):
            descriptor(1, tensor)


if __name__ == "__main__":
    unittest.main()
