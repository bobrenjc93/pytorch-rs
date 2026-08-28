import inspect
import re
import types
import unittest

import torch_rs as torch


class TensorEqualTests(unittest.TestCase):
    def assert_equal_result(self, left, right, expected):
        method_result = left.equal(right)
        function_result = torch.equal(left, right)
        self.assertIs(type(method_result), bool)
        self.assertIs(type(function_result), bool)
        self.assertIs(method_result, expected)
        self.assertIs(function_result, expected)

    def test_contiguous_strided_offset_and_empty_tensors(self):
        contiguous_offset = torch.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0]]
        )[1]
        strided = torch.tensor([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]).transpose(
            0, 1
        )
        offset = torch.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]]
        ).transpose(0, 1)[1]
        strided_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        offset_empty = strided_empty[1]

        self.assertTrue(contiguous_offset.is_contiguous())
        self.assertEqual(contiguous_offset.storage_offset(), 3)
        self.assertEqual(strided.stride(), (1, 2))
        self.assertEqual(offset.stride(), (2,))
        self.assertEqual(offset.storage_offset(), 1)
        self.assertEqual(strided_empty.stride(), (1, 3, 3))
        self.assertEqual(offset_empty.storage_offset(), 1)

        cases = (
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                True,
            ),
            (contiguous_offset, torch.tensor([1.0, 2.0, 3.0]), True),
            (contiguous_offset, torch.tensor([1.0, 2.0, 4.0]), False),
            (strided, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), True),
            (strided, torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]]), False),
            (offset, torch.tensor([20.0, 3.0, 4.0]), True),
            (offset, torch.tensor([10.0, 1.0, 2.0]), False),
            (torch.zeros((2, 0, 3)), torch.ones((2, 0, 3)), True),
            (strided_empty, torch.zeros((3, 0, 2)), True),
            (offset_empty, torch.zeros((0, 2)), True),
            (torch.zeros((0,)), torch.zeros((1, 0)), False),
        )
        for left, right, expected in cases:
            with self.subTest(
                left_shape=left.shape,
                left_stride=left.stride(),
                right_shape=right.shape,
                right_stride=right.stride(),
            ):
                self.assert_equal_result(left, right, expected)

    def test_shape_signed_zero_infinity_nan_and_autograd_metadata(self):
        nan = torch.tensor([float("nan")])
        cases = (
            (torch.tensor(1.0), torch.tensor([1.0]), False),
            (torch.tensor([0.0, -0.0]), torch.tensor([-0.0, 0.0]), True),
            (
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                True,
            ),
            (torch.tensor([float("inf")]), torch.tensor([-float("inf")]), False),
            (nan, nan, False),
            (nan, torch.tensor([float("nan")]), False),
            (
                torch.tensor([1.0, 2.0], requires_grad=True),
                torch.tensor([1.0, 2.0]),
                True,
            ),
        )
        for left, right, expected in cases:
            with self.subTest(left=left.tolist(), right=right.tolist()):
                self.assert_equal_result(left, right, expected)

    def test_callable_metadata_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "equal")
        bound = tensor.equal

        self.assertIs(type(torch.equal), types.BuiltinFunctionType)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertTrue(callable(torch.equal))
        self.assertTrue(callable(descriptor))
        self.assertTrue(callable(bound))
        self.assertEqual(torch.equal.__name__, "equal")
        self.assertEqual(descriptor.__name__, "equal")
        self.assertEqual(bound.__name__, "equal")
        self.assertIsNone(torch.equal.__text_signature__)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        for callable_object in (torch.equal, descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assertIs(descriptor(tensor, tensor), True)
        self.assertIn("equal", torch.__all__)

    def test_type_and_binding_errors(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.equal(),
                'equal() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.equal(tensor),
                'equal() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: torch.equal(None),
                "equal(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.equal(input=1),
                "equal(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.equal(tensor, tensor, tensor),
                "equal() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: torch.equal(None, tensor),
                "equal(): argument 'input' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: torch.equal(tensor, 1),
                "equal(): argument 'other' (position 2) must be Tensor, not int",
            ),
            (
                lambda: torch.equal(input=tensor, other=[]),
                "equal(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: torch.equal(foo=tensor, other=tensor),
                'equal() missing 2 required positional argument: "input", "other"',
            ),
            (
                lambda: torch.equal(tensor, tensor, extra=True),
                "equal() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.equal(tensor, tensor, other=tensor),
                "equal() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.equal(),
                'equal() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.equal(tensor, tensor),
                "equal() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.equal(None),
                "equal(): argument 'other' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.equal(input=tensor),
                'equal() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.equal(tensor, extra=True),
                "equal() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.equal(tensor, other=tensor),
                "equal() got multiple values for argument 'other'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
