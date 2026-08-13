import inspect
import operator
import re
import types
import unittest

import torch_rs as torch


class TensorTruthinessTests(unittest.TestCase):
    def assert_truth(self, tensor, expected):
        self.assertIs(bool(tensor), expected)
        self.assertIs(operator.truth(tensor), expected)
        self.assertIs(tensor.is_nonzero(), expected)

    def test_scalar_signed_zero_and_non_finite_values(self):
        cases = (
            (0.0, False),
            (-0.0, False),
            (1.0, True),
            (-2.5, True),
            (float("nan"), True),
            (float("inf"), True),
            (-float("inf"), True),
        )
        for value, expected in cases:
            tensor = torch.tensor(value)
            with self.subTest(value=value):
                self.assertEqual(tensor.shape, ())
                self.assert_truth(tensor, expected)

    def test_one_element_offset_views_use_their_strided_value(self):
        values = [0.0, -0.0, 3.0, -4.0, float("nan"), float("inf"), -float("inf")]
        expected = [False, False, True, True, True, True, True]
        transposed = torch.tensor([values]).transpose(0, 1)

        for index, truth in enumerate(expected):
            view = transposed[index]
            with self.subTest(index=index, value=values[index]):
                self.assertEqual(view.shape, (1,))
                self.assertEqual(view.stride(), (len(values),))
                self.assertEqual(view.storage_offset(), index)
                self.assert_truth(view, truth)

    def test_empty_and_multi_element_tensors_are_ambiguous(self):
        cases = (
            (
                torch.zeros((0,)),
                "Boolean value of Tensor with no values is ambiguous",
            ),
            (
                torch.zeros((2, 0, 3)).transpose(0, 2),
                "Boolean value of Tensor with no values is ambiguous",
            ),
            (
                torch.tensor([0.0, 0.0]),
                "Boolean value of Tensor with more than one value is ambiguous",
            ),
            (
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]).transpose(0, 1),
                "Boolean value of Tensor with more than one value is ambiguous",
            ),
        )
        for tensor, message in cases:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                for operation in (bool, operator.truth, lambda value: value.is_nonzero()):
                    with self.assertRaisesRegex(
                        RuntimeError, f"^{re.escape(message)}$"
                    ):
                        operation(tensor)

    def test_is_nonzero_no_argument_method_contract(self):
        tensor = torch.tensor(1.0)
        descriptor = inspect.getattr_static(torch.Tensor, "is_nonzero")
        bound = tensor.is_nonzero

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "is_nonzero")
        self.assertEqual(bound.__name__, "is_nonzero")
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertIsNone(descriptor.__doc__)
        self.assertIsNone(bound.__doc__)
        for callable_object in (descriptor, bound):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assertIs(descriptor(tensor), True)

        calls = (
            (
                lambda: tensor.is_nonzero(1),
                "Tensor.is_nonzero() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_nonzero(1, 2),
                "Tensor.is_nonzero() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_nonzero(dim=0),
                "Tensor.is_nonzero() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method Tensor.is_nonzero() needs an argument",
            ),
            (
                lambda: descriptor(tensor, 1),
                "Tensor.is_nonzero() takes no arguments (1 given)",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
