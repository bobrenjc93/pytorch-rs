import operator
import re
import unittest

import torch_rs as torch


class TensorTruthinessTests(unittest.TestCase):
    def assert_truth(self, tensor, expected):
        self.assertIs(bool(tensor), expected)
        self.assertIs(operator.truth(tensor), expected)

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
                with self.assertRaisesRegex(RuntimeError, f"^{re.escape(message)}$"):
                    bool(tensor)


if __name__ == "__main__":
    unittest.main()
