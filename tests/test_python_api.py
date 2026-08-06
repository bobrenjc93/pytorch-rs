import unittest

import torch_rs as torch


class PythonApiBaselineTests(unittest.TestCase):
    def test_readme_style_tensor_expression(self):
        x = torch.tensor([[-1.0, 2.0], [3.0, -4.0]])
        y = torch.ones([2, 2])
        result = (x + y).relu()

        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.tolist(), [[0.0, 3.0], [4.0, 0.0]])

    def test_scalar_reduction_and_item(self):
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).sum()
        self.assertEqual(value.shape, ())
        self.assertEqual(value.item(), 10.0)

    def test_matrix_multiplication_operator(self):
        left = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        right = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        output = left @ right
        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(output.tolist(), [[58.0, 64.0], [139.0, 154.0]])

    def test_ragged_input_is_rejected(self):
        with self.assertRaises(ValueError):
            torch.tensor([[1.0], [2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
