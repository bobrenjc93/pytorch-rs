"""Native boundaries inherited by the composed functional activation."""
import inspect
import unittest

import numpy as np
import torch_rs as torch
from torch_rs.nn.functional import tanhshrink


class FunctionalTanhshrinkTests(unittest.TestCase):
    def test_public_signature_and_composition(self):
        self.assertEqual(str(inspect.signature(tanhshrink)), "(input)")
        self.assertIs(torch.nn.functional.tanhshrink, tanhshrink)
        source = torch.tensor([-2.0, -0.0, 0.0, 0.5, 3.0], requires_grad=True)
        output = tanhshrink(input=source)
        expected = source - source.tanh()
        np.testing.assert_array_equal(np.asarray(output), np.asarray(expected))
        output.sum().backward()
        first_gradient = np.asarray(source.grad).copy()
        expected.sum().backward()
        np.testing.assert_array_equal(np.asarray(source.grad), 2 * first_gradient)

    def test_unsupported_grad_inputs_preserve_tanh_boundary_and_input(self):
        leaf = torch.tensor([[0.5, -1.0], [2.0, -3.0]], requires_grad=True)
        with torch.no_grad():
            no_grad_view = leaf[0]
        self.assertTrue(no_grad_view.is_leaf)
        rank_four_leaf = torch.ones((1, 1, 1, 2), requires_grad=True)
        cases = {
            "view_created_inside_no_grad": no_grad_view,
            "offset_view": leaf[1],
            "transposed_view": leaf.transpose(0, 1),
            "rank_four_nonleaf": rank_four_leaf + rank_four_leaf,
            "rank_five": torch.ones((1, 1, 1, 1, 2), requires_grad=True),
            "positive_infinity": torch.tensor([float("inf")], requires_grad=True),
            "negative_infinity": torch.tensor([float("-inf")], requires_grad=True),
            "nan": torch.tensor([float("nan")], requires_grad=True),
        }
        for name, source in cases.items():
            before = np.asarray(source).copy().view(np.uint32)
            metadata = (source.shape, source.stride(), source.data_ptr())
            with self.subTest(case=name):
                with self.assertRaisesRegex(
                    RuntimeError, r"^tanh\(\): autograd recording is not supported$"
                ):
                    tanhshrink(source)
                self.assertEqual(metadata, (source.shape, source.stride(), source.data_ptr()))
                np.testing.assert_array_equal(np.asarray(source).view(np.uint32), before)
                with torch.no_grad():
                    result = tanhshrink(source)
                self.assertFalse(result.requires_grad)
                self.assertTrue(result.is_leaf)
        self.assertIsNone(leaf.grad)
        # Rejection must leave the pre-existing graph usable.
        cases["rank_four_nonleaf"].sum().backward()
        self.assertEqual(rank_four_leaf.grad.tolist(), [[[[2.0, 2.0]]]])


if __name__ == "__main__":
    unittest.main()
