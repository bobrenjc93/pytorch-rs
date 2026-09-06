import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelCatReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.cat differentials require pinned PyTorch 2.13.0"
            )

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    @staticmethod
    def fresh_storage_observation(result, inputs):
        return tuple(
            (not result.is_set_to(input), result.data_ptr() != input.data_ptr())
            for input in inputs
            if input.numel()
        )

    def test_list_tuple_empty_operands_order_metadata_and_storage_match_pytorch_2_13(
        self,
    ):
        actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_base = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=reference_torch.float32,
        )
        actual_offset = actual_base.transpose(0, 1)[1]
        expected_offset = expected_base.transpose(0, 1)[1]

        cases = (
            (
                "list dim 0",
                [actual_offset, torch.tensor([]), torch.tensor([-0.0, 7.5])],
                [
                    expected_offset,
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([-0.0, 7.5], dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "tuple dim -1",
                (torch.tensor([]), torch.tensor([3.25]), torch.tensor([])),
                (
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([3.25], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                ),
                -1,
            ),
            (
                "single input",
                [torch.tensor([11.0, 13.0])],
                [reference_torch.tensor([11.0, 13.0], dtype=reference_torch.float32)],
                0,
            ),
            (
                "keywords",
                [torch.tensor([8.0]), torch.tensor([]), torch.tensor([9.0])],
                [
                    reference_torch.tensor([8.0], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([9.0], dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "axis alias",
                [torch.tensor([14.0]), torch.tensor([15.0])],
                [
                    reference_torch.tensor([14.0], dtype=reference_torch.float32),
                    reference_torch.tensor([15.0], dtype=reference_torch.float32),
                ],
                0,
            ),
        )
        for case, actual_inputs, expected_inputs, dimension in cases:
            with self.subTest(case=case):
                if case == "keywords":
                    actual = torch.cat(tensors=actual_inputs, dim=dimension, out=None)
                    expected = reference_torch.cat(
                        tensors=expected_inputs, dim=dimension, out=None
                    )
                elif case == "axis alias":
                    actual = torch.cat(actual_inputs, axis=dimension)
                    expected = reference_torch.cat(expected_inputs, axis=dimension)
                else:
                    actual = torch.cat(actual_inputs, dim=dimension)
                    expected = reference_torch.cat(expected_inputs, dim=dimension)
                self.assert_matches(actual, expected, case=case)
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

    def test_no_grad_grad_requiring_operands_match_pytorch_2_13(self):
        actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
        actual_right = torch.tensor([3.0], requires_grad=True)
        expected_left = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [3.0], dtype=reference_torch.float32, requires_grad=True
        )

        with torch.no_grad():
            actual = torch.cat([actual_left, actual_right], dim=-1)
        with reference_torch.no_grad():
            expected = reference_torch.cat([expected_left, expected_right], dim=-1)

        self.assert_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_left.grad)
        self.assertIsNone(actual_right.grad)
        self.assertIsNone(expected_left.grad)
        self.assertIsNone(expected_right.grad)

    def test_axis_dim_conflicts_match_pytorch_2_13(self):
        actual = [torch.tensor([1.0])]
        expected = [reference_torch.tensor([1.0], dtype=reference_torch.float32)]
        cases = (
            (
                "dim keyword",
                lambda: torch.cat(actual, dim=0, axis=0),
                lambda: reference_torch.cat(expected, dim=0, axis=0),
            ),
            (
                "dim positional",
                lambda: torch.cat(actual, 0, axis=0),
                lambda: reference_torch.cat(expected, 0, axis=0),
            ),
        )
        for case, actual_call, expected_call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    actual_call()
                with self.assertRaisesRegex(
                    TypeError,
                    r"^cat\(\) got an unexpected keyword argument 'axis'$",
                ):
                    expected_call()


if __name__ == "__main__":
    unittest.main()
