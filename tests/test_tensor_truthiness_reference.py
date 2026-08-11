import operator
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorTruthinessReferenceTests(unittest.TestCase):
    def assert_truth_matches(self, actual, expected):
        self.assertIs(type(bool(actual)), type(bool(expected)))
        self.assertEqual(bool(actual), bool(expected))
        self.assertIs(type(operator.truth(actual)), type(operator.truth(expected)))
        self.assertEqual(operator.truth(actual), operator.truth(expected))

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_scalar_values_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = (
            0.0,
            -0.0,
            1.0,
            -2.5,
            float("nan"),
            float("inf"),
            -float("inf"),
        )
        for value in values:
            actual = torch.tensor(value)
            expected = reference_torch.tensor(value, dtype=reference_torch.float32)
            with self.subTest(value=value):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assert_truth_matches(actual, expected)

    def test_one_element_strided_views_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [0.0, -0.0, 3.0, -4.0, float("nan"), float("inf"), -float("inf")]
        actual_source = torch.tensor([values]).transpose(0, 1)
        expected_source = reference_torch.tensor(
            [values], dtype=reference_torch.float32
        ).transpose(0, 1)

        for index in range(len(values)):
            actual = actual_source[index]
            expected = expected_source[index]
            with self.subTest(index=index, value=values[index]):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assert_truth_matches(actual, expected)

    def test_ambiguity_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        cases = (
            (torch.zeros((0,)), reference_torch.zeros((0,))),
            (
                torch.zeros((2, 0, 3)).transpose(0, 2),
                reference_torch.zeros((2, 0, 3)).transpose(0, 2),
            ),
            (torch.tensor([0.0, 0.0]), reference_torch.tensor([0.0, 0.0])),
            (
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]).transpose(0, 1),
                reference_torch.tensor([[0.0, 0.0], [0.0, 0.0]]).transpose(
                    0, 1
                ),
            ),
        )
        for actual, expected in cases:
            with self.subTest(shape=actual.shape, stride=actual.stride()):
                self.assert_error_matches(
                    lambda actual=actual: bool(actual),
                    lambda expected=expected: bool(expected),
                )


if __name__ == "__main__":
    unittest.main()
