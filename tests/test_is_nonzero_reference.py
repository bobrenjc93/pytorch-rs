import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchIsNonzeroReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_nonzero differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_result_matches(self, actual, expected):
        actual_results = (
            torch.is_nonzero(actual),
            torch.is_nonzero(input=actual),
            torch.is_nonzero(a=actual),
            torch.is_nonzero(x=actual),
        )
        expected_results = (
            reference_torch.is_nonzero(expected),
            reference_torch.is_nonzero(input=expected),
            reference_torch.is_nonzero(a=expected),
            reference_torch.is_nonzero(x=expected),
        )
        self.assertEqual(actual_results, expected_results)
        self.assertTrue(all(type(result) is bool for result in actual_results))

    def test_scalar_values_match_pytorch_2_13(self):
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
                self.assert_result_matches(actual, expected)

    def test_one_element_strided_views_match_pytorch_2_13(self):
        values = [0.0, -0.0, 3.0, -4.0, float("nan"), float("inf"), -float("inf")]
        actual_source = torch.tensor([values]).transpose(0, 1)
        expected_source = reference_torch.tensor(
            [values], dtype=reference_torch.float32
        ).transpose(0, 1)

        for index, value in enumerate(values):
            actual = actual_source[index]
            expected = expected_source[index]
            with self.subTest(index=index, value=value):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assert_result_matches(actual, expected)

    def test_ambiguity_errors_match_pytorch_2_13(self):
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
                for actual_call, expected_call in (
                    (
                        lambda actual=actual: torch.is_nonzero(actual),
                        lambda expected=expected: reference_torch.is_nonzero(expected),
                    ),
                    (
                        lambda actual=actual: torch.is_nonzero(input=actual),
                        lambda expected=expected: reference_torch.is_nonzero(
                            input=expected
                        ),
                    ),
                ):
                    self.assert_error_matches(actual_call, expected_call)

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.is_nonzero
        expected = reference_torch.is_nonzero

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        for function in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def test_binding_and_tensor_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.is_nonzero(), lambda: reference_torch.is_nonzero()),
            (
                lambda: torch.is_nonzero(actual, actual),
                lambda: reference_torch.is_nonzero(expected, expected),
            ),
            (
                lambda: torch.is_nonzero(actual, input=actual),
                lambda: reference_torch.is_nonzero(expected, input=expected),
            ),
            (
                lambda: torch.is_nonzero(actual, extra=True, input=actual),
                lambda: reference_torch.is_nonzero(
                    expected, extra=True, input=expected
                ),
            ),
            (
                lambda: torch.is_nonzero(actual, input=actual, extra=True),
                lambda: reference_torch.is_nonzero(
                    expected, input=expected, extra=True
                ),
            ),
            (
                lambda: torch.is_nonzero(extra=actual),
                lambda: reference_torch.is_nonzero(extra=expected),
            ),
            (
                lambda: torch.is_nonzero(1, extra=True),
                lambda: reference_torch.is_nonzero(1, extra=True),
            ),
            (
                lambda: torch.is_nonzero(input=[]),
                lambda: reference_torch.is_nonzero(input=[]),
            ),
            (
                lambda: torch.is_nonzero(a=1),
                lambda: reference_torch.is_nonzero(a=1),
            ),
            (
                lambda: torch.is_nonzero(x=[]),
                lambda: reference_torch.is_nonzero(x=[]),
            ),
            (
                lambda: torch.is_nonzero(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.is_nonzero(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
