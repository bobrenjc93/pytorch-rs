import inspect
import operator
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

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
        self.assertIs(type(actual.is_nonzero()), type(expected.is_nonzero()))
        self.assertEqual(actual.is_nonzero(), expected.is_nonzero())

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
                for actual_call, expected_call in (
                    (
                        lambda actual=actual: bool(actual),
                        lambda expected=expected: bool(expected),
                    ),
                    (
                        lambda actual=actual: operator.truth(actual),
                        lambda expected=expected: operator.truth(expected),
                    ),
                    (actual.is_nonzero, expected.is_nonzero),
                ):
                    self.assert_error_matches(actual_call, expected_call)

    def test_is_nonzero_method_contract_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor(1.0)
        expected = reference_torch.tensor(1.0)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "is_nonzero")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_nonzero"
        )

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "is_nonzero")
            self.assertIsNone(descriptor.__doc__)
            assert_no_argument_signature(self, descriptor, "(self, /)")
        actual_bound = actual.is_nonzero
        expected_bound = expected.is_nonzero
        for bound in (actual_bound, expected_bound):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "is_nonzero")
            self.assertIsNone(bound.__doc__)
            assert_no_argument_signature(self, bound, "()")

        self.assertEqual(actual_descriptor(actual), expected_descriptor(expected))
        for actual_call, expected_call in (
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (
                lambda: actual_bound(dim=0),
                lambda: expected_bound(dim=0),
            ),
            (
                lambda: actual_bound(input=actual),
                lambda: expected_bound(input=expected),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor(1)


if __name__ == "__main__":
    unittest.main()
