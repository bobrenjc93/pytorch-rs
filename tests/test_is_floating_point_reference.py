import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsFloatingPointReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_floating_point differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        offset_view = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        ).transpose(0, 1)[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            module.tensor(3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            offset_view,
            extreme_empty,
            leaf,
            tracked,
            leaf.grad,
        )

    def test_scalar_empty_strided_and_autograd_results_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_values = (
                    actual.is_floating_point(),
                    torch.is_floating_point(actual),
                    torch.is_floating_point(input=actual),
                )
                expected_values = (
                    expected.is_floating_point(),
                    reference_torch.is_floating_point(expected),
                    reference_torch.is_floating_point(input=expected),
                )
                self.assertEqual(actual_values, expected_values)
                self.assertTrue(all(type(value) is bool for value in actual_values))

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(
            torch.Tensor, "is_floating_point"
        )
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_floating_point"
        )
        actual_bound = actual_tensor.is_floating_point
        expected_bound = expected_tensor.is_floating_point

        for actual, expected, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual_bound, expected_bound, types.BuiltinMethodType),
            (
                torch.is_floating_point,
                reference_torch.is_floating_point,
                types.BuiltinFunctionType,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertIs(actual_descriptor(actual_tensor), True)
        self.assertIs(expected_descriptor(expected_tensor), True)

    def test_method_and_top_level_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        method_cases = (
            (
                lambda: actual.is_floating_point(1),
                lambda: expected.is_floating_point(1),
                "is_floating_point() takes no arguments (1 given)",
            ),
            (
                lambda: actual.is_floating_point(input=actual),
                lambda: expected.is_floating_point(input=expected),
                "is_floating_point() takes no keyword arguments",
            ),
        )
        for actual_call, expected_call, suffix in method_cases:
            messages = []
            for call in (actual_call, expected_call):
                with self.assertRaises(TypeError) as raised:
                    call()
                messages.append(str(raised.exception))
            for message in messages:
                self.assertTrue(message.endswith(suffix), message)

        cases = (
            (lambda: torch.is_floating_point(), lambda: reference_torch.is_floating_point()),
            (
                lambda: torch.is_floating_point(actual, actual),
                lambda: reference_torch.is_floating_point(expected, expected),
            ),
            (
                lambda: torch.is_floating_point(actual, input=actual),
                lambda: reference_torch.is_floating_point(
                    expected, input=expected
                ),
            ),
            (
                lambda: torch.is_floating_point(actual, extra=True),
                lambda: reference_torch.is_floating_point(expected, extra=True),
            ),
            (lambda: torch.is_floating_point(1), lambda: reference_torch.is_floating_point(1)),
            (
                lambda: torch.is_floating_point(input=[]),
                lambda: reference_torch.is_floating_point(input=[]),
            ),
            (
                lambda: torch.is_floating_point(
                    np.zeros((2, 3), dtype=np.float32)
                ),
                lambda: reference_torch.is_floating_point(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        actual_descriptor = inspect.getattr_static(
            torch.Tensor, "is_floating_point"
        )
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_floating_point"
        )
        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor(1)


if __name__ == "__main__":
    unittest.main()
