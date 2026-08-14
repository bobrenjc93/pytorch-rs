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
class TensorIsSameSizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_same_size differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_shape_cases(self, module):
        source = module.tensor(
            [
                [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ],
            dtype=module.float32,
        )
        square = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
        )
        return (
            (source, source),
            (source, source.detach()),
            (source, source.clone()),
            (source, module.zeros((2, 3, 2), dtype=module.float32)),
            (source, source.transpose(0, 2).transpose(0, 2)),
            (source, source.reshape((6, 2))),
            (source, source.transpose(1, 2)),
            (source[0], source[1]),
            (square, square.transpose(0, 1)),
            (
                module.tensor(3.0, dtype=module.float32),
                module.tensor(-8.0, dtype=module.float32),
            ),
            (
                module.tensor(3.0, dtype=module.float32),
                module.tensor([3.0], dtype=module.float32),
            ),
            (
                module.zeros((2, 0, 3), dtype=module.float32),
                module.ones((2, 0, 3), dtype=module.float32),
            ),
            (
                module.zeros((0,), dtype=module.float32),
                module.zeros((1, 0), dtype=module.float32),
            ),
        )

    def test_shape_results_match_pytorch_2_13(self):
        actual_cases = self.make_shape_cases(torch)
        expected_cases = self.make_shape_cases(reference_torch)
        for case, (
            (actual_left, actual_right),
            (expected_left, expected_right),
        ) in enumerate(zip(actual_cases, expected_cases, strict=True)):
            with self.subTest(case=case):
                self.assertEqual(actual_left.shape, tuple(expected_left.shape))
                self.assertEqual(actual_right.shape, tuple(expected_right.shape))
                actual_results = (
                    actual_left.is_same_size(actual_right),
                    torch.is_same_size(actual_left, actual_right),
                    torch.is_same_size(input=actual_left, other=actual_right),
                    torch.is_same_size(x1=actual_left, x2=actual_right),
                    torch.is_same_size(actual_left, x2=actual_right),
                )
                expected_results = (
                    expected_left.is_same_size(expected_right),
                    reference_torch.is_same_size(expected_left, expected_right),
                    reference_torch.is_same_size(
                        input=expected_left, other=expected_right
                    ),
                    reference_torch.is_same_size(x1=expected_left, x2=expected_right),
                    reference_torch.is_same_size(expected_left, x2=expected_right),
                )
                self.assertTrue(all(type(result) is bool for result in actual_results))
                self.assertEqual(actual_results, expected_results)

    def test_extreme_empty_and_autograd_behavior_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            extreme_empty = module.zeros((0,), dtype=module.float32).reshape(
                (2, 0, sys.maxsize)
            )
            independent_empty = module.ones((0,), dtype=module.float32).reshape(
                (2, 0, sys.maxsize)
            )

            leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            tracked = (leaf * 2.0).transpose(0, 1)
            independent = module.zeros(
                (2, 2),
                dtype=module.float32,
                requires_grad=True,
            )
            graph_before = (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                tracked.requires_grad,
                tracked.is_leaf,
                independent.requires_grad,
                independent.is_leaf,
                independent.grad,
            )
            results = (
                extreme_empty.is_same_size(independent_empty),
                module.is_same_size(input=extreme_empty, other=independent_empty),
                module.is_same_size(
                    x1=extreme_empty,
                    x2=independent_empty.transpose(0, 2),
                ),
                module.is_same_size(tracked, independent),
            )
            graph_after = (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                tracked.requires_grad,
                tracked.is_leaf,
                independent.requires_grad,
                independent.is_leaf,
                independent.grad,
            )
            tracked.sum().backward()
            outcomes.append(
                (
                    results,
                    graph_before,
                    graph_after,
                    leaf.grad.tolist(),
                    independent.grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "is_same_size")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_same_size"
        )

        for actual, expected, expected_type in (
            (
                torch.is_same_size,
                reference_torch.is_same_size,
                types.BuiltinFunctionType,
            ),
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (
                actual_tensor.is_same_size,
                expected_tensor.is_same_size,
                types.BuiltinMethodType,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertEqual(
            torch.is_same_size.__module__,
            torch.tensor.__module__,
        )
        self.assertEqual(
            reference_torch.is_same_size.__module__,
            reference_torch.tensor.__module__,
        )
        self.assertIn("is_same_size", torch.__all__)
        self.assertIn("is_same_size", reference_torch.__all__)
        self.assertEqual(
            actual_descriptor(actual_tensor, actual_tensor),
            expected_descriptor(expected_tensor, expected_tensor),
        )
        self.assertEqual(
            actual_tensor.is_same_size(other=actual_tensor),
            expected_tensor.is_same_size(other=expected_tensor),
        )
        self.assertEqual(
            torch.is_same_size(x1=actual_tensor, x2=actual_tensor),
            reference_torch.is_same_size(x1=expected_tensor, x2=expected_tensor),
        )

    def test_binding_and_non_tensor_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.is_same_size(), lambda: reference_torch.is_same_size()),
            (
                lambda: torch.is_same_size(actual),
                lambda: reference_torch.is_same_size(expected),
            ),
            (
                lambda: torch.is_same_size(other=actual),
                lambda: reference_torch.is_same_size(other=expected),
            ),
            (
                lambda: torch.is_same_size(actual, actual, actual),
                lambda: reference_torch.is_same_size(expected, expected, expected),
            ),
            (
                lambda: torch.is_same_size(1, actual),
                lambda: reference_torch.is_same_size(1, expected),
            ),
            (
                lambda: torch.is_same_size(actual, None),
                lambda: reference_torch.is_same_size(expected, None),
            ),
            (
                lambda: torch.is_same_size(input=[], other=actual),
                lambda: reference_torch.is_same_size(input=[], other=expected),
            ),
            (
                lambda: torch.is_same_size(input=actual, other=1),
                lambda: reference_torch.is_same_size(input=expected, other=1),
            ),
            (
                lambda: torch.is_same_size(x1=actual, x2=[]),
                lambda: reference_torch.is_same_size(x1=expected, x2=[]),
            ),
            (
                lambda: torch.is_same_size(np.zeros((2, 3), dtype=np.float32), actual),
                lambda: reference_torch.is_same_size(
                    np.zeros((2, 3), dtype=np.float32), expected
                ),
            ),
            (
                lambda: torch.is_same_size(actual, actual, extra=True),
                lambda: reference_torch.is_same_size(expected, expected, extra=True),
            ),
            (
                lambda: torch.is_same_size(actual, actual, input=actual),
                lambda: reference_torch.is_same_size(
                    expected, expected, input=expected
                ),
            ),
            (
                lambda: torch.is_same_size(actual, actual, other=actual),
                lambda: reference_torch.is_same_size(
                    expected, expected, other=expected
                ),
            ),
            (
                lambda: torch.is_same_size(actual, actual, x1=actual),
                lambda: reference_torch.is_same_size(expected, expected, x1=expected),
            ),
            (
                lambda: torch.is_same_size(actual, other=actual, x2=actual),
                lambda: reference_torch.is_same_size(
                    expected, other=expected, x2=expected
                ),
            ),
            (
                lambda: torch.is_same_size(input=actual, x1=actual, other=actual),
                lambda: reference_torch.is_same_size(
                    input=expected, x1=expected, other=expected
                ),
            ),
            (
                lambda: torch.is_same_size(input=1, other=actual, extra=True),
                lambda: reference_torch.is_same_size(
                    input=1, other=expected, extra=True
                ),
            ),
            (
                lambda: torch.is_same_size(input=actual, extra=True),
                lambda: reference_torch.is_same_size(input=expected, extra=True),
            ),
            (lambda: actual.is_same_size(), lambda: expected.is_same_size()),
            (
                lambda: actual.is_same_size(actual, actual),
                lambda: expected.is_same_size(expected, expected),
            ),
            (
                lambda: actual.is_same_size(actual, actual, actual),
                lambda: expected.is_same_size(expected, expected, expected),
            ),
            (
                lambda: actual.is_same_size(actual, other=actual),
                lambda: expected.is_same_size(expected, other=expected),
            ),
            (
                lambda: actual.is_same_size(tensor=actual),
                lambda: expected.is_same_size(tensor=expected),
            ),
            (
                lambda: actual.is_same_size(actual, extra=True),
                lambda: expected.is_same_size(expected, extra=True),
            ),
            (lambda: actual.is_same_size(1), lambda: expected.is_same_size(1)),
            (lambda: actual.is_same_size(None), lambda: expected.is_same_size(None)),
            (lambda: actual.is_same_size([]), lambda: expected.is_same_size([])),
            (
                lambda: actual.is_same_size(np.zeros((2, 3), dtype=np.float32)),
                lambda: expected.is_same_size(np.zeros((2, 3), dtype=np.float32)),
            ),
            (
                lambda: actual.is_same_size(other=1),
                lambda: expected.is_same_size(other=1),
            ),
            (
                lambda: actual.is_same_size(other=None),
                lambda: expected.is_same_size(other=None),
            ),
            (
                lambda: actual.is_same_size(other=[]),
                lambda: expected.is_same_size(other=[]),
            ),
            (
                lambda: actual.is_same_size(**{"other": 1, "extra": True}),
                lambda: expected.is_same_size(**{"other": 1, "extra": True}),
            ),
            (
                lambda: actual.is_same_size(**{"extra": True, "other": 1}),
                lambda: expected.is_same_size(**{"extra": True, "other": 1}),
            ),
            (
                lambda: actual.is_same_size(1, other=actual),
                lambda: expected.is_same_size(1, other=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
