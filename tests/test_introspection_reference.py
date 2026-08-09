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
class TensorIntrospectionReferenceTests(unittest.TestCase):
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

    def assert_introspection_matches(self, actual, expected, *, case):
        with self.subTest(case=case, shape=actual.shape):
            actual_values = (
                actual.ndim,
                actual.dim(),
                actual.ndimension(),
                actual.nelement(),
                actual.numel(),
                torch.numel(actual),
                torch.numel(input=actual),
            )
            expected_values = (
                expected.ndim,
                expected.dim(),
                expected.ndimension(),
                expected.nelement(),
                expected.numel(),
                reference_torch.numel(expected),
                reference_torch.numel(input=expected),
            )
            self.assertEqual(actual_values, expected_values)
            self.assertTrue(all(type(value) is int for value in actual_values))

    def test_seeded_shapes_and_metadata_views_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0xD1_213)
        shapes = [(), (0,), (1,), (2, 0, 3), (2, 3, 4), (1,) * 65]
        for _ in range(48):
            rank = int(rng.integers(0, 9))
            shapes.append(tuple(int(size) for size in rng.integers(0, 5, rank)))

        for case, shape in enumerate(shapes):
            actual = torch.zeros(shape)
            expected = reference_torch.zeros(shape)
            if len(shape) >= 2:
                actual = actual.transpose(0, -1)
                expected = expected.transpose(0, -1)
            if actual.shape and actual.shape[0] > 0 and case % 3 == 0:
                actual = actual[-1]
                expected = expected[-1]
            if case % 4 == 0:
                actual = actual.squeeze()
                expected = expected.squeeze()
            self.assert_introspection_matches(actual, expected, case=case)

        actual_extreme = torch.zeros((0,)).reshape((2, 0, sys.maxsize)).transpose(0, 2)
        expected_extreme = (
            reference_torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        self.assert_introspection_matches(
            actual_extreme, expected_extreme, case="extreme-empty-view"
        )

    def test_descriptors_signatures_and_bound_unbound_behavior_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 0, 3))
        expected = reference_torch.zeros((2, 0, 3))

        actual_ndim = inspect.getattr_static(torch.Tensor, "ndim")
        expected_ndim = inspect.getattr_static(reference_torch.Tensor, "ndim")
        self.assertIs(type(actual_ndim), types.GetSetDescriptorType)
        self.assertIs(type(expected_ndim), types.GetSetDescriptorType)
        self.assertEqual(actual_ndim.__name__, expected_ndim.__name__)
        self.assertEqual(
            actual_ndim.__get__(actual, torch.Tensor),
            expected_ndim.__get__(expected, reference_torch.Tensor),
        )
        self.assertIs(actual_ndim.__get__(None, torch.Tensor), actual_ndim)
        self.assertIs(
            expected_ndim.__get__(None, reference_torch.Tensor), expected_ndim
        )

        for name in ("dim", "ndimension", "nelement", "numel"):
            with self.subTest(name=name):
                actual_descriptor = inspect.getattr_static(torch.Tensor, name)
                expected_descriptor = inspect.getattr_static(reference_torch.Tensor, name)
                self.assertIs(type(actual_descriptor), types.MethodDescriptorType)
                self.assertIs(type(expected_descriptor), types.MethodDescriptorType)
                self.assertEqual(
                    actual_descriptor.__text_signature__,
                    expected_descriptor.__text_signature__,
                )
                self.assertEqual(
                    str(inspect.signature(actual_descriptor)),
                    str(inspect.signature(expected_descriptor)),
                )
                self.assertEqual(
                    str(inspect.signature(getattr(actual, name))),
                    str(inspect.signature(getattr(expected, name))),
                )
                self.assertEqual(actual_descriptor(actual), expected_descriptor(expected))

                self.assert_error_matches(
                    lambda name=name: getattr(actual, name)(1),
                    lambda name=name: getattr(expected, name)(1),
                )
                self.assert_error_matches(
                    lambda name=name: getattr(actual, name)(other=1),
                    lambda name=name: getattr(expected, name)(other=1),
                )
                with self.assertRaises(TypeError):
                    actual_descriptor()
                with self.assertRaises(TypeError):
                    expected_descriptor()
                with self.assertRaises(TypeError):
                    actual_descriptor(1)
                with self.assertRaises(TypeError):
                    expected_descriptor(1)

        self.assertIsNone(torch.numel.__text_signature__)
        self.assertIsNone(reference_torch.numel.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(torch.numel)
        with self.assertRaises(ValueError):
            inspect.signature(reference_torch.numel)

    def test_argument_and_type_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        cases = (
            (lambda: torch.numel(), lambda: reference_torch.numel()),
            (
                lambda: torch.numel(actual, actual),
                lambda: reference_torch.numel(expected, expected),
            ),
            (
                lambda: torch.numel(actual, input=actual),
                lambda: reference_torch.numel(expected, input=expected),
            ),
            (
                lambda: torch.numel(actual, extra=True),
                lambda: reference_torch.numel(expected, extra=True),
            ),
            (lambda: torch.numel(1), lambda: reference_torch.numel(1)),
            (lambda: torch.numel(input=[]), lambda: reference_torch.numel(input=[])),
            (
                lambda: torch.numel(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.numel(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
