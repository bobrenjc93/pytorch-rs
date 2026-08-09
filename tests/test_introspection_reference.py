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

    def test_dtype_introspection_matches_for_scalar_empty_and_strided_views(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        pairs = (
            (torch.tensor(3.5), reference_torch.tensor(3.5)),
            (torch.zeros((0,)), reference_torch.zeros((0,))),
            (torch.zeros((2, 0, 3)), reference_torch.zeros((2, 0, 3))),
            (
                torch.zeros((2, 3, 4)).transpose(0, 2),
                reference_torch.zeros((2, 3, 4)).transpose(0, 2),
            ),
            (
                torch.zeros((4, 3, 2)).transpose(0, 2)[1],
                reference_torch.zeros((4, 3, 2)).transpose(0, 2)[1],
            ),
        )
        for actual, expected in pairs:
            with self.subTest(shape=actual.shape, stride=actual.stride()):
                actual_values = (
                    actual.is_floating_point(),
                    torch.is_floating_point(actual),
                    torch.is_floating_point(input=actual),
                    actual.is_complex(),
                    torch.is_complex(actual),
                    torch.is_complex(input=actual),
                    actual.element_size(),
                )
                expected_values = (
                    expected.is_floating_point(),
                    reference_torch.is_floating_point(expected),
                    reference_torch.is_floating_point(input=expected),
                    expected.is_complex(),
                    reference_torch.is_complex(expected),
                    reference_torch.is_complex(input=expected),
                    expected.element_size(),
                )
                self.assertEqual(actual_values, expected_values)
                self.assertTrue(
                    all(type(value) is bool for value in actual_values[:6])
                )
                self.assertIs(type(actual_values[6]), int)

    def test_is_tensor_matches_for_tensors_and_arbitrary_objects(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_tensors = (
            torch.tensor(1.0),
            torch.zeros((0,)),
            torch.zeros((2, 3)).transpose(0, 1),
        )
        expected_tensors = (
            reference_torch.tensor(1.0),
            reference_torch.zeros((0,)),
            reference_torch.zeros((2, 3)).transpose(0, 1),
        )
        for actual, expected in zip(actual_tensors, expected_tensors):
            self.assertEqual(torch.is_tensor(actual), reference_torch.is_tensor(expected))
            self.assertIs(type(torch.is_tensor(actual)), bool)

        non_tensors = (
            None,
            object(),
            1,
            1.5,
            True,
            [],
            {"value": 1},
            np.zeros((2, 3), dtype=np.float32),
            torch.Tensor,
            torch.float32,
            torch.device("cpu"),
        )
        for value in non_tensors:
            with self.subTest(type=type(value).__name__):
                self.assertEqual(torch.is_tensor(value), reference_torch.is_tensor(value))
                self.assertIs(type(torch.is_tensor(value)), bool)

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

        for name in (
            "dim",
            "ndimension",
            "nelement",
            "numel",
            "is_floating_point",
            "is_complex",
            "element_size",
        ):
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

        actual_is_tensor = inspect.signature(torch.is_tensor)
        expected_is_tensor = inspect.signature(reference_torch.is_tensor)
        self.assertEqual(
            tuple((parameter.name, parameter.kind) for parameter in actual_is_tensor.parameters.values()),
            tuple((parameter.name, parameter.kind) for parameter in expected_is_tensor.parameters.values()),
        )

        for actual_function, expected_function in (
            (torch.is_floating_point, reference_torch.is_floating_point),
            (torch.is_complex, reference_torch.is_complex),
        ):
            self.assertIsNone(actual_function.__text_signature__)
            self.assertIsNone(expected_function.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual_function)
            with self.assertRaises(ValueError):
                inspect.signature(expected_function)

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

    def test_new_introspection_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))

        is_tensor_cases = (
            (lambda: torch.is_tensor(), lambda: reference_torch.is_tensor()),
            (
                lambda: torch.is_tensor(actual, actual),
                lambda: reference_torch.is_tensor(expected, expected),
            ),
            (
                lambda: torch.is_tensor(obj=actual),
                lambda: reference_torch.is_tensor(obj=expected),
            ),
            (
                lambda: torch.is_tensor(actual, obj=actual),
                lambda: reference_torch.is_tensor(expected, obj=expected),
            ),
            (
                lambda: torch.is_tensor(extra=actual),
                lambda: reference_torch.is_tensor(extra=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(is_tensor_cases):
            with self.subTest(function="is_tensor", case=case):
                self.assert_error_matches(actual_call, expected_call)

        for name in ("is_floating_point", "is_complex"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            cases = (
                (lambda f=actual_function: f(), lambda f=expected_function: f()),
                (
                    lambda f=actual_function: f(actual, actual),
                    lambda f=expected_function: f(expected, expected),
                ),
                (
                    lambda f=actual_function: f(actual, input=actual),
                    lambda f=expected_function: f(expected, input=expected),
                ),
                (
                    lambda f=actual_function: f(actual, extra=True),
                    lambda f=expected_function: f(expected, extra=True),
                ),
                (lambda f=actual_function: f(1), lambda f=expected_function: f(1)),
                (
                    lambda f=actual_function: f(input=[]),
                    lambda f=expected_function: f(input=[]),
                ),
                (
                    lambda f=actual_function: f(
                        np.zeros((2, 3), dtype=np.float32)
                    ),
                    lambda f=expected_function: f(
                        np.zeros((2, 3), dtype=np.float32)
                    ),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(function=name, case=case):
                    self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
