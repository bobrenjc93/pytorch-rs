import inspect
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAllCloseReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_allclose_calls_match(
        self, actual_left, actual_right, expected_left, expected_right, **kwargs
    ):
        actual_results = (
            actual_left.allclose(actual_right, **kwargs),
            torch.allclose(actual_left, actual_right, **kwargs),
        )
        expected_results = (
            expected_left.allclose(expected_right, **kwargs),
            reference_torch.allclose(expected_left, expected_right, **kwargs),
        )
        self.assertEqual(actual_results, expected_results)
        self.assertTrue(all(type(result) is bool for result in actual_results))
        self.assertTrue(all(type(result) is bool for result in expected_results))

    def test_supported_value_layout_and_scalar_broadcast_cases_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def tensor_pair(values_left, values_right):
            return (
                torch.tensor(values_left),
                torch.tensor(values_right),
                reference_torch.tensor(values_left, dtype=reference_torch.float32),
                reference_torch.tensor(values_right, dtype=reference_torch.float32),
            )

        actual_strided = torch.tensor(
            [[1.0, 4.0], [2.0, 5.000001], [3.0, 6.0]]
        ).transpose(0, 1)
        expected_strided = reference_torch.tensor(
            [[1.0, 4.0], [2.0, 5.000001], [3.0, 6.0]],
            dtype=reference_torch.float32,
        ).transpose(0, 1)
        actual_offset = torch.tensor(
            [[10.0, 20.0], [1.0, 2.000001], [3.0, 4.0]]
        ).transpose(0, 1)[1]
        expected_offset = reference_torch.tensor(
            [[10.0, 20.0], [1.0, 2.000001], [3.0, 4.0]],
            dtype=reference_torch.float32,
        ).transpose(0, 1)[1]
        actual_boundary = torch.tensor([np.float32(1.0000001e-08).item()])
        expected_boundary = reference_torch.tensor(
            [np.float32(1.0000001e-08).item()], dtype=reference_torch.float32
        )
        actual_boundary_reference = torch.tensor([np.float32(6.024795e-16).item()])
        expected_boundary_reference = reference_torch.tensor(
            [np.float32(6.024795e-16).item()], dtype=reference_torch.float32
        )

        cases = (
            (*tensor_pair([1.0, 1.000001], [1.0, 1.0]), {}),
            (*tensor_pair([1.0, 1.00002], [1.0, 1.0]), {}),
            (*tensor_pair([0.0, -0.0], [-0.0, 0.0]), {}),
            (
                *tensor_pair(
                    [float("inf"), -float("inf")],
                    [float("inf"), -float("inf")],
                ),
                {},
            ),
            (*tensor_pair([float("inf")], [1.0]), {"rtol": float("inf"), "atol": float("inf")}),
            (*tensor_pair([float("nan")], [float("nan")]), {}),
            (*tensor_pair([float("nan")], [float("nan")]), {"equal_nan": True}),
            (
                torch.zeros((2, 0, 3)),
                torch.ones((2, 0, 3)),
                reference_torch.zeros((2, 0, 3)),
                reference_torch.ones((2, 0, 3)),
                {},
            ),
            (
                actual_strided,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                expected_strided,
                reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=reference_torch.float32,
                ),
                {},
            ),
            (
                actual_offset,
                torch.tensor([20.0, 2.0, 4.0]),
                expected_offset,
                reference_torch.tensor([20.0, 2.0, 4.0], dtype=reference_torch.float32),
                {},
            ),
            (
                torch.tensor(1.0),
                torch.tensor([1.0, 1.0]),
                reference_torch.tensor(1.0),
                reference_torch.tensor([1.0, 1.0]),
                {},
            ),
            (
                torch.tensor([1.0, 1.0]),
                torch.tensor(1.0),
                reference_torch.tensor([1.0, 1.0]),
                reference_torch.tensor(1.0),
                {},
            ),
            (
                torch.zeros((0,)),
                torch.tensor(1.0),
                reference_torch.zeros((0,)),
                reference_torch.tensor(1.0),
                {},
            ),
            (actual_boundary, actual_boundary_reference, expected_boundary, expected_boundary_reference, {}),
        )
        for case, (actual_left, actual_right, expected_left, expected_right, kwargs) in enumerate(
            cases
        ):
            with self.subTest(case=case):
                self.assertEqual(actual_left.shape, expected_left.shape)
                self.assertEqual(actual_left.stride(), expected_left.stride())
                self.assertEqual(
                    actual_left.storage_offset(), expected_left.storage_offset()
                )
                self.assert_allclose_calls_match(
                    actual_left, actual_right, expected_left, expected_right, **kwargs
                )

    def test_numpy_and_python_tolerance_scalars_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        tolerance_cases = (
            {"rtol": True, "atol": False},
            {"rtol": 1, "atol": np.uint64(0)},
            {"rtol": np.float64(1e-5), "atol": np.float32(0.0)},
            {"rtol": np.complex64(1 + 2j), "atol": 0.0},
        )
        for kwargs in tolerance_cases:
            with self.subTest(kwargs=kwargs), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assert_allclose_calls_match(actual, actual, expected, expected, **kwargs)

    def test_callable_metadata_and_aliases_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        actual_other = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        expected_other = reference_torch.tensor([1.0])

        alias_cases = (
            (
                lambda: torch.allclose(input=actual, other=actual_other),
                lambda: reference_torch.allclose(input=expected, other=expected_other),
            ),
            (
                lambda: torch.allclose(x=actual, x2=actual_other),
                lambda: reference_torch.allclose(x=expected, x2=expected_other),
            ),
            (
                lambda: torch.allclose(a=actual, other=actual_other),
                lambda: reference_torch.allclose(a=expected, other=expected_other),
            ),
            (
                lambda: torch.allclose(x1=actual, x2=actual_other),
                lambda: reference_torch.allclose(x1=expected, x2=expected_other),
            ),
            (lambda: actual.allclose(x2=actual_other), lambda: expected.allclose(x2=expected_other)),
        )
        for case, (actual_call, expected_call) in enumerate(alias_cases):
            with self.subTest(case=case):
                self.assertIs(actual_call(), expected_call())

        callables = (
            (torch.allclose, reference_torch.allclose, types.BuiltinFunctionType),
            (
                inspect.getattr_static(torch.Tensor, "allclose"),
                inspect.getattr_static(reference_torch.Tensor, "allclose"),
                types.MethodDescriptorType,
            ),
            (actual.allclose, expected.allclose, types.BuiltinMethodType),
        )
        for actual_callable, expected_callable, callable_type in callables:
            with self.subTest(callable_type=callable_type.__name__):
                self.assertIs(type(actual_callable), callable_type)
                self.assertIs(type(expected_callable), callable_type)
                self.assertEqual(actual_callable.__name__, expected_callable.__name__)
                self.assertEqual(
                    actual_callable.__text_signature__,
                    expected_callable.__text_signature__,
                )
                actual_doc_signature = next(
                    line for line in actual_callable.__doc__.splitlines() if line
                )
                expected_doc_signature = next(
                    line for line in expected_callable.__doc__.splitlines() if line
                )
                self.assertEqual(actual_doc_signature, expected_doc_signature)

    def test_binding_tolerance_and_shape_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_other = torch.tensor([1.0])
        expected_other = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.allclose(), lambda: reference_torch.allclose()),
            (lambda: torch.allclose(actual), lambda: reference_torch.allclose(expected)),
            (lambda: torch.allclose(None, actual), lambda: reference_torch.allclose(None, expected)),
            (lambda: torch.allclose(actual, 1), lambda: reference_torch.allclose(expected, 1)),
            (
                lambda: torch.allclose(actual, actual, actual, actual, actual, actual),
                lambda: reference_torch.allclose(
                    expected, expected, expected, expected, expected, expected
                ),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=None),
                lambda: reference_torch.allclose(expected, expected, rtol=None),
            ),
            (
                lambda: torch.allclose(actual, actual, atol="1"),
                lambda: reference_torch.allclose(expected, expected, atol="1"),
            ),
            (
                lambda: torch.allclose(actual, actual, equal_nan=1),
                lambda: reference_torch.allclose(expected, expected, equal_nan=1),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=-1e-5, equal_nan=1),
                lambda: reference_torch.allclose(expected, expected, rtol=-1e-5, equal_nan=1),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=-1e-5),
                lambda: reference_torch.allclose(expected, expected, rtol=-1e-5),
            ),
            (
                lambda: torch.allclose(actual, actual, atol=float("nan")),
                lambda: reference_torch.allclose(expected, expected, atol=float("nan")),
            ),
            (
                lambda: torch.allclose(actual, actual, extra=True),
                lambda: reference_torch.allclose(expected, expected, extra=True),
            ),
            (
                lambda: torch.allclose(actual, actual, other=actual_other),
                lambda: reference_torch.allclose(expected, expected, other=expected_other),
            ),
            (
                lambda: torch.allclose(actual, actual, x=actual_other),
                lambda: reference_torch.allclose(expected, expected, x=expected_other),
            ),
            (
                lambda: torch.allclose(input=actual, other=actual, x2=actual_other),
                lambda: reference_torch.allclose(
                    input=expected, other=expected, x2=expected_other
                ),
            ),
            (
                lambda: torch.allclose(torch.zeros((2,)), torch.zeros((3,))),
                lambda: reference_torch.allclose(
                    reference_torch.zeros((2,)), reference_torch.zeros((3,))
                ),
            ),
            (lambda: actual.allclose(), lambda: expected.allclose()),
            (
                lambda: actual.allclose(actual, 1, 2, False, 3),
                lambda: expected.allclose(expected, 1, 2, False, 3),
            ),
            (lambda: actual.allclose(1), lambda: expected.allclose(1)),
            (lambda: actual.allclose(input=actual), lambda: expected.allclose(input=expected)),
            (
                lambda: actual.allclose(actual, other=actual_other),
                lambda: expected.allclose(expected, other=expected_other),
            ),
            (
                lambda: actual.allclose(actual, x2=actual_other),
                lambda: expected.allclose(expected, x2=expected_other),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_isclose_dtype_promotion_cuda_subclasses_and_broadcasting_stay_fail_closed(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        self.assertFalse(hasattr(torch, "isclose"))
        self.assertTrue(hasattr(reference_torch, "isclose"))
        self.assertIs(
            reference_torch.isclose(
                reference_torch.tensor([1.0]), reference_torch.tensor([1.0])
            ).dtype,
            reference_torch.bool,
        )

        unsupported_message = (
            "^allclose\\(\\): only exact native CPU float32 Tensor inputs with identical shapes "
            "or rank-0 scalar broadcasting are supported$"
        )
        with self.assertRaisesRegex(NotImplementedError, unsupported_message):
            torch.allclose(torch.ones((2, 1)), torch.ones((2, 3)))
        self.assertIs(
            reference_torch.allclose(reference_torch.ones((2, 1)), reference_torch.ones((2, 3))),
            True,
        )

        with self.assertRaisesRegex(NotImplementedError, unsupported_message):
            torch.allclose(
                torch.tensor([1.0]),
                reference_torch.tensor([1.0], dtype=reference_torch.float64),
            )
        with self.assertRaisesRegex(RuntimeError, "Float did not match Double"):
            reference_torch.allclose(
                reference_torch.tensor([1.0], dtype=reference_torch.float32),
                reference_torch.tensor([1.0], dtype=reference_torch.float64),
            )

        if reference_torch.cuda.is_available():
            expected_cuda = reference_torch.tensor([1.0], device="cuda")
            self.assertIs(reference_torch.allclose(expected_cuda, expected_cuda), True)
            with self.assertRaisesRegex(NotImplementedError, unsupported_message):
                torch.allclose(torch.tensor([1.0]), expected_cuda)

        class ExpectedSubclass(reference_torch.Tensor):
            pass

        expected_subclass = reference_torch.Tensor._make_subclass(
            ExpectedSubclass,
            reference_torch.tensor([1.0], dtype=reference_torch.float32),
            False,
        )
        self.assertIs(reference_torch.allclose(expected_subclass, expected_subclass), True)
        with self.assertRaisesRegex(NotImplementedError, unsupported_message):
            torch.allclose(torch.tensor([1.0]), expected_subclass)


if __name__ == "__main__":
    unittest.main()
