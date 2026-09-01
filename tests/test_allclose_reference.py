import unittest

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

    def test_supported_values_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def make_cases(module):
            contiguous_offset = module.tensor(
                [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0]]
            )[1]
            strided = module.tensor(
                [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
            ).transpose(0, 1)
            offset = module.tensor(
                [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]]
            ).transpose(0, 1)[1]
            strided_empty = module.zeros((2, 0, 3)).transpose(0, 2)
            return (
                (module.tensor(1.0), module.tensor(1.0), {}),
                (module.zeros((2, 0, 3)), module.ones((2, 0, 3)), {}),
                (contiguous_offset, module.tensor([1.0, 2.0, 3.0]), {}),
                (contiguous_offset, module.tensor([1.0, 2.0, 3.1]), {}),
                (
                    strided,
                    module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    {},
                ),
                (
                    strided,
                    module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.1]]),
                    {},
                ),
                (offset, module.tensor([20.0, 3.0, 4.0]), {}),
                (strided_empty, module.zeros((3, 0, 2)), {}),
                (module.tensor([0.0, -0.0]), module.tensor([-0.0, 0.0]), {}),
                (
                    module.tensor([1.000009]),
                    module.tensor([1.0]),
                    {"rtol": 1.0e-5, "atol": 0.0},
                ),
                (
                    module.tensor([1.00002]),
                    module.tensor([1.0]),
                    {"rtol": 1.0e-5, "atol": 0.0},
                ),
                (
                    module.tensor([1.0e-8]),
                    module.tensor([0.0]),
                    {"rtol": 0.0, "atol": 1.0e-8},
                ),
                (
                    module.tensor([1.1e-8]),
                    module.tensor([0.0]),
                    {"rtol": 0.0, "atol": 1.0e-8},
                ),
                (
                    module.tensor([float("inf"), -float("inf")]),
                    module.tensor([float("inf"), -float("inf")]),
                    {},
                ),
                (module.tensor([float("inf")]), module.tensor([-float("inf")]), {}),
                (
                    module.tensor([float("inf")]),
                    module.tensor([1.0]),
                    {"rtol": float("inf")},
                ),
                (
                    module.tensor([float("nan")]),
                    module.tensor([float("nan")]),
                    {},
                ),
                (
                    module.tensor([float("nan")]),
                    module.tensor([float("nan")]),
                    {"equal_nan": True},
                ),
                (
                    module.tensor([float("nan")]),
                    module.tensor([1.0]),
                    {"equal_nan": True},
                ),
            )

        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        for case, ((actual_left, actual_right, kwargs), (expected_left, expected_right, _)) in enumerate(
            zip(actual_cases, expected_cases)
        ):
            with self.subTest(case=case, kwargs=kwargs):
                self.assertEqual(actual_left.shape, expected_left.shape)
                self.assertEqual(actual_left.stride(), expected_left.stride())
                self.assertEqual(actual_right.shape, expected_right.shape)
                self.assertEqual(actual_right.stride(), expected_right.stride())
                self.assert_allclose_calls_match(
                    actual_left, actual_right, expected_left, expected_right, **kwargs
                )

    def test_input_nonmutation_and_grad_state_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def contract(module):
            leaf = module.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
            left = leaf.transpose(0, 1)
            right = module.tensor([[1.0, 3.0], [2.0, 4.0001]])
            left_before = left.clone()
            right_before = right.clone()
            first = module.allclose(left, right)
            second = left.allclose(right, atol=0.001, rtol=0.0)
            left_same = module.equal(left, left_before)
            right_same = module.equal(right, right_before)
            grad_before = leaf.grad is None
            grad_enabled_before = module.is_grad_enabled()
            with module.no_grad():
                no_grad_before = module.is_grad_enabled()
                no_grad_result = module.allclose(left, right, atol=0.001, rtol=0.0)
                no_grad_after = module.is_grad_enabled()
            grad_enabled_after = module.is_grad_enabled()
            grad_after = leaf.grad is None
            return (
                first,
                second,
                left_same,
                right_same,
                grad_before,
                grad_enabled_before,
                no_grad_before,
                no_grad_result,
                no_grad_after,
                grad_enabled_after,
                grad_after,
            )

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_binding_tolerance_and_equal_nan_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.allclose(), lambda: reference_torch.allclose()),
            (lambda: torch.allclose(actual), lambda: reference_torch.allclose(expected)),
            (
                lambda: torch.allclose(actual, actual, actual, actual, actual, actual),
                lambda: reference_torch.allclose(
                    expected, expected, expected, expected, expected, expected
                ),
            ),
            (lambda: actual.allclose(), lambda: expected.allclose()),
            (
                lambda: actual.allclose(actual, 1.0, 0.0, False, None),
                lambda: expected.allclose(expected, 1.0, 0.0, False, None),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=None),
                lambda: reference_torch.allclose(expected, expected, rtol=None),
            ),
            (
                lambda: torch.allclose(actual, actual, "1"),
                lambda: reference_torch.allclose(expected, expected, "1"),
            ),
            (
                lambda: actual.allclose(actual, 1.0e-5, "1"),
                lambda: expected.allclose(expected, 1.0e-5, "1"),
            ),
            (
                lambda: torch.allclose(actual, actual, atol=-1.0e-8),
                lambda: reference_torch.allclose(expected, expected, atol=-1.0e-8),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=float("nan")),
                lambda: reference_torch.allclose(expected, expected, rtol=float("nan")),
            ),
            (
                lambda: torch.allclose(actual, actual, equal_nan=1),
                lambda: reference_torch.allclose(expected, expected, equal_nan=1),
            ),
            (
                lambda: torch.allclose(actual, actual, extra=True),
                lambda: reference_torch.allclose(expected, expected, extra=True),
            ),
            (
                lambda: actual.allclose(actual, other=actual),
                lambda: expected.allclose(expected, other=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_shape_expansion_is_an_explicit_unsupported_boundary(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.zeros((2, 1))
        actual_right = torch.zeros((2, 3))
        expected_left = reference_torch.zeros((2, 1))
        expected_right = reference_torch.zeros((2, 3))

        with self.assertRaises(NotImplementedError):
            torch.allclose(actual_left, actual_right)
        self.assertIs(reference_torch.allclose(expected_left, expected_right), True)


if __name__ == "__main__":
    unittest.main()
