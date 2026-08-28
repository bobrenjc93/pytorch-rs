import inspect
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorEqualReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_equal_calls_match(self, actual_left, actual_right, expected_left, expected_right):
        actual_results = (
            actual_left.equal(actual_right),
            torch.equal(actual_left, actual_right),
        )
        expected_results = (
            expected_left.equal(expected_right),
            reference_torch.equal(expected_left, expected_right),
        )
        self.assertEqual(actual_results, expected_results)
        self.assertTrue(all(type(result) is bool for result in actual_results))

    def test_contiguous_strided_offset_and_empty_tensors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_contiguous_offset = torch.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0]]
        )[1]
        expected_contiguous_offset = reference_torch.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0]],
            dtype=reference_torch.float32,
        )[1]
        actual_strided = torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        ).transpose(0, 1)
        expected_strided = reference_torch.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            dtype=reference_torch.float32,
        ).transpose(0, 1)
        actual_offset = torch.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]]
        ).transpose(0, 1)[1]
        expected_offset = reference_torch.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]],
            dtype=reference_torch.float32,
        ).transpose(0, 1)[1]
        actual_strided_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_strided_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)

        cases = (
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                reference_torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=reference_torch.float32
                ),
                reference_torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=reference_torch.float32
                ),
            ),
            (
                actual_contiguous_offset,
                torch.tensor([1.0, 2.0, 3.0]),
                expected_contiguous_offset,
                reference_torch.tensor(
                    [1.0, 2.0, 3.0], dtype=reference_torch.float32
                ),
            ),
            (
                actual_contiguous_offset,
                torch.tensor([1.0, 2.0, 4.0]),
                expected_contiguous_offset,
                reference_torch.tensor(
                    [1.0, 2.0, 4.0], dtype=reference_torch.float32
                ),
            ),
            (
                actual_strided,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                expected_strided,
                reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=reference_torch.float32,
                ),
            ),
            (
                actual_strided,
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]]),
                expected_strided,
                reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 7.0]],
                    dtype=reference_torch.float32,
                ),
            ),
            (
                actual_offset,
                torch.tensor([20.0, 3.0, 4.0]),
                expected_offset,
                reference_torch.tensor(
                    [20.0, 3.0, 4.0], dtype=reference_torch.float32
                ),
            ),
            (
                torch.zeros((2, 0, 3)),
                torch.ones((2, 0, 3)),
                reference_torch.zeros((2, 0, 3)),
                reference_torch.ones((2, 0, 3)),
            ),
            (
                actual_strided_empty,
                torch.zeros((3, 0, 2)),
                expected_strided_empty,
                reference_torch.zeros((3, 0, 2)),
            ),
            (
                actual_strided_empty[1],
                torch.zeros((0, 2)),
                expected_strided_empty[1],
                reference_torch.zeros((0, 2)),
            ),
            (
                torch.zeros((0,)),
                torch.zeros((1, 0)),
                reference_torch.zeros((0,)),
                reference_torch.zeros((1, 0)),
            ),
        )
        for case, (actual_left, actual_right, expected_left, expected_right) in enumerate(
            cases
        ):
            with self.subTest(case=case):
                self.assertEqual(actual_left.shape, expected_left.shape)
                self.assertEqual(actual_left.stride(), expected_left.stride())
                self.assertEqual(
                    actual_left.storage_offset(), expected_left.storage_offset()
                )
                self.assertEqual(actual_right.shape, expected_right.shape)
                self.assertEqual(actual_right.stride(), expected_right.stride())
                self.assert_equal_calls_match(
                    actual_left, actual_right, expected_left, expected_right
                )

    def test_numerical_and_metadata_semantics_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_nan = torch.tensor([float("nan")])
        expected_nan = reference_torch.tensor(
            [float("nan")], dtype=reference_torch.float32
        )
        cases = (
            (
                torch.tensor(1.0),
                torch.tensor([1.0]),
                reference_torch.tensor(1.0),
                reference_torch.tensor([1.0]),
            ),
            (
                torch.tensor([0.0, -0.0]),
                torch.tensor([-0.0, 0.0]),
                reference_torch.tensor([0.0, -0.0]),
                reference_torch.tensor([-0.0, 0.0]),
            ),
            (
                torch.tensor([float("inf"), -float("inf")]),
                torch.tensor([float("inf"), -float("inf")]),
                reference_torch.tensor([float("inf"), -float("inf")]),
                reference_torch.tensor([float("inf"), -float("inf")]),
            ),
            (
                torch.tensor([float("inf")]),
                torch.tensor([-float("inf")]),
                reference_torch.tensor([float("inf")]),
                reference_torch.tensor([-float("inf")]),
            ),
            (actual_nan, actual_nan, expected_nan, expected_nan),
            (
                actual_nan,
                torch.tensor([float("nan")]),
                expected_nan,
                reference_torch.tensor([float("nan")]),
            ),
            (
                torch.tensor([1.0, 2.0], requires_grad=True),
                torch.tensor([1.0, 2.0]),
                reference_torch.tensor([1.0, 2.0], requires_grad=True),
                reference_torch.tensor([1.0, 2.0]),
            ),
        )
        for case, values in enumerate(cases):
            with self.subTest(case=case):
                self.assert_equal_calls_match(*values)

    def test_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        callables = (
            (torch.equal, reference_torch.equal, types.BuiltinFunctionType),
            (
                inspect.getattr_static(torch.Tensor, "equal"),
                inspect.getattr_static(reference_torch.Tensor, "equal"),
                types.MethodDescriptorType,
            ),
            (actual.equal, expected.equal, types.BuiltinMethodType),
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
                self.assertTrue(callable(actual_callable))
                self.assertTrue(callable(expected_callable))
                with self.assertRaises(ValueError):
                    inspect.signature(actual_callable)
                with self.assertRaises(ValueError):
                    inspect.signature(expected_callable)
                actual_doc_signature = next(
                    line for line in actual_callable.__doc__.splitlines() if line
                )
                expected_doc_signature = next(
                    line for line in expected_callable.__doc__.splitlines() if line
                )
                self.assertEqual(actual_doc_signature, expected_doc_signature)

    def test_binding_and_type_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.equal(), lambda: reference_torch.equal()),
            (lambda: torch.equal(actual), lambda: reference_torch.equal(expected)),
            (lambda: torch.equal(None), lambda: reference_torch.equal(None)),
            (lambda: torch.equal(input=1), lambda: reference_torch.equal(input=1)),
            (
                lambda: torch.equal(actual, actual, actual),
                lambda: reference_torch.equal(expected, expected, expected),
            ),
            (
                lambda: torch.equal(None, actual),
                lambda: reference_torch.equal(None, expected),
            ),
            (
                lambda: torch.equal(actual, 1),
                lambda: reference_torch.equal(expected, 1),
            ),
            (
                lambda: torch.equal(input=actual, other=[]),
                lambda: reference_torch.equal(input=expected, other=[]),
            ),
            (
                lambda: torch.equal(foo=actual, other=actual),
                lambda: reference_torch.equal(foo=expected, other=expected),
            ),
            (
                lambda: torch.equal(actual, actual, extra=True),
                lambda: reference_torch.equal(expected, expected, extra=True),
            ),
            (
                lambda: torch.equal(actual, actual, other=actual),
                lambda: reference_torch.equal(expected, expected, other=expected),
            ),
            (lambda: actual.equal(), lambda: expected.equal()),
            (
                lambda: actual.equal(actual, actual),
                lambda: expected.equal(expected, expected),
            ),
            (lambda: actual.equal(None), lambda: expected.equal(None)),
            (
                lambda: actual.equal(input=actual),
                lambda: expected.equal(input=expected),
            ),
            (
                lambda: actual.equal(actual, extra=True),
                lambda: expected.equal(expected, extra=True),
            ),
            (
                lambda: actual.equal(actual, other=actual),
                lambda: expected.equal(expected, other=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
