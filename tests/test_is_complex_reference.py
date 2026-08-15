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
class TensorIsComplexReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_complex differentials require pinned PyTorch 2.13.0"
            )

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

    def test_float32_results_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_values = (
                    actual.is_complex(),
                    torch.is_complex(actual),
                    torch.is_complex(input=actual),
                    torch.is_complex(a=actual),
                    torch.is_complex(x=actual),
                )
                expected_values = (
                    expected.is_complex(),
                    reference_torch.is_complex(expected),
                    reference_torch.is_complex(input=expected),
                    reference_torch.is_complex(a=expected),
                    reference_torch.is_complex(x=expected),
                )
                self.assertEqual(actual_values, expected_values)
                self.assertTrue(all(type(value) is bool for value in actual_values))

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "is_complex")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_complex"
        )
        actual_bound = actual_tensor.is_complex
        expected_bound = expected_tensor.is_complex

        for actual, expected, expected_type, python_313_signature in (
            (
                actual_descriptor,
                expected_descriptor,
                types.MethodDescriptorType,
                "(self, /)",
            ),
            (actual_bound, expected_bound, types.BuiltinMethodType, "()"),
            (
                torch.is_complex,
                reference_torch.is_complex,
                types.BuiltinFunctionType,
                None,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            for callable_object in (actual, expected):
                if sys.version_info >= (3, 13) and python_313_signature is not None:
                    self.assertEqual(callable_object.__text_signature__, "($self, /)")
                    self.assertEqual(
                        str(inspect.signature(callable_object)),
                        python_313_signature,
                    )
                else:
                    self.assertIsNone(callable_object.__text_signature__)
                    with self.assertRaises(ValueError):
                        inspect.signature(callable_object)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertIs(actual_descriptor(actual_tensor), False)
        self.assertIs(expected_descriptor(expected_tensor), False)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "is_complex")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_complex"
        )
        actual_bound = actual.is_complex
        expected_bound = expected.is_complex
        cases = (
            (lambda: actual.is_complex(1), lambda: expected.is_complex(1)),
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (lambda: actual.is_complex(1, 2), lambda: expected.is_complex(1, 2)),
            (
                lambda: actual.is_complex(input=actual),
                lambda: expected.is_complex(input=expected),
            ),
            (
                lambda: actual_bound(unexpected=True),
                lambda: expected_bound(unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, unexpected=True),
                lambda: expected_descriptor(expected, unexpected=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as actual_raised:
                    actual_call()
                with self.assertRaises(TypeError) as expected_raised:
                    expected_call()
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )

        top_level_cases = (
            (lambda: torch.is_complex(), lambda: reference_torch.is_complex()),
            (
                lambda: torch.is_complex(actual, actual),
                lambda: reference_torch.is_complex(expected, expected),
            ),
            (
                lambda: torch.is_complex(actual, input=actual),
                lambda: reference_torch.is_complex(expected, input=expected),
            ),
            (
                lambda: torch.is_complex(actual, extra=True, input=actual),
                lambda: reference_torch.is_complex(
                    expected, extra=True, input=expected
                ),
            ),
            (
                lambda: torch.is_complex(actual, input=actual, extra=True),
                lambda: reference_torch.is_complex(
                    expected, input=expected, extra=True
                ),
            ),
            (
                lambda: torch.is_complex(actual, extra=True),
                lambda: reference_torch.is_complex(expected, extra=True),
            ),
            (lambda: torch.is_complex(1), lambda: reference_torch.is_complex(1)),
            (
                lambda: torch.is_complex(input=[]),
                lambda: reference_torch.is_complex(input=[]),
            ),
            (
                lambda: torch.is_complex(a=1),
                lambda: reference_torch.is_complex(a=1),
            ),
            (
                lambda: torch.is_complex(x=[]),
                lambda: reference_torch.is_complex(x=[]),
            ),
            (
                lambda: torch.is_complex(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.is_complex(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(top_level_cases):
            with self.subTest(top_level_case=case):
                with self.assertRaises(Exception) as actual_raised:
                    actual_call()
                with self.assertRaises(Exception) as expected_raised:
                    expected_call()
                self.assertEqual(
                    type(actual_raised.exception).__name__,
                    type(expected_raised.exception).__name__,
                )
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )


if __name__ == "__main__":
    unittest.main()
