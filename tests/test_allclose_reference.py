import copy
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchAllcloseReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_allclose_calls_match(self, actual_left, actual_right, expected_left, expected_right, **kwargs):
        actual_result = torch.allclose(actual_left, actual_right, **kwargs)
        expected_result = reference_torch.allclose(expected_left, expected_right, **kwargs)
        self.assertEqual(actual_result, expected_result)
        self.assertIs(type(actual_result), bool)

    def test_same_shape_values_layouts_and_special_values_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        def make_cases(module):
            contiguous = module.tensor([[1.0, 2.0], [3.0, 4.0]])
            close = module.tensor([[1.0, 2.00001], [3.0, 4.0]])
            far = module.tensor([[1.0, 2.0001], [3.0, 4.0]])
            noncontiguous = module.tensor(
                [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
            ).transpose(0, 1)
            offset = module.tensor([[99.0, 98.0, 97.0], [1.0, 2.0, 3.0]])[1]
            strided_offset = module.tensor(
                [[99.0, 1.0], [98.0, 2.0], [97.0, 3.0]]
            ).transpose(0, 1)[1]
            return (
                ("scalar", module.tensor(1.0), module.tensor(1.0), {}),
                ("empty", module.zeros((2, 0, 3)), module.ones((2, 0, 3)), {}),
                (
                    "noncontiguous",
                    noncontiguous,
                    module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    {},
                ),
                ("offset", offset, module.tensor([1.0, 2.0, 3.0]), {}),
                ("strided offset", strided_offset, module.tensor([1.0, 2.0, 3.0]), {}),
                ("default tolerance true", contiguous, close, {}),
                ("default tolerance false", contiguous, far, {}),
                ("explicit rtol", contiguous, far, {"rtol": 6.0e-5}),
                ("explicit atol", contiguous, far, {"rtol": 0.0, "atol": 2.0e-4}),
                (
                    "signed zero",
                    module.tensor([0.0, -0.0]),
                    module.tensor([-0.0, 0.0]),
                    {},
                ),
                (
                    "same infinities",
                    module.tensor([float("inf"), -float("inf")]),
                    module.tensor([float("inf"), -float("inf")]),
                    {},
                ),
                (
                    "opposite infinities",
                    module.tensor([float("inf")]),
                    module.tensor([-float("inf")]),
                    {},
                ),
                (
                    "finite and infinity",
                    module.tensor([1.0]),
                    module.tensor([float("inf")]),
                    {"rtol": float("inf"), "atol": 0.0},
                ),
                (
                    "nan default",
                    module.tensor([float("nan")]),
                    module.tensor([float("nan")]),
                    {},
                ),
                (
                    "nan equal",
                    module.tensor([float("nan")]),
                    module.tensor([float("nan")]),
                    {"equal_nan": True},
                ),
            )

        for actual_case, expected_case in zip(make_cases(torch), make_cases(reference_torch)):
            case, actual_left, actual_right, kwargs = actual_case
            expected_case_name, expected_left, expected_right, expected_kwargs = expected_case
            with self.subTest(case=case):
                self.assertEqual(case, expected_case_name)
                self.assertEqual(kwargs, expected_kwargs)
                self.assertEqual(actual_left.shape, tuple(expected_left.shape))
                self.assertEqual(actual_left.stride(), tuple(expected_left.stride()))
                self.assertEqual(actual_left.storage_offset(), expected_left.storage_offset())
                self.assert_allclose_calls_match(
                    actual_left,
                    actual_right,
                    expected_left,
                    expected_right,
                    **kwargs,
                )

    def test_tolerance_binding_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor([3.0])
        actual_right = torch.tensor([2.0])
        expected_left = reference_torch.tensor([3.0])
        expected_right = reference_torch.tensor([2.0])

        calls = (
            {"rtol": True, "atol": False},
            {"rtol": np.float32(0.5), "atol": np.float64(0.0)},
        )
        for kwargs in calls:
            with self.subTest(kwargs=kwargs):
                self.assert_allclose_calls_match(
                    actual_left,
                    actual_right,
                    expected_left,
                    expected_right,
                    **kwargs,
                )

        self.assert_allclose_calls_match(
            actual_left,
            actual_right,
            expected_left,
            expected_right,
            rtol=0,
            atol=1.0,
            equal_nan=False,
        )
        self.assert_allclose_calls_match(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            reference_torch.tensor([1.0]),
            reference_torch.tensor([1.0]),
            rtol=0.0,
            atol=0.0,
            equal_nan=False,
        )

        error_cases = (
            (
                lambda: torch.allclose(actual_left, actual_right, rtol=-1.0),
                lambda: reference_torch.allclose(expected_left, expected_right, rtol=-1.0),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, atol=-1.0),
                lambda: reference_torch.allclose(expected_left, expected_right, atol=-1.0),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, rtol=float("nan")),
                lambda: reference_torch.allclose(
                    expected_left, expected_right, rtol=float("nan")
                ),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, atol=float("nan")),
                lambda: reference_torch.allclose(
                    expected_left, expected_right, atol=float("nan")
                ),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, rtol=None),
                lambda: reference_torch.allclose(expected_left, expected_right, rtol=None),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, atol="0"),
                lambda: reference_torch.allclose(expected_left, expected_right, atol="0"),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, equal_nan=1),
                lambda: reference_torch.allclose(
                    expected_left, expected_right, equal_nan=1
                ),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, 0, 0, 1),
                lambda: reference_torch.allclose(expected_left, expected_right, 0, 0, 1),
            ),
            (
                lambda: torch.allclose(),
                lambda: reference_torch.allclose(),
            ),
            (
                lambda: torch.allclose(actual_left),
                lambda: reference_torch.allclose(expected_left),
            ),
            (
                lambda: torch.allclose(actual_left, 1.0),
                lambda: reference_torch.allclose(expected_left, 1.0),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, out=None),
                lambda: reference_torch.allclose(expected_left, expected_right, out=None),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, input=actual_left),
                lambda: reference_torch.allclose(
                    expected_left, expected_right, input=expected_left
                ),
            ),
            (
                lambda: torch.allclose(actual_left, actual_right, other=actual_right),
                lambda: reference_torch.allclose(
                    expected_left, expected_right, other=expected_right
                ),
            ),
        )
        for actual_call, expected_call in error_cases:
            with self.subTest(call=expected_call):
                self.assert_error_matches(actual_call, expected_call)

    def test_metadata_exports_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.allclose
        expected = reference_torch.allclose
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        for callable_object in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(torch.__all__.count("allclose"), reference_torch.__all__.count("allclose"))
        self.assertIs(torch._C._VariableFunctionsClass.allclose, actual)
        self.assertEqual(
            torch._C._VariableFunctionsClass.__module__.replace("torch_rs._C", "torch._C"),
            reference_torch._C._VariableFunctionsClass.__module__,
        )
        self.assertIs(wildcard_namespace["allclose"], actual)
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol=protocol)), actual)


if __name__ == "__main__":
    unittest.main()
