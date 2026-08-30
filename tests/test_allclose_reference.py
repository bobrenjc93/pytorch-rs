import copy
import importlib
import inspect
import pickle
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AllCloseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.allclose differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_result_matches(self, actual_call, expected_call, *, case):
        with self.subTest(case=case):
            actual = actual_call()
            expected = expected_call()
            self.assertIs(type(actual), bool)
            self.assertEqual(actual, expected)

    def test_values_broadcast_views_empty_and_specials_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [
                [[99.0, 98.0, 97.0], [96.0, 95.0, 94.0]],
                [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            ]
        )[1].transpose(0, 1)
        expected_left = reference_torch.tensor(
            [
                [[99.0, 98.0, 97.0], [96.0, 95.0, 94.0]],
                [[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]],
            ],
            dtype=reference_torch.float32,
        )[1].transpose(0, 1)
        actual_right = torch.tensor([[1.0], [4.0], [7.0]])
        expected_right = reference_torch.tensor(
            [[1.0], [4.0], [7.0]],
            dtype=reference_torch.float32,
        )
        actual_extreme = torch.tensor([3.4028235e38])
        actual_opposite_extreme = torch.tensor([-1.0e38])
        expected_extreme = reference_torch.tensor(
            [3.4028235e38],
            dtype=reference_torch.float32,
        )
        expected_opposite_extreme = reference_torch.tensor(
            [-1.0e38],
            dtype=reference_torch.float32,
        )

        cases = (
            (
                "scalar true",
                lambda: torch.allclose(torch.tensor(1.0), torch.tensor(1.0)),
                lambda: reference_torch.allclose(
                    reference_torch.tensor(1.0), reference_torch.tensor(1.0)
                ),
            ),
            (
                "empty broadcast true",
                lambda: torch.allclose(torch.zeros((2, 0, 3)), torch.ones((1, 0, 1))),
                lambda: reference_torch.allclose(
                    reference_torch.zeros((2, 0, 3)),
                    reference_torch.ones((1, 0, 1)),
                ),
            ),
            (
                "offset noncontiguous broadcast false",
                lambda: torch.allclose(actual_left, actual_right),
                lambda: reference_torch.allclose(expected_left, expected_right),
            ),
            (
                "default example false",
                lambda: torch.allclose(
                    torch.tensor([10000.0, 1e-7]),
                    torch.tensor([10000.1, 1e-8]),
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([10000.0, 1e-7]),
                    reference_torch.tensor([10000.1, 1e-8]),
                ),
            ),
            (
                "default example true",
                lambda: torch.allclose(
                    torch.tensor([10000.0, 1e-8]),
                    torch.tensor([10000.1, 1e-9]),
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([10000.0, 1e-8]),
                    reference_torch.tensor([10000.1, 1e-9]),
                ),
            ),
            (
                "equal nan false",
                lambda: torch.allclose(
                    torch.tensor([1.0, float("nan")]),
                    torch.tensor([1.0, float("nan")]),
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([1.0, float("nan")]),
                    reference_torch.tensor([1.0, float("nan")]),
                ),
            ),
            (
                "equal nan true",
                lambda: torch.allclose(
                    torch.tensor([1.0, float("nan")]),
                    torch.tensor([1.0, float("nan")]),
                    equal_nan=True,
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([1.0, float("nan")]),
                    reference_torch.tensor([1.0, float("nan")]),
                    equal_nan=True,
                ),
            ),
            (
                "same signed infinities",
                lambda: torch.allclose(
                    torch.tensor([float("inf"), -float("inf")]),
                    torch.tensor([float("inf"), -float("inf")]),
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([float("inf"), -float("inf")]),
                    reference_torch.tensor([float("inf"), -float("inf")]),
                ),
            ),
            (
                "finite and infinity stay false under infinite tolerance",
                lambda: torch.allclose(
                    torch.tensor([1.0]), torch.tensor([float("inf")]), rtol=float("inf")
                ),
                lambda: reference_torch.allclose(
                    reference_torch.tensor([1.0]),
                    reference_torch.tensor([float("inf")]),
                    rtol=float("inf"),
                ),
            ),
            (
                "finite difference overflow stays false under infinite rtol",
                lambda: torch.allclose(
                    actual_extreme,
                    actual_opposite_extreme,
                    rtol=float("inf"),
                    atol=0.0,
                ),
                lambda: reference_torch.allclose(
                    expected_extreme,
                    expected_opposite_extreme,
                    rtol=float("inf"),
                    atol=0.0,
                ),
            ),
            (
                "finite difference overflow stays false under overflowed rtol",
                lambda: torch.allclose(
                    actual_extreme,
                    actual_opposite_extreme,
                    rtol=1.0e38,
                    atol=0.0,
                ),
                lambda: reference_torch.allclose(
                    expected_extreme,
                    expected_opposite_extreme,
                    rtol=1.0e38,
                    atol=0.0,
                ),
            ),
            (
                "finite difference overflow stays false under infinite atol",
                lambda: torch.allclose(
                    actual_extreme,
                    actual_opposite_extreme,
                    rtol=0.0,
                    atol=float("inf"),
                ),
                lambda: reference_torch.allclose(
                    expected_extreme,
                    expected_opposite_extreme,
                    rtol=0.0,
                    atol=float("inf"),
                ),
            ),
        )
        for case, actual_call, expected_call in cases:
            self.assert_result_matches(actual_call, expected_call, case=case)

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])

        calls = (
            (
                "input other",
                lambda: torch.allclose(input=actual, other=actual),
                lambda: reference_torch.allclose(input=expected, other=expected),
            ),
            (
                "x1 x2",
                lambda: torch.allclose(x1=actual, x2=actual),
                lambda: reference_torch.allclose(x1=expected, x2=expected),
            ),
            (
                "scalar tensor tolerances",
                lambda: torch.allclose(
                    actual,
                    actual,
                    rtol=torch.tensor(1e-5),
                    atol=torch.tensor(0.0),
                ),
                lambda: reference_torch.allclose(
                    expected,
                    expected,
                    rtol=reference_torch.tensor(1e-5),
                    atol=reference_torch.tensor(0.0),
                ),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_result_matches(actual_call, expected_call, case=case)

        error_cases = (
            (lambda: torch.allclose(), lambda: reference_torch.allclose()),
            (lambda: torch.allclose(actual), lambda: reference_torch.allclose(expected)),
            (lambda: torch.allclose(None, actual), lambda: reference_torch.allclose(None, expected)),
            (
                lambda: torch.allclose(actual, 1.0),
                lambda: reference_torch.allclose(expected, 1.0),
            ),
            (
                lambda: torch.allclose(actual, actual, 1, 2, False, 0),
                lambda: reference_torch.allclose(expected, expected, 1, 2, False, 0),
            ),
            (
                lambda: torch.allclose(actual, actual, out=None),
                lambda: reference_torch.allclose(expected, expected, out=None),
            ),
            (
                lambda: torch.allclose(a=actual, other=actual, extra=True),
                lambda: reference_torch.allclose(a=expected, other=expected, extra=True),
            ),
            (
                lambda: torch.allclose(input=actual, x2=actual, other=actual),
                lambda: reference_torch.allclose(input=expected, x2=expected, other=expected),
            ),
            (
                lambda: torch.allclose(a=actual, input=actual, other=actual),
                lambda: reference_torch.allclose(a=expected, input=expected, other=expected),
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
                lambda: torch.allclose(actual, actual, rtol=-1.0),
                lambda: reference_torch.allclose(expected, expected, rtol=-1.0),
            ),
            (
                lambda: torch.allclose(actual, actual, atol=float("nan")),
                lambda: reference_torch.allclose(expected, expected, atol=float("nan")),
            ),
            (
                lambda: torch.allclose(torch.zeros((2, 3)), torch.zeros((4,))),
                lambda: reference_torch.allclose(
                    reference_torch.zeros((2, 3)),
                    reference_torch.zeros((4,)),
                ),
            ),
        )
        for actual_call, expected_call in error_cases:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(actual_call, expected_call)

    def test_callable_metadata_exports_copy_pickle_and_reload_match_pytorch_2_13(self):
        actual = torch.allclose
        expected = reference_torch.allclose

        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertIs(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(torch.__all__.count("allclose"), reference_torch.__all__.count("allclose"))
        self.assertNotEqual(hasattr(torch, "isclose"), hasattr(reference_torch, "isclose"))

        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol=protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol=protocol)), expected)

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs import *", actual_wildcard)
        exec("from torch import *", expected_wildcard)
        self.assertEqual("allclose" in actual_wildcard, "allclose" in expected_wildcard)
        self.assertIs(actual_wildcard["allclose"], actual)
        self.assertIs(expected_wildcard["allclose"], expected)

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.allclose, actual)


if __name__ == "__main__":
    unittest.main()
