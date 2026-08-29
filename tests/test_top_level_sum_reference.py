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
class TopLevelSumReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.sum differentials require pinned PyTorch 2.13.0")

    def assert_scalar_matches(self, actual, expected, actual_input, expected_input, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertIsNot(actual, actual_input)
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertIsNot(expected, expected_input)
            self.assertFalse(expected.is_set_to(expected_input))
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(str(expected.device), "cpu")
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.asarray(actual).reshape(-1).view(np.uint32).item(),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32).item(),
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    @staticmethod
    def make_cases(module):
        dense = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = dense.transpose(0, 2)
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("negative zero", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_sum(module, source, form):
        if form == "positional":
            return module.sum(source)
        if form == "input":
            return module.sum(input=source)
        if form == "x":
            return module.sum(x=source)
        if form == "a":
            return module.sum(a=source)
        if form == "x1":
            return module.sum(x1=source)
        if form == "dtype none":
            return module.sum(source, dtype=None)
        if form == "dtype float32":
            return module.sum(input=source, dtype=module.float32)
        if form == "dtype float alias":
            return module.sum(x=source, dtype=module.float)
        raise AssertionError(f"unknown sum form: {form}")

    def test_default_form_values_metadata_storage_and_layout_match_pytorch_2_13(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "dtype none",
            "dtype float32",
            "dtype float alias",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            name, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(name, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_sum(torch, actual_input, form),
                    self.call_sum(reference_torch, expected_input, form),
                    actual_input,
                    expected_input,
                    case=(name, form),
                )

    def test_autograd_empty_offsets_and_no_grad_match_pytorch_2_13(self):
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = torch.sum(actual_leaf.transpose(0, 1), dtype=torch.float32)
        expected_loss = reference_torch.sum(
            expected_leaf.transpose(0, 1), dtype=reference_torch.float32
        )
        self.assert_scalar_matches(
            actual_loss, expected_loss, actual_leaf, expected_leaf, case="tracked"
        )
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        torch.sum(actual_empty.transpose(0, 2), dtype=None).backward()
        reference_torch.sum(expected_empty.transpose(0, 2), dtype=None).backward()
        self.assertEqual(actual_empty.grad.shape, tuple(expected_empty.grad.shape))
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = torch.sum(actual_leaf, dtype=torch.float)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sum(
                expected_leaf, dtype=reference_torch.float
            )
        self.assert_scalar_matches(
            actual_untracked,
            expected_untracked,
            actual_leaf,
            expected_leaf,
            case="no_grad",
        )

    def test_callable_metadata_documentation_exports_and_pickling_match_pytorch_2_13(self):
        actual = torch.sum
        expected = reference_torch.sum
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)

        actual_owner = actual.__reduce__()[1][0]
        expected_owner = expected.__reduce__()[1][0]
        self.assertEqual(actual_owner.__name__, expected_owner.__name__)
        self.assertEqual(actual_owner.__qualname__, expected_owner.__qualname__)
        self.assertIs(actual_owner, torch._C._VariableFunctionsClass)
        self.assertIs(actual_owner.sum, actual)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(actual, protocol=protocol)), actual
                )

        self.assertEqual(torch.__all__.count("sum"), reference_torch.__all__.count("sum"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sum"], actual)

    def test_unsupported_reduction_forms_and_cross_dtype_requests_stay_outside_surface(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        destination = torch.tensor([17.0, 19.0, 23.0])
        expected_destination = reference_torch.tensor(
            [17.0, 19.0, 23.0], dtype=reference_torch.float32
        )
        unsupported = (
            (lambda: torch.sum(actual, 0), lambda: reference_torch.sum(expected, 0)),
            (
                lambda: torch.sum(actual, None),
                lambda: reference_torch.sum(expected, None),
            ),
            (
                lambda: torch.sum(actual, dim=0),
                lambda: reference_torch.sum(expected, dim=0),
            ),
            (
                lambda: torch.sum(actual, dim=None),
                lambda: reference_torch.sum(expected, dim=None),
            ),
            (
                lambda: torch.sum(actual, 0, False),
                lambda: reference_torch.sum(expected, 0, False),
            ),
            (
                lambda: torch.sum(actual, dim=0, keepdim=True),
                lambda: reference_torch.sum(expected, dim=0, keepdim=True),
            ),
            (
                lambda: torch.sum(actual, 0, out=destination),
                lambda: reference_torch.sum(expected, 0, out=expected_destination),
            ),
            (
                lambda: torch.sum(actual, dtype=reference_torch.float64),
                lambda: reference_torch.sum(expected, dtype=reference_torch.float64),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(unsupported):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    actual_call()
                expected_call()
                self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])

    def test_non_tensor_input_boundaries_match_pytorch_2_13(self):
        cases = (
            (lambda: torch.sum(1), lambda: reference_torch.sum(1)),
            (lambda: torch.sum(input=[]), lambda: reference_torch.sum(input=[])),
            (
                lambda: torch.sum(1, dtype=None),
                lambda: reference_torch.sum(1, dtype=None),
            ),
            (
                lambda: torch.sum(input=[], dim=0),
                lambda: reference_torch.sum(input=[], dim=0),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
