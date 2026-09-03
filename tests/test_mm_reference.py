import copy
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

from tests.test_mm import mm_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class MmReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.mm differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

        actual_values = np.asarray(actual, dtype=np.float32)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, classifications=True):
            np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(expected_values))
            finite_or_infinite = ~np.isnan(expected_values)
            np.testing.assert_array_equal(
                np.signbit(actual_values[finite_or_infinite]),
                np.signbit(expected_values[finite_or_infinite]),
            )
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    @staticmethod
    def call_mm(module, left, right, form):
        if form == "positional":
            return module.mm(left, right)
        if form == "canonical keywords":
            return module.mm(input=left, mat2=right)
        if form == "input alias x":
            return module.mm(x=left, mat2=right)
        if form == "input alias a":
            return module.mm(a=left, mat2=right)
        if form == "input alias x1":
            return module.mm(x1=left, mat2=right)
        if form == "out none":
            return module.mm(left, right, out=None)
        raise AssertionError(f"unknown torch.mm call form: {form}")

    def test_rank_two_values_layouts_and_edge_cases_match_pytorch_2_13(self):
        actual_cases = mm_cases(torch)
        expected_cases = mm_cases(reference_torch)
        forms = (
            "positional",
            "canonical keywords",
            "input alias x",
            "input alias a",
            "input alias x1",
            "out none",
        )
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_matches(
                    self.call_mm(torch, actual_left, actual_right, form),
                    self.call_mm(reference_torch, expected_left, expected_right, form),
                    case=(case, form),
                )

    def test_supported_binding_and_rank_errors_match_pytorch_2_13(self):
        actual = torch.tensor([[1.0]])
        expected = reference_torch.tensor([[1.0]], dtype=reference_torch.float32)
        for case, (actual_call, expected_call) in enumerate(
            (
                (lambda: torch.mm(), lambda: reference_torch.mm()),
                (
                    lambda: torch.mm(actual, actual, actual),
                    lambda: reference_torch.mm(expected, expected, expected),
                ),
                (lambda: torch.mm([], actual), lambda: reference_torch.mm([], expected)),
                (lambda: torch.mm(actual, []), lambda: reference_torch.mm(expected, [])),
                (
                    lambda: torch.mm(input=None, mat2=actual),
                    lambda: reference_torch.mm(input=None, mat2=expected),
                ),
                (
                    lambda: torch.mm(input=actual, mat2=None),
                    lambda: reference_torch.mm(input=expected, mat2=None),
                ),
                (
                    lambda: torch.mm(x1=actual, x2=actual),
                    lambda: reference_torch.mm(x1=expected, x2=expected),
                ),
                (
                    lambda: torch.mm(torch.ones((2,)), torch.ones((2, 2))),
                    lambda: reference_torch.mm(
                        reference_torch.ones((2,)),
                        reference_torch.ones((2, 2)),
                    ),
                ),
                (
                    lambda: torch.mm(torch.ones((2, 2)), torch.ones((2,))),
                    lambda: reference_torch.mm(
                        reference_torch.ones((2, 2)),
                        reference_torch.ones((2,)),
                    ),
                ),
                (
                    lambda: torch.mm(torch.ones((1, 2, 2)), torch.ones((2, 2))),
                    lambda: reference_torch.mm(
                        reference_torch.ones((1, 2, 2)),
                        reference_torch.ones((2, 2)),
                    ),
                ),
                (
                    lambda: torch.mm(torch.zeros((2, 3)), torch.zeros((4, 2))),
                    lambda: reference_torch.mm(
                        reference_torch.zeros((2, 3)),
                        reference_torch.zeros((4, 2)),
                    ),
                ),
            )
        ):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    @staticmethod
    def callable_contract(module):
        function = module.mm
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.mm is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("mm"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["mm"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )


if __name__ == "__main__":
    unittest.main()
