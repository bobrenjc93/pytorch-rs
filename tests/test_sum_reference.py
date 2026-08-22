import inspect
import pickle
import re
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

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_scalar_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, value=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.detach().cpu().numpy(),
                rtol=2.0e-5,
                atol=2.0e-5,
                equal_nan=True,
            )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = base.transpose(0, 2)
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("negative zero", module.tensor(-0.0, dtype=module.float32)),
            ("empty", module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_sum(module, source, form):
        if form == "positional":
            return module.sum(source)
        if form == "input":
            return module.sum(input=source)
        if form in {"x", "a", "x1"}:
            return module.sum(**{form: source})
        if form == "dtype none":
            return module.sum(source, dtype=None)
        if form == "dtype float32":
            return module.sum(input=source, dtype=module.float32)
        raise AssertionError(f"unknown call form: {form}")

    def test_values_scalar_metadata_empty_and_noncontiguous_match_pytorch_2_13(self):
        forms = ("positional", "input", "x", "a", "x1", "dtype none", "dtype float32")
        for actual_case, expected_case in zip(
            self.make_cases(torch), self.make_cases(reference_torch), strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_sum(torch, actual_input, form),
                    self.call_sum(reference_torch, expected_input, form),
                    case=(case, form),
                )

    def test_autograd_repeated_backward_empty_and_no_grad_match_pytorch_2_13(self):
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = torch.sum(actual_leaf.transpose(0, 1), dtype=torch.float32)
        expected_loss = reference_torch.sum(
            expected_leaf.transpose(0, 1), dtype=reference_torch.float32
        )
        self.assert_scalar_matches(actual_loss, expected_loss, case="tracked")
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
        torch.sum(actual_empty.transpose(0, 2)).backward()
        reference_torch.sum(expected_empty.transpose(0, 2)).backward()
        self.assertEqual(actual_empty.grad.shape, tuple(expected_empty.grad.shape))
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = torch.sum(actual_leaf)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sum(expected_leaf)
        self.assert_scalar_matches(actual_untracked, expected_untracked, case="no_grad")

    @staticmethod
    def callable_contract(module):
        function = module.sum
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
            "owner_callable_identity": owner.sum is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("sum"),
            "wildcard_identity": wildcard_namespace["sum"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_ownership_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    @staticmethod
    def mode_observation(module):
        tensor = module.tensor([1.0, 2.0], dtype=module.float32, requires_grad=True)
        marker = object()
        observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                observations.append(
                    (
                        func is module.sum,
                        types == (),
                        len(args),
                        kwargs is None,
                        None if kwargs is None else tuple(kwargs),
                    )
                )
                return marker

        calls = (
            lambda: module.sum(tensor),
            lambda: module.sum(input=tensor),
            lambda: module.sum(x=tensor, dtype=module.float32),
            lambda: module.sum(tensor, dtype=None),
        )
        for call in calls:
            with RecordingMode():
                if call() is not marker:
                    raise AssertionError("TorchFunctionMode result was not returned")
        return observations

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_observation(torch), self.mode_observation(reference_torch)
        )

    def test_supported_form_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        cases = (
            (lambda: torch.sum(), lambda: reference_torch.sum()),
            (lambda: torch.sum(1), lambda: reference_torch.sum(1)),
            (lambda: torch.sum(input=[]), lambda: reference_torch.sum(input=[])),
            (lambda: torch.sum(dtype=None), lambda: reference_torch.sum(dtype=None)),
            (
                lambda: torch.sum(actual, input=actual),
                lambda: reference_torch.sum(expected, input=expected),
            ),
            (
                lambda: torch.sum(actual, dtype=1),
                lambda: reference_torch.sum(expected, dtype=1),
            ),
            (
                lambda: torch.sum(actual, 0, False, torch.float32),
                lambda: reference_torch.sum(
                    expected, 0, False, reference_torch.float32
                ),
            ),
        )
        for actual_call, expected_call in cases:
            self.assert_error_matches(actual_call, expected_call)

    def test_dimension_keepdim_out_and_other_dtypes_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        cases = (
            (lambda: torch.sum(actual, 0), lambda: reference_torch.sum(expected, 0)),
            (
                lambda: torch.sum(actual, dim=0),
                lambda: reference_torch.sum(expected, dim=0),
            ),
            (
                lambda: torch.sum(actual, keepdim=True),
                lambda: reference_torch.sum(expected, dim=None, keepdim=True),
            ),
            (
                lambda: torch.sum(actual, out=None),
                lambda: reference_torch.sum(expected, dim=None, out=None),
            ),
            (
                lambda: torch.sum(actual, dtype=reference_torch.float64),
                lambda: reference_torch.sum(expected, dtype=reference_torch.float64),
            ),
        )
        for actual_call, expected_call in cases:
            with self.assertRaises(TypeError):
                actual_call()
            expected_call()


if __name__ == "__main__":
    unittest.main()
