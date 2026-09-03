import copy
import importlib
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


def mm_layout_cases(module):
    offset_left = module.tensor(
        np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist(),
        dtype=module.float32,
    )[1]
    offset_right = module.tensor(
        np.arange(12, dtype=np.float32).reshape(2, 3, 2).tolist(),
        dtype=module.float32,
    )[1]
    strided_left = module.tensor(
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
    ).transpose(0, 1)
    strided_right = module.tensor(
        [[7.0, 9.0, 11.0], [8.0, 10.0, 12.0]], dtype=module.float32
    ).transpose(0, 1)

    return (
        (
            "square",
            module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32),
            module.tensor([[-5.0, 6.0], [7.0, -8.0]], dtype=module.float32),
        ),
        (
            "rectangular",
            module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
            ),
            module.tensor(
                [
                    [7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0],
                ],
                dtype=module.float32,
            ),
        ),
        (
            "empty rows",
            module.zeros((0, 3), dtype=module.float32),
            module.ones((3, 4), dtype=module.float32),
        ),
        (
            "empty columns",
            module.ones((2, 3), dtype=module.float32),
            module.zeros((3, 0), dtype=module.float32),
        ),
        (
            "empty inner",
            module.ones((2, 0), dtype=module.float32),
            module.zeros((0, 3), dtype=module.float32),
        ),
        ("offset", offset_left, offset_right),
        ("noncontiguous", strided_left, strided_right),
        (
            "signed zero",
            module.tensor([[-0.0], [0.0]], dtype=module.float32),
            module.tensor([[1.0, -1.0]], dtype=module.float32),
        ),
        (
            "nan inf",
            module.tensor(
                [
                    [float("inf"), 1.0],
                    [float("-inf"), -1.0],
                    [float("nan"), 2.0],
                ],
                dtype=module.float32,
            ),
            module.tensor([[1.0, -1.0], [0.5, 1.0]], dtype=module.float32),
        ),
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchMmReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.mm differentials require pinned PyTorch 2.13.0")

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

        actual_values = np.asarray(actual)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, classifications=True):
            np.testing.assert_array_equal(np.isnan(actual_values), np.isnan(expected_values))
            non_nan = ~np.isnan(expected_values)
            np.testing.assert_array_equal(
                np.signbit(actual_values[non_nan]), np.signbit(expected_values[non_nan])
            )
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

    def test_rank_two_results_layouts_and_special_values_match_pytorch_2_13(self):
        actual_cases = mm_layout_cases(torch)
        expected_cases = mm_layout_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_left, actual_right = actual_case
            expected_name, expected_left, expected_right = expected_case
            self.assertEqual(case, expected_name)
            calls = (
                (
                    "positional",
                    lambda: torch.mm(actual_left, actual_right),
                    lambda: reference_torch.mm(expected_left, expected_right),
                ),
                (
                    "canonical keywords",
                    lambda: torch.mm(input=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(input=expected_left, mat2=expected_right),
                ),
                (
                    "input alias x",
                    lambda: torch.mm(x=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(x=expected_left, mat2=expected_right),
                ),
                (
                    "input alias a",
                    lambda: torch.mm(a=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(a=expected_left, mat2=expected_right),
                ),
                (
                    "input alias x1",
                    lambda: torch.mm(x1=actual_left, mat2=actual_right),
                    lambda: reference_torch.mm(x1=expected_left, mat2=expected_right),
                ),
                (
                    "out none",
                    lambda: torch.mm(actual_left, actual_right, out=None),
                    lambda: reference_torch.mm(expected_left, expected_right, out=None),
                ),
            )
            for style, actual_call, expected_call in calls:
                self.assert_matches(actual_call(), expected_call(), case=(case, style))

    def test_rank_two_shape_errors_match_pytorch_2_13(self):
        for left_shape, right_shape in (
            ((2, 3), (4, 2)),
            ((0, 3), (4, 0)),
        ):
            actual_left = torch.zeros(left_shape)
            actual_right = torch.zeros(right_shape)
            expected_left = reference_torch.zeros(left_shape)
            expected_right = reference_torch.zeros(right_shape)
            with self.subTest(left=left_shape, right=right_shape):
                with self.assertRaises(Exception) as actual_raised:
                    torch.mm(actual_left, actual_right)
                with self.assertRaises(Exception) as expected_raised:
                    reference_torch.mm(expected_left, expected_right)
                self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
                self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def callable_contract(self, module, package_name):
        function = module.mm
        owner = function.__reduce__()[1][0]
        direct_namespace = {}
        exec(f"from {package_name} import mm", direct_namespace)
        wildcard_namespace = {}
        exec(f"from {package_name} import *", wildcard_namespace)
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
            "direct_import_identity": direct_namespace["mm"] is function,
            "wildcard_identity": wildcard_namespace["mm"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_import_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch, "torch_rs"),
            self.callable_contract(reference_torch, "torch"),
        )

    def test_reload_preserves_top_level_mm_callable(self):
        function = torch.mm
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.mm, function)


if __name__ == "__main__":
    unittest.main()
