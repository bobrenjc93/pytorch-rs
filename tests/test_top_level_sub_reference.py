import copy
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
class TopLevelSubReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.sub differentials require pinned PyTorch 2.13.0")

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
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_layouts_ieee_empties_and_argument_forms_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])

        for name in ("sub", "subtract"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            for case, actual_call, expected_call in (
                (
                    "positional tensors",
                    lambda: actual_function(actual_left, actual_right),
                    lambda: expected_function(expected_left, expected_right),
                ),
                (
                    "canonical keywords",
                    lambda: actual_function(input=actual_left, other=actual_right),
                    lambda: expected_function(input=expected_left, other=expected_right),
                ),
                (
                    "aliases",
                    lambda: actual_function(x1=actual_left, x2=actual_right),
                    lambda: expected_function(x1=expected_left, x2=expected_right),
                ),
                (
                    "out none",
                    lambda: actual_function(actual_left, actual_right, out=None),
                    lambda: expected_function(expected_left, expected_right, out=None),
                ),
                (
                    "alpha int one",
                    lambda: actual_function(actual_left, actual_right, alpha=1),
                    lambda: expected_function(expected_left, expected_right, alpha=1),
                ),
                (
                    "alpha numpy float one",
                    lambda: actual_function(
                        actual_left, actual_right, alpha=np.float32(1.0)
                    ),
                    lambda: expected_function(
                        expected_left, expected_right, alpha=np.float32(1.0)
                    ),
                ),
                (
                    "tensor/scalar",
                    lambda: actual_function(actual_left[1], np.float32(-0.0)),
                    lambda: expected_function(expected_left[1], np.float32(-0.0)),
                ),
                (
                    "scalar/tensor",
                    lambda: actual_function(np.int64(3), actual_left[1]),
                    lambda: expected_function(np.int64(3), expected_left[1]),
                ),
            ):
                self.assert_matches(
                    actual_call(), expected_call(), case=(name, case)
                )

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.sub(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.sub(
                expected_empty, reference_torch.ones((1, 1, 2))
            ),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.sub(-0.0, torch.tensor(values)),
            reference_torch.sub(-0.0, reference_torch.tensor(values)),
            case="signed zero and non-finites",
        )

    def test_scalar_autograd_and_no_grad_match_pytorch_2_13_for_default_alpha(self):
        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.subtract(actual_scalar, 4.0).sum().backward()
        reference_torch.subtract(expected_scalar, 4.0).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_scalar.grad), expected_scalar.grad.numpy()
        )

        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.subtract(4.0, actual_scalar).sum().backward()
        reference_torch.subtract(4.0, expected_scalar).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_scalar.grad), expected_scalar.grad.numpy()
        )

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.sub(2.0, actual_no_grad)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sub(2.0, expected_no_grad)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad")
        self.assertTrue(torch.sub(actual_no_grad, 2.0).requires_grad)
        self.assertTrue(reference_torch.sub(expected_no_grad, 2.0).requires_grad)

    def test_pytorch_matching_errors_for_supported_schema_boundaries(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.sub([], actual), lambda: reference_torch.sub([], expected)),
            (lambda: torch.sub(actual, []), lambda: reference_torch.sub(expected, [])),
            (
                lambda: torch.sub(input=None, other=actual),
                lambda: reference_torch.sub(input=None, other=expected),
            ),
            (
                lambda: torch.sub(actual, actual, x2=actual),
                lambda: reference_torch.sub(expected, expected, x2=expected),
            ),
            (
                lambda: torch.sub(actual, actual, dtype=torch.float32),
                lambda: reference_torch.sub(
                    expected, expected, dtype=reference_torch.float32
                ),
            ),
            (
                lambda: torch.sub(actual, np.uint64(2**63)),
                lambda: reference_torch.sub(expected, np.uint64(2**63)),
            ),
            (
                lambda: torch.sub(actual, True),
                lambda: reference_torch.sub(expected, True),
            ),
            (
                lambda: torch.sub(True, actual),
                lambda: reference_torch.sub(True, expected),
            ),
            (
                lambda: torch.sub(actual, actual, alpha=True),
                lambda: reference_torch.sub(expected, expected, alpha=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module, name):
        function = getattr(module, name)
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
            "owner_callable_identity": getattr(owner, name) is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count(name),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace[name] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        for name in ("sub", "subtract"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

    def test_deliberately_unsupported_out_alpha_and_scalar_only_are_rejected(self):
        tensor = torch.tensor([1.0])
        destination = torch.tensor([17.0])
        for function in (torch.sub, torch.subtract):
            with self.subTest(function=function.__name__, option="out"):
                with self.assertRaisesRegex(RuntimeError, "out.*not supported"):
                    function(tensor, tensor, out=destination)
                self.assertEqual(destination.tolist(), [17.0])
            with self.subTest(function=function.__name__, option="alpha"):
                with self.assertRaisesRegex(NotImplementedError, "alpha=1"):
                    function(tensor, tensor, alpha=2)
            with self.subTest(function=function.__name__, option="scalar"):
                with self.assertRaisesRegex(TypeError, "scalar-scalar subtraction"):
                    function(2, 3)


if __name__ == "__main__":
    unittest.main()
