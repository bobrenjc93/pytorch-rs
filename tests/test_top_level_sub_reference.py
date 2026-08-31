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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelSubtractionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.sub differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
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
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_supported_values_layouts_empty_and_defaults_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])

        calls = (
            (
                "sub tensors",
                lambda: torch.sub(actual_left, actual_right),
                lambda: reference_torch.sub(expected_left, expected_right),
            ),
            (
                "sub keywords",
                lambda: torch.sub(input=actual_left, other=actual_right),
                lambda: reference_torch.sub(
                    input=expected_left, other=expected_right
                ),
            ),
            (
                "sub aliases",
                lambda: torch.sub(x1=actual_left, x2=actual_right, alpha=1),
                lambda: reference_torch.sub(
                    x1=expected_left, x2=expected_right, alpha=1
                ),
            ),
            (
                "sub out none",
                lambda: torch.sub(actual_left, actual_right, out=None),
                lambda: reference_torch.sub(expected_left, expected_right, out=None),
            ),
            (
                "subtract tensors",
                lambda: torch.subtract(actual_left, actual_right),
                lambda: reference_torch.subtract(expected_left, expected_right),
            ),
            (
                "subtract alpha one",
                lambda: torch.subtract(actual_left, actual_right, alpha=np.float32(1.0)),
                lambda: reference_torch.subtract(
                    expected_left, expected_right, alpha=np.float32(1.0)
                ),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_offset = actual_left[1]
        expected_offset = expected_left[1]
        for scalar in (-2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            self.assert_matches(
                torch.sub(actual_offset, scalar),
                reference_torch.sub(expected_offset, scalar),
                case=("tensor/scalar", repr(scalar)),
            )
            self.assert_matches(
                torch.subtract(scalar, actual_offset),
                reference_torch.subtract(scalar, expected_offset),
                case=("scalar/tensor", repr(scalar)),
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
            torch.sub(torch.tensor(values), -0.0),
            reference_torch.sub(reference_torch.tensor(values), -0.0),
            case="signed zero and non-finites",
        )
        self.assert_matches(
            torch.subtract(-0.0, torch.tensor(values)),
            reference_torch.subtract(-0.0, reference_torch.tensor(values)),
            case="reflected signed zero and non-finites",
        )

    def test_autograd_and_no_grad_match_pytorch_2_13_for_supported_forms(self):
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )
        actual_output = torch.sub(
            actual_left.transpose(0, 1), actual_right.transpose(0, 1)
        )
        expected_output = reference_torch.sub(
            expected_left.transpose(0, 1), expected_right.transpose(0, 1)
        )
        self.assert_matches(actual_output, expected_output, case="tracked views")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_scalar = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_scalar = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.subtract(4.0, actual_scalar).sum().backward()
        reference_torch.subtract(4.0, expected_scalar).sum().backward()
        self.assert_matches(
            actual_scalar.grad, expected_scalar.grad, case="scalar-first gradient"
        )

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.sub(2.0, actual_no_grad.transpose(0, 1))
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sub(
                2.0, expected_no_grad.transpose(0, 1)
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        marker = object()
        observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("sub", "subtract"):
            function = getattr(module, name)
            calls = (
                (lambda function=function: function(left, right), None),
                (lambda function=function: function(left, 4.0), None),
                (lambda function=function: function(4.0, left), None),
                (lambda function=function: function(left, True), None),
                (lambda function=function: function(False, left), None),
                (
                    lambda function=function: function(
                        input=left, other=right, alpha=1, out=None
                    ),
                    ("input", "other", "alpha", "out"),
                ),
                (
                    lambda function=function: function(left, right, alpha=True),
                    ("alpha",),
                ),
            )
            for call, keywords in calls:
                mode = RecordingMode()
                with mode:
                    result = call()
                func, dispatch_types, args, kwargs = mode.calls[0]
                observations.append(
                    (
                        name,
                        result is marker,
                        func is function,
                        dispatch_types == (),
                        len(args),
                        kwargs is None,
                        kwargs is not None and tuple(kwargs) == keywords,
                    )
                )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for name in ("sub", "subtract"):
            function = getattr(module, name)
            for call in (
                lambda value, function=function: function(value, right),
                lambda value, function=function: function(left, value),
                lambda value, function=function: function(left, right, alpha=value),
                lambda value, function=function: function(left, right, out=value),
            ):
                value = Override()
                Override.calls.clear()
                result = call(value)
                func, dispatch_types, args, kwargs = Override.calls[0]
                override_observations.append(
                    (
                        name,
                        result is marker,
                        func is function,
                        dispatch_types == (Override,),
                        len(args),
                        kwargs is None,
                        kwargs is not None and tuple(kwargs),
                    )
                )

        invalid_observations = []
        for function in (module.sub, module.subtract):
            for call in (
                lambda function=function: function([], right),
                lambda function=function: function(left, []),
                lambda function=function: function(left, right, alpha=[]),
                lambda function=function: function(left, right, out=[]),
            ):
                invalid_mode = RecordingMode()
                try:
                    with invalid_mode:
                        call()
                except Exception as error:
                    invalid_observations.append(
                        (type(error).__name__, len(invalid_mode.calls))
                    )

        return observations, override_observations, invalid_observations

    def test_torch_function_dispatch_subset_matches_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        observed = {}
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        for name in ("sub", "subtract"):
            function = getattr(module, name)
            owner = function.__reduce__()[1][0]
            try:
                inspect.signature(function)
            except Exception as error:
                signature_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                )
            else:
                signature_error = None
            observed[name] = {
                "type": type(function).__name__,
                "is_builtin": type(function) is types.BuiltinFunctionType,
                "name": function.__name__,
                "qualname": function.__qualname__,
                "module": function.__module__,
                "doc": function.__doc__,
                "text_signature": function.__text_signature__,
                "owner_name": owner.__name__,
                "owner_qualname": owner.__qualname__,
                "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
                "owner_path_identity": owner is module._C._VariableFunctionsClass,
                "owner_callable_identity": getattr(owner, name) is function,
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "all_count": module.__all__.count(name),
                "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
                "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
                "wildcard_identity": wildcard_namespace[name] is function,
                "pickle_identities": tuple(
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
                "signature_error": signature_error,
            }
        observed["distinct_aliases"] = module.sub is not module.subtract
        return observed

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )
        self.assertIs(importlib.reload(torch).sub, torch.sub)

    def test_pytorch_matching_errors_for_supported_binding_boundaries(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.sub(), lambda: reference_torch.sub()),
            (lambda: torch.sub(actual), lambda: reference_torch.sub(expected)),
            (
                lambda: torch.sub(actual, actual, actual),
                lambda: reference_torch.sub(expected, expected, expected),
            ),
            (
                lambda: torch.sub([], actual),
                lambda: reference_torch.sub([], expected),
            ),
            (
                lambda: torch.sub(actual, []),
                lambda: reference_torch.sub(expected, []),
            ),
            (
                lambda: torch.sub(actual, actual, input=actual),
                lambda: reference_torch.sub(expected, expected, input=expected),
            ),
            (
                lambda: torch.sub(actual, actual, alpha=[]),
                lambda: reference_torch.sub(expected, expected, alpha=[]),
            ),
            (
                lambda: torch.sub(actual, True),
                lambda: reference_torch.sub(expected, True),
            ),
            (
                lambda: torch.sub(False, actual),
                lambda: reference_torch.sub(False, expected),
            ),
            (
                lambda: torch.sub(actual, np.uint64(2**63)),
                lambda: reference_torch.sub(expected, np.uint64(2**63)),
            ),
            (
                lambda: torch.sub(actual, 2**64),
                lambda: reference_torch.sub(expected, 2**64),
            ),
            (
                lambda: torch.sub(-(2**63) - 1, actual),
                lambda: reference_torch.sub(-(2**63) - 1, expected),
            ),
            (
                lambda: torch.sub(torch.zeros((2, 3)), torch.zeros((4, 2))),
                lambda: reference_torch.sub(
                    reference_torch.zeros((2, 3)),
                    reference_torch.zeros((4, 2)),
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
