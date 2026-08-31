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
class TopLevelAddReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.add differentials require pinned PyTorch 2.13.0")

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
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    @staticmethod
    def make_binary_case(module):
        left = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        ).transpose(0, 2)
        right = module.tensor([[[10.0], [20.0], [30.0]]], dtype=module.float32)
        return left, right

    def test_supported_values_metadata_and_defaults_match_pytorch_2_13(self):
        actual_left, actual_right = self.make_binary_case(torch)
        expected_left, expected_right = self.make_binary_case(reference_torch)
        calls = (
            (
                "positional tensors",
                lambda: torch.add(actual_left, actual_right),
                lambda: reference_torch.add(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.add(input=actual_left, other=actual_right),
                lambda: reference_torch.add(input=expected_left, other=expected_right),
            ),
            (
                "legacy aliases",
                lambda: torch.add(x1=actual_left, x2=actual_right),
                lambda: reference_torch.add(x1=expected_left, x2=expected_right),
            ),
            (
                "alpha integer default",
                lambda: torch.add(actual_left, actual_right, alpha=1),
                lambda: reference_torch.add(expected_left, expected_right, alpha=1),
            ),
            (
                "alpha float default",
                lambda: torch.add(actual_left, actual_right, alpha=np.float32(1.0)),
                lambda: reference_torch.add(
                    expected_left, expected_right, alpha=np.float32(1.0)
                ),
            ),
            (
                "out none",
                lambda: torch.add(actual_left, actual_right, out=None),
                lambda: reference_torch.add(expected_left, expected_right, out=None),
            ),
            (
                "tensor/scalar",
                lambda: torch.add(actual_left[1], -2.5),
                lambda: reference_torch.add(expected_left[1], -2.5),
            ),
            (
                "scalar/tensor",
                lambda: torch.add(np.int64(3), actual_left[1]),
                lambda: reference_torch.add(np.int64(3), expected_left[1]),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32
        ).transpose(0, 2)
        self.assert_matches(
            torch.add(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.add(
                expected_empty,
                reference_torch.ones((1, 1, 2), dtype=reference_torch.float32),
            ),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        actual_special = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected_special = reference_torch.tensor(
            special_bits.view(np.float32), dtype=reference_torch.float32
        )
        self.assert_matches(
            torch.add(-0.0, actual_special),
            reference_torch.add(-0.0, expected_special),
            case="signed zero and non-finites",
        )
        nan_scalar = np.asarray([0x7FA1_2345], dtype=np.uint32).view(np.float32)[0]
        nan_tensor_bits = np.asarray(
            [0x7FC5_4321, 0x0000_0000, 0x8000_0000], dtype=np.uint32
        )
        actual_nan_tensor = torch.tensor(memoryview(nan_tensor_bits.view(np.float32)))
        expected_nan_tensor = reference_torch.tensor(
            nan_tensor_bits.view(np.float32), dtype=reference_torch.float32
        )
        self.assert_matches(
            torch.add(nan_scalar, actual_nan_tensor),
            reference_torch.add(nan_scalar, expected_nan_tensor),
            case="scalar/tensor NaN payloads",
        )
        self.assert_matches(
            torch.add(actual_nan_tensor, nan_scalar),
            reference_torch.add(expected_nan_tensor, nan_scalar),
            case="tensor/scalar NaN payloads",
        )

    def test_supported_scalar_autograd_and_no_grad_match_pytorch_2_13(self):
        actual_leaf = torch.tensor(
            [[2.0, -3.0], [5.0, -7.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[2.0, -3.0], [5.0, -7.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        torch.add(1.5, actual_leaf.transpose(0, 1)).sum().backward()
        reference_torch.add(1.5, expected_leaf.transpose(0, 1)).sum().backward()
        self.assert_matches(actual_leaf.grad, expected_leaf.grad, case="scalar autograd")

        actual_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        actual_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        expected_left = reference_torch.tensor(
            [[1.0, 2.0]], dtype=reference_torch.float32, requires_grad=True
        )
        expected_right = reference_torch.tensor(
            [[3.0], [4.0]], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_untracked = torch.add(
                actual_left.transpose(0, 1), actual_right.transpose(0, 1)
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.add(
                expected_left.transpose(0, 1), expected_right.transpose(0, 1)
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad")

    def torch_function_dispatch_observation(self, module):
        left = module.tensor([2.0], dtype=module.float32)
        right = module.tensor([3.0], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.add
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
            (
                lambda: function(input=left, other=right, alpha=1, out=None),
                ("input", "other", "alpha", "out"),
            ),
            (lambda: function(left, right, alpha=2), ("alpha",)),
            (lambda: function(left, right, out=destination), ("out",)),
        )
        for call, keywords in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == keywords,
                )
            )

        override_events = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call in (
            lambda value: function(value, right),
            lambda value: function(left, value),
            lambda value: function(value, 4.0),
            lambda value: function(left, right, alpha=value),
            lambda value: function(left, right, out=value),
        ):
            Override.calls.clear()
            result = call(Override())
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_events.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (Override,),
                    len(args),
                    kwargs is None,
                    kwargs is not None and tuple(kwargs),
                )
            )

        return mode_observations, override_events

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.torch_function_dispatch_observation(torch),
            self.torch_function_dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        function = module.add
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
            "owner_callable_identity": owner.add is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("add"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["add"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_exports_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def test_unsupported_boundaries_are_explicit(self):
        actual = torch.tensor([1.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^add\(\): tensor/tensor autograd recording is not supported$",
        ):
            torch.add(torch.tensor([1.0], requires_grad=True), actual)
        with self.assertRaisesRegex(
            NotImplementedError, r"^add\(\): non-default alpha is not supported$"
        ):
            torch.add(actual, actual, alpha=2)
        with self.assertRaisesRegex(
            RuntimeError, r"^Boolean alpha only supported for Boolean results\.$"
        ):
            torch.add(actual, actual, alpha=True)
        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError, r"^add\(\): the 'out' argument is not supported$"
        ):
            torch.add(actual, actual, out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        with self.assertRaisesRegex(TypeError, "scalar-scalar addition"):
            torch.add(2, 3)
        for name in ("add_", "sub", "subtract"):
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)
        for name in ("add", "add_"):
            self.assertFalse(hasattr(torch.Tensor, name))


if __name__ == "__main__":
    unittest.main()
