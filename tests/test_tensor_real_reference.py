import inspect
import json
import pickle
import re
import subprocess
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
class TensorRealReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.real differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        scalar_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = module.tensor(memoryview(scalar_bits.view(np.float32)))
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        return (
            *(scalar_storage[index] for index in range(len(scalar_bits))),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            leaf,
            (leaf * 3.0).transpose(0, 1)[1],
        )

    def identity_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        pointer = tensor.data_ptr()
        result = tensor.real
        return {
            "identity": result is tensor,
            "metadata": metadata,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
                str(result.dtype),
                str(result.device),
                str(result.layout),
                result.requires_grad,
                result.is_leaf,
            ),
            "pointer_unchanged": result.data_ptr() == pointer,
            "bits": np.asarray(result.detach()).reshape(-1).view(np.uint32).copy(),
        }

    def test_scalar_empty_offset_strided_leaf_and_non_leaf_tensors_match(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_contract = self.identity_contract(actual)
                expected_contract = self.identity_contract(expected)
                np.testing.assert_array_equal(
                    actual_contract.pop("bits"), expected_contract.pop("bits")
                )
                self.assertEqual(actual_contract, expected_contract)

    def test_leaf_and_non_leaf_autograd_identity_matches(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            leaf_result = leaf.real
            non_leaf = (leaf_result * 3.0).transpose(0, 1)[1]
            graph_before = (
                non_leaf.requires_grad,
                non_leaf.is_leaf,
                tuple(non_leaf.shape),
                non_leaf.stride(),
                non_leaf.storage_offset(),
            )
            pointer = non_leaf.data_ptr()
            result = non_leaf.real
            graph_after = (
                result.requires_grad,
                result.is_leaf,
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
            )
            result.sum().backward()
            gradient = leaf.grad
            outcomes.append(
                (
                    leaf_result is leaf,
                    result is non_leaf,
                    result.data_ptr() == pointer,
                    graph_before,
                    graph_after,
                    leaf.real is leaf,
                    leaf.grad is gradient,
                    np.asarray(gradient).copy(),
                )
            )

        self.assertEqual(outcomes[0][:-1], outcomes[1][:-1])
        np.testing.assert_array_equal(outcomes[0][-1], outcomes[1][-1])

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.real unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "real")
        tensor = module.tensor([1.0], dtype=module.float32)
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.real is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_identity": descriptor.__get__(tensor, module.Tensor) is tensor,
            "receiver_error": self.error(lambda: descriptor.__get__(1, int)),
        }

    def test_descriptor_ownership_documentation_and_identity_match(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "real")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

recording = RecordingMode(marker)
with recording:
    intercepted = tensor.real
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = tensor.real

sys.setrecursionlimit(80)
class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

lower = RecordingMode(marker)
upper = DecliningMode()
try:
    with lower:
        with upper:
            tensor.real
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_self": function.__self__ is descriptor,
    "function_equals_descriptor_get": function == descriptor.__get__,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded_identity": forwarded is tensor,
    "declining_error": declining_error,
    "declining_calls": upper.calls,
    "lower_skipped": len(lower.calls) == 0,
    "ordinary_identity": tensor.real is tensor,
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    @staticmethod
    def call_top_level(module, tensor, keyword):
        if keyword is None:
            return module.real(tensor)
        return module.real(**{keyword: tensor})

    def top_level_identity_contract(self, tensor, caller):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.data_ptr(),
        )
        bits = np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()
        result = caller(tensor)
        return {
            "identity": result is tensor,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
                str(result.dtype),
                str(result.device),
                str(result.layout),
                result.requires_grad,
                result.is_leaf,
                result.data_ptr(),
            ),
            "bits": np.asarray(result.detach()).reshape(-1).view(np.uint32).copy(),
        }

    def test_top_level_scalar_empty_offset_strided_leaf_and_non_leaf_match(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for keyword in (None, "input", "x", "a", "x1"):
                with self.subTest(case=case, keyword=keyword, shape=actual.shape):
                    actual_contract = self.top_level_identity_contract(
                        actual,
                        lambda value, keyword=keyword: self.call_top_level(
                            torch, value, keyword
                        ),
                    )
                    expected_contract = self.top_level_identity_contract(
                        expected,
                        lambda value, keyword=keyword: self.call_top_level(
                            reference_torch, value, keyword
                        ),
                    )
                    np.testing.assert_array_equal(
                        actual_contract.pop("bits"), expected_contract.pop("bits")
                    )
                    self.assertEqual(actual_contract, expected_contract)

    def class_shadowing_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        original = module.Tensor.real
        marker = object()

        try:
            module.Tensor.real = property(lambda _self: marker)
            return {
                "property_is_shadowed": tensor.real is marker,
                "top_level_positional_identity": module.real(tensor) is tensor,
                "top_level_keyword_identity": module.real(input=tensor) is tensor,
                "top_level_ignores_shadow": module.real(tensor) is not marker,
            }
        finally:
            module.Tensor.real = original

    def test_top_level_real_ignores_tensor_class_shadowing_like_pytorch(self):
        self.assertEqual(
            self.class_shadowing_contract(torch),
            self.class_shadowing_contract(reference_torch),
        )

    def test_top_level_leaf_and_non_leaf_autograd_identity_matches(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            leaf_result = module.real(leaf)
            non_leaf = (leaf_result * 3.0).transpose(0, 1)[1]
            graph_before = (
                non_leaf.requires_grad,
                non_leaf.is_leaf,
                tuple(non_leaf.shape),
                non_leaf.stride(),
                non_leaf.storage_offset(),
            )
            pointer = non_leaf.data_ptr()
            result = module.real(non_leaf)
            graph_after = (
                result.requires_grad,
                result.is_leaf,
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
            )
            result.sum().backward()
            gradient = leaf.grad
            outcomes.append(
                (
                    leaf_result is leaf,
                    result is non_leaf,
                    result.data_ptr() == pointer,
                    graph_before,
                    graph_after,
                    module.real(leaf) is leaf,
                    leaf.grad is gradient,
                    np.asarray(gradient).copy(),
                )
            )

        self.assertEqual(outcomes[0][:-1], outcomes[1][:-1])
        np.testing.assert_array_equal(outcomes[0][-1], outcomes[1][-1])

    def top_level_callable_contract(self, module):
        function = module.real
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
            "owner_callable_identity": owner.real is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("real"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["real"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_metadata_and_wildcard_import_match(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    def top_level_dispatch_observation(self, module, keyword):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            intercepted = (
                module.real(tensor)
                if keyword is None
                else module.real(**{keyword: tensor})
            )
        function, dispatch_types, args, kwargs = mode.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.real(a=tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module.real(x=value)
        override_function, override_types, override_args, override_kwargs = (
            override_calls[0]
        )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining_mode = DecliningMode()
        try:
            with declining_mode:
                module.real(tensor)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted is marker,
            "call_count": len(mode.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_identity": function is module.real,
            "types_empty": dispatch_types == (),
            "args_original": (
                (len(args) == 1 and args[0] is tensor)
                if keyword is None
                else args == ()
            ),
            "kwargs_original": (
                kwargs is None
                if keyword is None
                else len(kwargs) == 1 and kwargs.get(keyword) is tensor
            ),
            "forwarding_order": order,
            "forwarded_is_receiver": forwarded is tensor,
            "override_result": override_result is marker,
            "override_function": override_function is module.real,
            "override_types": override_types == (Override,),
            "override_args": override_args == (),
            "override_kwargs": len(override_kwargs) == 1
            and override_kwargs.get("x") is value,
            "declining_error": declining_error,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_top_level_torch_function_dispatch_matches_pytorch_2_13(self):
        for keyword in (None, "input", "x", "a", "x1"):
            with self.subTest(keyword=keyword):
                self.assertEqual(
                    self.top_level_dispatch_observation(torch, keyword),
                    self.top_level_dispatch_observation(reference_torch, keyword),
                )

    def test_top_level_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.real(), lambda: reference_torch.real()),
            (
                lambda: torch.real(actual, actual),
                lambda: reference_torch.real(expected, expected),
            ),
            (
                lambda: torch.real(actual, input=actual),
                lambda: reference_torch.real(expected, input=expected),
            ),
            (
                lambda: torch.real(actual, out=None),
                lambda: reference_torch.real(expected, out=None),
            ),
            (
                lambda: torch.real(actual, out=torch.tensor([0.0])),
                lambda: reference_torch.real(
                    expected,
                    out=reference_torch.tensor([0.0]),
                ),
            ),
            (
                lambda: torch.real(actual, dtype=torch.float32),
                lambda: reference_torch.real(expected, dtype=reference_torch.float32),
            ),
            (
                lambda: torch.real(actual, device="cpu"),
                lambda: reference_torch.real(expected, device="cpu"),
            ),
            (
                lambda: torch.real(1, extra=True),
                lambda: reference_torch.real(1, extra=True),
            ),
            (lambda: torch.real(input=[]), lambda: reference_torch.real(input=[])),
            (lambda: torch.real(a=1), lambda: reference_torch.real(a=1)),
            (lambda: torch.real(x=[]), lambda: reference_torch.real(x=[])),
            (lambda: torch.real(x1=None), lambda: reference_torch.real(x1=None)),
            (
                lambda: torch.real(a=actual, x=actual),
                lambda: reference_torch.real(a=expected, x=expected),
            ),
            (
                lambda: torch.real(x=actual, a=actual),
                lambda: reference_torch.real(x=expected, a=expected),
            ),
            (
                lambda: torch.real(input=actual, x1=actual),
                lambda: reference_torch.real(input=expected, x1=expected),
            ),
            (
                lambda: torch.real(x=actual, x1=actual),
                lambda: reference_torch.real(x=expected, x1=expected),
            ),
            (
                lambda: torch.real(x1=actual, x=actual),
                lambda: reference_torch.real(x1=expected, x=expected),
            ),
            (
                lambda: torch.real(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.real(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_scope_adds_real_without_complex_support(self):
        self.assertTrue(hasattr(torch, "real"))
        self.assertTrue(hasattr(reference_torch, "real"))
        self.assertTrue(hasattr(torch, "imag"))
        self.assertTrue(hasattr(reference_torch, "imag"))
        self.assertFalse(hasattr(torch, "real_"))
        self.assertFalse(hasattr(torch.Tensor, "real_"))
        self.assertFalse(hasattr(torch, "complex64"))
        self.assertTrue(hasattr(reference_torch, "complex64"))


if __name__ == "__main__":
    unittest.main()
