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
class TensorResolveNegReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "resolve_neg differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            leaf,
            tracked,
            (
                module.tensor(-3.5, dtype=module.float32),
                module.zeros((2, 0, 3), dtype=module.float32),
                source.neg(),
                strided_view,
                offset_view,
                extreme_empty,
                module.tensor(memoryview(special_bits.view(np.float32))),
                leaf,
                tracked,
                tracked.detach(),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return tuple(
            np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist()
        )

    def contract(self, tensor, resolver):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        bits = self.value_bits(tensor)
        result = resolver(tensor)
        return {
            "receiver_is_clear": tensor.is_neg() is False,
            "result_is_receiver": result is tensor,
            "result_is_clear": result.is_neg() is False,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
                str(result.dtype),
                str(result.device),
                result.requires_grad,
                result.is_leaf,
            ),
            "bits_unchanged": bits == self.value_bits(result),
        }

    def top_level_contract(self, module, tensor, keyword):
        return self.contract(
            tensor,
            lambda value: module.resolve_neg(value)
            if keyword is None
            else module.resolve_neg(**{keyword: value}),
        )

    def test_supported_clear_bit_path_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                self.assertEqual(
                    self.contract(actual, lambda value: value.resolve_neg()),
                    self.contract(expected, lambda value: value.resolve_neg()),
                )

        actual_tracked.resolve_neg().sum().backward()
        expected_tracked.resolve_neg().sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_top_level_clear_bit_call_forms_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for keyword in (None, "input", "x", "a", "x1"):
                with self.subTest(
                    case=case,
                    keyword=keyword,
                    shape=actual.shape,
                    stride=actual.stride(),
                ):
                    self.assertEqual(
                        self.top_level_contract(torch, actual, keyword),
                        self.top_level_contract(reference_torch, expected, keyword),
                    )

        torch.resolve_neg(x=actual_tracked).sum().backward()
        reference_torch.resolve_neg(x=expected_tracked).sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_reference_lazy_negative_view_paths_materialize_and_preserve_graph(self):
        self.assertFalse(hasattr(torch, "_neg_view"))
        for form, resolver in (
            ("method", lambda value: value.resolve_neg()),
            ("top-level", reference_torch.resolve_neg),
        ):
            with self.subTest(form=form):
                source = reference_torch.tensor(
                    [1.0, -2.0, 3.0],
                    dtype=reference_torch.float32,
                    requires_grad=True,
                )
                negative_view = reference_torch._neg_view(source)

                self.assertIs(source.is_neg(), False)
                self.assertIs(negative_view.is_neg(), True)
                self.assertTrue(negative_view._is_view())
                self.assertIs(negative_view._base, source)
                self.assertEqual(
                    negative_view.untyped_storage().data_ptr(),
                    source.untyped_storage().data_ptr(),
                )
                self.assertEqual(
                    type(negative_view.grad_fn).__name__, "NegViewBackward0"
                )

                resolved = resolver(negative_view)

                self.assertIsNot(resolved, negative_view)
                self.assertIsNot(resolved, source)
                self.assertIs(resolved.is_neg(), False)
                self.assertFalse(resolved._is_view())
                self.assertIsNone(resolved._base)
                self.assertNotEqual(
                    resolved.untyped_storage().data_ptr(),
                    negative_view.untyped_storage().data_ptr(),
                )
                self.assertEqual(resolved.tolist(), negative_view.tolist())
                self.assertTrue(resolved.requires_grad)
                self.assertFalse(resolved.is_leaf)
                self.assertEqual(type(resolved.grad_fn).__name__, "CloneBackward0")

                resolved.sum().backward()
                self.assertEqual(source.grad.tolist(), [-1.0, -1.0, -1.0])

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("resolve_neg unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "resolve_neg")
        bound = tensor.resolve_neg
        calls = (
            lambda: tensor.resolve_neg(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.resolve_neg(1, 2),
            lambda: tensor.resolve_neg(input=tensor),
            lambda: bound(unexpected=True),
            lambda: descriptor(tensor, unexpected=True),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "signatures": tuple(
                self.signature_outcome(callable_object)
                for callable_object in (descriptor, bound)
            ),
            "call_errors": tuple(self.error(call) for call in calls),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def top_level_callable_contract(self, module):
        function = module.resolve_neg
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
            "owner_callable_identity": owner.resolve_neg is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("resolve_neg"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["resolve_neg"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    def test_top_level_binding_error_precedence_matches_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.resolve_neg(), lambda: reference_torch.resolve_neg()),
            (
                lambda: torch.resolve_neg(actual, actual),
                lambda: reference_torch.resolve_neg(expected, expected),
            ),
            (
                lambda: torch.resolve_neg(actual, input=actual),
                lambda: reference_torch.resolve_neg(expected, input=expected),
            ),
            (
                lambda: torch.resolve_neg(actual, extra=True, input=actual),
                lambda: reference_torch.resolve_neg(
                    expected, extra=True, input=expected
                ),
            ),
            (
                lambda: torch.resolve_neg(actual, input=actual, extra=True),
                lambda: reference_torch.resolve_neg(
                    expected, input=expected, extra=True
                ),
            ),
            (
                lambda: torch.resolve_neg(extra=actual),
                lambda: reference_torch.resolve_neg(extra=expected),
            ),
            (
                lambda: torch.resolve_neg(1, extra=True),
                lambda: reference_torch.resolve_neg(1, extra=True),
            ),
            (
                lambda: torch.resolve_neg(input=[]),
                lambda: reference_torch.resolve_neg(input=[]),
            ),
            (
                lambda: torch.resolve_neg(a=1),
                lambda: reference_torch.resolve_neg(a=1),
            ),
            (
                lambda: torch.resolve_neg(x=[]),
                lambda: reference_torch.resolve_neg(x=[]),
            ),
            (
                lambda: torch.resolve_neg(x1=None),
                lambda: reference_torch.resolve_neg(x1=None),
            ),
            (
                lambda: torch.resolve_neg(a=actual, x=actual),
                lambda: reference_torch.resolve_neg(a=expected, x=expected),
            ),
            (
                lambda: torch.resolve_neg(x=actual, a=actual),
                lambda: reference_torch.resolve_neg(x=expected, a=expected),
            ),
            (
                lambda: torch.resolve_neg(input=actual, x1=actual),
                lambda: reference_torch.resolve_neg(
                    input=expected, x1=expected
                ),
            ),
            (
                lambda: torch.resolve_neg(x=actual, x1=actual),
                lambda: reference_torch.resolve_neg(x=expected, x1=expected),
            ),
            (
                lambda: torch.resolve_neg(x1=actual, x=actual),
                lambda: reference_torch.resolve_neg(x1=expected, x=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "resolve_neg")
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
    intercepted = tensor.resolve_neg()
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
        forwarded = tensor.resolve_neg()

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
            tensor.resolve_neg()
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
    "function_is_descriptor": function is descriptor,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded_is_receiver": forwarded is tensor,
    "forwarded_is_clear": forwarded.is_neg() is False,
    "declining_error": declining_error,
    "declining_calls": upper.calls,
    "lower_skipped": len(lower.calls) == 0,
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
                module.resolve_neg(tensor)
                if keyword is None
                else module.resolve_neg(**{keyword: tensor})
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
                forwarded = module.resolve_neg(a=tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module.resolve_neg(x=value)
        override_function, override_types, override_args, override_kwargs = (
            override_calls[0]
        )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining_mode = DecliningMode()
        try:
            with declining_mode:
                module.resolve_neg(tensor)
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
            "function_identity": function is module.resolve_neg,
            "types_empty": dispatch_types == (),
            "args_original": (len(args) == 1 and args[0] is tensor)
            if keyword is None
            else args == (),
            "kwargs_original": kwargs is None
            if keyword is None
            else len(kwargs) == 1 and kwargs.get(keyword) is tensor,
            "forwarding_order": order,
            "forwarded_is_receiver": forwarded is tensor,
            "forwarded_is_clear": forwarded.is_neg() is False,
            "override_result": override_result is marker,
            "override_function": override_function is module.resolve_neg,
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

    def test_scope_adds_callable_without_lazy_negative_views(self):
        self.assertTrue(hasattr(torch, "resolve_neg"))
        self.assertTrue(hasattr(reference_torch, "resolve_neg"))
        self.assertFalse(hasattr(torch, "_neg_view"))
        self.assertTrue(hasattr(reference_torch, "_neg_view"))


if __name__ == "__main__":
    unittest.main()
