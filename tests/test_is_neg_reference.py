import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsNegReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_neg differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
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
        return leaf, tracked, (
            module.tensor(-3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            source.neg(),
            strided_view,
            offset_view,
            extreme_empty,
            leaf,
            tracked,
            tracked.detach(),
        )

    def contract(self, tensor, checker=None):
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
        result = tensor.is_neg() if checker is None else checker(tensor)
        return {
            "result": result,
            "result_type": type(result).__name__,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def top_level_contract(self, module, tensor, keyword):
        return self.contract(
            tensor,
            lambda value: module.is_neg(value)
            if keyword is None
            else module.is_neg(**{keyword: value}),
        )

    def test_supported_tensors_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                self.assertEqual(self.contract(actual), self.contract(expected))

        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_top_level_supported_tensor_forms_match_pytorch_2_13(self):
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

        self.assertIs(torch.is_neg(actual_tracked), False)
        self.assertIs(reference_torch.is_neg(expected_tracked), False)
        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_reference_lazy_negative_view_sets_the_bit(self):
        source = reference_torch.tensor(
            [1.0, -2.0, 3.0], dtype=reference_torch.float32
        )
        negative_view = reference_torch._neg_view(source)

        self.assertIs(source.is_neg(), False)
        self.assertTrue(negative_view._is_view())
        self.assertIs(negative_view._base, source)
        self.assertEqual(
            negative_view.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )
        self.assertEqual(negative_view.tolist(), [-1.0, 2.0, -3.0])
        self.assertIs(negative_view.is_neg(), True)
        self.assertIs(reference_torch.is_neg(source), False)
        self.assertIs(reference_torch.is_neg(negative_view), True)
        self.assertFalse(hasattr(torch, "_neg_view"))

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_neg unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def callable_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "is_neg")
        bound = tensor.is_neg
        calls = (
            lambda: tensor.is_neg(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.is_neg(1, 2),
            lambda: tensor.is_neg(input=tensor),
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
            "descriptor_result": descriptor(tensor),
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
        function = module.is_neg
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
            "owner_callable_identity": owner.is_neg is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("is_neg"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["is_neg"] is function,
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
            (lambda: torch.is_neg(), lambda: reference_torch.is_neg()),
            (
                lambda: torch.is_neg(actual, actual),
                lambda: reference_torch.is_neg(expected, expected),
            ),
            (
                lambda: torch.is_neg(actual, input=actual),
                lambda: reference_torch.is_neg(expected, input=expected),
            ),
            (
                lambda: torch.is_neg(actual, extra=True, input=actual),
                lambda: reference_torch.is_neg(
                    expected, extra=True, input=expected
                ),
            ),
            (
                lambda: torch.is_neg(actual, input=actual, extra=True),
                lambda: reference_torch.is_neg(
                    expected, input=expected, extra=True
                ),
            ),
            (
                lambda: torch.is_neg(extra=actual),
                lambda: reference_torch.is_neg(extra=expected),
            ),
            (
                lambda: torch.is_neg(1, extra=True),
                lambda: reference_torch.is_neg(1, extra=True),
            ),
            (
                lambda: torch.is_neg(input=[]),
                lambda: reference_torch.is_neg(input=[]),
            ),
            (
                lambda: torch.is_neg(a=1),
                lambda: reference_torch.is_neg(a=1),
            ),
            (
                lambda: torch.is_neg(x=[]),
                lambda: reference_torch.is_neg(x=[]),
            ),
            (
                lambda: torch.is_neg(x1=None),
                lambda: reference_torch.is_neg(x1=None),
            ),
            (
                lambda: torch.is_neg(a=actual, x=actual),
                lambda: reference_torch.is_neg(a=expected, x=expected),
            ),
            (
                lambda: torch.is_neg(x=actual, a=actual),
                lambda: reference_torch.is_neg(x=expected, a=expected),
            ),
            (
                lambda: torch.is_neg(input=actual, x1=actual),
                lambda: reference_torch.is_neg(input=expected, x1=expected),
            ),
            (
                lambda: torch.is_neg(x=actual, x1=actual),
                lambda: reference_torch.is_neg(x=expected, x1=expected),
            ),
            (
                lambda: torch.is_neg(x1=actual, x=actual),
                lambda: reference_torch.is_neg(x1=expected, x=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def keyword_lookup_observation(self, module):
        class RaisingKey(str):
            def __eq__(self, other):
                raise RuntimeError("keyword equality exploded")

            __hash__ = str.__hash__

        outcomes = []
        for alias in ("input", "x", "a", "x1"):
            key = RaisingKey(alias)
            outcomes.append(
                self.error(
                    lambda key=key: module.is_neg(
                        **{key: module.tensor([1.0], dtype=module.float32)}
                    )
                )
            )
        return outcomes

    def test_legacy_keyword_lookup_suppression_matches_pytorch_2_13(self):
        self.assertEqual(
            self.keyword_lookup_observation(torch),
            self.keyword_lookup_observation(reference_torch),
        )

    def disabled_override_observation(self, module_name):
        source = r'''
import importlib
import json

import torch as reference_torch

module = importlib.import_module(MODULE)

class Disabled:
    __torch_function__ = reference_torch._C._disabled_torch_function_impl

class InstanceDisabled:
    pass

instance_disabled = InstanceDisabled()
instance_disabled.__torch_function__ = (
    reference_torch._C._disabled_torch_function_impl
)

class PropertyDisabled:
    @property
    def __torch_function__(self):
        return reference_torch._C._disabled_torch_function_impl

def error(action):
    try:
        action()
    except Exception as exception:
        return [type(exception).__name__, str(exception)]
    return None

forms = [
    error(lambda: module.is_neg(Disabled())),
    error(lambda: module.is_neg(input=Disabled())),
    error(lambda: module.is_neg(x=Disabled())),
    error(lambda: module.is_neg(a=Disabled())),
    error(lambda: module.is_neg(x1=Disabled())),
    error(lambda: module.is_neg(Disabled)),
    error(lambda: module.is_neg(instance_disabled)),
    error(lambda: module.is_neg(PropertyDisabled())),
]

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return object()

mode = RecordingMode()
with mode:
    mode_error = error(lambda: module.is_neg(Disabled()))

print(json.dumps({
    "forms": forms,
    "mode_error": mode_error,
    "mode_calls": mode.calls,
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=(
                f"{module_name} disabled override subprocess exited with "
                f"{result.returncode}: stdout={result.stdout!r}, "
                f"stderr={result.stderr!r}"
            ),
        )
        return json.loads(result.stdout)

    def test_disabled_override_sentinel_matches_pytorch_2_13(self):
        self.assertEqual(
            self.disabled_override_observation("torch_rs"),
            self.disabled_override_observation("torch"),
        )

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "is_neg")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.is_neg()
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
                forwarded = tensor.is_neg()

        return {
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
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
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
                module.is_neg(tensor)
                if keyword is None
                else module.is_neg(**{keyword: tensor})
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
                forwarded = module.is_neg(a=tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module.is_neg(x=value)
        override_function, override_types, override_args, override_kwargs = (
            override_calls[0]
        )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining_mode = DecliningMode()
        try:
            with declining_mode:
                module.is_neg(tensor)
        except Exception as error:
            declining_mode_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_mode_error = None

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            module.is_neg(DecliningOverride())
        except Exception as error:
            declining_override_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_override_error = None

        fallback_calls = []

        class FallbackMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                fallback_calls.append(("mode", func, types, args, kwargs))
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_calls.append(("override", func, types, args, kwargs))
                return marker

        fallback_value = FallbackOverride()
        with FallbackMode():
            fallback_result = module.is_neg(input=fallback_value)

        return {
            "intercepted": intercepted is marker,
            "call_count": len(mode.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_identity": function is module.is_neg,
            "types_empty": dispatch_types == (),
            "args_original": (len(args) == 1 and args[0] is tensor)
            if keyword is None
            else args == (),
            "kwargs_original": kwargs is None
            if keyword is None
            else len(kwargs) == 1 and kwargs.get(keyword) is tensor,
            "forwarding_order": order,
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
            "override_result": override_result is marker,
            "override_function": override_function is module.is_neg,
            "override_types": override_types == (Override,),
            "override_args": override_args == (),
            "override_kwargs": len(override_kwargs) == 1
            and override_kwargs.get("x") is value,
            "declining_mode_error": declining_mode_error,
            "declining_override_error": declining_override_error,
            "fallback_result": fallback_result is marker,
            "fallback_order": [call[0] for call in fallback_calls],
            "fallback_functions": all(
                call[1] is module.is_neg for call in fallback_calls
            ),
            "fallback_types": all(
                call[2] == (FallbackOverride,) for call in fallback_calls
            ),
            "fallback_args": all(call[3] == () for call in fallback_calls),
            "fallback_kwargs": all(
                len(call[4]) == 1 and call[4].get("input") is fallback_value
                for call in fallback_calls
            ),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_top_level_torch_function_dispatch_matches_pytorch_2_13(self):
        for keyword in (None, "input", "x", "a", "x1"):
            with self.subTest(keyword=keyword):
                self.assertEqual(
                    self.top_level_dispatch_observation(torch, keyword),
                    self.top_level_dispatch_observation(reference_torch, keyword),
                )

    def test_scope_adds_only_the_top_level_entry_point(self):
        self.assertTrue(hasattr(torch, "is_neg"))
        self.assertTrue(hasattr(reference_torch, "is_neg"))
        self.assertFalse(hasattr(torch, "_neg_view"))
        self.assertTrue(hasattr(reference_torch, "_neg_view"))


if __name__ == "__main__":
    unittest.main()
