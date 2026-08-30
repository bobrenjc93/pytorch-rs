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
class TensorConjReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("conj differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_bits(tensor):
        if 0 in tensor.shape:
            return None
        if isinstance(tensor, torch.Tensor):
            return tuple(
                np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist()
            )
        return tuple(tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).tolist())

    def make_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        source = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = source.transpose(0, 2)
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
        special_values = special_bits.view(np.float32)
        special = (
            module.tensor(memoryview(special_values))
            if module is torch
            else module.tensor(special_values, dtype=module.float32)
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("special values", special),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
        )

    def contract(self, tensor, caller):
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
            tensor.is_conj(),
        )
        bits = self.tensor_bits(tensor)
        result = caller(tensor)
        return {
            "result_is_receiver": result is tensor,
            "result_is_set_to_detached": result.is_set_to(tensor.detach()),
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
                result.is_conj(),
            ),
            "result_is_clear": result.is_conj() is False,
            "bits_unchanged": bits == self.tensor_bits(result),
        }

    @staticmethod
    def call_top_level(module, tensor, keyword):
        if keyword is None:
            return module.conj(tensor)
        return module.conj(**{keyword: tensor})

    def test_real_tensor_identity_contract_matches_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            with self.subTest(case=case, form="method"):
                self.assertEqual(
                    self.contract(actual, lambda value: value.conj()),
                    self.contract(expected, lambda value: value.conj()),
                )
            for keyword in (None, "input", "x", "a", "x1"):
                with self.subTest(case=case, keyword=keyword):
                    self.assertEqual(
                        self.contract(
                            actual,
                            lambda value, keyword=keyword: self.call_top_level(
                                torch, value, keyword
                            ),
                        ),
                        self.contract(
                            expected,
                            lambda value, keyword=keyword: self.call_top_level(
                                reference_torch, value, keyword
                            ),
                        ),
                    )

    @staticmethod
    def grad_array(tensor):
        if isinstance(tensor, torch.Tensor):
            return np.asarray(tensor).copy()
        return tensor.detach().cpu().numpy().copy()

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        non_leaf = (leaf * 2.0).transpose(0, 1)

        with module.no_grad():
            no_grad_method = non_leaf.conj()
            no_grad_top_level = module.conj(non_leaf)

        module.conj(non_leaf).sum().backward()
        first_grad = self.grad_array(leaf.grad)

        reusable_leaf = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        reusable_loss = reusable_leaf.transpose(0, 0).conj().sum()
        reusable_loss.backward()
        reusable_loss.backward()
        repeated_grad = self.grad_array(reusable_leaf.grad)

        return {
            "leaf_identity": leaf.conj() is leaf,
            "leaf_requires_grad": leaf.conj().requires_grad,
            "leaf_is_leaf": leaf.conj().is_leaf,
            "non_leaf_identity": non_leaf.conj() is non_leaf,
            "non_leaf_requires_grad": non_leaf.conj().requires_grad,
            "non_leaf_is_leaf": non_leaf.conj().is_leaf,
            "no_grad_method_identity": no_grad_method is non_leaf,
            "no_grad_top_level_identity": no_grad_top_level is non_leaf,
            "no_grad_requires_grad": (
                no_grad_method.requires_grad,
                no_grad_top_level.requires_grad,
            ),
            "no_grad_is_leaf": (
                no_grad_method.is_leaf,
                no_grad_top_level.is_leaf,
            ),
            "first_grad": first_grad.tolist(),
            "repeated_grad": repeated_grad.tolist(),
        }

    def test_leaf_non_leaf_no_grad_and_repeated_backward_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_outcome(torch), self.autograd_outcome(reference_torch)
        )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("conj unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "conj")
        bound = tensor.conj
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
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def top_level_callable_contract(self, module):
        function = module.conj
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
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
            "owner_callable_identity": owner.conj is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature": self.signature_outcome(function),
            "all_count": module.__all__.count("conj"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["conj"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_wildcard_import_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    def test_binding_error_precedence_matches_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.conj(), lambda: reference_torch.conj()),
            (
                lambda: torch.conj(actual, actual),
                lambda: reference_torch.conj(expected, expected),
            ),
            (
                lambda: torch.conj(actual, input=actual),
                lambda: reference_torch.conj(expected, input=expected),
            ),
            (
                lambda: torch.conj(actual, out=None),
                lambda: reference_torch.conj(expected, out=None),
            ),
            (
                lambda: torch.conj(actual, dtype=torch.float32),
                lambda: reference_torch.conj(expected, dtype=reference_torch.float32),
            ),
            (
                lambda: torch.conj(actual, device="cpu"),
                lambda: reference_torch.conj(expected, device="cpu"),
            ),
            (
                lambda: torch.conj(1, extra=True),
                lambda: reference_torch.conj(1, extra=True),
            ),
            (
                lambda: torch.conj(input=[]),
                lambda: reference_torch.conj(input=[]),
            ),
            (lambda: torch.conj(a=1), lambda: reference_torch.conj(a=1)),
            (lambda: torch.conj(x=[]), lambda: reference_torch.conj(x=[])),
            (
                lambda: torch.conj(x1=None),
                lambda: reference_torch.conj(x1=None),
            ),
            (
                lambda: torch.conj(a=actual, x=actual),
                lambda: reference_torch.conj(a=expected, x=expected),
            ),
            (
                lambda: torch.conj(x=actual, a=actual),
                lambda: reference_torch.conj(x=expected, a=expected),
            ),
            (
                lambda: torch.conj(input=actual, x1=actual),
                lambda: reference_torch.conj(input=expected, x1=expected),
            ),
            (
                lambda: torch.conj(x=actual, x1=actual),
                lambda: reference_torch.conj(x=expected, x1=expected),
            ),
            (
                lambda: torch.conj(x1=actual, x=actual),
                lambda: reference_torch.conj(x1=expected, x=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def mode_dispatch_observation(self, module_name):
        source = r"""
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "conj")
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
    intercepted = tensor.conj()
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
        forwarded = tensor.conj()

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
    "forwarded_is_clear": forwarded.is_conj() is False,
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}))
"""
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_method_torch_function_mode_dispatch_matches_pytorch_2_13(self):
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
                module.conj(tensor)
                if keyword is None
                else module.conj(**{keyword: tensor})
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
                forwarded = module.conj(a=tensor)

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        value = Override()
        override_result = module.conj(x=value)
        override_function, override_types, override_args, override_kwargs = (
            override_calls[0]
        )

        return {
            "intercepted": intercepted is marker,
            "call_count": len(mode.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_identity": function is module.conj,
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
            "forwarded_is_clear": forwarded.is_conj() is False,
            "override_result": override_result is marker,
            "override_function": override_function is module.conj,
            "override_types": override_types == (Override,),
            "override_args": override_args == (),
            "override_kwargs": len(override_kwargs) == 1
            and override_kwargs.get("x") is value,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_top_level_torch_function_dispatch_matches_pytorch_2_13(self):
        for keyword in (None, "input", "x", "a", "x1"):
            with self.subTest(keyword=keyword):
                self.assertEqual(
                    self.top_level_dispatch_observation(torch, keyword),
                    self.top_level_dispatch_observation(reference_torch, keyword),
                )

    def test_complex_and_physical_conj_paths_remain_outside_native_scope(self):
        self.assertTrue(hasattr(torch, "conj"))
        self.assertTrue(hasattr(reference_torch, "conj"))
        self.assertFalse(hasattr(torch, "conj_physical"))
        self.assertTrue(hasattr(reference_torch, "conj_physical"))
        self.assertFalse(hasattr(torch.Tensor, "conj_"))
        self.assertFalse(hasattr(torch.Tensor, "conj_physical"))
        self.assertFalse(hasattr(torch, "complex64"))
        self.assertTrue(hasattr(reference_torch, "complex64"))

        source = reference_torch.tensor(
            [1.0 + 2.0j, -3.0 + 4.0j],
            dtype=reference_torch.complex64,
            requires_grad=True,
        )
        conjugate_view = source.conj()
        self.assertIs(source.is_conj(), False)
        self.assertIs(conjugate_view.is_conj(), True)
        self.assertTrue(conjugate_view._is_view())
        self.assertIs(conjugate_view._base, source)
        self.assertEqual(
            conjugate_view.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )
        self.assertEqual(type(conjugate_view.grad_fn).__name__, "ConjBackward0")


if __name__ == "__main__":
    unittest.main()
