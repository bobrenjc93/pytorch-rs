import copy
import importlib
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AsarrayReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("asarray differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided = source.transpose(0, 1)
        special_bits = np.asarray(
            (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            ("empty view", module.zeros((2, 0, 3), dtype=module.float32)[1]),
            ("strided view", strided[1]),
            ("leaf", leaf),
            ("tracked view", tracked),
            ("special bits", module.tensor(memoryview(special_bits.view(np.float32)))),
        )

    def option_cases(self, module):
        return (
            {},
            {"dtype": None},
            {"dtype": module.float32},
            {"dtype": module.float},
            {"device": None},
            {"device": "cpu"},
            {"device": module.device("cpu")},
            {"copy": None},
            {"copy": False},
            {"requires_grad": None},
            {
                "dtype": module.float32,
                "device": module.device("cpu"),
                "copy": False,
                "requires_grad": None,
            },
        )

    def tensor_state(self, module, tensor):
        if module is reference_torch:
            values = tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).tolist()
        else:
            values = np.asarray(tensor).reshape(-1).view(np.uint32).tolist()
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "data_ptr": tensor.data_ptr(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "output_nr": tensor.output_nr,
            "values": values,
        }

    def comparable_tensor_state(self, module, tensor):
        state = self.tensor_state(module, tensor)
        state.pop("data_ptr")
        return state

    def asarray_identity_contract(self, module, tensor, options):
        before = self.tensor_state(module, tensor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = module.asarray(tensor, **options)
        after = self.tensor_state(module, tensor)
        return {
            "same_object": result is tensor,
            "same_pointer": result.data_ptr() == before["data_ptr"],
            "same_logical_storage": result.is_set_to(tensor),
            "result_state": self.comparable_tensor_state(module, result),
            "source_unchanged": before == after,
        }

    def test_identity_aliasing_and_metadata_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        actual_options = self.option_cases(torch)
        expected_options = self.option_cases(reference_torch)

        for (case, actual), (_, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for actual_kwargs, expected_kwargs in zip(
                actual_options, expected_options, strict=True
            ):
                with self.subTest(case=case, options=actual_kwargs):
                    actual_contract = self.asarray_identity_contract(
                        torch, actual, actual_kwargs
                    )
                    expected_contract = self.asarray_identity_contract(
                        reference_torch, expected, expected_kwargs
                    )
                    self.assertEqual(actual_contract, expected_contract)

    def test_autograd_identity_aliasing_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = (leaf * 3.0).transpose(0, 1)[1]
            before = self.tensor_state(module, source)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = module.asarray(
                    source,
                    dtype=module.float32,
                    device="cpu",
                    copy=False,
                    requires_grad=None,
                )
            after = self.tensor_state(module, result)
            result.sum().backward()
            outcomes.append(
                (
                    result is source,
                    before["data_ptr"] == after["data_ptr"],
                    {key: value for key, value in before.items() if key != "data_ptr"},
                    {key: value for key, value in after.items() if key != "data_ptr"},
                    np.asarray(leaf.grad).reshape(-1).tolist(),
                    module.asarray(leaf.grad) is leaf.grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def callable_contract(self, module):
        function = module.asarray
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
            "owner_callable_identity": owner.asarray is function,
            "native_has_function": hasattr(module._C, "asarray"),
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("asarray"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["asarray"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

        old = torch.asarray
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertFalse(hasattr(native, "asarray"))
        self.assertIs(torch.asarray, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.asarray, old)

    def mode_dispatch_observation(self, module, case):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        if case == "positional":
            call = lambda: module.asarray(tensor)
        elif case == "keyword":
            call = lambda: module.asarray(obj=tensor, dtype=module.float32)
        elif case == "sequence":
            data = [1.0, 2.0]
            call = lambda: module.asarray(data)
        elif case == "device":
            data = [1.0]
            call = lambda: module.asarray(data, device="cuda")
        elif case == "copy":
            call = lambda: module.asarray(tensor, copy=True)
        else:
            call = lambda: module.asarray(tensor, requires_grad=True)

        mode = RecordingMode()
        with mode:
            result = call()
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
                forwarded = module.asarray(
                    obj=tensor,
                    dtype=module.float32,
                    copy=False,
                    requires_grad=None,
                )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            with DecliningMode():
                module.asarray(tensor)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": result is marker,
            "call_count": len(mode.calls),
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_identity": function is module.asarray,
            "types_empty": dispatch_types == (),
            "args_len": len(args),
            "kwargs_keys": None if kwargs is None else tuple(kwargs.keys()),
            "forwarding_order": order,
            "forwarded_is_tensor": forwarded is tensor,
            "declining_error": declining_error,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        for case in ("positional", "keyword", "sequence", "device", "copy", "requires_grad"):
            with self.subTest(case=case):
                self.assertEqual(
                    self.mode_dispatch_observation(torch, case),
                    self.mode_dispatch_observation(reference_torch, case),
                )

    def error(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("expected call to fail")

    def test_binding_errors_match_pytorch_2_13_for_exposed_schema(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.asarray(), lambda: reference_torch.asarray()),
            (
                lambda: torch.asarray(actual, actual),
                lambda: reference_torch.asarray(expected, expected),
            ),
            (
                lambda: torch.asarray(actual, obj=actual),
                lambda: reference_torch.asarray(expected, obj=expected),
            ),
            (
                lambda: torch.asarray(data=actual),
                lambda: reference_torch.asarray(data=expected),
            ),
            (
                lambda: torch.asarray(actual, data=actual),
                lambda: reference_torch.asarray(expected, data=expected),
            ),
            (
                lambda: torch.asarray(actual, out=None),
                lambda: reference_torch.asarray(expected, out=None),
            ),
            (
                lambda: torch.asarray(actual, pin_memory=False),
                lambda: reference_torch.asarray(expected, pin_memory=False),
            ),
            (
                lambda: torch.asarray(actual, copy=1),
                lambda: reference_torch.asarray(expected, copy=1),
            ),
            (
                lambda: torch.asarray(actual, requires_grad=1),
                lambda: reference_torch.asarray(expected, requires_grad=1),
            ),
            (
                lambda: torch.asarray(actual, dtype=1),
                lambda: reference_torch.asarray(expected, dtype=1),
            ),
            (
                lambda: torch.asarray(actual, device=1.5),
                lambda: reference_torch.asarray(expected, device=1.5),
            ),
            (
                lambda: torch.asarray(actual, device=""),
                lambda: reference_torch.asarray(expected, device=""),
            ),
            (
                lambda: torch.asarray(actual, device="banana"),
                lambda: reference_torch.asarray(expected, device="banana"),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(case=actual_call):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def fresh_warning_observation(self, module_name):
        script = f"""
import json
import warnings

import {module_name} as module

tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    result = module.asarray(tensor)
print(json.dumps({{
    "identity": result is tensor,
    "requires_grad": result.requires_grad,
    "warnings": [
        [warning.category.__name__, str(warning.message)]
        for warning in caught
    ],
}}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            cwd=".",
            stdout=subprocess.PIPE,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_requires_grad_default_warning_matches_pytorch_2_13(self):
        expected = self.fresh_warning_observation("torch")
        self.assertEqual(len(expected["warnings"]), 1)
        self.assertEqual(self.fresh_warning_observation("torch_rs"), expected)


if __name__ == "__main__":
    unittest.main()
