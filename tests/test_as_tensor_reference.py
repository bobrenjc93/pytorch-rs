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
class AsTensorReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "as_tensor differentials require pinned PyTorch 2.13.0"
            )

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
            {"dtype": module.float32, "device": module.device("cpu")},
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

    def as_tensor_identity_contract(self, module, tensor, options):
        before = self.tensor_state(module, tensor)
        result = module.as_tensor(tensor, **options)
        after = self.tensor_state(module, tensor)
        return {
            "same_object": result is tensor,
            "same_pointer": result.data_ptr() == before["data_ptr"],
            "same_logical_storage": result.is_set_to(tensor),
            "result_state": self.comparable_tensor_state(module, result),
            "source_unchanged": before == after,
        }

    def as_tensor_float_scalar_contract(self, module, value, options):
        first = module.as_tensor(value, **options)
        second = module.as_tensor(value, **options)
        return {
            "fresh_object": first is not second,
            "fresh_storage": first.data_ptr() != second.data_ptr(),
            "not_set_to_duplicate": not first.is_set_to(second),
            "state": self.comparable_tensor_state(module, first),
            "numel": first.numel(),
            "grad_is_none": first.grad is None,
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
                    actual_contract = self.as_tensor_identity_contract(
                        torch, actual, actual_kwargs
                    )
                    expected_contract = self.as_tensor_identity_contract(
                        reference_torch, expected, expected_kwargs
                    )
                    self.assertEqual(actual_contract, expected_contract)

    def test_python_float_scalar_creation_matches_pytorch_2_13(self):
        values = (
            0.0,
            -0.0,
            1.25,
            -3.5,
            1e39,
            -1e39,
            float("inf"),
            float("-inf"),
            float("nan"),
        )
        actual_options = self.option_cases(torch)
        expected_options = self.option_cases(reference_torch)
        for value in values:
            for actual_kwargs, expected_kwargs in zip(
                actual_options, expected_options, strict=True
            ):
                with self.subTest(value=repr(value), options=actual_kwargs):
                    actual_contract = self.as_tensor_float_scalar_contract(
                        torch, value, actual_kwargs
                    )
                    expected_contract = self.as_tensor_float_scalar_contract(
                        reference_torch, value, expected_kwargs
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
            result = module.as_tensor(source, dtype=module.float32, device="cpu")
            after = self.tensor_state(module, result)
            result.sum().backward()
            outcomes.append(
                (
                    result is source,
                    before["data_ptr"] == after["data_ptr"],
                    {key: value for key, value in before.items() if key != "data_ptr"},
                    {key: value for key, value in after.items() if key != "data_ptr"},
                    np.asarray(leaf.grad).reshape(-1).tolist(),
                    module.as_tensor(leaf.grad) is leaf.grad,
                )
            )

        self.assertEqual(outcomes[0], outcomes[1])

    def callable_contract(self, module):
        function = module.as_tensor
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
            "owner_callable_identity": owner.as_tensor is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("as_tensor"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["as_tensor"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_imports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(self.callable_contract(torch), self.callable_contract(reference_torch))

        old = torch.as_tensor
        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.as_tensor, old)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.as_tensor, old)

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
            call = lambda: module.as_tensor(tensor)
        elif case == "keyword":
            call = lambda: module.as_tensor(data=tensor, dtype=module.float32)
        elif case == "scalar":
            data = 1.25
            call = lambda: module.as_tensor(data)
        elif case == "sequence":
            data = [1.0, 2.0]
            call = lambda: module.as_tensor(data)
        else:
            data = [1.0]
            call = lambda: module.as_tensor(data, device="cuda")

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
                forwarded = module.as_tensor(data=tensor, dtype=module.float32)

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        try:
            with DecliningMode():
                module.as_tensor(tensor)
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
            "function_identity": function is module.as_tensor,
            "types_empty": dispatch_types == (),
            "args_len": len(args),
            "kwargs_keys": None if kwargs is None else tuple(kwargs.keys()),
            "forwarding_order": order,
            "forwarded_is_tensor": forwarded is tensor,
            "declining_error": declining_error,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        for case in ("positional", "keyword", "scalar", "sequence", "device"):
            with self.subTest(case=case):
                self.assertEqual(
                    self.mode_dispatch_observation(torch, case),
                    self.mode_dispatch_observation(reference_torch, case),
                )

    def test_binding_errors_match_pytorch_2_13_for_exposed_schema(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.as_tensor(), lambda: reference_torch.as_tensor()),
            (
                lambda: torch.as_tensor(actual, actual),
                lambda: reference_torch.as_tensor(expected, expected),
            ),
            (
                lambda: torch.as_tensor(actual, data=actual),
                lambda: reference_torch.as_tensor(expected, data=expected),
            ),
            (
                lambda: torch.as_tensor(actual, out=None),
                lambda: reference_torch.as_tensor(expected, out=None),
            ),
            (
                lambda: torch.as_tensor(actual, pin_memory=False),
                lambda: reference_torch.as_tensor(expected, pin_memory=False),
            ),
            (
                lambda: torch.as_tensor(actual, copy=False),
                lambda: reference_torch.as_tensor(expected, copy=False),
            ),
            (
                lambda: torch.as_tensor(1.0, requires_grad=True),
                lambda: reference_torch.as_tensor(1.0, requires_grad=True),
            ),
            (
                lambda: torch.as_tensor(actual, dtype=1),
                lambda: reference_torch.as_tensor(expected, dtype=1),
            ),
            (
                lambda: torch.as_tensor(actual, device=1.5),
                lambda: reference_torch.as_tensor(expected, device=1.5),
            ),
            (
                lambda: torch.as_tensor(actual, device=""),
                lambda: reference_torch.as_tensor(expected, device=""),
            ),
            (
                lambda: torch.as_tensor(actual, device="banana"),
                lambda: reference_torch.as_tensor(expected, device="banana"),
            ),
        )
        for actual_call, expected_call in cases:
            with self.subTest(case=actual_call):
                with self.assertRaises(Exception) as actual_raised:
                    actual_call()
                with self.assertRaises(Exception) as expected_raised:
                    expected_call()
                self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
                self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))


if __name__ == "__main__":
    unittest.main()
