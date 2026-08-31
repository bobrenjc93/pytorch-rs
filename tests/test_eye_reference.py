import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EyeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("eye differentials require pinned PyTorch 2.13.0")

    def tensor_observation(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
        }

    def tensor_metadata(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
        }

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_square_rectangular_and_zero_size_results_match_pytorch_2_13(self):
        cases = (
            lambda module: module.eye(3),
            lambda module: module.eye(2, 4),
            lambda module: module.eye(n=4, m=2),
            lambda module: module.eye(n=0),
            lambda module: module.eye(3, 0),
            lambda module: module.eye(0, 3),
        )
        for create in cases:
            with self.subTest(create=create):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

    def test_dtype_device_and_requires_grad_metadata_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {
                "dtype": module.float32,
                "device": module.device("cpu"),
                "requires_grad": True,
            },
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            with self.subTest(options=actual_options):
                with torch.no_grad():
                    actual = torch.eye(2, 3, **actual_options)
                with reference_torch.no_grad():
                    expected = reference_torch.eye(2, 3, **expected_options)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

    def test_default_out_layout_and_fresh_storage_match_pytorch_2_13(self):
        cases = (
            lambda module: module.eye(3, out=None),
            lambda module: module.eye(2, 4, layout=None),
            lambda module: module.eye(4, 2, out=None, layout=module.strided),
            lambda module: module.eye(0, out=None, layout=module.strided),
            lambda module: module.eye(3, 0, out=None, layout=None),
            lambda module: module.eye(0, 3, out=None, layout=module.strided),
        )
        for create in cases:
            with self.subTest(create=create):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

        actual_first = torch.eye(2, out=None, layout=torch.strided)
        actual_second = torch.eye(2, out=None, layout=torch.strided)
        expected_first = reference_torch.eye(
            2, out=None, layout=reference_torch.strided
        )
        expected_second = reference_torch.eye(
            2, out=None, layout=reference_torch.strided
        )
        self.assertEqual(
            actual_first.data_ptr() != actual_second.data_ptr(),
            expected_first.data_ptr() != expected_second.data_ptr(),
        )

    def normalize_mode_value(self, module, value):
        if value is module.strided:
            return "torch.strided"
        if isinstance(value, dict):
            return {
                key: self.normalize_mode_value(module, item)
                for key, item in value.items()
            }
        return value

    def mode_dispatch_observation(self, module):
        function = module.eye
        marker = object()
        intercepted = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func is function,
                        types,
                        args,
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        for call in (
            lambda: function(2),
            lambda: function(2, out=None, layout=module.strided),
            lambda: function(2**63, out=None, layout=module.strided),
            lambda: function(n=2, m=3, out=None, layout=module.strided),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
                restored_inside = (
                    module.overrides._get_current_function_mode_stack() == [mode]
                )
            intercepted.append(
                (
                    result is marker,
                    tuple(
                        (
                            func_is_function,
                            types,
                            args,
                            self.normalize_mode_value(module, kwargs),
                            stack_depth,
                        )
                        for func_is_function, types, args, kwargs, stack_depth in mode.calls
                    ),
                    restored_inside,
                    module.overrides._get_current_function_mode_stack() == [],
                )
        )

        forwarding_events = []
        test_case = self

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_events.append(
                    (
                        self.label,
                        tuple(
                            mode.label
                            for mode in module.overrides._get_current_function_mode_stack()
                        ),
                        func is function,
                        types,
                        args,
                        test_case.normalize_mode_value(module, kwargs),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                forwarded = function(2, out=None, layout=module.strided)
                nested_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, upper]
                )
            lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )
        stack_empty = module.overrides._get_current_function_mode_stack() == []

        native_error_mode = ForwardingMode("native-error")
        with native_error_mode:
            try:
                function(-1, out=None, layout=module.strided)
            except Exception as error:
                native_error = (type(error).__name__, str(error), error.args)
            else:
                native_error = None
            native_error_restored = (
                module.overrides._get_current_function_mode_stack()
                == [native_error_mode]
            )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            try:
                function(2)
            except Exception as error:
                declining_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                    error.args[1:] if len(error.args) > 1 else (),
                )
            else:
                declining_error = None
            declining_restored = (
                module.overrides._get_current_function_mode_stack() == [declining]
            )

        return (
            intercepted,
            forwarding_events,
            self.tensor_observation(module, forwarded),
            nested_restored,
            lower_restored,
            stack_empty,
            native_error,
            native_error_restored,
            declining_error,
            declining_restored,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation(torch),
            self.mode_dispatch_observation(reference_torch),
        )

    def test_integer_protocol_inputs_match_pytorch_2_13(self):
        cases = (
            lambda: (IntSubclass(2),),
            lambda: (np.int64(2), np.uint32(3)),
            lambda: (IndexDimension(2), IndexDimension(3)),
        )
        for argument_factory in cases:
            actual_arguments = argument_factory()
            expected_arguments = argument_factory()
            with self.subTest(arguments=actual_arguments):
                actual = torch.eye(*actual_arguments)
                expected = reference_torch.eye(*expected_arguments)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )

    def test_negative_and_dimension_overflow_errors_match_pytorch_2_13(self):
        exact_cases = (
            lambda module: module.eye(-1),
            lambda module: module.eye(1, -2),
            lambda module: module.eye(2**63),
            lambda module: module.eye(1, 2**63),
            lambda module: module.eye(np.uint64(2**63)),
            lambda module: module.eye(IndexDimension(2**63)),
            lambda module: module.eye(sys.maxsize, 2),
            lambda module: module.eye(sys.maxsize, 3),
            lambda module: module.eye(sys.maxsize, 1),
            lambda module: module.eye(sys.maxsize // 4 + 1, 1),
        )
        for call in exact_cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch), lambda: call(reference_torch)
                )

    def test_zero_size_huge_dimension_metadata_matches_pytorch_2_13(self):
        cases = (
            lambda module: module.eye(0, sys.maxsize),
            lambda module: module.eye(sys.maxsize, 0),
        )
        for create in cases:
            with self.subTest(create=create):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_metadata(torch, actual),
                    self.tensor_metadata(reference_torch, expected),
                )

    def callable_contract(self, module):
        function = module.eye
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
            "owner_callable_identity": owner.eye is function,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("eye"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["eye"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
