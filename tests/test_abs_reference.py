import builtins
import copy
import ctypes
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAbsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.abs differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_bits(tensor):
        if isinstance(tensor, torch.Tensor):
            return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)
        return tensor.detach().cpu().numpy().reshape(-1).view(np.uint32)

    @staticmethod
    def raw_storage_bits(tensor):
        storage = (ctypes.c_uint32 * tensor.numel()).from_address(tensor.data_ptr())
        return tuple(storage)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_tensor_matches(self, actual, expected, *, case, raw_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            if raw_bits:
                self.assertEqual(
                    self.raw_storage_bits(actual), self.raw_storage_bits(expected)
                )
            else:
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
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
        channels_last = module.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("empty singleton trailing", module.zeros((0, 1), dtype=module.float32)),
            ("empty singleton middle", module.zeros((0, 1, 2), dtype=module.float32)),
            (
                "empty singleton surrounding",
                module.zeros((1, 0, 1), dtype=module.float32),
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            ("IEEE edges", special),
        )

    @staticmethod
    def tensor_from_float32_array(module, values, *, requires_grad=False):
        if module is torch:
            return module.tensor(memoryview(values), requires_grad=requires_grad)
        return module.tensor(
            values, dtype=module.float32, requires_grad=requires_grad
        )

    @classmethod
    def autograd_case(cls, module, case):
        if case == "scalar":
            leaf = module.tensor(
                -3.0, dtype=module.float32, requires_grad=True
            )
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1], None
        if case == "offset":
            leaf = module.tensor(
                [[-2.0, -0.0, 1.0], [2.0, -4.0, 8.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            return leaf, leaf.transpose(0, 1)[1], None
        if case == "noncontiguous":
            values = np.asarray(
                [[-3.0, -0.0, 2.0], [4.0, -5.0, 0.0]], dtype=np.float32
            )
            leaf = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 1), None
        if case == "weighted edge":
            input_bits = np.asarray(
                (
                    0x0000_0000,
                    0x8000_0000,
                    0x0000_0001,
                    0x8000_0001,
                    0x0080_0000,
                    0x8080_0000,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x7F7F_FFFF,
                    0xFF7F_FFFF,
                    0x7F80_0000,
                    0xFF80_0000,
                    0x7F81_2345,
                    0xFF81_2345,
                    0x7FC1_2345,
                    0xFFC5_4321,
                ),
                dtype=np.uint32,
            )
            weight_bits = np.asarray(
                (
                    0x3F80_0000,
                    0xBF80_0000,
                    0x3F00_0000,
                    0x3F00_0000,
                    0x0000_0001,
                    0x0000_0001,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x3E80_0000,
                    0x3E80_0000,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x3F80_0000,
                    0xBF80_0000,
                    0x7FC0_1234,
                    0xFFC0_5678,
                ),
                dtype=np.uint32,
            )
            leaf = cls.tensor_from_float32_array(
                module, input_bits.view(np.float32), requires_grad=True
            )
            weights = cls.tensor_from_float32_array(
                module, weight_bits.view(np.float32)
            )
            return leaf, leaf, weights
        raise AssertionError(f"unknown abs autograd case: {case}")

    @staticmethod
    def supported_calls(tensor):
        return (
            ("abs", tensor.abs),
            ("absolute", tensor.absolute),
            ("operator", lambda: builtins.abs(tensor)),
        )

    @staticmethod
    def call_top_level(module, tensor, name, form):
        function = getattr(module, name)
        if form == "positional":
            return function(tensor)
        if form == "out none":
            return function(tensor, out=None)
        if form == "alias and out none":
            return function(x=tensor, out=None)
        return function(**{form: tensor})

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual_calls = self.supported_calls(actual)
            expected_calls = self.supported_calls(expected)
            for (form, actual_call), (expected_form, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                self.assertEqual(form, expected_form)
                actual_output = actual_call()
                expected_output = expected_call()
                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(case, form),
                    raw_bits=case == "IEEE edges",
                )
                self.assertFalse(actual_output.is_set_to(actual))
                self.assertFalse(expected_output.is_set_to(expected))
                if actual.numel():
                    self.assertNotEqual(actual_output.data_ptr(), actual.data_ptr())
                    self.assertNotEqual(
                        expected_output.data_ptr(), expected.data_ptr()
                    )

    def test_top_level_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for name in ("abs", "absolute"):
            for (case, actual), (expected_case, expected) in zip(
                actual_cases, expected_cases, strict=True
            ):
                self.assertEqual(case, expected_case)
                for form in forms:
                    actual_output = self.call_top_level(torch, actual, name, form)
                    expected_output = self.call_top_level(
                        reference_torch, expected, name, form
                    )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(name, case, form),
                        raw_bits=case == "IEEE edges",
                    )
                    self.assertFalse(actual_output.is_set_to(actual))
                    self.assertFalse(expected_output.is_set_to(expected))
                if actual.numel():
                    self.assertNotEqual(actual_output.data_ptr(), actual.data_ptr())
                    self.assertNotEqual(
                        expected_output.data_ptr(), expected.data_ptr()
                    )

    def test_autograd_scalar_empty_offset_and_noncontiguous_match_pytorch_2_13(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_leaf, actual_input, actual_weights = self.autograd_case(torch, case)
            expected_leaf, expected_input, expected_weights = self.autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.abs()
            expected_output = expected_input.abs()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "forward")
            )

            if actual_weights is None:
                actual_loss = actual_output if case == "scalar" else actual_output.sum()
                expected_loss = (
                    expected_output if case == "scalar" else expected_output.sum()
                )
            else:
                actual_loss = (actual_output * actual_weights).sum()
                expected_loss = (expected_output * expected_weights).sum()
            actual_loss.backward()
            expected_loss.backward()
            self.assert_tensor_matches(
                actual_leaf.grad, expected_leaf.grad, case=(case, "gradient")
            )

    def test_abs_backward_signed_zero_nan_and_infinity_bits_match_pytorch_2_13(self):
        results = []
        for module in (torch, reference_torch):
            leaf, source, weights = self.autograd_case(module, "weighted edge")
            output = source.abs()
            (output * weights).sum().backward()
            results.append((output, leaf.grad))

        self.assert_tensor_matches(
            results[0][0], results[1][0], case="edge forward", raw_bits=True
        )
        self.assert_tensor_matches(
            results[0][1], results[1][1], case="edge gradient", raw_bits=True
        )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("absolute-value method unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module, name):
        tensor = module.tensor([-4.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, name)
        bound = getattr(tensor, name)
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
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: getattr(tensor, name)(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: getattr(tensor, name)(1, 2),
                    lambda: getattr(tensor, name)(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        for name in ("abs", "absolute"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

    @staticmethod
    def alias_and_serialization_contract(module):
        tensor = module.tensor([-4.0], dtype=module.float32)
        abs_descriptor = inspect.getattr_static(module.Tensor, "abs")
        absolute_descriptor = inspect.getattr_static(module.Tensor, "absolute")
        operator_descriptor = inspect.getattr_static(module.Tensor, "__abs__")
        try:
            inspect.getattr_static(abs_descriptor.__objclass__, "__abs__")
        except AttributeError:
            base_has_operator = False
        else:
            base_has_operator = True

        copy_contract = []
        for name in ("abs", "absolute", "__abs__"):
            descriptor = inspect.getattr_static(module.Tensor, name)
            bound = getattr(tensor, name)
            copy_contract.append(
                (
                    name,
                    copy.copy(descriptor) is descriptor,
                    copy.deepcopy(descriptor) is descriptor,
                    copy.copy(bound) is bound,
                    copy.deepcopy(bound) is bound,
                    tuple(
                        (
                            pickle.loads(pickle.dumps(descriptor, protocol))
                            is descriptor,
                            pickle.loads(ForkingPickler.dumps(descriptor, protocol))
                            is descriptor,
                        )
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                    ),
                )
            )

        return {
            "operator_is_abs": operator_descriptor is abs_descriptor,
            "operator_is_not_absolute": operator_descriptor
            is not absolute_descriptor,
            "methods_are_distinct": abs_descriptor is not absolute_descriptor,
            "operator_in_tensor_dict": module.Tensor.__dict__["__abs__"]
            is abs_descriptor,
            "method_names_not_in_tensor_dict": (
                "abs" not in module.Tensor.__dict__,
                "absolute" not in module.Tensor.__dict__,
            ),
            "base_has_operator": base_has_operator,
            "operator_bound_metadata": (
                tensor.__abs__.__name__,
                tensor.__abs__.__qualname__,
                tensor.__abs__.__doc__,
                tensor.__abs__.__text_signature__,
                tensor.__abs__.__module__,
                type(tensor.__abs__).__name__,
            ),
            "copy_contract": tuple(copy_contract),
        }

    def test_alias_identity_copying_and_pickling_match_pytorch_2_13(self):
        self.assertEqual(
            self.alias_and_serialization_contract(torch),
            self.alias_and_serialization_contract(reference_torch),
        )

    def top_level_callable_contract(self, module, name):
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
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertIsNot(torch.abs, torch.absolute)
        self.assertIsNot(reference_torch.abs, reference_torch.absolute)
        for name in ("abs", "absolute"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.top_level_callable_contract(torch, name),
                    self.top_level_callable_contract(reference_torch, name),
                )

    def top_level_dispatch_observation(self, module, name):
        tensor = module.tensor([-4.0], dtype=module.float32)
        tracked = module.tensor([-4.0], dtype=module.float32, requires_grad=True)
        destination = module.tensor([0.0], dtype=module.float32)
        function = getattr(module, name)
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tracked, out=destination), ("input", "out")),
        )
        for call, keyword_names in mode_calls:
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
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(input=value), "input"),
            (lambda value: function(tensor, out=value), "out"),
            (lambda value: function(x=value, out=None), "x"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), out=DerivedOverride())

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function(),
            lambda: function([], out=destination),
            lambda: function(tensor, out=[]),
            lambda: function(tensor, extra=True),
            lambda: function(tensor, tensor),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forward_order,
            tuple(np.asarray(forwarded).reshape(-1)),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_top_level_torch_function_dispatch_matches_pytorch_2_13(self):
        for name in ("abs", "absolute"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.top_level_dispatch_observation(torch, name),
                    self.top_level_dispatch_observation(reference_torch, name),
                )

    def test_top_level_declining_override_diagnostics_match_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        for name in ("abs", "absolute"):
            with self.subTest(name=name, argument="input"):
                self.assert_error_matches(
                    lambda name=name: getattr(torch, name)(Override()),
                    lambda name=name: getattr(reference_torch, name)(Override()),
                )
            with self.subTest(name=name, argument="out"):
                self.assert_error_matches(
                    lambda name=name: getattr(torch, name)(
                        torch.tensor([1.0]), out=Override()
                    ),
                    lambda name=name: getattr(reference_torch, name)(
                        reference_torch.tensor([1.0]), out=Override()
                    ),
                )

    def test_top_level_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([-4.0])
        expected = reference_torch.tensor([-4.0])
        for name in ("abs", "absolute"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            cases = (
                (lambda: actual_function(), lambda: expected_function()),
                (
                    lambda: actual_function(actual, actual),
                    lambda: expected_function(expected, expected),
                ),
                (
                    lambda: actual_function(actual, input=actual),
                    lambda: expected_function(expected, input=expected),
                ),
                (
                    lambda: actual_function(out=actual),
                    lambda: expected_function(out=expected),
                ),
                (
                    lambda: actual_function(1, extra=True),
                    lambda: expected_function(1, extra=True),
                ),
                (
                    lambda: actual_function(input=[]),
                    lambda: expected_function(input=[]),
                ),
                (
                    lambda: actual_function(actual, out=[]),
                    lambda: expected_function(expected, out=[]),
                ),
                (
                    lambda: actual_function(actual, extra=True, out=[]),
                    lambda: expected_function(expected, extra=True, out=[]),
                ),
                (
                    lambda: actual_function(actual, extra=True),
                    lambda: expected_function(expected, extra=True),
                ),
                (
                    lambda: actual_function(input=actual, a=actual),
                    lambda: expected_function(input=expected, a=expected),
                ),
                (
                    lambda: actual_function(a=actual, x=actual, out=None),
                    lambda: expected_function(a=expected, x=expected, out=None),
                ),
                (
                    lambda: actual_function(x=actual, a=actual, out=None),
                    lambda: expected_function(x=expected, a=expected, out=None),
                ),
                (
                    lambda: actual_function(actual, dtype=torch.float32),
                    lambda: expected_function(expected, dtype=reference_torch.float32),
                ),
                (
                    lambda: actual_function(actual, device=torch.device("cpu")),
                    lambda: expected_function(
                        expected, device=reference_torch.device("cpu")
                    ),
                ),
                (
                    lambda: actual_function(np.zeros((2, 3), dtype=np.float32)),
                    lambda: expected_function(np.zeros((2, 3), dtype=np.float32)),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(name=name, case=case):
                    self.assert_error_matches(actual_call, expected_call)

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import builtins
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([-4.0], dtype=module.float32)
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label, order):
        self.label = label
        self.order = order

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.order.append(self.label)
        return func(*args, **(kwargs or {}))

forms = (
    ("abs", "abs"),
    ("absolute", "absolute"),
    ("operator", "abs"),
)

def invoke(form):
    if form == "abs":
        return tensor.abs()
    if form == "absolute":
        return tensor.absolute()
    return builtins.abs(tensor)

def invoke_invalid(form):
    if form == "abs":
        return tensor.abs(1)
    if form == "absolute":
        return tensor.absolute(1)
    return builtins.abs(tensor, 1)

sys.setrecursionlimit(80)
observations = {}
for form, descriptor_name in forms:
    descriptor = inspect.getattr_static(module.Tensor, descriptor_name)
    recording = RecordingMode(marker)
    with recording:
        intercepted = invoke(form)
    function, dispatch_types, args, kwargs = recording.calls[0]

    order = []
    with ForwardingMode("lower", order):
        with ForwardingMode("upper", order):
            forwarded = invoke(form)

    declining = RecordingMode(NotImplemented)
    try:
        with declining:
            invoke(form)
    except Exception as error:
        declining_error = [type(error).__name__, str(error)]
    else:
        declining_error = None

    invalid = RecordingMode(marker)
    try:
        with invalid:
            invoke_invalid(form)
    except Exception as error:
        invalid_error = [type(error).__name__, str(error)]
    else:
        invalid_error = None

    observations[form] = {
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
        "forwarded": forwarded.tolist(),
        "declining_error": declining_error,
        "declining_calls": len(declining.calls),
        "invalid_error": invalid_error,
        "invalid_calls": len(invalid.calls),
        "stack_depth": len(module.overrides._get_current_function_mode_stack()),
    }

print(json.dumps(observations, sort_keys=True))
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

    def test_autograd_no_grad_detach_and_unsupported_surface_match_pytorch_2_13(self):
        actual = torch.tensor([-2.0, -0.0, 3.0], requires_grad=True)
        expected = reference_torch.tensor(
            [-2.0, -0.0, 3.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_calls = (
            *self.supported_calls(actual),
            *(
                (
                    f"torch.{name} {form}",
                    lambda name=name, form=form: self.call_top_level(
                        torch, actual, name, form
                    ),
                )
                for name in ("abs", "absolute")
                for form in (
                    "positional",
                    "input",
                    "x",
                    "a",
                    "x1",
                    "out none",
                    "alias and out none",
                )
            ),
        )
        expected_calls = (
            *self.supported_calls(expected),
            *(
                (
                    f"torch.{name} {form}",
                    lambda name=name, form=form: self.call_top_level(
                        reference_torch, expected, name, form
                    ),
                )
                for name in ("abs", "absolute")
                for form in (
                    "positional",
                    "input",
                    "x",
                    "a",
                    "x1",
                    "out none",
                    "alias and out none",
                )
            ),
        )
        for (form, actual_call), (expected_form, expected_call) in zip(
            actual_calls, expected_calls, strict=True
        ):
            self.assertEqual(form, expected_form)
            with self.subTest(form=form, mode="recording"):
                actual_output = actual_call()
                expected_output = expected_call()
                self.assert_tensor_matches(
                    actual_output, expected_output, case=(form, "recording")
                )
                actual_output.sum().backward()
                expected_output.sum().backward()
                self.assert_tensor_matches(
                    actual.grad, expected.grad, case=(form, "accumulated gradient")
                )

            with torch.no_grad():
                actual_no_grad = actual_call()
            with reference_torch.no_grad():
                expected_no_grad = expected_call()
            self.assert_tensor_matches(
                actual_no_grad, expected_no_grad, case=(form, "no_grad")
            )

        actual_detached = actual.detach()
        expected_detached = expected.detach()
        actual_detached_calls = (
            *self.supported_calls(actual_detached),
            *(
                (
                    f"torch.{name} {form}",
                    lambda name=name, form=form: self.call_top_level(
                        torch, actual_detached, name, form
                    ),
                )
                for name in ("abs", "absolute")
                for form in (
                    "positional",
                    "input",
                    "x",
                    "a",
                    "x1",
                    "out none",
                    "alias and out none",
                )
            ),
        )
        expected_detached_calls = (
            *self.supported_calls(expected_detached),
            *(
                (
                    f"torch.{name} {form}",
                    lambda name=name, form=form: self.call_top_level(
                        reference_torch, expected_detached, name, form
                    ),
                )
                for name in ("abs", "absolute")
                for form in (
                    "positional",
                    "input",
                    "x",
                    "a",
                    "x1",
                    "out none",
                    "alias and out none",
                )
            ),
        )
        for (form, actual_call), (expected_form, expected_call) in zip(
            actual_detached_calls,
            expected_detached_calls,
            strict=True,
        ):
            self.assertEqual(form, expected_form)
            self.assert_tensor_matches(
                actual_call(), expected_call(), case=(form, "detached")
            )

        for name in ("abs", "absolute"):
            self.assertTrue(hasattr(torch, name))
            self.assertTrue(hasattr(reference_torch, name))
        self.assertTrue(hasattr(torch.Tensor, "absolute"))
        self.assertTrue(hasattr(torch.Tensor, "__abs__"))
        for name in ("abs_", "absolute_"):
            self.assertFalse(hasattr(torch.Tensor, name))
            self.assertTrue(hasattr(reference_torch.Tensor, name))
        self.assertFalse(hasattr(torch, "abs_"))
        self.assertTrue(hasattr(reference_torch, "abs_"))
        self.assertFalse(hasattr(torch, "absolute_"))
        self.assertFalse(hasattr(reference_torch, "absolute_"))


if __name__ == "__main__":
    unittest.main()
