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
    def call_abs(module, tensor, form):
        if form == "method":
            return tensor.abs()
        if form == "positional":
            return module.abs(tensor)
        if form == "out none":
            return module.abs(tensor, out=None)
        if form == "alias and out none":
            return module.abs(x=tensor, out=None)
        return module.abs(**{form: tensor})

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "method",
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for (case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            for form in forms:
                actual_output = self.call_abs(torch, actual, form)
                expected_output = self.call_abs(reference_torch, expected, form)
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

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.abs unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([-4.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "abs")
        bound = tensor.abs
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
                    lambda: tensor.abs(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.abs(1, 2),
                    lambda: tensor.abs(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    def top_level_callable_contract(self, module):
        function = module.abs
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
            "owner_callable_identity": owner.abs is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("abs"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["abs"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([-4.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "abs")
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
    intercepted = tensor.abs()
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
        forwarded = tensor.abs()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.abs()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.abs(1)
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

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
    "forwarded": forwarded.tolist(),
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "invalid_error": invalid_error,
    "invalid_calls": len(invalid.calls),
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}, sort_keys=True))
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

    def top_level_dispatch_observation(self, module):
        tensor = module.tensor([-4.0])
        destination = module.tensor([17.0])
        function = module.abs
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
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
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

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)
        forwarded_observation = (
            forwarded.requires_grad,
            forwarded.is_leaf,
            tuple(forwarded.shape),
            forwarded.stride(),
            forwarded.storage_offset(),
            tuple(self.tensor_bits(forwarded)),
        )

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
            forwarding_order,
            forwarded_observation,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_top_level_modes_and_subclass_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_dispatch_observation(torch),
            self.top_level_dispatch_observation(reference_torch),
        )

    def test_top_level_declining_override_diagnostics_match_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assertEqual(
            self.error(lambda: torch.abs(Override())),
            self.error(lambda: reference_torch.abs(Override())),
        )
        self.assertEqual(
            self.error(lambda: torch.abs(torch.tensor([-4.0]), out=Override())),
            self.error(
                lambda: reference_torch.abs(
                    reference_torch.tensor([-4.0]), out=Override()
                )
            ),
        )

    def test_top_level_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([-4.0])
        expected = reference_torch.tensor([-4.0])
        cases = (
            (lambda: torch.abs(), lambda: reference_torch.abs()),
            (
                lambda: torch.abs(actual, actual),
                lambda: reference_torch.abs(expected, expected),
            ),
            (
                lambda: torch.abs(actual, input=actual),
                lambda: reference_torch.abs(expected, input=expected),
            ),
            (
                lambda: torch.abs(out=actual),
                lambda: reference_torch.abs(out=expected),
            ),
            (
                lambda: torch.abs(extra=actual),
                lambda: reference_torch.abs(extra=expected),
            ),
            (
                lambda: torch.abs(1, extra=True),
                lambda: reference_torch.abs(1, extra=True),
            ),
            (lambda: torch.abs(input=[]), lambda: reference_torch.abs(input=[])),
            (
                lambda: torch.abs(actual, out=[]),
                lambda: reference_torch.abs(expected, out=[]),
            ),
            (
                lambda: torch.abs(actual, extra=True, out=[]),
                lambda: reference_torch.abs(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.abs(actual, extra=True),
                lambda: reference_torch.abs(expected, extra=True),
            ),
            (
                lambda: torch.abs(input=actual, a=actual),
                lambda: reference_torch.abs(input=expected, a=expected),
            ),
            (
                lambda: torch.abs(a=actual, x=actual, out=None),
                lambda: reference_torch.abs(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.abs(x=actual, a=actual, out=None),
                lambda: reference_torch.abs(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.abs(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.abs(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_inference_boundary_and_unsupported_surface_are_explicit(self):
        actual = torch.tensor([-2.0, -0.0, 3.0], requires_grad=True)
        expected = reference_torch.tensor(
            [-2.0, -0.0, 3.0], dtype=reference_torch.float32, requires_grad=True
        )
        for form in ("method", "positional", "input", "x", "a", "x1", "out none"):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^abs\(\): autograd recording is not supported$",
                ):
                    self.call_abs(torch, actual, form)
                self.assertTrue(
                    self.call_abs(reference_torch, expected, form).requires_grad
                )

            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    actual_no_grad = self.call_abs(torch, actual, form)
                with reference_torch.no_grad():
                    expected_no_grad = self.call_abs(
                        reference_torch, expected, form
                    )
                self.assert_tensor_matches(
                    actual_no_grad,
                    expected_no_grad,
                    case=(form, "no_grad"),
                )

            with self.subTest(form=form, mode="detached"):
                actual_detached = self.call_abs(torch, actual.detach(), form)
                expected_detached = self.call_abs(
                    reference_torch, expected.detach(), form
                )
                self.assert_tensor_matches(
                    actual_detached,
                    expected_detached,
                    case=(form, "detached"),
                )

        destination = torch.tensor([17.0, 19.0, 23.0])
        pointer = destination.data_ptr()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^abs\(\): the 'out' argument is not supported$",
        ):
            torch.abs(actual, out=destination)
        self.assertEqual(destination.data_ptr(), pointer)
        self.assertEqual(destination.tolist(), [17.0, 19.0, 23.0])
        self.assertIsNone(actual.grad)

        expected_input = reference_torch.tensor(
            [-2.0, -0.0, 3.0], dtype=reference_torch.float32
        )
        expected_out = reference_torch.tensor(
            [17.0, 19.0, 23.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.abs(expected_input, out=expected_out), expected_out
        )
        self.assertEqual(expected_out.tolist(), [2.0, 0.0, 3.0])

        self.assertTrue(hasattr(torch, "abs"))
        self.assertTrue(hasattr(reference_torch, "abs"))
        self.assertIn("abs", torch.__all__)
        self.assertFalse(hasattr(torch, "absolute"))
        self.assertTrue(hasattr(reference_torch, "absolute"))
        for name in ("absolute", "abs_", "absolute_"):
            self.assertFalse(hasattr(torch.Tensor, name))
            self.assertTrue(hasattr(reference_torch.Tensor, name))
        for name in ("abs_", "absolute_"):
            self.assertFalse(hasattr(torch, name))
            self.assertNotIn(name, torch.__all__)


if __name__ == "__main__":
    unittest.main()
