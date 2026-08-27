import copy
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
class TensorRsqrtReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.rsqrt differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def tensor_cases(module):
        base = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
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
                0x4080_0000,
                0xC080_0000,
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
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            module.zeros((0, 1), dtype=module.float32),
            module.zeros((0, 1, 2), dtype=module.float32),
            module.zeros((1, 0, 1), dtype=module.float32),
            strided[1],
            strided,
            module.tensor(memoryview(special_bits.view(np.float32))),
        )

    @staticmethod
    def call_top_level(module, tensor, form):
        if form == "positional":
            return module.rsqrt(tensor)
        if form == "out none":
            return module.rsqrt(tensor, out=None)
        if form == "alias and out none":
            return module.rsqrt(x=tensor, out=None)
        return module.rsqrt(**{form: tensor})

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            actual_before = (
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32).copy()
            )
            expected_before = (
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )
            actual_output = actual.rsqrt()
            expected_output = expected.rsqrt()
            self.assert_tensor_matches(actual_output, expected_output, case=case)
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                actual_before,
            )
            np.testing.assert_array_equal(
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
                expected_before,
            )

    def test_seeded_float32_bits_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xA513_213)
        input_bits = rng.integers(0, 2**32, size=4096, dtype=np.uint32)
        for case, shape in enumerate(((4096,), (64, 64), (8, 16, 32))):
            values = input_bits.view(np.float32)
            actual = torch.tensor(memoryview(values)).reshape(shape)
            expected = reference_torch.tensor(memoryview(values)).reshape(shape)
            self.assert_tensor_matches(
                actual.rsqrt(), expected.rsqrt(), case=(case, shape)
            )

    def test_top_level_forms_and_unary_layouts_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for case, (actual_input, expected_input) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for form in forms:
                actual = self.call_top_level(torch, actual_input, form)
                expected = self.call_top_level(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(actual, expected, case=(case, form))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.rsqrt unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([4.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "rsqrt")
        bound = tensor.rsqrt
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
                    lambda: tensor.rsqrt(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.rsqrt(1, 2),
                    lambda: tensor.rsqrt(input=tensor),
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
        function = module.rsqrt
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
            "owner_callable_identity": owner.rsqrt is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("rsqrt"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["rsqrt"] is function,
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
tensor = module.tensor([4.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "rsqrt")
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
    intercepted = tensor.rsqrt()
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
        forwarded = tensor.rsqrt()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.rsqrt()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.rsqrt(1)
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

    @staticmethod
    def top_level_dispatch_observation(module):
        tensor = module.tensor([4.0], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.rsqrt
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        for call in (
            lambda: function(tensor),
            lambda: function(input=tensor),
            lambda: function(x=tensor),
            lambda: function(input=tensor, out=destination),
        ):
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
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call in (
            lambda value: function(value),
            lambda value: function(input=value),
            lambda value: function(tensor, out=value),
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
                )
            )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)

        return (
            mode_observations,
            override_observations,
            order,
            forwarded.tolist(),
        )

    def test_top_level_override_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_dispatch_observation(torch),
            self.top_level_dispatch_observation(reference_torch),
        )

    def test_top_level_binding_and_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.rsqrt(), lambda: reference_torch.rsqrt()),
            (
                lambda: torch.rsqrt(actual, actual),
                lambda: reference_torch.rsqrt(expected, expected),
            ),
            (
                lambda: torch.rsqrt(actual, input=actual),
                lambda: reference_torch.rsqrt(expected, input=expected),
            ),
            (
                lambda: torch.rsqrt(out=actual),
                lambda: reference_torch.rsqrt(out=expected),
            ),
            (
                lambda: torch.rsqrt(1, extra=True),
                lambda: reference_torch.rsqrt(1, extra=True),
            ),
            (
                lambda: torch.rsqrt(input=[]),
                lambda: reference_torch.rsqrt(input=[]),
            ),
            (
                lambda: torch.rsqrt(actual, out=[]),
                lambda: reference_torch.rsqrt(expected, out=[]),
            ),
            (
                lambda: torch.rsqrt(actual, extra=True, out=[]),
                lambda: reference_torch.rsqrt(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.rsqrt(actual, extra=True),
                lambda: reference_torch.rsqrt(expected, extra=True),
            ),
            (
                lambda: torch.rsqrt(input=actual, a=actual),
                lambda: reference_torch.rsqrt(input=expected, a=expected),
            ),
            (
                lambda: torch.rsqrt(a=actual, x=actual, out=None),
                lambda: reference_torch.rsqrt(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.rsqrt(x=actual, a=actual, out=None),
                lambda: reference_torch.rsqrt(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.rsqrt(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.rsqrt(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                actual_error = self.error(actual_call)
                expected_error = self.error(expected_call)
                self.assertEqual(actual_error, expected_error)

    def test_inference_only_and_unsupported_boundaries_remain_explicit(self):
        actual = torch.tensor([4.0], requires_grad=True)
        for call in (actual.rsqrt, lambda: torch.rsqrt(actual)):
            with self.assertRaisesRegex(
                RuntimeError,
                r"^rsqrt\(\): autograd recording is not supported$",
            ):
                call()

        expected = reference_torch.tensor([4.0], requires_grad=True)
        self.assertTrue(expected.rsqrt().requires_grad)
        self.assertTrue(reference_torch.rsqrt(expected).requires_grad)

        with torch.no_grad():
            actual_no_grad = torch.rsqrt(actual, out=None)
        with reference_torch.no_grad():
            expected_no_grad = reference_torch.rsqrt(expected, out=None)
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^rsqrt\(\): the 'out' argument is not supported$",
        ):
            torch.rsqrt(torch.tensor([4.0]), out=destination)
        self.assertEqual(destination.tolist(), [17.0])

        self.assertTrue(hasattr(torch, "rsqrt"))
        self.assertIn("rsqrt", torch.__all__)
        self.assertTrue(hasattr(reference_torch, "rsqrt"))
        self.assertFalse(hasattr(torch.Tensor, "rsqrt_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "rsqrt_"))


if __name__ == "__main__":
    unittest.main()
