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

if __package__:
    from .test_sigmoid import SPECIAL_INPUT_BITS, SPECIAL_OUTPUT_BITS
else:
    from test_sigmoid import SPECIAL_INPUT_BITS, SPECIAL_OUTPUT_BITS

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSigmoidReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.sigmoid differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

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
            np.testing.assert_allclose(
                self.tensor_values(actual),
                self.tensor_values(expected),
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
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
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            ("empty singleton trailing", module.zeros((0, 1), dtype=module.float32)),
            (
                "empty singleton middle",
                module.zeros((0, 1, 2), dtype=module.float32),
            ),
            (
                "empty singleton surrounding",
                module.zeros((1, 0, 1), dtype=module.float32),
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            (
                "numerical edges",
                module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            ),
        )

    @staticmethod
    def call_top_level_sigmoid(module, tensor, form):
        if form == "positional":
            return module.sigmoid(tensor)
        if form == "out none":
            return module.sigmoid(tensor, out=None)
        if form == "alias and out none":
            return module.sigmoid(x=tensor, out=None)
        return module.sigmoid(**{form: tensor})

    def test_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual = actual_input.sigmoid()
            expected = expected_input.sigmoid()
            self.assert_tensor_matches(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )

    def test_top_level_values_layouts_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            for form in forms:
                actual = self.call_top_level_sigmoid(torch, actual_input, form)
                expected = self.call_top_level_sigmoid(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )

    def test_seeded_float32_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x5160_213)
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(
                tuple(int(value) for value in rng.integers(0, 9, size=rank))
            )

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-100.0, 100.0, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
            values = values.reshape(shape)

            actual_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            actual = actual_input.sigmoid()
            expected = expected_input.sigmoid()

            with self.subTest(case=case, shape=shape):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                np.testing.assert_allclose(
                    self.tensor_values(actual),
                    self.tensor_values(expected),
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.sigmoid unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.25], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "sigmoid")
        bound = tensor.sigmoid
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
                    lambda: tensor.sigmoid(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.sigmoid(1, 2),
                    lambda: tensor.sigmoid(input=tensor),
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
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.25], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "sigmoid")
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
    intercepted = tensor.sigmoid()
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
        forwarded = tensor.sigmoid()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.sigmoid()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.sigmoid(1)
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

    def top_level_callable_contract(self, module):
        function = module.sigmoid
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
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.sigmoid is function,
            "all_count": module.__all__.count("sigmoid"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sigmoid"] is function,
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
    def top_level_dispatch_observation(module):
        tensor = module.tensor([1.25], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.sigmoid
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
            dispatched, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    dispatched is function,
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
            dispatched, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    dispatched is function,
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
            forwarded.tolist(),
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
            self.error(lambda: torch.sigmoid(Override())),
            self.error(lambda: reference_torch.sigmoid(Override())),
        )
        self.assertEqual(
            self.error(
                lambda: torch.sigmoid(torch.tensor([1.25]), out=Override())
            ),
            self.error(
                lambda: reference_torch.sigmoid(
                    reference_torch.tensor([1.25]), out=Override()
                )
            ),
        )

    def test_top_level_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.25])
        expected = reference_torch.tensor([1.25])
        cases = (
            (lambda: torch.sigmoid(), lambda: reference_torch.sigmoid()),
            (
                lambda: torch.sigmoid(actual, actual),
                lambda: reference_torch.sigmoid(expected, expected),
            ),
            (
                lambda: torch.sigmoid(actual, input=actual),
                lambda: reference_torch.sigmoid(expected, input=expected),
            ),
            (
                lambda: torch.sigmoid(out=actual),
                lambda: reference_torch.sigmoid(out=expected),
            ),
            (
                lambda: torch.sigmoid(extra=actual),
                lambda: reference_torch.sigmoid(extra=expected),
            ),
            (
                lambda: torch.sigmoid(1, extra=True),
                lambda: reference_torch.sigmoid(1, extra=True),
            ),
            (
                lambda: torch.sigmoid(input=[]),
                lambda: reference_torch.sigmoid(input=[]),
            ),
            (
                lambda: torch.sigmoid(actual, out=[]),
                lambda: reference_torch.sigmoid(expected, out=[]),
            ),
            (
                lambda: torch.sigmoid(actual, extra=True, out=[]),
                lambda: reference_torch.sigmoid(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.sigmoid(actual, extra=True),
                lambda: reference_torch.sigmoid(expected, extra=True),
            ),
            (
                lambda: torch.sigmoid(input=actual, a=actual),
                lambda: reference_torch.sigmoid(input=expected, a=expected),
            ),
            (
                lambda: torch.sigmoid(a=actual, x=actual, out=None),
                lambda: reference_torch.sigmoid(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.sigmoid(x=actual, a=actual, out=None),
                lambda: reference_torch.sigmoid(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.sigmoid(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.sigmoid(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_inference_only_autograd_boundary_is_explicit(self):
        actual_leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        expected_leaf = reference_torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32).reshape(2, 3, 4),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_input = actual_leaf.transpose(0, 2)[1]
        expected_input = expected_leaf.transpose(0, 2)[1]

        for call in (
            actual_input.sigmoid,
            lambda: torch.sigmoid(actual_input),
            lambda: torch.sigmoid(actual_input, out=None),
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"^sigmoid\(\): autograd recording is not supported$"
            ):
                call()
        self.assertTrue(expected_input.sigmoid().requires_grad)
        self.assertTrue(reference_torch.sigmoid(expected_input).requires_grad)

        with torch.no_grad():
            actual_no_grad = actual_input.sigmoid()
        with reference_torch.no_grad():
            expected_no_grad = expected_input.sigmoid()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        with torch.no_grad():
            actual_top_level_no_grad = torch.sigmoid(actual_input, out=None)
        with reference_torch.no_grad():
            expected_top_level_no_grad = reference_torch.sigmoid(
                expected_input, out=None
            )
        self.assert_tensor_matches(
            actual_top_level_no_grad,
            expected_top_level_no_grad,
            case="top-level no_grad",
        )

        actual_detached = actual_input.detach().sigmoid()
        expected_detached = expected_input.detach().sigmoid()
        self.assert_tensor_matches(actual_detached, expected_detached, case="detached")

    def test_concrete_out_functional_and_inplace_boundaries_remain_explicit(self):
        actual_input = torch.tensor([1.25, -1.25], requires_grad=True)
        actual_out = torch.tensor([17.0, 19.0])
        actual_out_pointer = actual_out.data_ptr()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): the 'out' argument is not supported$",
        ):
            torch.sigmoid(actual_input, out=actual_out)
        self.assertEqual(actual_out.data_ptr(), actual_out_pointer)
        self.assertEqual(actual_out.tolist(), [17.0, 19.0])
        self.assertIsNone(actual_input.grad)

        expected_input = reference_torch.tensor(
            [1.25, -1.25], dtype=reference_torch.float32
        )
        expected_out = reference_torch.tensor(
            [17.0, 19.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.sigmoid(expected_input, out=expected_out),
            expected_out,
        )
        np.testing.assert_allclose(
            expected_out.numpy(),
            np.asarray([0.7772999, 0.22270013], dtype=np.float32),
            rtol=1.0e-6,
        )

        self.assertTrue(hasattr(torch, "sigmoid"))
        self.assertTrue(hasattr(reference_torch, "sigmoid"))
        self.assertIn("sigmoid", torch.__all__)
        self.assertFalse(hasattr(torch.nn.functional, "sigmoid"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(torch, "sigmoid_"))
        self.assertNotIn("sigmoid_", torch.__all__)


if __name__ == "__main__":
    unittest.main()
