import copy
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

if __package__:
    from .test_log import LOG_DOC, SPECIAL_OUTPUT_BITS, TOP_LEVEL_LOG_DOC, make_cases
else:
    from test_log import LOG_DOC, SPECIAL_OUTPUT_BITS, TOP_LEVEL_LOG_DOC, make_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorLogReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.log differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    @staticmethod
    def call_log(module, tensor, form):
        if form == "method":
            return tensor.log()
        if form == "positional":
            return module.log(tensor)
        if form == "out none":
            return module.log(tensor, out=None)
        if form == "alias and out none":
            return module.log(x=tensor, out=None)
        return module.log(**{form: tensor})

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
            if case == "numerical edges" or (
                isinstance(case, tuple) and case[0] == "numerical edges"
            ):
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                )
            else:
                np.testing.assert_allclose(
                    self.tensor_values(actual),
                    self.tensor_values(expected),
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    def test_scalar_empty_contiguous_offset_strided_and_edges_match_pytorch_2_13(self):
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
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
        for (case, actual_input, actual_stride), (
            expected_case,
            expected_input,
            expected_stride,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(actual_stride, expected_stride)
            for form in forms:
                actual = self.call_log(torch, actual_input, form)
                expected = self.call_log(reference_torch, expected_input, form)
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(
                        expected.data_ptr(), expected_input.data_ptr()
                    )
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )

    def test_requires_grad_inputs_match_pytorch_2_13_inside_no_grad(self):
        actual_leaf = torch.tensor(
            [[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        expected_leaf = reference_torch.tensor(
            [[0.25, 0.5, 1.0], [2.0, 4.0, 8.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]
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
        for form in forms:
            with torch.no_grad():
                actual = self.call_log(torch, actual_input, form)
            with reference_torch.no_grad():
                expected = self.call_log(reference_torch, expected_input, form)
            self.assert_tensor_matches(actual, expected, case=form)
            self.assertIsNone(actual_leaf.grad)
            self.assertIsNone(expected_leaf.grad)

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("torch.log unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.25], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "log")
        bound = tensor.log
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
            "descriptor_copy": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy": copy.deepcopy(descriptor) is descriptor,
            "pickle": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "forking_pickle": tuple(
                pickle.loads(ForkingPickler.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.log(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.log(1, 2),
                    lambda: tensor.log(input=tensor),
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

    def top_level_callable_contract(self, module):
        function = module.log
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
            "owner_callable_identity": owner.log is function,
            "all_count": module.__all__.count("log"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["log"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "forking_pickle_identities": tuple(
                pickle.loads(ForkingPickler.dumps(function, protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    @staticmethod
    def method_mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
tracked = module.tensor([1.0], dtype=module.float32, requires_grad=True)
descriptor = inspect.getattr_static(module.Tensor, "log")
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
    intercepted = tracked.log()
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
        forwarded = tensor.log()

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.log(1)
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
    "args": len(args) == 1 and args[0] is tracked,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
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

    def test_method_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.method_mode_dispatch_observation("torch_rs"),
            self.method_mode_dispatch_observation("torch"),
        )

    @staticmethod
    def top_level_dispatch_observation(module):
        tensor = module.tensor([1.0], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.log
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
            self.error(lambda: torch.log(Override())),
            self.error(lambda: reference_torch.log(Override())),
        )
        self.assertEqual(
            self.error(lambda: torch.log(torch.tensor([1.0]), out=Override())),
            self.error(
                lambda: reference_torch.log(
                    reference_torch.tensor([1.0]), out=Override()
                )
            ),
        )

    def test_top_level_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.log(), lambda: reference_torch.log()),
            (
                lambda: torch.log(actual, actual),
                lambda: reference_torch.log(expected, expected),
            ),
            (
                lambda: torch.log(actual, input=actual),
                lambda: reference_torch.log(expected, input=expected),
            ),
            (
                lambda: torch.log(out=actual),
                lambda: reference_torch.log(out=expected),
            ),
            (
                lambda: torch.log(extra=actual),
                lambda: reference_torch.log(extra=expected),
            ),
            (
                lambda: torch.log(1, extra=True),
                lambda: reference_torch.log(1, extra=True),
            ),
            (lambda: torch.log(input=[]), lambda: reference_torch.log(input=[])),
            (
                lambda: torch.log(actual, out=[]),
                lambda: reference_torch.log(expected, out=[]),
            ),
            (
                lambda: torch.log(actual, extra=True, out=[]),
                lambda: reference_torch.log(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.log(actual, extra=True),
                lambda: reference_torch.log(expected, extra=True),
            ),
            (
                lambda: torch.log(input=actual, a=actual),
                lambda: reference_torch.log(input=expected, a=expected),
            ),
            (
                lambda: torch.log(a=actual, x=actual, out=None),
                lambda: reference_torch.log(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.log(x=actual, a=actual, out=None),
                lambda: reference_torch.log(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.log(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.log(np.zeros((2, 3), dtype=np.float32)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_deliberately_unsupported_boundaries_remain_narrow(self):
        tracked = torch.tensor([1.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): autograd recording is not supported$",
        ):
            tracked.log()
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): autograd recording is not supported$",
        ):
            torch.log(tracked)
        reference_tracked = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32, requires_grad=True
        )
        self.assertTrue(reference_tracked.log().requires_grad)
        self.assertTrue(reference_torch.log(reference_tracked).requires_grad)

        destination = torch.tensor([17.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^log\(\): the 'out' argument is not supported$",
        ):
            torch.log(torch.tensor([1.0]), out=destination)
        self.assertEqual(destination.tolist(), [17.0])
        expected_destination = reference_torch.tensor(
            [17.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.log(reference_torch.tensor([1.0]), out=expected_destination),
            expected_destination,
        )

        self.assertTrue(hasattr(torch.Tensor, "log"))
        self.assertFalse(hasattr(torch.Tensor, "log_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "log_"))
        with self.assertRaisesRegex(
            TypeError, "type 'torch_rs.Tensor' is not an acceptable base type"
        ):
            class TensorSubclass(torch.Tensor):
                pass

        class ReferenceSubclass(reference_torch.Tensor):
            pass

        self.assertTrue(issubclass(ReferenceSubclass, reference_torch.Tensor))
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0], device="cuda")
        self.assertEqual(reference_torch.device("cuda").type, "cuda")


if __name__ == "__main__":
    unittest.main()
