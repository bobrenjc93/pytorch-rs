import inspect
import json
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIntrospectionReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_introspection_matches(self, actual, expected, *, case):
        with self.subTest(case=case, shape=actual.shape):
            actual_values = (
                actual.ndim,
                actual.dim(),
                actual.ndimension(),
                actual.nelement(),
                actual.numel(),
                torch.numel(actual),
                torch.numel(input=actual),
                actual.element_size(),
                actual.nbytes,
            )
            expected_values = (
                expected.ndim,
                expected.dim(),
                expected.ndimension(),
                expected.nelement(),
                expected.numel(),
                reference_torch.numel(expected),
                reference_torch.numel(input=expected),
                expected.element_size(),
                expected.nbytes,
            )
            self.assertEqual(actual_values, expected_values)
            self.assertTrue(all(type(value) is int for value in actual_values))
            self.assertEqual(actual.nbytes, actual.numel() * actual.element_size())
            self.assertEqual(
                expected.nbytes, expected.numel() * expected.element_size()
            )

    def test_seeded_shapes_and_metadata_views_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0xD1_213)
        shapes = [(), (0,), (1,), (2, 0, 3), (2, 3, 4), (1,) * 65]
        for _ in range(48):
            rank = int(rng.integers(0, 9))
            shapes.append(tuple(int(size) for size in rng.integers(0, 5, rank)))

        for case, shape in enumerate(shapes):
            actual = torch.zeros(shape)
            expected = reference_torch.zeros(shape)
            if len(shape) >= 2:
                actual = actual.transpose(0, -1)
                expected = expected.transpose(0, -1)
            if actual.shape and actual.shape[0] > 0 and case % 3 == 0:
                actual = actual[-1]
                expected = expected[-1]
            if case % 4 == 0:
                actual = actual.squeeze()
                expected = expected.squeeze()
            self.assert_introspection_matches(actual, expected, case=case)

        actual_extreme = torch.zeros((0,)).reshape((2, 0, sys.maxsize)).transpose(0, 2)
        expected_extreme = (
            reference_torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        self.assert_introspection_matches(
            actual_extreme, expected_extreme, case="extreme-empty-view"
        )

        values = [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
        ]
        actual_offset = torch.tensor(values).transpose(0, 1)[1]
        expected_offset = reference_torch.tensor(
            values, dtype=reference_torch.float32
        ).transpose(0, 1)[1]
        self.assertGreater(actual_offset.storage_offset(), 0)
        self.assertLess(actual_offset.nbytes, torch.tensor(values).nbytes)
        self.assert_introspection_matches(
            actual_offset, expected_offset, case="noncontiguous-offset-view"
        )

        actual_huge_offset = torch.zeros((sys.maxsize, 0))[sys.maxsize - 1]
        expected_huge_offset = reference_torch.zeros((sys.maxsize, 0))[
            sys.maxsize - 1
        ]
        self.assertGreater(actual_huge_offset.storage_offset(), 0)
        self.assert_introspection_matches(
            actual_huge_offset,
            expected_huge_offset,
            case="extreme-empty-offset-view",
        )

    def test_descriptors_signatures_and_bound_unbound_behavior_match(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 0, 3))
        expected = reference_torch.zeros((2, 0, 3))

        for name in ("ndim", "nbytes"):
            with self.subTest(name=name):
                actual_property = inspect.getattr_static(torch.Tensor, name)
                expected_property = inspect.getattr_static(reference_torch.Tensor, name)
                self.assertIs(type(actual_property), types.GetSetDescriptorType)
                self.assertIs(type(expected_property), types.GetSetDescriptorType)
                self.assertEqual(actual_property.__name__, expected_property.__name__)
                self.assertEqual(
                    actual_property.__get__(actual, torch.Tensor),
                    expected_property.__get__(expected, reference_torch.Tensor),
                )
                self.assertIs(
                    actual_property.__get__(None, torch.Tensor), actual_property
                )
                self.assertIs(
                    expected_property.__get__(None, reference_torch.Tensor),
                    expected_property,
                )

        actual_nbytes = inspect.getattr_static(torch.Tensor, "nbytes")
        expected_nbytes = inspect.getattr_static(reference_torch.Tensor, "nbytes")
        self.assertEqual(actual_nbytes.__doc__, expected_nbytes.__doc__)

        for name in ("dim", "ndimension", "nelement", "numel", "element_size"):
            with self.subTest(name=name):
                actual_descriptor = inspect.getattr_static(torch.Tensor, name)
                expected_descriptor = inspect.getattr_static(reference_torch.Tensor, name)
                self.assertIs(type(actual_descriptor), types.MethodDescriptorType)
                self.assertIs(type(expected_descriptor), types.MethodDescriptorType)
                self.assertEqual(
                    actual_descriptor.__text_signature__,
                    expected_descriptor.__text_signature__,
                )
                for descriptor in (actual_descriptor, expected_descriptor):
                    assert_no_argument_signature(self, descriptor, "(self, /)")
                for bound in (getattr(actual, name), getattr(expected, name)):
                    assert_no_argument_signature(self, bound, "()")
                self.assertEqual(actual_descriptor(actual), expected_descriptor(expected))
                if name in ("numel", "element_size"):
                    self.assertEqual(
                        actual_descriptor.__doc__, expected_descriptor.__doc__
                    )
                    self.assertEqual(
                        getattr(actual, name).__doc__, getattr(expected, name).__doc__
                    )
                    self.assertEqual(
                        actual_descriptor.__objclass__.__name__,
                        expected_descriptor.__objclass__.__name__,
                    )
                    self.assertEqual(
                        actual_descriptor.__objclass__.__module__,
                        expected_descriptor.__objclass__.__module__,
                    )
                    self.assertEqual(
                        getattr(actual, name)(), getattr(expected, name)()
                    )
                if name != "element_size":
                    self.assert_error_matches(
                        lambda name=name: getattr(actual, name)(1),
                        lambda name=name: getattr(expected, name)(1),
                    )
                    self.assert_error_matches(
                        lambda name=name: getattr(actual, name)(other=1),
                        lambda name=name: getattr(expected, name)(other=1),
                    )
                with self.assertRaises(TypeError):
                    actual_descriptor()
                with self.assertRaises(TypeError):
                    expected_descriptor()
                with self.assertRaises(TypeError):
                    actual_descriptor(1)
                with self.assertRaises(TypeError):
                    expected_descriptor(1)

        actual_descriptor = inspect.getattr_static(torch.Tensor, "element_size")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "element_size"
        )
        actual_bound = actual.element_size
        expected_bound = expected.element_size
        call_pairs = (
            (
                lambda: actual.element_size(1),
                lambda: expected.element_size(1),
            ),
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (
                lambda: actual.element_size(unexpected=True),
                lambda: expected.element_size(unexpected=True),
            ),
            (
                lambda: actual_bound(unexpected=True),
                lambda: expected_bound(unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, unexpected=True),
                lambda: expected_descriptor(expected, unexpected=True),
            ),
            (
                lambda: actual.element_size(1, unexpected=True),
                lambda: expected.element_size(1, unexpected=True),
            ),
            (
                lambda: actual_bound(1, unexpected=True),
                lambda: expected_bound(1, unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, 1, unexpected=True),
                lambda: expected_descriptor(expected, 1, unexpected=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(method="element_size", invalid_call=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertIsNone(torch.numel.__text_signature__)
        self.assertIsNone(reference_torch.numel.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(torch.numel)
        with self.assertRaises(ValueError):
            inspect.signature(reference_torch.numel)

    @staticmethod
    def numel_mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.zeros((2, 3), dtype=module.float32).transpose(0, 1)
descriptor = inspect.getattr_static(module.Tensor, "numel")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append(
            (
                func,
                types,
                args,
                kwargs,
                len(module.overrides._get_current_function_mode_stack()),
            )
        )
        return self.result

recording = RecordingMode(marker)
with recording:
    intercepted = tensor.numel()
function, dispatch_types, args, kwargs, handler_stack_depth = recording.calls[0]

forwarding_events = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.events.append(
            [
                self.label,
                len(module.overrides._get_current_function_mode_stack()),
            ]
        )
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower", forwarding_events):
    with ForwardingMode("upper", forwarding_events):
        forwarded = tensor.numel()

old_recursion_limit = sys.getrecursionlimit()
declining = RecordingMode(NotImplemented)
lower = RecordingMode(marker)
try:
    sys.setrecursionlimit(80)
    with lower:
        try:
            with declining:
                tensor.numel()
        except Exception as error:
            declining_error = [type(error).__name__, str(error)]
            declining_stack_inside_lower = len(
                module.overrides._get_current_function_mode_stack()
            )
        else:
            declining_error = None
            declining_stack_inside_lower = None
finally:
    sys.setrecursionlimit(old_recursion_limit)

class RaisingMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0
        self.handler_stack_depth = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        self.handler_stack_depth = len(
            module.overrides._get_current_function_mode_stack()
        )
        raise ValueError("numel mode failed")

recovery_events = []
raising = RaisingMode()
with ForwardingMode("lower", recovery_events):
    try:
        with raising:
            tensor.numel()
    except Exception as error:
        raising_error = [type(error).__name__, str(error)]
        raising_stack_inside_lower = len(
            module.overrides._get_current_function_mode_stack()
        )
    else:
        raising_error = None
        raising_stack_inside_lower = None
    recovered = tensor.numel()

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.numel(1)
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

print(json.dumps({
    "descriptor_type": type(descriptor).__name__,
    "descriptor_repr": repr(descriptor),
    "descriptor_name": descriptor.__name__,
    "descriptor_qualname": descriptor.__qualname__,
    "descriptor_doc": descriptor.__doc__,
    "descriptor_owner_name": descriptor.__objclass__.__name__,
    "descriptor_owner_module": descriptor.__objclass__.__module__,
    "descriptor_identity": module.Tensor.numel is descriptor,
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_is_descriptor": function is descriptor,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "handler_stack_depth": handler_stack_depth,
    "forwarding_events": forwarding_events,
    "forwarded": forwarded,
    "forwarded_type": type(forwarded).__name__,
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "declining_handler_depths": sorted({call[4] for call in declining.calls}),
    "declining_stack_inside_lower": declining_stack_inside_lower,
    "lower_skipped": len(lower.calls) == 0,
    "raising_error": raising_error,
    "raising_calls": raising.calls,
    "raising_handler_stack_depth": raising.handler_stack_depth,
    "raising_stack_inside_lower": raising_stack_inside_lower,
    "recovery_events": recovery_events,
    "recovered": recovered,
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

    def test_numel_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.numel_mode_dispatch_observation("torch_rs"),
            self.numel_mode_dispatch_observation("torch"),
        )

    @staticmethod
    def legacy_numel_tight_headroom_observation(module_name):
        source = r'''
import importlib
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
marker = object()
sys.setrecursionlimit(80)

class StatefulMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        if self.calls == 1:
            return NotImplemented
        return marker

def call_from_wrappers(remaining):
    if remaining:
        return call_from_wrappers(remaining - 1)
    return tensor.numel()

mode = StatefulMode()
try:
    with mode:
        wrapper_depth = 72
        result = call_from_wrappers(wrapper_depth)
except Exception as error:
    outcome = ["error", type(error).__name__, str(error)]
else:
    outcome = ["result", result is marker]

print(json.dumps({
    "wrapper_depth": wrapper_depth,
    "calls": mode.calls,
    "outcome": outcome,
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    @unittest.skipUnless(
        sys.version_info < (3, 12),
        "legacy CPython mode recursion accounting only applies before 3.12",
    )
    def test_legacy_numel_tight_headroom_matches_pytorch_2_13(self):
        actual = self.legacy_numel_tight_headroom_observation("torch_rs")
        expected = self.legacy_numel_tight_headroom_observation("torch")
        self.assertEqual(actual, expected)
        self.assertEqual(actual["wrapper_depth"], 72)
        self.assertEqual(actual["calls"], 1)
        self.assertEqual(actual["outcome"][:2], ["error", "RecursionError"])
        self.assertRegex(
            actual["outcome"][2], r"^maximum recursion depth exceeded"
        )
        self.assertEqual(actual["stack_depth"], 0)

    def test_nbytes_assignment_behavior_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        errors = []
        for module in (torch, reference_torch):
            tensor = module.tensor([1.0, 2.0])
            module_errors = []
            for action in ("set", "delete"):
                try:
                    if action == "set":
                        tensor.nbytes = 99
                    else:
                        del tensor.nbytes
                except Exception as error:
                    module_errors.append((type(error).__name__, str(error)))
                else:
                    self.fail(f"{module.__name__} allowed nbytes to be {action}")
            errors.append(module_errors)

        for actual, expected in zip(errors[0], errors[1], strict=True):
            self.assertEqual(actual[0], expected[0])
            for _, message in (actual, expected):
                self.assertIn("attribute 'nbytes'", message)
                self.assertTrue(message.endswith("objects is not writable"))

    def test_argument_and_type_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        cases = (
            (lambda: torch.numel(), lambda: reference_torch.numel()),
            (
                lambda: torch.numel(actual, actual),
                lambda: reference_torch.numel(expected, expected),
            ),
            (
                lambda: torch.numel(actual, input=actual),
                lambda: reference_torch.numel(expected, input=expected),
            ),
            (
                lambda: torch.numel(actual, extra=True),
                lambda: reference_torch.numel(expected, extra=True),
            ),
            (lambda: torch.numel(1), lambda: reference_torch.numel(1)),
            (lambda: torch.numel(input=[]), lambda: reference_torch.numel(input=[])),
            (
                lambda: torch.numel(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.numel(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
