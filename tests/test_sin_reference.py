import inspect
import json
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
class SinReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.sin differentials require pinned PyTorch 2.13.0"
            )

    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def test_autograd_preserves_scalar_empty_and_strided_history(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_scalar = torch.tensor(1.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(1.5, requires_grad=True)
        actual_scalar_output = actual_scalar.sin()
        expected_scalar_output = expected_scalar.sin()
        self.assertFalse(actual_scalar_output.is_set_to(actual_scalar))
        self.assertFalse(expected_scalar_output.is_set_to(expected_scalar))
        self.assert_metadata_matches(
            actual_scalar_output,
            expected_scalar_output,
            case="scalar output",
        )
        actual_scalar_output.backward()
        expected_scalar_output.backward()
        self.assert_metadata_matches(
            actual_scalar.grad,
            expected_scalar.grad,
            case="scalar gradient",
        )
        self.assertEqual(
            np.asarray(actual_scalar.grad).view(np.uint32).item(),
            expected_scalar.grad.detach().numpy().view(np.uint32).item(),
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty_output = actual_empty.sin()
        expected_empty_output = expected_empty.sin()
        self.assertFalse(actual_empty_output.is_set_to(actual_empty))
        self.assertFalse(expected_empty_output.is_set_to(expected_empty))
        self.assert_metadata_matches(
            actual_empty_output,
            expected_empty_output,
            case="empty output",
        )
        actual_empty_output.sum().backward()
        expected_empty_output.sum().backward()
        self.assert_metadata_matches(
            actual_empty.grad,
            expected_empty.grad,
            case="empty gradient",
        )
        self.assertEqual(actual_empty.grad.tolist(), expected_empty.grad.tolist())

        values = [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]]
        weights = [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_output = actual_leaf.transpose(0, 1).sin()
        expected_output = expected_leaf.transpose(0, 1).sin()
        self.assertFalse(actual_output.is_set_to(actual_leaf))
        self.assertFalse(expected_output.is_set_to(expected_leaf))
        self.assert_metadata_matches(
            actual_output,
            expected_output,
            case="strided output",
        )
        (actual_output * torch.tensor(weights)).sum().backward()
        (expected_output * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.detach().numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )

    def test_vjp_matches_grad_output_times_cosine_of_saved_input(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x3F80_0000,
                0xC000_0000,
                0x4049_0FDB,
                0x5015_02F9,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x0000_0000,
                0x7F80_0000,
                0x3F80_0000,
                0xBF80_0000,
            ),
            dtype=np.uint32,
        )
        input_values = input_bits.view(np.float32)
        weight_values = weight_bits.view(np.float32)
        actual_leaf = torch.tensor(memoryview(input_values), requires_grad=True)
        expected_leaf = reference_torch.tensor(input_values, requires_grad=True)
        actual_weights = torch.tensor(memoryview(weight_values))
        expected_weights = reference_torch.tensor(weight_values)

        (actual_leaf.sin() * actual_weights).sum().backward()
        (expected_leaf.sin() * expected_weights).sum().backward()
        expected_formula = expected_weights * expected_leaf.detach().cos()
        np.testing.assert_array_equal(
            expected_leaf.grad.detach().numpy().view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_formula.numpy().view(np.uint32),
        )

    def test_detach_no_grad_and_freed_graph_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)

        actual_detached_input = actual_leaf.detach().transpose(0, 1).sin()
        expected_detached_input = expected_leaf.detach().transpose(0, 1).sin()
        self.assert_metadata_matches(
            actual_detached_input,
            expected_detached_input,
            case="detached input",
        )

        actual_tracked = actual_leaf.transpose(0, 1).sin()
        expected_tracked = expected_leaf.transpose(0, 1).sin()
        actual_detached_output = actual_tracked.detach()
        expected_detached_output = expected_tracked.detach()
        self.assert_metadata_matches(
            actual_detached_output,
            expected_detached_output,
            case="detached output",
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.transpose(0, 1).sin()
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.transpose(0, 1).sin()
        self.assert_metadata_matches(
            actual_untracked,
            expected_untracked,
            case="no_grad output",
        )

        with torch.no_grad():
            actual_no_grad_view = actual_leaf.transpose(0, 1)
        with reference_torch.no_grad():
            expected_no_grad_view = expected_leaf.transpose(0, 1)
        actual_boundary_output = actual_no_grad_view.sin()
        expected_boundary_output = expected_no_grad_view.sin()
        self.assert_metadata_matches(
            actual_boundary_output,
            expected_boundary_output,
            case="operation after no_grad view",
        )
        actual_boundary_loss = actual_boundary_output.sum()
        expected_boundary_loss = expected_boundary_output.sum()
        actual_boundary_loss.backward()
        expected_boundary_loss.backward()
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)
        with self.assertRaises(RuntimeError) as expected_boundary_raised:
            expected_boundary_loss.backward()
        with self.assertRaises(RuntimeError) as actual_boundary_raised:
            actual_boundary_loss.backward()
        self.assertEqual(
            str(actual_boundary_raised.exception),
            str(expected_boundary_raised.exception),
        )

        actual_loss = actual_tracked.sum()
        expected_loss = expected_tracked.sum()
        actual_loss.backward()
        expected_loss.backward()
        np.testing.assert_allclose(
            np.asarray(actual_leaf.grad),
            expected_leaf.grad.detach().numpy(),
            rtol=2.0e-6,
            atol=0.0,
        )
        with self.assertRaises(RuntimeError) as expected_raised:
            expected_loss.backward()
        with self.assertRaises(RuntimeError) as actual_raised:
            actual_loss.backward()
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.sin unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "sin")
        bound = tensor.sin
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
                    lambda: tensor.sin(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.sin(1, 2),
                    lambda: tensor.sin(input=tensor),
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
tensor = module.tensor([0.5], dtype=module.float32, requires_grad=True)
descriptor = inspect.getattr_static(module.Tensor, "sin")
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
    intercepted = tensor.sin()
function, dispatch_types, args, kwargs = recording.calls[0]

forwarding_events = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.events.append(
            [self.label, len(module.overrides._get_current_function_mode_stack())]
        )
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower", forwarding_events):
    with ForwardingMode("upper", forwarding_events):
        forwarded = tensor.sin()

extreme = module.zeros((0,), dtype=module.float32).reshape((0, sys.maxsize, 3))
bypass = RecordingMode(marker)
with bypass:
    bypassed = extreme.sin()

old_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        try:
            tensor.sin()
        except Exception as error:
            declining_error = [type(error).__name__, str(error)]
            declining_stack_inside = len(
                module.overrides._get_current_function_mode_stack()
            )
        else:
            declining_error = None
            declining_stack_inside = None
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
        raise ValueError("sin mode failed")

recovery_events = []
raising = RaisingMode()
with ForwardingMode("lower", recovery_events):
    try:
        with raising:
            tensor.sin()
    except Exception as error:
        raising_error = [type(error).__name__, str(error)]
        raising_stack_inside_lower = len(
            module.overrides._get_current_function_mode_stack()
        )
    else:
        raising_error = None
        raising_stack_inside_lower = None
    recovered = tensor.sin()

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.sin(1)
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
    "forwarding_events": forwarding_events,
    "forwarded": forwarded.tolist(),
    "bypassed": bypassed is marker,
    "bypass_calls": len(bypass.calls),
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "declining_stack_inside": declining_stack_inside,
    "raising_error": raising_error,
    "raising_calls": raising.calls,
    "raising_handler_stack_depth": raising.handler_stack_depth,
    "raising_stack_inside_lower": raising_stack_inside_lower,
    "recovery_events": recovery_events,
    "recovered": recovered.tolist(),
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
    def autograd_name_error(module):
        probability = module.tensor(
            [-1.0], dtype=module.float32, requires_grad=True
        ).sin()
        try:
            module.nn.functional.dropout(
                module.tensor([1.0], dtype=module.float32),
                p=probability,
                training=False,
            )
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError(
            "dropout unexpectedly accepted a negative sine probability"
        )

    def test_sin_backward_name_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_name_error(torch),
            self.autograd_name_error(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
