import inspect
import json
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorConstDataPtrReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "const_data_ptr differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def ordinary_storage_outcome(self, module):
        scalar = module.tensor(2.5, dtype=module.float32)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        offset = source[2]
        strided = source.transpose(0, 1)[1]
        empty = module.zeros((3, 0, 4), dtype=module.float32)[2]
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        autograd_view = (leaf * 3.0).transpose(0, 1)
        tensors = (
            scalar,
            source,
            offset,
            strided,
            empty,
            strided.detach(),
            leaf,
            autograd_view,
        )

        observations = []
        for tensor in tensors:
            state_before = (
                tensor.tolist(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            )
            pointer = tensor.const_data_ptr()
            state_after = (
                tensor.tolist(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            )
            observations.append(
                (
                    type(pointer).__name__,
                    pointer == tensor.data_ptr(),
                    pointer == tensor.const_data_ptr(),
                    state_before,
                    state_after,
                )
            )

        autograd_view.sum().backward()
        return (
            observations,
            offset.const_data_ptr() - source.const_data_ptr(),
            strided.const_data_ptr() - source.const_data_ptr(),
            empty.const_data_ptr(),
            leaf.grad.tolist(),
            leaf.grad.const_data_ptr() == leaf.grad.data_ptr(),
        )

    def test_ordinary_storage_behavior_matches_pytorch_2_13(self):
        actual = self.ordinary_storage_outcome(torch)
        expected = self.ordinary_storage_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertTrue(all(item[1] and item[2] for item in actual[0]))
        self.assertTrue(all(item[3] == item[4] for item in actual[0]))
        self.assertEqual(actual[1:4], (32, 4, 0))

    def test_descriptor_documentation_signature_and_errors_match(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(
            torch.Tensor, "const_data_ptr"
        )
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "const_data_ptr"
        )
        actual_bound = actual.const_data_ptr
        expected_bound = expected.const_data_ptr

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual_bound, expected_bound, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                self.signature_outcome(actual_callable),
                self.signature_outcome(expected_callable),
            )

        self.assertEqual(
            actual_descriptor.__qualname__, expected_descriptor.__qualname__
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )

        call_pairs = (
            (
                lambda: actual.const_data_ptr(1),
                lambda: expected.const_data_ptr(1),
            ),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (
                lambda: actual.const_data_ptr(dim=0),
                lambda: expected.const_data_ptr(dim=0),
            ),
            (
                lambda: actual_bound(unexpected=True),
                lambda: expected_bound(unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, unexpected=True),
                lambda: expected_descriptor(expected, unexpected=True),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(invalid_call=case):
                self.assert_error_matches(actual_call, expected_call)

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "const_data_ptr")
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
    intercepted = tensor.const_data_ptr()
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
        forwarded = tensor.const_data_ptr()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
lower = RecordingMode(marker)
try:
    with lower:
        with declining:
            tensor.const_data_ptr()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

nested_declining = RecordingMode(NotImplemented)
nested_lower = RecordingMode(marker)
def call_from_python_wrapper():
    return tensor.const_data_ptr()

try:
    with nested_lower:
        with nested_declining:
            call_from_python_wrapper()
except Exception as error:
    nested_declining_error = [type(error).__name__, str(error)]
else:
    nested_declining_error = None

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
    "forwarded_matches": forwarded == tensor.data_ptr(),
    "forwarded_type": type(forwarded).__name__,
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "lower_skipped": len(lower.calls) == 0,
    "nested_declining_error": nested_declining_error,
    "nested_declining_calls": len(nested_declining.calls),
    "nested_lower_skipped": len(nested_lower.calls) == 0,
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

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def handler_recursion_error_observation(self, module_name):
        source = r'''
import importlib
import json

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)

class ExplodingRecursionError(RecursionError):
    def __str__(self):
        raise LookupError("exception stringification must not run")

observations = []
for error in (
    RecursionError("maximum recursion depth exceeded"),
    ExplodingRecursionError("maximum recursion depth exceeded"),
):
    class StatefulMode(module.overrides.TorchFunctionMode):
        def __init__(self):
            self.calls = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.calls += 1
            if self.calls == 1:
                return NotImplemented
            raise error

    mode = StatefulMode()
    try:
        with mode:
            tensor.const_data_ptr()
    except Exception as caught:
        observations.append({
            "type": type(caught).__name__,
            "args": caught.args,
            "same_object": caught is error,
            "calls": mode.calls,
        })
    else:
        observations.append(None)

print(json.dumps({
    "observations": observations,
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

    def test_handler_recursion_errors_propagate_unchanged(self):
        self.assertEqual(
            self.handler_recursion_error_observation("torch_rs"),
            self.handler_recursion_error_observation("torch"),
        )

    def test_reference_cow_boundary_remains_explicitly_unsupported(self):
        self.assertFalse(hasattr(torch, "_lazy_clone"))
        self.assertFalse(hasattr(torch.Tensor, "_lazy_clone"))

        source = reference_torch.arange(
            12, dtype=reference_torch.float32
        ).reshape(3, 4)
        cow = source._lazy_clone()
        shared_pointer = source.const_data_ptr()
        self.assertEqual(cow.const_data_ptr(), shared_pointer)

        cow_view = cow.transpose(0, 1)[1]
        self.assertEqual(
            cow_view.const_data_ptr() - shared_pointer,
            cow_view.storage_offset() * cow_view.element_size(),
        )
        self.assertEqual(source.tolist(), cow.tolist())

        source_write_pointer = source.data_ptr()
        self.assertNotEqual(source_write_pointer, shared_pointer)
        self.assertEqual(source.const_data_ptr(), source_write_pointer)
        self.assertEqual(cow.const_data_ptr(), shared_pointer)
        self.assertEqual(cow.data_ptr(), shared_pointer)
        self.assertEqual(source.tolist(), cow.tolist())


if __name__ == "__main__":
    unittest.main()
