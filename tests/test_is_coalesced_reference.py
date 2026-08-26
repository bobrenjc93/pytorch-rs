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
class TensorIsCoalescedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_coalesced differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        noncontiguous = source.transpose(0, 1)
        offset = noncontiguous[1]
        return leaf, (
            module.tensor(3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            noncontiguous,
            offset,
            leaf,
            tracked,
            leaf.grad,
        )

    def metadata(self, tensor):
        return (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.tolist(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
            repr(tensor),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_coalesced unexpectedly succeeded")

    def strided_contract(self, module, tensor):
        before = self.metadata(tensor)
        error = self.error(tensor.is_coalesced)
        after = self.metadata(tensor)
        return {
            "error": error,
            "layout_is_canonical_strided": tensor.layout is module.strided,
            "metadata_unchanged": before == after,
        }

    def test_supported_strided_states_match_pytorch_2_13(self):
        actual_leaf, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_cases = self.tensor_cases(reference_torch)

        self.assertFalse(actual_cases[2].is_contiguous())
        self.assertFalse(expected_cases[2].is_contiguous())
        self.assertGreater(actual_cases[3].storage_offset(), 0)
        self.assertGreater(expected_cases[3].storage_offset(), 0)
        actual_gradient = actual_leaf.grad
        expected_gradient = expected_leaf.grad
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.strided_contract(torch, actual),
                    self.strided_contract(reference_torch, expected),
                )
        self.assertIs(actual_leaf.grad, actual_gradient)
        self.assertIs(expected_leaf.grad, expected_gradient)
        self.assertEqual(actual_gradient.tolist(), expected_gradient.tolist())

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "is_coalesced")
        bound = tensor.is_coalesced
        calls = (
            lambda: tensor.is_coalesced(),
            lambda: descriptor(tensor),
            lambda: tensor.is_coalesced(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.is_coalesced(1, 2),
            lambda: tensor.is_coalesced(input=tensor),
            lambda: bound(unexpected=True),
            lambda: descriptor(tensor, unexpected=True),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
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
            "errors": tuple(self.error(call) for call in calls),
        }

    def test_callable_metadata_documentation_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "is_coalesced")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return marker

recording = RecordingMode()
with recording:
    intercepted = tensor.is_coalesced()
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

try:
    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            tensor.is_coalesced()
except Exception as error:
    forwarded_error = [type(error).__name__, str(error)]
else:
    forwarded_error = None

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
    "forwarded_error": forwarded_error,
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

    def test_reference_sparse_coo_behavior_bounds_unsupported_surface(self):
        for name in ("sparse_coo_tensor", "sparse_coo"):
            self.assertFalse(hasattr(torch, name))
        for name in ("coalesce", "_coalesced_", "indices", "_indices"):
            self.assertFalse(hasattr(torch.Tensor, name))

        indices = reference_torch.tensor(
            [[0, 1], [1, 0]], dtype=reference_torch.int64
        )
        values = reference_torch.tensor([3.0, 4.0])
        descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_coalesced"
        )
        for expected, tensor in (
            (
                True,
                reference_torch.sparse_coo_tensor(
                    indices,
                    values,
                    (2, 2),
                    is_coalesced=True,
                    check_invariants=True,
                ),
            ),
            (
                False,
                reference_torch.sparse_coo_tensor(
                    indices,
                    values,
                    (2, 2),
                    is_coalesced=False,
                    check_invariants=True,
                ),
            ),
        ):
            with self.subTest(expected=expected):
                self.assertIs(tensor.layout, reference_torch.sparse_coo)
                self.assertIs(tensor.is_coalesced(), expected)
                self.assertIs(descriptor(tensor), expected)


if __name__ == "__main__":
    unittest.main()
