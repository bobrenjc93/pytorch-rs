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
class TensorResolveConjReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "resolve_conj differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            leaf,
            tracked,
            (
                module.tensor(-3.5, dtype=module.float32),
                module.zeros((2, 0, 3), dtype=module.float32),
                source.neg(),
                strided_view,
                offset_view,
                extreme_empty,
                module.tensor(memoryview(special_bits.view(np.float32))),
                leaf,
                tracked,
                tracked.detach(),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return tuple(
            np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist()
        )

    def contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        bits = self.value_bits(tensor)
        result = tensor.resolve_conj()
        return {
            "receiver_is_clear": tensor.is_conj() is False,
            "result_is_receiver": result is tensor,
            "result_is_clear": result.is_conj() is False,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
                str(result.dtype),
                str(result.device),
                result.requires_grad,
                result.is_leaf,
            ),
            "bits_unchanged": bits == self.value_bits(result),
        }

    def test_supported_clear_bit_path_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                self.assertEqual(self.contract(actual), self.contract(expected))

        actual_tracked.resolve_conj().sum().backward()
        expected_tracked.resolve_conj().sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_reference_conjugate_view_path_materializes_and_preserves_graph(self):
        self.assertFalse(hasattr(torch, "complex64"))
        source = reference_torch.tensor(
            [1.0 + 2.0j, -3.0 + 4.0j],
            dtype=reference_torch.complex64,
            requires_grad=True,
        )
        conjugate_view = source.conj()

        self.assertIs(source.is_conj(), False)
        self.assertIs(conjugate_view.is_conj(), True)
        self.assertTrue(conjugate_view._is_view())
        self.assertIs(conjugate_view._base, source)
        self.assertEqual(
            conjugate_view.untyped_storage().data_ptr(),
            source.untyped_storage().data_ptr(),
        )
        self.assertEqual(type(conjugate_view.grad_fn).__name__, "ConjBackward0")

        resolved = conjugate_view.resolve_conj()

        self.assertIsNot(resolved, conjugate_view)
        self.assertIsNot(resolved, source)
        self.assertIs(resolved.is_conj(), False)
        self.assertFalse(resolved._is_view())
        self.assertIsNone(resolved._base)
        self.assertNotEqual(
            resolved.untyped_storage().data_ptr(),
            conjugate_view.untyped_storage().data_ptr(),
        )
        self.assertEqual(resolved.tolist(), conjugate_view.tolist())
        self.assertTrue(resolved.requires_grad)
        self.assertFalse(resolved.is_leaf)
        self.assertEqual(type(resolved.grad_fn).__name__, "CloneBackward0")

        resolved.real.sum().backward()
        self.assertEqual(source.grad.tolist(), [1.0 + 0.0j, 1.0 + 0.0j])

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.resolve_conj unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "resolve_conj")
        bound = tensor.resolve_conj
        calls = (
            lambda: tensor.resolve_conj(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.resolve_conj(1, 2),
            lambda: tensor.resolve_conj(input=tensor),
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
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "signatures": tuple(
                self.signature_outcome(callable_object)
                for callable_object in (descriptor, bound)
            ),
            "call_errors": tuple(self.error(call) for call in calls),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def test_callable_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "resolve_conj")
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
    intercepted = tensor.resolve_conj()
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
        forwarded = tensor.resolve_conj()

sys.setrecursionlimit(80)
class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

lower = RecordingMode(marker)
upper = DecliningMode()
try:
    with lower:
        with upper:
            tensor.resolve_conj()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

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
    "forwarded_is_receiver": forwarded is tensor,
    "forwarded_is_clear": forwarded.is_conj() is False,
    "declining_error": declining_error,
    "declining_calls": upper.calls,
    "lower_skipped": len(lower.calls) == 0,
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

    def test_scope_is_intentionally_narrower_than_pytorch(self):
        self.assertFalse(hasattr(torch, "resolve_conj"))
        self.assertTrue(hasattr(reference_torch, "resolve_conj"))
        self.assertFalse(hasattr(torch, "complex64"))
        self.assertTrue(hasattr(reference_torch, "complex64"))


if __name__ == "__main__":
    unittest.main()
