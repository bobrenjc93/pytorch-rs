import inspect
import json
import subprocess
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorDequantizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.dequantize() differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        multi_output = tracked.unbind()[1]
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
                source,
                strided_view,
                offset_view,
                extreme_empty,
                module.tensor(memoryview(special_bits.view(np.float32))),
                leaf,
                tracked,
                multi_output,
                tracked.detach(),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return tuple(np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist())

    def identity_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.is_quantized,
            tensor.requires_grad,
        )
        bits = self.value_bits(tensor)
        result = tensor.dequantize()
        return {
            "result_is_receiver": result is tensor,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
                str(result.dtype),
                str(result.device),
                str(result.layout),
                result.is_quantized,
                result.requires_grad,
            ),
            "bits_unchanged": bits == self.value_bits(result),
            "is_non_quantized": result.is_quantized is False,
        }

    def test_supported_non_quantized_identity_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        _expected_leaf, _expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.identity_contract(actual),
                    self.identity_contract(expected),
                )

        # PyTorch 2.13 replaces a grad-tracked receiver's edge with an
        # unsupported dequantize edge even though it returns the same Python
        # object. The requested surface is the stricter exact identity, whose
        # graph preservation is covered directly in test_dequantize.py.
        actual_tracked.dequantize().sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])

    def no_grad_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)
        with module.no_grad():
            result = tracked.dequantize()
        result.sum().backward()
        return {
            "result_is_receiver": result is tracked,
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "output_nr": result.output_nr,
            "shape": tuple(result.shape),
            "stride": tuple(result.stride()),
            "offset": result.storage_offset(),
            "pointer_unchanged": result.data_ptr() == tracked.data_ptr(),
            "gradient": leaf.grad.tolist(),
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def test_no_grad_graph_identity_matches_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_contract(torch),
            self.no_grad_contract(reference_torch),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.dequantize unexpectedly accepted an invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "dequantize")
        bound = tensor.dequantize
        calls = (
            lambda: tensor.dequantize(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.dequantize(1, 2),
            lambda: tensor.dequantize(input=tensor),
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
            "class_identity": module.Tensor.dequantize is descriptor,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "bound_result_is_receiver": bound() is tensor,
            "errors": tuple(self.error(call) for call in calls),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
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
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "dequantize")
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
    intercepted = tensor.dequantize()
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
        forwarded = tensor.dequantize()

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
            tensor.dequantize()
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

    def test_reference_quantized_tensors_bound_the_unsupported_path(self):
        self.assertFalse(hasattr(torch, "dequantize"))
        self.assertTrue(hasattr(reference_torch, "dequantize"))
        for name in (
            "quantize_per_tensor",
            "quantize_per_channel",
            "qint8",
            "quint8",
            "qint32",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))

        source = reference_torch.tensor(
            [[-1.0, 0.0, 0.5], [2.0, 4.0, -3.0]],
            dtype=reference_torch.float32,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cases = (
                (
                    "quint8 per-tensor",
                    reference_torch.quantize_per_tensor(
                        source, 0.5, 2, reference_torch.quint8
                    ),
                    [[-1.0, 0.0, 0.5], [2.0, 4.0, -1.0]],
                ),
                (
                    "qint8 per-tensor",
                    reference_torch.quantize_per_tensor(
                        source, 0.25, -3, reference_torch.qint8
                    ),
                    source.tolist(),
                ),
                (
                    "qint32 per-tensor",
                    reference_torch.quantize_per_tensor(
                        source, 0.125, 17, reference_torch.qint32
                    ),
                    source.tolist(),
                ),
                (
                    "qint8 per-channel",
                    reference_torch.quantize_per_channel(
                        source,
                        reference_torch.tensor([0.5, 0.25]),
                        reference_torch.tensor([0, -2]),
                        0,
                        reference_torch.qint8,
                    ),
                    [[-1.0, 0.0, 0.5], [2.0, 4.0, -3.0]],
                ),
            )

        for case, quantized, expected_values in cases:
            with self.subTest(case=case):
                self.assertTrue(quantized.is_quantized)
                self.assertIs(quantized.layout, reference_torch.strided)

                result = quantized.dequantize()

                self.assertIsNot(result, quantized)
                self.assertIs(result.dtype, reference_torch.float32)
                self.assertIs(result.layout, reference_torch.strided)
                self.assertFalse(result.is_quantized)
                self.assertEqual(result.shape, quantized.shape)
                self.assertEqual(result.stride(), quantized.stride())
                self.assertEqual(result.tolist(), expected_values)
                self.assertNotEqual(
                    result.untyped_storage().data_ptr(),
                    quantized.untyped_storage().data_ptr(),
                )
                self.assertFalse(result.requires_grad)
                self.assertTrue(result.is_leaf)
                self.assertIsNone(result.grad_fn)
                self.assertIs(result.dequantize(), result)

        transposed = cases[1][1].transpose(0, 1)
        materialized = transposed.dequantize()
        self.assertFalse(transposed.is_contiguous())
        self.assertTrue(materialized.is_contiguous())
        self.assertEqual(transposed.stride(), (1, 3))
        self.assertEqual(materialized.stride(), (2, 1))
        self.assertEqual(
            materialized.tolist(),
            [[-1.0, 2.0], [0.0, 4.0], [0.5, -3.0]],
        )


if __name__ == "__main__":
    unittest.main()
