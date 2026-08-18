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
            "is_leaf": result.is_leaf,
            "retains_grad": result.retains_grad,
            "output_nr": result.output_nr,
        }

    def test_supported_non_quantized_identity_matches_pytorch_2_13(self):
        _actual_leaf, _actual_tracked, actual_cases = self.tensor_cases(torch)
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

    def grad_enabled_contract(self, module, case):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "leaf":
            tensor = leaf
        elif case == "non-leaf view":
            tensor = (leaf * 2.0).transpose(0, 1)
        else:
            tensor = (leaf * 2.0).unbind()[1]
        graph_before = (tensor.is_leaf, tensor.output_nr)
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
        )

        result = tensor.dequantize()
        try:
            result.sum().backward()
        except Exception as error:
            backward_error = type(error).__name__, str(error)
        else:
            backward_error = None

        return {
            "result_is_receiver": result is tensor,
            "graph_before": graph_before,
            "is_leaf": result.is_leaf,
            "output_nr": result.output_nr,
            "requires_grad": result.requires_grad,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
            ),
            "backward_error": backward_error,
        }

    def test_grad_enabled_unsupported_edge_matches_pytorch_2_13(self):
        for case in ("leaf", "non-leaf view", "multi-output view"):
            with self.subTest(case=case):
                self.assertEqual(
                    self.grad_enabled_contract(torch, case),
                    self.grad_enabled_contract(reference_torch, case),
                )

    def populated_leaf_grad_contract(self, module):
        leaf = module.tensor(2.0, dtype=module.float32, requires_grad=True)
        leaf.backward()
        result = leaf.dequantize()
        gradient = leaf.grad
        try:
            leaf.backward()
        except Exception as error:
            backward_error = type(error).__name__, str(error)
        else:
            backward_error = None
        return {
            "result_is_receiver": result is leaf,
            "is_leaf": leaf.is_leaf,
            "output_nr": leaf.output_nr,
            "gradient": gradient.tolist(),
            "gradient_is_stable": leaf.grad is gradient,
            "backward_error": backward_error,
        }

    def test_populated_leaf_gradient_survives_the_edge_replacement(self):
        self.assertEqual(
            self.populated_leaf_grad_contract(torch),
            self.populated_leaf_grad_contract(reference_torch),
        )

    def moved_leaf_contract(self, module):
        leaf = module.tensor(
            [2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        pending = (leaf * 4.0).sum()
        (leaf * 5.0).sum().backward()
        gradient = leaf.grad
        result = leaf.dequantize()

        def outcome(action):
            try:
                action()
            except Exception as error:
                return type(error).__name__, str(error)
            return None

        first_error = outcome(pending.backward)
        gradient_after_error = leaf.grad
        second_error = outcome(pending.backward)
        return {
            "result_is_receiver": result is leaf,
            "is_leaf": leaf.is_leaf,
            "first_error": first_error,
            "second_error": second_error,
            "gradient_before": gradient.tolist(),
            "gradient_after": gradient_after_error.tolist(),
            "gradient_is_stable": gradient_after_error is gradient,
        }

    def test_pending_leaf_edges_are_invalidated_like_pytorch_2_13(self):
        self.assertEqual(
            self.moved_leaf_contract(torch),
            self.moved_leaf_contract(reference_torch),
        )

    def failing_backward_contract(self, module):
        def outcome(action):
            try:
                action()
            except Exception as error:
                return type(error).__name__, str(error)
            return None

        leaf = module.tensor([0.5], dtype=module.float32, requires_grad=True)
        loss = leaf.dequantize().sin().sum()
        chain_errors = outcome(loss.backward), outcome(loss.backward)

        failing_leaf = module.tensor(
            0.0, dtype=module.float32, requires_grad=True
        )
        independent_leaf = module.tensor(
            3.0, dtype=module.float32, requires_grad=True
        )
        branched_loss = failing_leaf.dequantize().sin() * independent_leaf
        branch_first_error = outcome(branched_loss.backward)
        committed = independent_leaf.grad
        branch_second_error = outcome(branched_loss.backward)

        older_leaf = module.tensor(
            2.0, dtype=module.float32, requires_grad=True
        )
        freed_branch = older_leaf.sin()
        freed_branch.backward()
        newer_leaf = module.tensor(
            0.5, dtype=module.float32, requires_grad=True
        )
        failing_branch = newer_leaf.dequantize().sin()
        ordered_loss = failing_branch * freed_branch

        return {
            "chain_errors": chain_errors,
            "branch_first_error": branch_first_error,
            "branch_second_error": branch_second_error,
            "committed_gradient": committed.tolist(),
            "committed_gradient_is_stable": independent_leaf.grad is committed,
            "ordered_error": outcome(ordered_loss.backward),
        }

    def test_failing_backward_lifecycle_matches_pytorch_2_13(self):
        self.assertEqual(
            self.failing_backward_contract(torch),
            self.failing_backward_contract(reference_torch),
        )

    def no_grad_contract(self, module, case):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tensor = leaf if case == "leaf" else (leaf * 3.0).transpose(0, 1)
        graph_before = (tensor.is_leaf, tensor.output_nr)
        with module.no_grad():
            result = tensor.dequantize()
        result.sum().backward()
        return {
            "result_is_receiver": result is tensor,
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "output_nr": result.output_nr,
            "graph_before": graph_before,
            "shape": tuple(result.shape),
            "stride": tuple(result.stride()),
            "offset": result.storage_offset(),
            "pointer_unchanged": result.data_ptr() == tensor.data_ptr(),
            "gradient": leaf.grad.tolist(),
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def test_no_grad_graph_identity_matches_pytorch_2_13(self):
        for case in ("leaf", "non-leaf view"):
            with self.subTest(case=case):
                self.assertEqual(
                    self.no_grad_contract(torch, case),
                    self.no_grad_contract(reference_torch, case),
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
