import importlib
import inspect
import json
import math
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
class TensorIsDistributedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.is_distributed() differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module, device="cpu"):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            device=device,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked_view = produced.transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
            device=device,
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32, device=device)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        channels_last = module.zeros(
            (2, 3, 4, 5), dtype=module.float32, device=device
        ).contiguous(memory_format=module.channels_last)
        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)
        return leaf, tracked_view, (
            *(
                module.tensor(value, dtype=module.float32, device=device)
                for value in (
                    -math.inf,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    math.inf,
                    math.nan,
                )
            ),
            source,
            module.zeros((2, 0, 3), dtype=module.float32, device=device),
            extreme_empty,
            channels_last,
            strided_view,
            offset_view,
            leaf,
            produced,
            tracked_view,
            produced.detach(),
            tracked_view.detach(),
            no_grad_output,
            no_grad_view,
        )

    def contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        result = tensor.is_distributed()
        return {
            "result": result,
            "result_type": type(result).__name__,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                str(tensor.dtype),
                str(tensor.device),
                str(tensor.layout),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            ),
        }

    def test_supported_local_cpu_tensors_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(self.contract(actual), self.contract(expected))
                self.assertIs(actual.is_distributed(), False)

        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())
        self.assertEqual(
            self.contract(actual_leaf.grad), self.contract(expected_leaf.grad)
        )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is unavailable",
    )
    def test_reference_local_cuda_tensors_are_also_not_distributed(self):
        device = reference_torch.device("cuda", 0)
        device_name = reference_torch.cuda.get_device_name(device)
        leaf, tracked, cases = self.tensor_cases(reference_torch, device)
        for case, tensor in enumerate(cases):
            with self.subTest(case=case, shape=tensor.shape, gpu=device_name):
                self.assertEqual(tensor.device.type, "cuda")
                self.assertIs(tensor.is_distributed(), False)
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        reference_torch.cuda.synchronize(device)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_distributed unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "is_distributed")
        bound = tensor.is_distributed
        calls = (
            lambda: tensor.is_distributed(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.is_distributed(1, 2),
            lambda: tensor.is_distributed(unexpected=True),
            lambda: bound(unexpected=True),
            lambda: descriptor(tensor, unexpected=True),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "descriptor_doc": descriptor.__doc__,
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
            "class_identity": module.Tensor.is_distributed is descriptor,
            "descriptor_result": descriptor(tensor),
            "bound_result": bound(),
            "errors": tuple(self.error(call) for call in calls),
        }

    def test_callable_metadata_nulls_and_errors_match_pytorch_2_13(self):
        actual = self.callable_contract(torch)
        expected = self.callable_contract(reference_torch)
        self.assertEqual(actual, expected)
        self.assertIsNone(actual["descriptor_doc"])
        self.assertIsNone(actual["bound_doc"])

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "is_distributed")
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
    intercepted = tensor.is_distributed()
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
        forwarded = tensor.is_distributed()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
lower = RecordingMode(marker)
try:
    with lower:
        with declining:
            tensor.is_distributed()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

rejected = RecordingMode(marker)
try:
    with rejected:
        tensor.is_distributed(unexpected=True)
except Exception as error:
    rejected_error = [type(error).__name__, str(error)]
else:
    rejected_error = None

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
    "forwarded": forwarded,
    "forwarded_type": type(forwarded).__name__,
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "lower_skipped": len(lower.calls) == 0,
    "rejected_error": rejected_error,
    "rejected_calls": len(rejected.calls),
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

    def test_distributed_tensor_and_execution_apis_stay_out_of_scope(self):
        self.assertFalse(hasattr(torch, "is_distributed"))
        self.assertTrue(hasattr(reference_torch, "is_distributed"))
        for name in (
            "DeviceMesh",
            "ProcessGroup",
            "all_reduce",
            "init_process_group",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.distributed, name))
                self.assertTrue(hasattr(reference_torch.distributed, name))

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.distributed.tensor")
        distributed_tensor = importlib.import_module("torch.distributed.tensor")
        self.assertTrue(hasattr(distributed_tensor, "DTensor"))
        self.assertTrue(hasattr(distributed_tensor, "DeviceMesh"))


if __name__ == "__main__":
    unittest.main()
