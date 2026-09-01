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
class TensorReciprocalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.reciprocal differentials require pinned PyTorch 2.13.0"
            )

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
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def tensor_cases(module):
        base = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            module.zeros((0, 1), dtype=module.float32),
            module.zeros((0, 1, 2), dtype=module.float32),
            module.zeros((1, 0, 1), dtype=module.float32),
            strided[1],
            strided,
            module.tensor(memoryview(special_bits.view(np.float32))),
        )

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(4.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None
        if case == "offset":
            leaf = module.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                dtype=module.float32,
                requires_grad=True,
            )
            return leaf, leaf[1], None
        if case == "noncontiguous":
            leaf = module.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
                dtype=module.float32,
                requires_grad=True,
            )
            input = leaf.transpose(0, 2)[1]
            weights = module.tensor(
                np.arange(1, 7, dtype=np.float32).reshape(3, 2).tolist(),
                dtype=module.float32,
            )
            return leaf, input, weights
        raise AssertionError(f"unknown reciprocal autograd case: {case}")

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            actual_output = actual.reciprocal()
            expected_output = expected.reciprocal()
            self.assert_tensor_matches(actual_output, expected_output, case=case)
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))

    def test_autograd_scalar_empty_offset_and_noncontiguous_match_pytorch_2_13(
        self,
    ):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_leaf, actual_input, actual_weights = self.autograd_case(torch, case)
            expected_leaf, expected_input, expected_weights = self.autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.reciprocal()
            expected_output = expected_input.reciprocal()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "forward")
            )

            if actual_weights is None:
                actual_loss = actual_output if case == "scalar" else actual_output.sum()
                expected_loss = (
                    expected_output if case == "scalar" else expected_output.sum()
                )
            else:
                actual_loss = (actual_output * actual_weights).sum()
                expected_loss = (expected_output * expected_weights).sum()
            actual_loss.backward()
            expected_loss.backward()
            self.assert_tensor_matches(
                actual_leaf.grad,
                expected_leaf.grad,
                case=(case, "gradient"),
            )

    def test_autograd_special_values_match_pytorch_2_13_bitwise(self):
        input_bits = np.asarray(
            (
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x00800000,
                0x80800000,
                0x3E800000,
                0x3F800000,
                0x40000000,
                0x40800000,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
                0x7F812345,
                0xFF812345,
                0x7FC12345,
                0xFFC54321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F800000,
                0xBF800000,
                0x00000000,
                0x80000000,
                0x7F800000,
                0xFF800000,
                0x3F000000,
                0xBF000000,
                0x3F800000,
                0xBF800000,
                0x3F800000,
                0xBF800000,
                0x3F800000,
                0xBF800000,
                0x3F800000,
                0xBF800000,
                0x7FC01234,
                0xFFC05678,
            ),
            dtype=np.uint32,
        )
        tensors = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            output = leaf.reciprocal()
            (output * weights).sum().backward()
            tensors.append((output, leaf.grad))

        self.assert_tensor_matches(
            tensors[0][0], tensors[1][0], case="special forward"
        )
        self.assert_tensor_matches(
            tensors[0][1], tensors[1][1], case="special gradient"
        )

    def test_autograd_accumulation_and_graph_freeing_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            accumulated = module.tensor(
                [1.0, 2.0, -4.0], dtype=module.float32, requires_grad=True
            )
            accumulated.reciprocal().sum().backward()
            first = np.asarray(accumulated.grad).copy()
            accumulated.reciprocal().sum().backward()
            second = np.asarray(accumulated.grad).copy()

            freed = module.tensor(
                [1.0, 2.0, -4.0], dtype=module.float32, requires_grad=True
            )
            loss = freed.reciprocal().sum()
            loss.backward()
            second_backward_error = self.error(loss.backward)
            snapshots.append((first, second, second_backward_error))

        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
        self.assertEqual(snapshots[0][2], snapshots[1][2])

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.reciprocal unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([4.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "reciprocal")
        bound = tensor.reciprocal
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
                    lambda: tensor.reciprocal(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.reciprocal(1, 2),
                    lambda: tensor.reciprocal(input=tensor),
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
tensor = module.tensor([4.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "reciprocal")
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
    intercepted = tensor.reciprocal()
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
        forwarded = tensor.reciprocal()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.reciprocal()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.reciprocal(1)
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

    def test_no_grad_and_unsupported_boundaries_remain_explicit(self):
        actual = torch.tensor([2.0], requires_grad=True)
        expected = reference_torch.tensor([2.0], requires_grad=True)
        self.assertTrue(actual.reciprocal().requires_grad)
        self.assertTrue(expected.reciprocal().requires_grad)

        with torch.no_grad():
            actual_no_grad = actual.reciprocal()
        with reference_torch.no_grad():
            expected_no_grad = expected.reciprocal()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        self.assert_tensor_matches(
            actual.detach().reciprocal(),
            expected.detach().reciprocal(),
            case="detach",
        )

        self.assertFalse(hasattr(torch.Tensor, "reciprocal_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "reciprocal_"))
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaises(RuntimeError):
            torch.tensor([2.0], device="cuda")
        self.assertEqual(reference_torch.device("cuda").type, "cuda")


if __name__ == "__main__":
    unittest.main()
