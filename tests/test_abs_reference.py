import ctypes
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
class TensorAbsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.abs differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_bits(tensor):
        if isinstance(tensor, torch.Tensor):
            return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)
        return tensor.detach().cpu().numpy().reshape(-1).view(np.uint32)

    @staticmethod
    def raw_storage_bits(tensor):
        storage = (ctypes.c_uint32 * tensor.numel()).from_address(tensor.data_ptr())
        return tuple(storage)

    def assert_tensor_matches(self, actual, expected, *, case, raw_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            if raw_bits:
                self.assertEqual(
                    self.raw_storage_bits(actual), self.raw_storage_bits(expected)
                )
            else:
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
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
        special_values = special_bits.view(np.float32)
        special = (
            module.tensor(memoryview(special_values))
            if module is torch
            else module.tensor(special_values, dtype=module.float32)
        )
        channels_last = module.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("empty singleton trailing", module.zeros((0, 1), dtype=module.float32)),
            ("empty singleton middle", module.zeros((0, 1, 2), dtype=module.float32)),
            (
                "empty singleton surrounding",
                module.zeros((1, 0, 1), dtype=module.float32),
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            ("IEEE edges", special),
        )

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual_output = actual.abs()
            expected_output = expected.abs()
            self.assert_tensor_matches(
                actual_output,
                expected_output,
                case=case,
                raw_bits=case == "IEEE edges",
            )
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))
            if actual.numel():
                self.assertNotEqual(actual_output.data_ptr(), actual.data_ptr())
                self.assertNotEqual(expected_output.data_ptr(), expected.data_ptr())

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.abs unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([-4.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "abs")
        bound = tensor.abs
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
                    lambda: tensor.abs(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.abs(1, 2),
                    lambda: tensor.abs(input=tensor),
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
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([-4.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "abs")
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
    intercepted = tensor.abs()
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
        forwarded = tensor.abs()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.abs()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.abs(1)
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

    def test_finite_owned_scalar_autograd_bits_match_pytorch_2_13(self):
        cases = (
            0x4000_0000,
            0xC000_0000,
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7F7F_FFFF,
            0xFF7F_FFFF,
        )
        for input_bits in cases:
            with self.subTest(input_bits=f"0x{input_bits:08x}"):
                value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
                actual_leaf = torch.tensor(value, requires_grad=True)
                expected_leaf = reference_torch.tensor(
                    value, dtype=reference_torch.float32, requires_grad=True
                )
                actual_output = actual_leaf.abs()
                expected_output = expected_leaf.abs()

                self.assert_tensor_matches(
                    actual_output, expected_output, case="forward", raw_bits=True
                )
                self.assertEqual(type(expected_output.grad_fn).__name__, "AbsBackward0")
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        actual_output
                    ),
                    ", grad_fn=<AbsBackward0>",
                )

                actual_output.backward()
                expected_output.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case="gradient",
                    raw_bits=True,
                )
                self.assertEqual(
                    self.error(actual_output.backward),
                    self.error(expected_output.backward),
                )

    def test_weighted_composition_and_accumulation_match_pytorch_2_13(self):
        weighted_cases = (
            (0x4000_0000, 0xC040_0000),
            (0xC000_0000, 0xC040_0000),
            (0x0000_0000, 0xC040_0000),
            (0x8000_0000, 0xC040_0000),
        )
        for input_bits, weight_bits in weighted_cases:
            value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
            weight = np.asarray(weight_bits, dtype=np.uint32).view(np.float32).item()
            actual_leaf = torch.tensor(value, requires_grad=True)
            expected_leaf = reference_torch.tensor(
                value, dtype=reference_torch.float32, requires_grad=True
            )
            (actual_leaf.abs() * weight).backward()
            (expected_leaf.abs() * weight).backward()
            self.assert_tensor_matches(
                actual_leaf.grad,
                expected_leaf.grad,
                case=(f"0x{input_bits:08x}", f"0x{weight_bits:08x}"),
                raw_bits=True,
            )

        gradients = []
        for module in (torch, reference_torch):
            composed = module.tensor(
                -0.5, dtype=module.float32, requires_grad=True
            )
            composed.abs().sin().backward()

            accumulated = module.tensor(
                -2.0, dtype=module.float32, requires_grad=True
            )
            (accumulated.abs() * -3.0).backward()
            first = self.tensor_bits(accumulated.grad).copy()
            (accumulated.abs() * 0.5).backward()
            second = self.tensor_bits(accumulated.grad).copy()
            gradients.append(
                (self.tensor_bits(composed.grad).copy(), first, second)
            )

        for actual, expected in zip(gradients[0], gradients[1], strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_autograd_modes_boundaries_and_unsupported_surface_are_explicit(self):
        actual_scalar = torch.tensor(-0.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(
            -0.5, dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_no_grad_scalar = actual_scalar.abs()
        with reference_torch.no_grad():
            expected_no_grad_scalar = expected_scalar.abs()
        self.assert_tensor_matches(
            actual_no_grad_scalar,
            expected_no_grad_scalar,
            case="scalar no_grad",
            raw_bits=True,
        )
        self.assert_tensor_matches(
            actual_scalar.detach().abs(),
            expected_scalar.detach().abs(),
            case="scalar detached",
            raw_bits=True,
        )

        higher_order = torch.tensor(-0.25, requires_grad=True)
        loss = higher_order.abs()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        loss.backward()
        self.assertEqual(higher_order.grad.item(), -1.0)

        message = r"^abs\(\): autograd recording is not supported$"
        for bits in (0x7F80_0000, 0xFF80_0000, 0x7FC1_2345, 0xFFC5_4321):
            value = np.asarray(bits, dtype=np.uint32).view(np.float32).item()
            actual_nonfinite = torch.tensor(value, requires_grad=True)
            with self.subTest(nonfinite=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_nonfinite.abs()
            self.assertIsNone(actual_nonfinite.grad)
            actual_nonfinite.sum().backward()
            self.assertEqual(actual_nonfinite.grad.item(), 1.0)

        actual = torch.tensor([-2.0, -0.0, 3.0], requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, message):
            actual.abs()

        view_base = torch.tensor([-0.5], requires_grad=True)
        scalar_view = view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            scalar_view.abs()
        scalar_view.backward()
        self.assertEqual(view_base.grad.tolist(), [1.0])

        nonleaf_base = torch.tensor(-0.5, requires_grad=True)
        nonleaf = nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            nonleaf.abs()
        nonleaf.backward()
        np.testing.assert_allclose(
            nonleaf_base.grad.item(), np.cos(np.float32(0.5)), rtol=0.0, atol=0.0
        )

        with torch.no_grad():
            no_grad_view = actual[0]
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            no_grad_view.abs()

        expected = reference_torch.tensor(
            [-2.0, -0.0, 3.0], dtype=reference_torch.float32, requires_grad=True
        )

        with torch.no_grad():
            actual_no_grad = actual.abs()
        with reference_torch.no_grad():
            expected_no_grad = expected.abs()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        actual_detached = actual.detach().abs()
        expected_detached = expected.detach().abs()
        self.assert_tensor_matches(
            actual_detached, expected_detached, case="detached"
        )

        actual.sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(actual.grad),
            self.tensor_bits(reference_torch.ones_like(expected)),
        )

        for name in ("abs", "absolute"):
            self.assertFalse(hasattr(torch, name))
            self.assertTrue(hasattr(reference_torch, name))
        for name in ("absolute", "abs_", "absolute_"):
            self.assertFalse(hasattr(torch.Tensor, name))
            self.assertTrue(hasattr(reference_torch.Tensor, name))


if __name__ == "__main__":
    unittest.main()
