import copy
import inspect
import json
import pickle
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
class TensorExp2ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.exp2 differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def values(tensor):
        if isinstance(tensor, torch.Tensor):
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)

        actual_values = self.values(actual)
        expected_values = self.values(expected)
        with self.subTest(case=case, values=True):
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32),
                    expected_values.reshape(-1).view(np.uint32),
                )
            else:
                np.testing.assert_array_equal(
                    np.isnan(actual_values), np.isnan(expected_values)
                )
                np.testing.assert_array_equal(
                    np.isposinf(actual_values), np.isposinf(expected_values)
                )
                np.testing.assert_array_equal(
                    np.isneginf(actual_values), np.isneginf(expected_values)
                )
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
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
                0x4000_0000,
                0xC000_0000,
                0x42FF_FFFF,
                0x4300_0000,
                0xC314_FFFF,
                0xC315_0000,
                0xC315_FFFF,
                0xC316_0000,
                0xC316_0001,
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
            np.linspace(-3.0, 4.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-5.0, 6.0, 720, dtype=np.float32)
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
            actual_before = self.values(actual).reshape(-1).view(np.uint32).copy()
            expected_before = self.values(expected).reshape(-1).view(np.uint32).copy()
            actual_output = actual.exp2()
            expected_output = expected.exp2()
            self.assert_tensor_matches(
                actual_output,
                expected_output,
                case=case,
                exact_bits=case == "IEEE edges",
            )
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))
            if actual.numel():
                self.assertNotEqual(actual_output.data_ptr(), actual.data_ptr())
                self.assertNotEqual(expected_output.data_ptr(), expected.data_ptr())
            np.testing.assert_array_equal(
                self.values(actual).reshape(-1).view(np.uint32), actual_before
            )
            np.testing.assert_array_equal(
                self.values(expected).reshape(-1).view(np.uint32), expected_before
            )

    def test_seeded_float32_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xE221_213)
        input_bits = rng.integers(0, 2**32, size=4096, dtype=np.uint32)
        values = input_bits.view(np.float32)
        for case, shape in enumerate(((4096,), (64, 64), (8, 16, 32))):
            actual = torch.tensor(memoryview(values)).reshape(shape)
            expected = reference_torch.tensor(memoryview(values)).reshape(shape)
            self.assert_tensor_matches(
                actual.exp2(), expected.exp2(), case=(case, shape)
            )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.exp2 unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "exp2")
        bound = tensor.exp2
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
            "copy_identity": (
                copy.copy(descriptor) is descriptor,
                copy.deepcopy(descriptor) is descriptor,
            ),
            "pickle_identity": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.exp2(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.exp2(1, 2),
                    lambda: tensor.exp2(input=tensor),
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
tensor = module.tensor([0.5], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "exp2")
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
    intercepted = tensor.exp2()
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
        forwarded = tensor.exp2()

old_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.exp2()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None
finally:
    sys.setrecursionlimit(old_recursion_limit)

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.exp2(1)
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

    def test_inference_only_and_unsupported_boundaries_remain_explicit(self):
        actual = torch.tensor([0.5, 1.0], requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^exp2\(\): autograd recording is not supported$",
        ):
            actual.exp2()

        expected = reference_torch.tensor(
            [0.5, 1.0], dtype=reference_torch.float32, requires_grad=True
        )
        self.assertTrue(expected.exp2().requires_grad)

        with torch.no_grad():
            actual_no_grad = actual.exp2()
        with reference_torch.no_grad():
            expected_no_grad = expected.exp2()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        self.assert_tensor_matches(
            actual.detach().exp2(),
            expected.detach().exp2(),
            case="detached",
        )

        self.assertFalse(hasattr(torch, "exp2"))
        self.assertNotIn("exp2", torch.__all__)
        self.assertTrue(hasattr(reference_torch, "exp2"))
        self.assertFalse(hasattr(torch.Tensor, "exp2_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "exp2_"))
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaises(RuntimeError):
            torch.tensor([0.5], device="cuda")
        self.assertEqual(reference_torch.device("cuda").type, "cuda")


if __name__ == "__main__":
    unittest.main()
