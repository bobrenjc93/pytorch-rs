import ctypes
import inspect
import json
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .test_round import SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_round import SPECIAL_OUTPUT_BITS, make_cases

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorRoundReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.round differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

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
                self.tensor_values(actual).reshape(-1).view(np.uint32),
                self.tensor_values(expected).reshape(-1).view(np.uint32),
            )

    def test_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = make_cases(torch)
        expected_cases = make_cases(reference_torch)
        for (case, actual_input, actual_stride), (
            expected_case,
            expected_input,
            expected_stride,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            self.assertEqual(actual_stride, expected_stride)
            actual = actual_input.round()
            expected = expected_input.round()
            self.assert_tensor_matches(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )

    def test_seeded_float32_bit_patterns_match_pytorch_2_13_exactly(self):
        rng = np.random.default_rng(0xA0_0D_213)
        bits = rng.integers(0, 1 << 32, size=4096, dtype=np.uint32)
        actual_input = torch.tensor(memoryview(bits.view(np.float32)))
        expected_input = reference_torch.tensor(memoryview(bits.view(np.float32)))
        self.assert_tensor_matches(
            actual_input.round(), expected_input.round(), case="seeded raw bits"
        )

    def test_round_is_independent_of_ambient_floating_point_rounding_mode(self):
        try:
            runtime = ctypes.CDLL(None)
            fegetround = runtime.fegetround
            fesetround = runtime.fesetround
            nearbyintf = runtime.nearbyintf
        except (AttributeError, OSError):
            self.skipTest("the platform C runtime does not expose fenv controls")
        fegetround.argtypes = []
        fegetround.restype = ctypes.c_int
        fesetround.argtypes = [ctypes.c_int]
        fesetround.restype = ctypes.c_int
        nearbyintf.argtypes = [ctypes.c_float]
        nearbyintf.restype = ctypes.c_float

        input_bits = np.asarray(
            [0xC020_0000, 0xC013_3333, 0x3F33_3333, 0x3FC0_0000],
            dtype=np.uint32,
        )
        expected_bits = np.asarray(
            [0xC000_0000, 0xC000_0000, 0x3F80_0000, 0x4000_0000],
            dtype=np.uint32,
        )
        actual_input = torch.tensor(memoryview(input_bits.view(np.float32)))
        reference_input = reference_torch.tensor(
            memoryview(input_bits.view(np.float32))
        )
        negative_probe = ctypes.c_float(-2.3)
        positive_probe = ctypes.c_float(0.7)
        original_rounding = fegetround()

        try:
            if fesetround(0) != 0 or fegetround() != 0:
                self.skipTest("the platform does not expose round-to-nearest as zero")
            actual_nearest = actual_input.round()
            reference_nearest = reference_input.round()

            downward_rounding = None
            for candidate in (0x400, 0x80_0000):
                if fesetround(candidate) != 0 or fegetround() != candidate:
                    continue
                if (
                    nearbyintf(negative_probe) == -3.0
                    and nearbyintf(positive_probe) == 0.0
                ):
                    downward_rounding = candidate
                    break
            if downward_rounding is None:
                self.skipTest("the platform does not expose a downward rounding mode")

            actual_downward = actual_input.round()
            self.assertEqual(fegetround(), downward_rounding)
            reference_downward = reference_input.round()
            self.assertEqual(fegetround(), downward_rounding)
        finally:
            self.assertEqual(fesetround(original_rounding), 0)

        for case, actual, expected in (
            ("nearest", actual_nearest, reference_nearest),
            ("downward", actual_downward, reference_downward),
            ("actual modes", actual_downward, actual_nearest),
            ("reference modes", reference_downward, reference_nearest),
        ):
            self.assert_tensor_matches(actual, expected, case=case)
            np.testing.assert_array_equal(
                self.tensor_values(actual).view(np.uint32), expected_bits
            )

    def test_detached_and_no_grad_results_match_pytorch_2_13(self):
        values = (
            np.linspace(-5.5, 5.5, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_source = actual_leaf.transpose(0, 2)[1]
        expected_source = expected_leaf.transpose(0, 2)[1]

        self.assert_tensor_matches(
            actual_source.detach().round(),
            expected_source.detach().round(),
            case="detached",
        )
        with torch.no_grad():
            actual = actual_source.round()
        with reference_torch.no_grad():
            expected = expected_source.round()
        self.assert_tensor_matches(actual, expected, case="no_grad")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_metadata(self, module):
        tensor = module.tensor([1.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "round")
        bound = tensor.round
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
            "default_bits": self.tensor_values(bound()).view(np.uint32).tolist(),
        }

    def test_default_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_metadata(torch),
            self.callable_metadata(reference_torch),
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import re
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.5], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "round")
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
    intercepted = tensor.round()
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
        forwarded = tensor.round()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.round()
except Exception as error:
    declining_error = [
        type(error).__name__,
        re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
    ]
else:
    declining_error = None

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_is_descriptor": function is descriptor,
    "types": dispatch_types == (),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
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


if __name__ == "__main__":
    unittest.main()
