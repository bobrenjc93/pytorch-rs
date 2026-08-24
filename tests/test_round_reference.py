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
    from .test_round import ROUND_DOC, SPECIAL_OUTPUT_BITS, make_cases
else:
    from test_round import ROUND_DOC, SPECIAL_OUTPUT_BITS, make_cases

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

    def test_seeded_float32_values_match_pytorch_2_13_exactly(self):
        rng = np.random.default_rng(0xA0D0_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(
                tuple(int(value) for value in rng.integers(0, 9, size=rank))
            )

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-1.0e6, 1.0e6, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
                half_count = values[5::23].size
                values[5::23] = (
                    rng.integers(-4096, 4097, size=half_count).astype(np.float32)
                    + np.float32(0.5)
                )
            values = values.reshape(shape)

            actual_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            self.assert_tensor_matches(
                actual_input.round(), expected_input.round(), case=(case, shape)
            )

    def test_ties_to_even_are_independent_of_the_ambient_rounding_mode(self):
        try:
            runtime = ctypes.CDLL(None)
            fegetround = runtime.fegetround
            fesetround = runtime.fesetround
        except (AttributeError, OSError):
            self.skipTest("the platform C runtime does not expose fenv controls")
        fegetround.argtypes = []
        fegetround.restype = ctypes.c_int
        fesetround.argtypes = [ctypes.c_int]
        fesetround.restype = ctypes.c_int

        input_bits = np.asarray(
            [0xC020_0000, 0xBF00_0000, 0x3F00_0000, 0x3FC0_0000, 0x4020_0000],
            dtype=np.uint32,
        )
        expected_bits = np.asarray(
            [0xC000_0000, 0x8000_0000, 0x0000_0000, 0x4000_0000, 0x4000_0000],
            dtype=np.uint32,
        )
        original_rounding = fegetround()

        try:
            alternate_rounding = None
            for candidate in (0x400, 0x80_0000):
                if fesetround(candidate) == 0 and fegetround() == candidate:
                    alternate_rounding = candidate
                    break
            if alternate_rounding is None:
                self.skipTest("the platform does not expose a downward rounding mode")

            actual = torch.tensor(memoryview(input_bits.view(np.float32))).round()
            self.assertEqual(fegetround(), alternate_rounding)
            expected = reference_torch.tensor(
                input_bits.view(np.float32), dtype=reference_torch.float32
            ).round()
            self.assertEqual(fegetround(), alternate_rounding)
        finally:
            self.assertEqual(fesetround(original_rounding), 0)

        np.testing.assert_array_equal(
            self.tensor_values(actual).view(np.uint32), expected_bits
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected).view(np.uint32), expected_bits
        )

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

    def test_supported_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(self.callable_metadata(torch), self.callable_metadata(reference_torch))
        self.assertEqual(
            inspect.getattr_static(torch.Tensor, "round").__doc__, ROUND_DOC
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.5], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "round")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __repr__(self):
        return "RecordingMode()"

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
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.round(0)
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
    "invalid_error_type": invalid_error[0] if invalid_error else None,
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

    def test_inference_only_autograd_boundary_is_explicit(self):
        values = np.linspace(-3.75, 3.75, 24, dtype=np.float32).reshape(2, 3, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_input = actual_leaf.transpose(0, 2)[1]
        expected_input = expected_leaf.transpose(0, 2)[1]

        with self.assertRaisesRegex(
            RuntimeError, r"^round\(\): autograd recording is not supported$"
        ):
            actual_input.round()
        self.assertTrue(expected_input.round().requires_grad)

        with torch.no_grad():
            actual_no_grad = actual_input.round()
        with reference_torch.no_grad():
            expected_no_grad = expected_input.round()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        actual_detached = actual_input.detach().round()
        expected_detached = expected_input.detach().round()
        self.assert_tensor_matches(actual_detached, expected_detached, case="detached")

    def test_explicit_decimals_top_level_and_inplace_boundaries_remain_explicit(self):
        actual = torch.tensor([1.25])
        expected = reference_torch.tensor([1.25])
        for call in (
            lambda: actual.round(0),
            lambda: actual.round(decimals=0),
            lambda: actual.round(decimals=1),
        ):
            with self.assertRaises(TypeError):
                call()
        self.assertEqual(expected.round(decimals=0).tolist(), [1.0])
        self.assertEqual(expected.round(decimals=1).tolist(), [1.2000000476837158])
        self.assertFalse(hasattr(torch, "round"))
        self.assertNotIn("round", torch.__all__)
        self.assertTrue(hasattr(reference_torch, "round"))
        self.assertFalse(hasattr(torch.Tensor, "round_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "round_"))
        self.assertFalse(hasattr(torch, "round_"))
        self.assertNotIn("round_", torch.__all__)


if __name__ == "__main__":
    unittest.main()
