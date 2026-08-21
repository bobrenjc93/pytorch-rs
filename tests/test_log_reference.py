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
class TensorLogReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("log differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            actual_values = np.asarray(actual, dtype=np.float32).reshape(-1)
            expected_values = expected.detach().cpu().numpy().reshape(-1)
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )
            np.testing.assert_array_equal(
                np.isinf(actual_values), np.isinf(expected_values)
            )
            np.testing.assert_array_equal(
                np.signbit(actual_values[np.isinf(actual_values)]),
                np.signbit(expected_values[np.isinf(expected_values)]),
            )
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.view(np.uint32), expected_values.view(np.uint32)
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
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x007FFFFF,
                0x807FFFFF,
                0x00800000,
                0x80800000,
                0x3F000000,
                0x3F800000,
                0x40000000,
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
        return (
            module.tensor(1.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            module.tensor(memoryview(special_bits.view(np.float32))),
        )

    def test_values_layout_dtype_and_device_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            self.assert_tensor_matches(
                actual.log(),
                expected.log(),
                case=case,
                exact_bits=case == len(actual_cases) - 1,
            )

    def test_seeded_random_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x1062_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(20):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 8, size=rank)))

        special = np.asarray(
            [
                -np.inf,
                -np.finfo(np.float32).max,
                -1.0,
                -np.nextafter(np.float32(0), np.float32(1)),
                -0.0,
                0.0,
                np.nextafter(np.float32(0), np.float32(1)),
                np.finfo(np.float32).tiny,
                0.5,
                1.0,
                2.0,
                np.finfo(np.float32).max,
                np.inf,
                np.nan,
            ],
            dtype=np.float32,
        )
        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = np.exp(rng.uniform(-100.0, 88.0, size=elements)).astype(
                np.float32
            )
            if elements:
                values[::3] *= np.float32(-1.0)
                values[::7] = rng.choice(special, size=values[::7].size)
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
                actual_input.log(), expected_input.log(), case=(case, shape)
            )

    def test_denormal_flush_mode_matches_pytorch_2_13_and_is_restored(self):
        subnormal_bits = np.resize(
            np.asarray(
                (0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF),
                dtype=np.uint32,
            ),
            64,
        )
        values = memoryview(subnormal_bits.view(np.float32))

        reference_torch.set_flush_denormal(False)
        preserved_actual_input = torch.tensor(values)
        preserved_expected_input = reference_torch.tensor(values)
        try:
            self.assertTrue(reference_torch.set_flush_denormal(True))
            flushed_actual_input = torch.tensor(values)
            flushed_expected_input = reference_torch.tensor(values)
            preserved_actual_output = preserved_actual_input.log()
            preserved_expected_output = preserved_expected_input.log()
            flushed_actual_output = flushed_actual_input.log()
            flushed_expected_output = flushed_expected_input.log()
        finally:
            reference_torch.set_flush_denormal(False)

        self.assert_tensor_matches(
            preserved_actual_output,
            preserved_expected_output,
            case="stored before enabling denormal flushing",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            np.asarray(preserved_actual_output).view(np.uint32),
            np.resize(
                np.asarray(
                    (0xC2CE_8ED0, 0x7FC0_0000, 0xC2AE_AC50, 0x7FC0_0000),
                    dtype=np.uint32,
                ),
                subnormal_bits.shape,
            ),
        )
        self.assert_tensor_matches(
            flushed_actual_input,
            flushed_expected_input,
            case="constructed with denormal flushing",
            exact_bits=True,
        )
        self.assert_tensor_matches(
            flushed_actual_output,
            flushed_expected_output,
            case="logged after construction with denormal flushing",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            np.asarray(flushed_actual_output).view(np.uint32),
            np.full(subnormal_bits.shape, 0xFF80_0000, dtype=np.uint32),
        )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("log unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "log")
        bound = tensor.log
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
                    lambda: tensor.log(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.log(1, 2),
                    lambda: tensor.log(input=tensor),
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

    def test_no_grad_matches_while_recording_is_deliberately_unsupported(self):
        actual_leaf = torch.tensor([0.5, 1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [0.5, 1.0, 2.0], requires_grad=True
        )
        with self.assertRaisesRegex(
            RuntimeError, r"^log\(\): autograd recording is not supported$"
        ):
            actual_leaf.log()
        expected_recording = expected_leaf.log()
        self.assertTrue(expected_recording.requires_grad)
        self.assertFalse(expected_recording.is_leaf)

        with torch.no_grad():
            actual = actual_leaf.log()
        with reference_torch.no_grad():
            expected = expected_leaf.log()
        self.assert_tensor_matches(actual, expected, case="no_grad")

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "log")
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
    intercepted = tensor.log()
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
        forwarded = tensor.log()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.log()
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
