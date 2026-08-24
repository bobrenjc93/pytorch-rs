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
class TensorTanhReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.tanh differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        actual_values = self.tensor_values(actual)
        expected_values = self.tensor_values(expected)
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=3.0 * np.finfo(np.float32).eps,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )
            zero_mask = expected_values == 0
            np.testing.assert_array_equal(
                actual_values[zero_mask].view(np.uint32),
                expected_values[zero_mask].view(np.uint32),
            )
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32),
                    expected_values.reshape(-1).view(np.uint32),
                )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
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
        channels_last = module.tensor(
            np.linspace(-3.0, 3.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-3.0, 3.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
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
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(1.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf, None

        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "contiguous":
            weights = module.tensor(values.tolist(), dtype=module.float32)
            return leaf, leaf, weights
        if case == "offset":
            source = leaf[1]
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 12, dtype=np.float32)
                .reshape(3, 4)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        if case == "strided":
            source = leaf.transpose(0, 2)
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 24, dtype=np.float32)
                .reshape(4, 3, 2)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        raise AssertionError(f"unknown Tensor.tanh autograd case: {case}")

    def test_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual = actual_input.tanh()
            expected = expected_input.tanh()
            self.assert_tensor_matches(
                actual,
                expected,
                case=case,
                exact_bits=case == "numerical edges",
            )
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def test_seeded_random_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x7A4E_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(24):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 9, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(0.0, 5.0, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
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
                actual_input.tanh(),
                expected_input.tanh(),
                case=(case, shape),
            )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.tanh unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "tanh")
        bound = tensor.tanh
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
                    lambda: tensor.tanh(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.tanh(1, 2),
                    lambda: tensor.tanh(input=tensor),
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
tensor = module.tensor([0.5], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "tanh")
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
    intercepted = tensor.tanh()
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
        forwarded = tensor.tanh()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.tanh()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.tanh(1)
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

    def test_autograd_scalar_empty_contiguous_offset_and_strided_match_pytorch_2_13(
        self,
    ):
        for case in ("scalar", "empty", "contiguous", "offset", "strided"):
            actual_leaf, actual_input, actual_weights = self.autograd_case(torch, case)
            expected_leaf, expected_input, expected_weights = self.autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.tanh()
            expected_output = expected_input.tanh()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "forward")
            )
            self.assertFalse(actual_output.is_set_to(actual_input))
            self.assertFalse(expected_output.is_set_to(expected_input))

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

    def test_autograd_special_values_match_pytorch_2_13_bits(self):
        input_bits = np.asarray(
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
                0x4000_0000,
                0xC000_0000,
                0x4120_0000,
                0xC120_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x4000_0000,
                0xC000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        tensors = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            output = leaf.tanh()
            (output * weights).sum().backward()
            tensors.append((output, leaf.grad))

        self.assert_tensor_matches(
            tensors[0][0], tensors[1][0], case="special forward", exact_bits=True
        )
        self.assert_tensor_matches(
            tensors[0][1], tensors[1][1], case="special gradient", exact_bits=True
        )

    def test_backward_uses_saved_forward_result_across_rounding_mode_changes(self):
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

        original_rounding = fegetround()
        input_bits = np.asarray([0x411C_49E2], dtype=np.uint32)

        try:
            if fesetround(0) != 0 or fegetround() != 0:
                self.skipTest("the platform does not expose round-to-nearest as zero")

            downward_rounding = None
            for candidate in (0x400, 0x80_0000):
                if fesetround(candidate) != 0 or fegetround() != candidate:
                    continue
                probe = torch.tensor(memoryview(input_bits.view(np.float32))).tanh()
                probe_bits = np.asarray(probe, dtype=np.float32).view(np.uint32).item()
                if probe_bits == 0x3F7F_FFFF:
                    downward_rounding = candidate
                    break
            if downward_rounding is None:
                self.skipTest("native tanh is not sensitive to available fenv modes")

            self.assertEqual(fesetround(0), 0)
            actual_leaf = torch.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            expected_leaf = reference_torch.tensor(
                input_bits.view(np.float32), requires_grad=True
            )
            actual_output = actual_leaf.tanh()
            expected_output = expected_leaf.tanh()

            self.assertEqual(fesetround(downward_rounding), 0)
            actual_output.backward()
            expected_output.backward()
        finally:
            self.assertEqual(fesetround(original_rounding), 0)

        actual_output_bits = (
            np.asarray(actual_output, dtype=np.float32).view(np.uint32).item()
        )
        expected_output_bits = expected_output.detach().numpy().view(np.uint32).item()
        actual_gradient_bits = (
            np.asarray(actual_leaf.grad, dtype=np.float32).view(np.uint32).item()
        )
        expected_gradient_bits = (
            expected_leaf.grad.detach().numpy().view(np.uint32).item()
        )
        self.assertEqual(actual_output_bits, 0x3F80_0000)
        self.assertEqual(actual_output_bits, expected_output_bits)
        self.assertEqual(actual_gradient_bits, 0x8000_0000)
        self.assertEqual(actual_gradient_bits, expected_gradient_bits)

    def test_tanh_backward_node_identity_matches_pytorch_2_13(self):
        errors = []
        for module in (torch, reference_torch):
            probability = module.tensor(
                [-1.0], dtype=module.float32, requires_grad=True
            ).tanh()
            try:
                module.nn.functional.dropout(
                    module.tensor([1.0], dtype=module.float32),
                    p=probability,
                    training=False,
                )
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail("dropout unexpectedly accepted a tanh probability")

        self.assertEqual(errors[0], errors[1])
        self.assertIn("grad_fn=<TanhBackward0>", errors[0][1])

    def test_composition_accumulation_repeated_backward_and_no_grad_match(self):
        snapshots = []
        for module in (torch, reference_torch):
            composed = module.tensor(
                [-1.0, 0.5, 2.0], dtype=module.float32, requires_grad=True
            )
            composed.sin().tanh().sum().backward()
            composed_gradient = np.asarray(composed.grad, dtype=np.float32).copy()

            accumulated = module.tensor(
                [-1.0, 0.0, 1.0, 4.0],
                dtype=module.float32,
                requires_grad=True,
            )
            accumulated.tanh().sum().backward()
            first = np.asarray(accumulated.grad, dtype=np.float32).copy()
            accumulated.tanh().sum().backward()
            second = np.asarray(accumulated.grad, dtype=np.float32).copy()

            freed = module.tensor(
                [-1.0, 0.0, 1.0], dtype=module.float32, requires_grad=True
            )
            loss = freed.tanh().sum()
            loss.backward()
            repeated_backward = self.error(loss.backward)
            snapshots.append((composed_gradient, first, second, repeated_backward))

        for index in range(3):
            np.testing.assert_allclose(
                snapshots[0][index],
                snapshots[1][index],
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
            )
        self.assertEqual(snapshots[0][3], snapshots[1][3])

        actual_leaf = torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            actual = actual_leaf.transpose(0, 1)[1].tanh()
        with reference_torch.no_grad():
            expected = expected_leaf.transpose(0, 1)[1].tanh()
        self.assert_tensor_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_leaf.grad)
        self.assertTrue(actual_leaf.tanh().requires_grad)

        self.assert_tensor_matches(
            actual_leaf.detach().tanh(),
            expected_leaf.detach().tanh(),
            case="detached input",
        )

    def test_unsupported_functional_and_inplace_boundaries_remain_explicit(self):
        self.assertFalse(hasattr(torch, "tanh"))
        self.assertTrue(hasattr(reference_torch, "tanh"))
        self.assertFalse(hasattr(torch.nn.functional, "tanh"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "tanh"))
        self.assertFalse(hasattr(torch.Tensor, "tanh_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "tanh_"))


if __name__ == "__main__":
    unittest.main()
