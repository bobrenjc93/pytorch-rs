import inspect
import json
import pickle
import subprocess
import sys
import types
import unittest
from multiprocessing.reduction import ForkingPickler

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ExpReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.exp differentials require pinned PyTorch 2.13.0"
            )

    def assert_tensor_matches(
        self, actual, expected, *, case, exact_non_nan_bits=False
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        actual_values = np.asarray(actual, dtype=np.float32)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, values=True):
            if exact_non_nan_bits:
                np.testing.assert_array_equal(
                    np.isnan(actual_values), np.isnan(expected_values)
                )
                non_nan = ~np.isnan(expected_values.reshape(-1))
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32)[non_nan],
                    expected_values.reshape(-1).view(np.uint32)[non_nan],
                )
            else:
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(1.5, dtype=module.float32, requires_grad=True)
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
        raise AssertionError(f"unknown Tensor.exp autograd case: {case}")

    def test_seeded_random_shapes_and_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0xE11E_213)
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))

        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(
                tuple(int(value) for value in rng.integers(0, 9, size=rank))
            )

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            selector = rng.integers(0, 4, size=elements)
            values = np.empty(elements, dtype=np.float32)
            values[selector == 0] = rng.uniform(
                -105.0, 89.5, size=np.count_nonzero(selector == 0)
            )
            values[selector == 1] = rng.normal(
                0.0, 24.0, size=np.count_nonzero(selector == 1)
            )
            values[selector == 2] = rng.uniform(
                -1.0e-4, 1.0e-4, size=np.count_nonzero(selector == 2)
            )
            values[selector == 3] = rng.choice(
                np.array(
                    [
                        -104.0,
                        -103.5,
                        -103.0,
                        -100.0,
                        -88.0,
                        -smallest_subnormal,
                        -0.0,
                        0.0,
                        smallest_subnormal,
                        1.0,
                        88.0,
                        88.75,
                        89.0,
                    ],
                    dtype=np.float32,
                ),
                size=np.count_nonzero(selector == 3),
            )
            values = values.reshape(shape)

            native_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            native_output = native_input.exp()
            native = np.asarray(native_output, dtype=np.float32)
            expected_tensor = reference_torch.tensor(
                values, dtype=reference_torch.float32
            ).exp()
            expected = expected_tensor.cpu().numpy()

            with self.subTest(case=case, shape=shape):
                self.assertEqual(native_output.shape, expected_tensor.shape)
                self.assertEqual(native_output.stride(), expected_tensor.stride())
                self.assertIs(native_output.dtype, torch.float32)
                self.assertEqual(native_output.device, torch.device("cpu"))
                np.testing.assert_allclose(
                    native,
                    expected,
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )

    def test_scalar_empty_offset_and_strided_results_keep_fresh_storage(self):
        actual_base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        expected_base = reference_torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        )
        actual_strided = actual_base.transpose(0, 2)
        expected_strided = expected_base.transpose(0, 2)
        actual_cases = (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", actual_strided[1]),
            ("strided", actual_strided),
        )
        expected_cases = (
            ("scalar", reference_torch.tensor(-0.0)),
            (
                "empty",
                reference_torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            ),
            ("offset", expected_strided[1]),
            ("strided", expected_strided),
        )

        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            actual_output = actual_input.exp()
            expected_output = expected_input.exp()
            self.assert_tensor_matches(actual_output, expected_output, case=case)
            self.assertFalse(actual_output.is_set_to(actual_input))
            self.assertFalse(expected_output.is_set_to(expected_input))

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.exp unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "exp")
        bound = tensor.exp
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
                    lambda: tensor.exp(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.exp(1, 2),
                    lambda: tensor.exp(input=tensor),
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
    def descriptor_pickle_contract(module):
        descriptor = inspect.getattr_static(module.Tensor, "exp")
        return tuple(
            (
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor,
                pickle.loads(ForkingPickler.dumps(descriptor, protocol))
                is descriptor,
            )
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
        )

    def test_descriptor_pickle_round_trips_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_pickle_contract(torch),
            self.descriptor_pickle_contract(reference_torch),
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([0.5], dtype=module.float32, requires_grad=True)
descriptor = inspect.getattr_static(module.Tensor, "exp")
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
    intercepted = tensor.exp()
function, dispatch_types, args, kwargs = recording.calls[0]

forwarding_events = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label, events):
        self.label = label
        self.events = events

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.events.append(
            [self.label, len(module.overrides._get_current_function_mode_stack())]
        )
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower", forwarding_events):
    with ForwardingMode("upper", forwarding_events):
        forwarded = tensor.exp()

extreme = module.zeros((0,), dtype=module.float32).reshape((0, sys.maxsize, 3))
bypass = RecordingMode(marker)
with bypass:
    bypassed = extreme.exp()

old_recursion_limit = sys.getrecursionlimit()
sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        try:
            tensor.exp()
        except Exception as error:
            declining_error = [type(error).__name__, str(error)]
            declining_stack_inside = len(
                module.overrides._get_current_function_mode_stack()
            )
        else:
            declining_error = None
            declining_stack_inside = None
finally:
    sys.setrecursionlimit(old_recursion_limit)

class RaisingMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0
        self.handler_stack_depth = None

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        self.handler_stack_depth = len(
            module.overrides._get_current_function_mode_stack()
        )
        raise ValueError("exp mode failed")

recovery_events = []
raising = RaisingMode()
with ForwardingMode("lower", recovery_events):
    try:
        with raising:
            tensor.exp()
    except Exception as error:
        raising_error = [type(error).__name__, str(error)]
        raising_stack_inside_lower = len(
            module.overrides._get_current_function_mode_stack()
        )
    else:
        raising_error = None
        raising_stack_inside_lower = None
    recovered = tensor.exp()

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.exp(1)
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

native = tensor.detach().exp()
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
    "forwarding_events": forwarding_events,
    "forwarded_matches_native": forwarded.tolist() == native.tolist(),
    "bypassed": bypassed is marker,
    "bypass_calls": len(bypass.calls),
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "declining_stack_inside": declining_stack_inside,
    "raising_error": raising_error,
    "raising_calls": raising.calls,
    "raising_handler_stack_depth": raising.handler_stack_depth,
    "raising_stack_inside_lower": raising_stack_inside_lower,
    "recovery_events": recovery_events,
    "recovered_matches_native": recovered.tolist() == native.tolist(),
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
            actual_output = actual_input.exp()
            expected_output = expected_input.exp()
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
                0xFF80_0000,
                0xC2D0_0000,
                0xC2CF_0000,
                0xC2CE_0000,
                0xC2C8_0000,
                0xC2B0_0000,
                0xBF80_0000,
                0x8000_0000,
                0x0000_0000,
                0x3F80_0000,
                0x4120_0000,
                0x42A0_0000,
                0x42B0_0000,
                0x42B1_8000,
                0x42B2_0000,
                0x7F80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x7F80_0000,
                0xBF80_0000,
                0x3F00_0000,
                0xBF00_0000,
                0x0000_0001,
                0xFF80_0000,
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x0000_0001,
                0x3E80_0000,
                0x4000_0000,
                0xBF80_0000,
                0x3F80_0000,
                0x0000_0000,
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
            output = leaf.exp()
            (output * weights).sum().backward()
            tensors.append((output, leaf.grad))

        self.assert_tensor_matches(
            tensors[0][0],
            tensors[1][0],
            case="special forward",
            exact_non_nan_bits=True,
        )
        self.assert_tensor_matches(
            tensors[0][1],
            tensors[1][1],
            case="special gradient",
            exact_non_nan_bits=True,
        )

    def test_exp_backward_node_identity_matches_pytorch_2_13(self):
        errors = []
        for module in (torch, reference_torch):
            probability = module.tensor(
                [1.0], dtype=module.float32, requires_grad=True
            ).exp()
            try:
                module.nn.functional.dropout(
                    module.tensor([1.0], dtype=module.float32),
                    p=probability,
                    training=False,
                )
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail("dropout unexpectedly accepted an exponential probability")

        self.assertEqual(errors[0], errors[1])
        self.assertIn("grad_fn=<ExpBackward0>", errors[0][1])

    def test_composition_accumulation_repeated_backward_and_no_grad_match(self):
        snapshots = []
        for module in (torch, reference_torch):
            composed = module.tensor(
                [-1.0, 0.5, 2.0], dtype=module.float32, requires_grad=True
            )
            composed.sin().exp().sum().backward()
            composed_gradient = np.asarray(composed.grad, dtype=np.float32).copy()

            accumulated = module.tensor(
                [-1.0, 0.0, 1.0, 4.0],
                dtype=module.float32,
                requires_grad=True,
            )
            accumulated.exp().sum().backward()
            first = np.asarray(accumulated.grad, dtype=np.float32).copy()
            accumulated.exp().sum().backward()
            second = np.asarray(accumulated.grad, dtype=np.float32).copy()

            freed = module.tensor(
                [-1.0, 0.0, 1.0], dtype=module.float32, requires_grad=True
            )
            loss = freed.exp().sum()
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
            actual = actual_leaf.transpose(0, 1)[1].exp()
        with reference_torch.no_grad():
            expected = expected_leaf.transpose(0, 1)[1].exp()
        self.assert_tensor_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_leaf.grad)
        self.assertTrue(actual_leaf.exp().requires_grad)

        self.assert_tensor_matches(
            actual_leaf.detach().exp(),
            expected_leaf.detach().exp(),
            case="detached input",
        )


if __name__ == "__main__":
    unittest.main()
