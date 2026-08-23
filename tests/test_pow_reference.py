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
class TensorPowReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.pow differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
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
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(-3.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1], None

        leaf = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            source = leaf[1]
            weights = module.tensor(
                np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)
            weights = module.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2).tolist(),
                dtype=module.float32,
            )
            return leaf, source, weights
        raise AssertionError(f"unknown pow autograd case: {case}")

    def test_values_layouts_and_fresh_storage_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for (case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual_output = actual.pow(2)
            expected_output = expected.pow(2)
            self.assert_tensor_matches(actual_output, expected_output, case=case)
            self.assertFalse(actual_output.is_set_to(actual))
            self.assertFalse(expected_output.is_set_to(expected))

    def test_seeded_random_values_match_pytorch_2_13_bitwise(self):
        rng = np.random.default_rng(0x50_A2E)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(16):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 8, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-1.0e20, 1.0e20, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
            values = values.reshape(shape)
            actual_input = (
                torch.zeros(shape, dtype=torch.float32)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            self.assert_tensor_matches(
                actual_input.pow(2), expected_input.pow(2), case=(case, shape)
            )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_leaf, actual_input, actual_weights = self.autograd_case(torch, case)
            expected_leaf, expected_input, expected_weights = self.autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.pow(2)
            expected_output = expected_input.pow(2)
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
                actual_leaf.grad, expected_leaf.grad, case=(case, "gradient")
            )

        snapshots = []
        for module in (torch, reference_torch):
            accumulated = module.tensor(
                [2.0, -3.0], dtype=module.float32, requires_grad=True
            )
            accumulated.pow(2).sum().backward()
            first = np.asarray(accumulated.grad).copy()
            accumulated.pow(exponent=2.0).sum().backward()
            second = np.asarray(accumulated.grad).copy()

            freed = module.tensor(
                [2.0, -3.0], dtype=module.float32, requires_grad=True
            )
            loss = freed.pow(2).sum()
            loss.backward()
            try:
                loss.backward()
            except Exception as error:
                second_backward = type(error).__name__, str(error)
            else:
                self.fail("a freed graph unexpectedly supported repeated backward")
            snapshots.append((first, second, second_backward))

        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
        self.assertEqual(snapshots[0][2], snapshots[1][2])

        for case in ("scalar", "empty", "offset", "noncontiguous"):
            _, actual_input, _ = self.autograd_case(torch, case)
            _, expected_input, _ = self.autograd_case(reference_torch, case)
            with torch.no_grad():
                actual_output = actual_input.pow(2)
            with reference_torch.no_grad():
                expected_output = expected_input.pow(2)
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "no_grad")
            )
            self.assertFalse(actual_output.is_set_to(actual_input))
            self.assertFalse(expected_output.is_set_to(expected_input))

    def test_pow_backward_edge_bits_match_pytorch_2_13(self):
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
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
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x3F00_0000,
                0x3F00_0000,
                0x0000_0001,
                0x0000_0001,
                0x0000_0000,
                0x8000_0000,
                0x3E80_0000,
                0x3E80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        results = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            output = leaf.pow(2)
            (output * weights).sum().backward()
            results.append((output, leaf.grad))

        self.assert_tensor_matches(
            results[0][0], results[1][0], case="edge forward"
        )
        self.assert_tensor_matches(
            results[0][1], results[1][1], case="edge gradient"
        )

    def test_pow_backward_node_identity_matches_pytorch_2_13(self):
        errors = []
        for module in (torch, reference_torch):
            probability = module.tensor(
                [2.0], dtype=module.float32, requires_grad=True
            ).pow(2)
            try:
                module.nn.functional.dropout(
                    module.tensor([1.0], dtype=module.float32),
                    p=probability,
                    training=False,
                )
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail("dropout unexpectedly accepted a powered tensor probability")

        self.assertEqual(errors[0], errors[1])
        self.assertIn("grad_fn=<PowBackward0>", errors[0][1])

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.pow unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([2.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "pow")
        bound = tensor.pow
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
                    lambda: tensor.pow(),
                    lambda: bound(),
                    lambda: descriptor(tensor),
                    lambda: tensor.pow(2, 3),
                    lambda: tensor.pow(2, exponent=2),
                    lambda: tensor.pow(other=2),
                    lambda: tensor.pow(object()),
                    lambda: tensor.pow(exponent=object()),
                    lambda: descriptor(),
                    lambda: descriptor(1, 2),
                    lambda: descriptor(self=tensor, exponent=2),
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

module = importlib.import_module(MODULE)
tensor = module.tensor([2.0, -3.0], dtype=module.float32)
tensor_exponent = module.tensor(2.0, dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "pow")
marker = object()

class RecordingMode(module.overrides.TorchFunctionMode):
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append((func, types, args, kwargs))
        return self.result

observations = []
for style in ("positional", "keyword", "non_two", "tensor"):
    recording = RecordingMode(marker)
    with recording:
        if style == "positional":
            intercepted = tensor.pow(2)
        elif style == "keyword":
            intercepted = tensor.pow(exponent=2)
        elif style == "non_two":
            intercepted = tensor.pow(3)
        else:
            intercepted = tensor.pow(tensor_exponent)
    function, dispatch_types, args, kwargs = recording.calls[0]
    observations.append({
        "style": style,
        "intercepted": intercepted is marker,
        "call_count": len(recording.calls),
        "function_type": type(function).__name__,
        "function_name": function.__name__,
        "function_qualname": function.__qualname__,
        "function_is_descriptor": function is descriptor,
        "types": [value.__name__ for value in dispatch_types],
        "receiver": args[0] is tensor,
        "argument_count": len(args),
        "positional_exponent": (
            len(args) == 2
            and (args[1] is tensor_exponent if style == "tensor" else args[1] in (2, 3))
        ),
        "keyword_exponent": kwargs is not None and kwargs.get("exponent") == 2,
        "kwargs_is_none": kwargs is None,
    })

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = tensor.pow(exponent=2)

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.pow(object())
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

print(json.dumps({
    "observations": observations,
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
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

    def test_deliberately_unsupported_exponents_and_neighbor_apis(self):
        actual = torch.tensor([2.0], dtype=torch.float32)
        expected = reference_torch.tensor([2.0], dtype=reference_torch.float32)
        for actual_exponent, expected_exponent in (
            (3, 3),
            (2.0000000000000004, 2.0000000000000004),
            (2 + 0j, 2 + 0j),
            (torch.tensor(2.0), reference_torch.tensor(2.0)),
        ):
            with self.subTest(exponent=repr(actual_exponent)):
                with self.assertRaises(NotImplementedError):
                    actual.pow(actual_exponent)
                self.assertEqual(expected.pow(expected_exponent).numel(), 1)

        self.assertFalse(hasattr(torch, "pow"))
        self.assertTrue(hasattr(reference_torch, "pow"))
        self.assertFalse(hasattr(torch.Tensor, "pow_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "pow_"))


if __name__ == "__main__":
    unittest.main()
