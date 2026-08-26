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
class TensorImagReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.imag differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        scalar_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = module.tensor(memoryview(scalar_bits.view(np.float32)))
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        return (
            *(scalar_storage[index] for index in range(len(scalar_bits))),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            module.zeros((2, 3, 4, 5), dtype=module.float32).contiguous(
                memory_format=module.channels_last
            ),
            leaf,
            (leaf * 3.0).transpose(0, 1)[1],
        )

    def metadata(self, tensor, module):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.layout),
            tensor.is_contiguous(),
            tensor.is_contiguous(memory_format=module.channels_last),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def error_contract(self, action):
        try:
            action()
        except Exception as error:
            return error, (type(error).__name__, str(error), error.args)
        self.fail("Tensor.imag unexpectedly accepted the operation")

    def read_contract(self, tensor, module):
        metadata = self.metadata(tensor, module)
        pointer = tensor.data_ptr()
        alias = tensor.detach()
        before_bits = np.asarray(alias).reshape(-1).view(np.uint32).copy()
        errors = [self.error_contract(lambda: tensor.imag) for _ in range(3)]
        after_bits = np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()
        return {
            "errors": [contract for _, contract in errors],
            "fresh_errors": all(
                first is not second
                for index, (first, _) in enumerate(errors)
                for second, _ in errors[index + 1 :]
            ),
            "metadata": metadata,
            "metadata_unchanged": self.metadata(tensor, module) == metadata,
            "pointer_unchanged": tensor.data_ptr() == pointer,
            "storage_unchanged": tensor.is_set_to(alias),
            "before_bits": before_bits,
            "after_bits": after_bits,
        }

    def test_errors_and_state_match_for_every_supported_tensor_shape(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_contract = self.read_contract(actual, torch)
                expected_contract = self.read_contract(expected, reference_torch)
                for key in ("before_bits", "after_bits"):
                    np.testing.assert_array_equal(
                        actual_contract.pop(key), expected_contract.pop(key)
                    )
                self.assertEqual(actual_contract, expected_contract)

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        before = (self.metadata(leaf, module), self.metadata(non_leaf, module))
        first_errors = (
            self.error_contract(lambda: leaf.imag)[1],
            self.error_contract(lambda: non_leaf.imag)[1],
        )
        after = (self.metadata(leaf, module), self.metadata(non_leaf, module))

        non_leaf.sum().backward()
        gradient = leaf.grad
        gradient_before = self.metadata(gradient, module)
        second_errors = (
            self.error_contract(lambda: leaf.imag)[1],
            self.error_contract(lambda: non_leaf.imag)[1],
        )
        gradient_after = self.metadata(gradient, module)
        return {
            "first_errors": first_errors,
            "metadata_before": before,
            "metadata_after": after,
            "second_errors": second_errors,
            "gradient_identity": leaf.grad is gradient,
            "gradient_metadata_before": gradient_before,
            "gradient_metadata_after": gradient_after,
            "gradient": np.asarray(gradient).copy(),
        }

    def test_failed_reads_preserve_leaf_and_non_leaf_autograd_state(self):
        actual = self.autograd_contract(torch)
        expected = self.autograd_contract(reference_torch)
        np.testing.assert_array_equal(
            actual.pop("gradient"), expected.pop("gradient")
        )
        self.assertEqual(actual, expected)

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "imag")
        tensor = module.tensor([1.0], dtype=module.float32)
        replacement = module.tensor([2.0], dtype=module.float32)
        value_error = self.error_contract(
            lambda: descriptor.__get__(tensor, module.Tensor)
        )[1]
        receiver_errors = (
            self.error_contract(lambda: descriptor.__get__(1, int))[1],
            self.error_contract(lambda: descriptor.__set__(1, replacement))[1],
            self.error_contract(lambda: descriptor.__delete__(1))[1],
        )
        mutation_errors = []
        for action in (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__delete__(tensor),
        ):
            mutation_errors.append(self.error_contract(action))
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_getset": type(descriptor) is types.GetSetDescriptorType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "doc": descriptor.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "has_text_signature": hasattr(descriptor, "__text_signature__"),
            "repr": repr(descriptor),
            "class_identity": module.Tensor.imag is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value_error": value_error,
            "receiver_errors": receiver_errors,
            "mutation_errors": [contract for _, contract in mutation_errors],
            "fresh_mutation_errors": all(
                first is not second
                for index, (first, _) in enumerate(mutation_errors)
                for second, _ in mutation_errors[index + 1 :]
            ),
            "value_unchanged": tensor.tolist() == [1.0],
        }

    def test_descriptor_and_mutation_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "imag")
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
    intercepted = tensor.imag
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

try:
    with ForwardingMode("lower"):
        with ForwardingMode("upper"):
            tensor.imag
except Exception as error:
    forwarding_error = [type(error).__name__, str(error), error.args]
else:
    forwarding_error = None

sys.setrecursionlimit(80)
class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

lower = RecordingMode(marker)
upper = DecliningMode()
try:
    with lower:
        with upper:
            tensor.imag
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

replacement = module.tensor([2.0], dtype=module.float32)
mutation_mode = RecordingMode(marker)
mutation_errors = []
for action in (
    lambda: setattr(tensor, "imag", replacement),
    lambda: delattr(tensor, "imag"),
    lambda: descriptor.__set__(tensor, replacement),
    lambda: descriptor.__delete__(tensor),
):
    try:
        with mutation_mode:
            action()
    except Exception as error:
        mutation_errors.append([type(error).__name__, str(error), error.args])
    else:
        mutation_errors.append(None)

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_self": function.__self__ is descriptor,
    "function_equals_descriptor_get": function == descriptor.__get__,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarding_error": forwarding_error,
    "declining_error": declining_error,
    "declining_calls": upper.calls,
    "lower_skipped": len(lower.calls) == 0,
    "mutation_errors": mutation_errors,
    "mutation_mode_calls": len(mutation_mode.calls),
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

    def test_broader_complex_surface_remains_deliberately_unsupported(self):
        self.assertTrue(hasattr(reference_torch, "imag"))
        self.assertFalse(hasattr(torch, "imag"))
        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(reference_torch, name))
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
