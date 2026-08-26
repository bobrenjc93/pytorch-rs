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
        scalar_storage = module.tensor(
            memoryview(scalar_bits.view(np.float32)), dtype=module.float32
        )
        base = module.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 3)
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        return (
            *(scalar_storage[index] for index in range(len(scalar_bits))),
            base,
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            base.contiguous(memory_format=module.channels_last),
            module.zeros((2, 3, 4, 5, 6), dtype=module.float32).contiguous(
                memory_format=module.channels_last_3d
            ),
            leaf,
            (leaf * 3.0).transpose(0, 1)[1],
            module.zeros((1, 0, 1, 1, 1, 1), dtype=module.float32),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return error
        self.fail("Tensor.imag unexpectedly accepted the operation")

    def error_read_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        pointer = tensor.data_ptr()
        alias = tensor.detach()
        bits = np.asarray(alias).reshape(-1).view(np.uint32).copy()
        errors = [self.error(lambda: tensor.imag) for _ in range(3)]
        return {
            "errors": tuple((type(error).__name__, str(error)) for error in errors),
            "fresh_errors": len({id(error) for error in errors}) == len(errors),
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                str(tensor.dtype),
                str(tensor.device),
                str(tensor.layout),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.output_nr,
            ),
            "pointer_unchanged": tensor.data_ptr() == pointer,
            "storage_alias_unchanged": tensor.is_set_to(alias),
            "bits_unchanged": np.array_equal(
                np.asarray(tensor.detach()).reshape(-1).view(np.uint32), bits
            ),
        }

    def test_float32_errors_and_side_effect_boundaries_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.error_read_contract(actual),
                    self.error_read_contract(expected),
                )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        before = (
            leaf.requires_grad,
            leaf.is_leaf,
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            tuple(non_leaf.shape),
            non_leaf.stride(),
            non_leaf.storage_offset(),
        )
        errors = (
            self.error(lambda: leaf.imag),
            self.error(lambda: non_leaf.imag),
        )
        after = (
            leaf.requires_grad,
            leaf.is_leaf,
            non_leaf.requires_grad,
            non_leaf.is_leaf,
            tuple(non_leaf.shape),
            non_leaf.stride(),
            non_leaf.storage_offset(),
        )
        non_leaf.sum().backward()
        gradient = leaf.grad
        final_error = self.error(lambda: leaf.imag)
        return {
            "before": before,
            "after": after,
            "errors": tuple((type(error).__name__, str(error)) for error in errors),
            "gradient": np.asarray(gradient).copy(),
            "gradient_identity_preserved": leaf.grad is gradient,
            "final_error": (type(final_error).__name__, str(final_error)),
        }

    def test_autograd_graph_preservation_matches_pytorch_2_13(self):
        actual = self.autograd_contract(torch)
        expected = self.autograd_contract(reference_torch)
        np.testing.assert_array_equal(
            actual.pop("gradient"), expected.pop("gradient")
        )
        self.assertEqual(actual, expected)

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "imag")
        tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
        replacement = module.tensor([2.0], dtype=module.float32)
        metadata = (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        actions = (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        )
        mutation_errors = tuple(
            (type(error).__name__, str(error))
            for error in (self.error(action) for action in actions)
        )
        receiver_errors = tuple(
            (type(error).__name__, str(error))
            for error in (
                self.error(lambda: descriptor.__get__(1, int)),
                self.error(lambda: descriptor.__set__(1, replacement)),
                self.error(lambda: descriptor.__delete__(1)),
            )
        )
        read_error = self.error(lambda: descriptor.__get__(tensor, module.Tensor))
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
            "repr": repr(descriptor),
            "class_identity": module.Tensor.imag is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "read_error": (type(read_error).__name__, str(read_error)),
            "mutation_errors": mutation_errors,
            "receiver_errors": receiver_errors,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
            ),
        }

    def test_descriptor_and_mutation_semantics_match_pytorch_2_13(self):
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
tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
replacement = module.tensor([2.0], dtype=module.float32)
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
    forwarding_error = [type(error).__name__, str(error)]
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

mutation_calls = []
for action in (
    lambda: setattr(tensor, "imag", replacement),
    lambda: delattr(tensor, "imag"),
    lambda: descriptor.__set__(tensor, replacement),
    lambda: descriptor.__delete__(tensor),
):
    mode = RecordingMode(marker)
    try:
        with mode:
            action()
    except Exception as error:
        outcome = [type(error).__name__, str(error)]
    else:
        outcome = None
    mutation_calls.append([len(mode.calls), outcome])

errors = []
for _ in range(3):
    try:
        tensor.imag
    except Exception as error:
        errors.append(error)

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
    "mutation_calls": mutation_calls,
    "fresh_errors": len({id(error) for error in errors}) == len(errors),
    "ordinary_errors": [[type(error).__name__, str(error)] for error in errors],
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

    def test_torch_function_mode_semantics_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def test_scope_keeps_complex_dtypes_and_top_level_imag_unsupported(self):
        self.assertTrue(hasattr(torch.Tensor, "imag"))
        self.assertTrue(hasattr(reference_torch.Tensor, "imag"))
        self.assertFalse(hasattr(torch, "imag"))
        self.assertTrue(hasattr(reference_torch, "imag"))
        self.assertNotIn("imag", torch.__all__)
        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(dtype=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))


if __name__ == "__main__":
    unittest.main()
