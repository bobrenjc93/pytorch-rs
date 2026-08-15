import inspect
import json
import subprocess
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorAdjointReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.adjoint differentials require pinned PyTorch 2.13.0")

    def tensor_cases(self, module):
        values = [float(value) for value in range(120)]
        dense = module.tensor(values, dtype=module.float32).reshape(2, 3, 4, 5)
        return (
            module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
            ),
            dense,
            module.zeros((2, 0, 3), dtype=module.float32),
            dense.transpose(0, 3),
            dense.transpose(0, 3)[1],
            module.zeros((3, 0, 2), dtype=module.float32).transpose(0, 2)[1],
        )

    def view_contract(self, tensor):
        result = tensor.adjoint()
        restored = result.adjoint()
        return {
            "identity": result is tensor,
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "is_contiguous": result.is_contiguous(),
            "pointer_preserved": result.data_ptr() == tensor.data_ptr(),
            "matches_transpose": result.is_set_to(tensor.transpose(-2, -1)),
            "matches_mh": result.is_set_to(tensor.mH),
            "is_conj": result.is_conj(),
            "values": result.tolist(),
            "restored_identity": restored is tensor,
            "restored_metadata": restored.is_set_to(tensor),
        }

    def test_matrix_batched_empty_offset_and_strided_views_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(self.view_contract(actual), self.view_contract(expected))

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.adjoint unexpectedly accepted the operation")

    def descriptor_contract(self, module):
        descriptor = inspect.getattr_static(module.Tensor, "adjoint")
        tensor = module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32)
        bound = tensor.adjoint
        invalid_calls = (
            lambda: tensor.adjoint(1),
            lambda: tensor.adjoint(1, 2),
            lambda: tensor.adjoint(dim=0),
            lambda: bound(1),
            lambda: bound(dim=0),
            lambda: descriptor(tensor, 1),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
            lambda: descriptor.__get__(1, int),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "is_method_descriptor": type(descriptor) is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "is_builtin_method": type(bound) is types.BuiltinMethodType,
            "callable": callable(descriptor),
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "repr": repr(descriptor),
            "class_identity": module.Tensor.adjoint is descriptor,
            "class_get_identity": descriptor.__get__(None, module.Tensor)
            is descriptor,
            "value": self.view_contract(descriptor(tensor)),
            "empty_kwargs_value": self.view_contract(bound(**{})),
            "invalid_errors": tuple(self.error(action) for action in invalid_calls),
        }

    def test_tensorbase_descriptor_and_invalid_calls_match_pytorch_2_13(self):
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
import warnings

module = importlib.import_module(MODULE)
descriptor = inspect.getattr_static(module.Tensor, "adjoint")
matrix = module.tensor(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
)
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
    intercepted = matrix.adjoint()
function, dispatch_types, args, kwargs = recording.calls[0]

scalar = module.tensor(2.5, dtype=module.float32, requires_grad=True)
scalar_marker = object()
with warnings.catch_warnings(record=True) as scalar_warnings:
    warnings.simplefilter("always")
    with RecordingMode(scalar_marker):
        scalar_intercepted = scalar.adjoint()

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = matrix.adjoint()

sys.setrecursionlimit(80)
class DecliningMode(module.overrides.TorchFunctionMode):
    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return NotImplemented

lower = RecordingMode(object())
upper = DecliningMode()
try:
    with lower:
        with upper:
            matrix.adjoint()
except Exception as error:
    nested_error = [type(error).__name__, str(error)]
else:
    nested_error = None

scalar_declining = DecliningMode()
with warnings.catch_warnings(record=True) as declining_warnings:
    warnings.simplefilter("always")
    try:
        with scalar_declining:
            scalar.adjoint()
    except Exception as error:
        scalar_declining_error = [type(error).__name__, str(error)]
    else:
        scalar_declining_error = None

print(json.dumps({
    "intercepted": intercepted is marker,
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_is_descriptor": function is descriptor,
    "types": len(dispatch_types) == 1 and dispatch_types[0] is module.Tensor,
    "args": len(args) == 1 and args[0] is matrix,
    "kwargs_is_none": kwargs is None,
    "scalar_intercepted": scalar_intercepted is scalar_marker,
    "scalar_warning_count": len(scalar_warnings),
    "forwarding_order": order,
    "forwarded_shape": list(forwarded.shape),
    "forwarded_stride": list(forwarded.stride()),
    "forwarded_values": forwarded.tolist(),
    "nested_error": nested_error,
    "upper_calls": upper.calls,
    "lower_skipped": len(lower.calls) == 0,
    "scalar_declining_error": scalar_declining_error,
    "scalar_declining_calls": scalar_declining.calls,
    "scalar_declining_warning_count": len(declining_warnings),
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

    def test_scalar_warning_vector_error_and_extreme_metadata_match_pytorch_2_13(self):
        script = r'''
import importlib, json, sys, warnings
torch = importlib.import_module(MODULE)
outputs = []
scalar = torch.tensor(2.5, dtype=torch.float32, requires_grad=True)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    first = scalar.adjoint()
    second = scalar.adjoint()
outputs.append({
    "count": len(caught),
    "category": caught[0].category.__name__,
    "message": str(caught[0].message),
    "filename": caught[0].filename,
    "lineno": caught[0].lineno,
    "identity": first is scalar and second is scalar,
    "requires_grad": first.requires_grad,
    "is_leaf": first.is_leaf,
})
try:
    torch.zeros((3,), dtype=torch.float32).adjoint()
except Exception as error:
    outputs.append({"error": type(error).__name__, "message": str(error)})
try:
    torch.zeros((sys.maxsize, 0, sys.maxsize), dtype=torch.float32).adjoint()
except Exception as error:
    outputs.append({"error": type(error).__name__, "message": str(error)})
offset = torch.zeros((sys.maxsize, 0, 1), dtype=torch.float32)[sys.maxsize - 1]
result = offset.adjoint()
outputs.append({
    "shape": list(result.shape),
    "stride": list(result.stride()),
    "storage_offset": result.storage_offset(),
    "pointer_preserved": result.data_ptr() == offset.data_ptr(),
    "values": result.tolist(),
})
print(json.dumps(outputs))
'''

        def run(module):
            result = subprocess.run(
                [sys.executable, "-c", f"MODULE = {module!r}\n" + script],
                check=True,
                capture_output=True,
                text=True,
            )
            return json.loads(result.stdout)

        self.assertEqual(run("torch_rs"), run("torch"))

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        weights = module.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=module.float32
        )
        view = leaf.adjoint()
        view_contract = (
            view.requires_grad,
            view.is_leaf,
            tuple(view.shape),
            view.stride(),
            view.storage_offset(),
            view.data_ptr() == leaf.data_ptr(),
            view.is_set_to(leaf.mH),
            view.is_conj(),
        )
        (view * weights).sum().backward()

        empty = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
        empty_view = empty.adjoint()
        empty_view.sum().backward()

        no_grad_source = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        with module.no_grad():
            no_grad_view = no_grad_source.adjoint()
        no_grad_contract = (
            no_grad_view.requires_grad,
            no_grad_view.is_leaf,
            tuple(no_grad_view.shape),
            no_grad_view.stride(),
            no_grad_view.storage_offset(),
            no_grad_view.data_ptr() == no_grad_source.data_ptr(),
        )
        (no_grad_view * no_grad_view).sum().backward()

        return {
            "view": view_contract,
            "gradient": leaf.grad.tolist(),
            "empty_view": (
                tuple(empty_view.shape),
                empty_view.stride(),
                empty_view.storage_offset(),
                empty_view.data_ptr() == empty.data_ptr(),
            ),
            "empty_gradient": (
                tuple(empty.grad.shape),
                empty.grad.stride(),
                empty.grad.numel(),
            ),
            "no_grad_view": no_grad_contract,
            "no_grad_source_gradient": no_grad_source.grad is None,
        }

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch),
            self.autograd_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
