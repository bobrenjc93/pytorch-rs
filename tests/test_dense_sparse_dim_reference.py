import inspect
import json
import subprocess
import sys
import types
import unittest
import warnings

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorDenseSparseDimReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "dense_dim/sparse_dim differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = module.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=module.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return leaf, tracked, (
            module.tensor(3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            strided,
            offset,
            extreme_empty,
            leaf,
            tracked,
        )

    def metadata_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.is_sparse,
            tensor.is_sparse_csr,
        )
        dense_dimensions = tensor.dense_dim()
        sparse_dimensions = tensor.sparse_dim()
        return {
            "dense_dim": dense_dimensions,
            "sparse_dim": sparse_dimensions,
            "dense_type": type(dense_dimensions).__name__,
            "sparse_type": type(sparse_dimensions).__name__,
            "sum_is_rank": dense_dimensions + sparse_dimensions == tensor.ndim,
            "metadata_unchanged": metadata
            == (
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                str(tensor.dtype),
                str(tensor.device),
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.is_sparse,
                tensor.is_sparse_csr,
            ),
        }

    def test_supported_strided_tensors_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.metadata_contract(actual),
                    self.metadata_contract(expected),
                )

        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("dimension metadata method unexpectedly accepted an invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module, name):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, name)
        bound = getattr(tensor, name)
        if name == "dense_dim":
            direct_positional = lambda: tensor.dense_dim(1)
            direct_multiple = lambda: tensor.dense_dim(1, 2)
            direct_keyword = lambda: tensor.dense_dim(input=tensor)
        else:
            direct_positional = lambda: tensor.sparse_dim(1)
            direct_multiple = lambda: tensor.sparse_dim(1, 2)
            direct_keyword = lambda: tensor.sparse_dim(input=tensor)
        calls = (
            direct_positional,
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            direct_multiple,
            direct_keyword,
            lambda: bound(unexpected=True),
            lambda: descriptor(tensor, unexpected=True),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
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
            "descriptor_result": descriptor(tensor),
            "bound_result": bound(),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "errors": tuple(self.error(call) for call in calls),
        }

    def test_callable_metadata_documentation_and_errors_match_pytorch_2_13(self):
        for name in ("dense_dim", "sparse_dim"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

    def mode_dispatch_observation(self, module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.0], dtype=module.float32)
observations = {}

for name in ("dense_dim", "sparse_dim"):
    descriptor = inspect.getattr_static(module.Tensor, name)
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
        intercepted = getattr(tensor, name)()
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
            forwarded = getattr(tensor, name)()

    sys.setrecursionlimit(80)
    declining = RecordingMode(NotImplemented)
    lower = RecordingMode(marker)
    try:
        with lower:
            with declining:
                getattr(tensor, name)()
    except Exception as error:
        declining_error = [type(error).__name__, str(error)]
    else:
        declining_error = None

    observations[name] = {
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
        "forwarded": forwarded,
        "forwarded_type": type(forwarded).__name__,
        "declining_error": declining_error,
        "declining_calls": len(declining.calls),
        "lower_skipped": len(lower.calls) == 0,
        "stack_depth": len(module.overrides._get_current_function_mode_stack()),
    }

print(json.dumps(observations, sort_keys=True))
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

    def test_reference_coo_and_csr_bound_unsupported_sparse_dimensions(self):
        for name in ("sparse_coo_tensor", "sparse_csr_tensor"):
            self.assertFalse(hasattr(torch, name))
        for name in ("sparse_coo", "sparse_csr"):
            self.assertFalse(hasattr(torch, name))

        indices = reference_torch.tensor(
            [[0, 1], [1, 0]], dtype=reference_torch.int64
        )
        coo = reference_torch.sparse_coo_tensor(
            indices,
            reference_torch.tensor([3.0, 4.0]),
            (2, 2),
            check_invariants=True,
        )
        hybrid_coo = reference_torch.sparse_coo_tensor(
            indices,
            reference_torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            (2, 2, 3),
            check_invariants=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            crow_indices = reference_torch.tensor(
                [0, 1, 2], dtype=reference_torch.int64
            )
            column_indices = reference_torch.tensor(
                [1, 0], dtype=reference_torch.int64
            )
            csr = reference_torch.sparse_csr_tensor(
                crow_indices,
                column_indices,
                reference_torch.tensor([3.0, 4.0]),
                size=(2, 2),
                check_invariants=True,
            )
            hybrid_csr = reference_torch.sparse_csr_tensor(
                crow_indices,
                column_indices,
                reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
                ),
                size=(2, 2, 3),
                check_invariants=True,
            )

        dense_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "dense_dim"
        )
        sparse_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "sparse_dim"
        )
        for tensor, layout, dense_dimensions in (
            (coo, reference_torch.sparse_coo, 0),
            (hybrid_coo, reference_torch.sparse_coo, 1),
            (csr, reference_torch.sparse_csr, 0),
            (hybrid_csr, reference_torch.sparse_csr, 1),
        ):
            with self.subTest(layout=layout, shape=tensor.shape):
                self.assertIs(tensor.layout, layout)
                self.assertEqual(tensor.dense_dim(), dense_dimensions)
                self.assertEqual(tensor.sparse_dim(), 2)
                self.assertEqual(
                    dense_descriptor(tensor), dense_dimensions
                )
                self.assertEqual(sparse_descriptor(tensor), 2)
                self.assertEqual(
                    tensor.dense_dim() + tensor.sparse_dim(), tensor.ndim
                )


if __name__ == "__main__":
    unittest.main()
