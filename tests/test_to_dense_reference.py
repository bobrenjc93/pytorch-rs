import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorToDenseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.to_dense() differentials require pinned PyTorch 2.13.0"
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
        return (
            leaf,
            tracked,
            (
                module.tensor(-3.5, dtype=module.float32),
                module.zeros((2, 0, 3), dtype=module.float32),
                source,
                strided,
                offset,
                leaf,
                tracked,
                tracked.detach(),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return tuple(np.asarray(tensor.detach()).reshape(-1).view(np.uint32).tolist())

    def identity_contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        bits = self.value_bits(tensor)
        result = tensor.to_dense()
        return {
            "result_is_receiver": result is tensor,
            "receiver_is_strided": str(tensor.layout) == "torch.strided",
            "receiver_is_sparse": tensor.is_sparse,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
                str(result.dtype),
                str(result.device),
                str(result.layout),
                result.requires_grad,
                result.is_leaf,
            ),
            "bits_unchanged": bits == self.value_bits(result),
        }

    def test_supported_strided_identity_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.identity_contract(actual),
                    self.identity_contract(expected),
                )

        actual_tracked.to_dense().sum().backward()
        expected_tracked.to_dense().sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def no_grad_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)
        with module.no_grad():
            result = tracked.to_dense()
        result.sum().backward()
        return {
            "result_is_receiver": result is tracked,
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "shape": tuple(result.shape),
            "stride": tuple(result.stride()),
            "offset": result.storage_offset(),
            "pointer_unchanged": result.data_ptr() == tracked.data_ptr(),
            "gradient": leaf.grad.tolist(),
            "grad_mode_restored": module.is_grad_enabled(),
        }

    def test_no_grad_identity_matches_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_contract(torch),
            self.no_grad_contract(reference_torch),
        )

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to_dense")
        bound = tensor.to_dense

        def signature_outcome(callable_object):
            try:
                return "signature", str(inspect.signature(callable_object))
            except Exception as error:
                return "error", type(error).__name__

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
                signature_outcome(descriptor),
                signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "bound_result_is_receiver": bound() is tensor,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def test_descriptor_ownership_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to_dense")
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
            intercepted = tensor.to_dense()
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
                forwarded = tensor.to_dense()

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.to_dense()
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_is_descriptor": function is descriptor,
            "types_empty": dispatch_types == (),
            "args": len(args) == 1 and args[0] is tensor,
            "kwargs_is_none": kwargs is None,
            "forwarding_order": order,
            "forwarded_is_receiver": forwarded is tensor,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_reference_sparse_tensor_bounds_unsupported_conversion_path(self):
        self.assertFalse(hasattr(torch, "sparse_coo_tensor"))
        indices = reference_torch.tensor([[0, 1, 1], [1, 0, 2]])
        values = reference_torch.tensor(
            [3.0, 4.0, 5.0],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        sparse = reference_torch.sparse_coo_tensor(
            indices,
            values,
            (2, 3),
            check_invariants=True,
        )

        self.assertIs(sparse.layout, reference_torch.sparse_coo)
        self.assertTrue(sparse.is_sparse)
        dense = sparse.to_dense()

        self.assertIsNot(dense, sparse)
        self.assertIs(dense.layout, reference_torch.strided)
        self.assertFalse(dense.is_sparse)
        self.assertEqual(dense.shape, (2, 3))
        self.assertEqual(dense.stride(), (3, 1))
        self.assertEqual(dense.tolist(), [[0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
        self.assertTrue(dense.requires_grad)
        self.assertFalse(dense.is_leaf)
        self.assertEqual(type(dense.grad_fn).__name__, "ToDenseBackward0")

        dense.sum().backward()
        self.assertEqual(values.grad.tolist(), [1.0, 1.0, 1.0])

    def test_scope_excludes_dtype_and_masked_grad_overloads(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        unsupported_calls = (
            lambda: actual.to_dense(torch.float32),
            lambda: actual.to_dense(dtype=torch.float32),
            lambda: actual.to_dense(masked_grad=False),
        )
        for call in unsupported_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        self.assertIs(expected.to_dense(dtype=None), expected)
        self.assertIs(expected.to_dense(masked_grad=False), expected)
        self.assertFalse(hasattr(torch, "to_dense"))
        self.assertFalse(hasattr(reference_torch, "to_dense"))


if __name__ == "__main__":
    unittest.main()
