import inspect
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
class TensorConjReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("conj differentials require pinned PyTorch 2.13.0")

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
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            leaf,
            tracked,
            (
                module.tensor(-3.5, dtype=module.float32),
                module.zeros((2, 0, 3), dtype=module.float32),
                source.neg(),
                strided_view,
                offset_view,
                extreme_empty,
                module.tensor(memoryview(special_bits.view(np.float32))),
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

        result = tensor.conj()

        return {
            "result_is_receiver": result is tensor,
            "result_is_clear": result.is_conj() is False,
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

    def test_real_float32_identity_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(
                case=case,
                shape=actual.shape,
                stride=actual.stride(),
            ):
                self.assertEqual(
                    self.identity_contract(actual),
                    self.identity_contract(expected),
                )

        actual_tracked.conj().sum().backward()
        expected_tracked.conj().sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def no_grad_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)[1]
        metadata = (
            tuple(tracked.shape),
            tuple(tracked.stride()),
            tracked.storage_offset(),
            tracked.data_ptr(),
            tracked.requires_grad,
            tracked.is_leaf,
        )

        with module.no_grad():
            result = tracked.conj()

        result.sum().backward()
        return {
            "result_is_receiver": result is tracked,
            "metadata_unchanged": metadata
            == (
                tuple(result.shape),
                tuple(result.stride()),
                result.storage_offset(),
                result.data_ptr(),
                result.requires_grad,
                result.is_leaf,
            ),
            "result_is_clear": result.is_conj() is False,
            "gradient": leaf.grad.tolist(),
        }

    def test_no_grad_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.no_grad_contract(torch),
            self.no_grad_contract(reference_torch),
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("conj unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "conj")
        bound = tensor.conj
        calls = (
            lambda: tensor.conj(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.conj(1, 2),
            lambda: tensor.conj(input=tensor),
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
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_result_is_receiver": descriptor(tensor) is tensor,
            "signatures": tuple(
                self.signature_outcome(callable_object)
                for callable_object in (descriptor, bound)
            ),
            "call_errors": tuple(self.error(call) for call in calls),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def test_callable_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32, requires_grad=True)
        descriptor = inspect.getattr_static(module.Tensor, "conj")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.conj()
        function, dispatch_types, args, kwargs = recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with module.no_grad():
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = tensor.conj()

        return {
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
            "forwarded_is_receiver": forwarded is tensor,
            "forwarded_requires_grad": forwarded.requires_grad,
            "forwarded_is_clear": forwarded.is_conj() is False,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_forwarding_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_deliberate_scope_is_method_only_and_real_only(self):
        self.assertTrue(hasattr(torch.Tensor, "conj"))
        self.assertFalse(hasattr(torch, "conj"))
        self.assertTrue(hasattr(reference_torch, "conj"))
        self.assertFalse(hasattr(torch.Tensor, "conj_physical"))
        self.assertTrue(hasattr(reference_torch.Tensor, "conj_physical"))
        self.assertFalse(hasattr(torch, "complex64"))
        self.assertTrue(hasattr(reference_torch, "complex64"))


if __name__ == "__main__":
    unittest.main()
