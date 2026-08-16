import inspect
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsInferenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_inference differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked_view = produced.transpose(0, 1)
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
        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)
        return leaf, tracked_view, (
            source,
            leaf,
            produced,
            strided_view,
            offset_view,
            tracked_view,
            tracked_view.detach(),
            module.zeros((2, 0, 3), dtype=module.float32),
            extreme_empty,
            no_grad_output,
            no_grad_view,
        )

    def contract(self, tensor):
        metadata = (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        result = tensor.is_inference()
        return {
            "result": result,
            "result_type": type(result).__name__,
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
            ),
        }

    def test_ordinary_supported_tensors_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape):
                actual_contract = self.contract(actual)
                self.assertEqual(actual_contract, self.contract(expected))
                self.assertIs(actual_contract["result"], False)

        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_reference_inference_mode_sets_and_preserves_true_state(self):
        ordinary = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32, requires_grad=True
        )
        with reference_torch.inference_mode():
            inference_leaf = reference_torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            inference_output = inference_leaf * 2.0
            inference_transpose = inference_output.transpose(0, 1)
            inference_detached = inference_transpose.detach()
            inference_empty = reference_torch.zeros(
                (2, 0, 3), dtype=reference_torch.float32
            )
            with reference_torch.no_grad():
                inference_no_grad = inference_leaf * 3.0

            inference_tensors = (
                inference_leaf,
                inference_output,
                inference_transpose,
                inference_detached,
                inference_empty,
                inference_no_grad,
            )
            self.assertIs(ordinary.is_inference(), False)
            self.assertTrue(
                all(tensor.is_inference() is True for tensor in inference_tensors)
            )

        self.assertIs(ordinary.is_inference(), False)
        self.assertTrue(
            all(tensor.is_inference() is True for tensor in inference_tensors)
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_inference unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", inspect.signature(callable_object)
        except Exception as error:
            return "error", type(error)

    def callable_contract(self, module):
        tensor = module.tensor([1.0])
        descriptor = inspect.getattr_static(module.Tensor, "is_inference")
        bound = tensor.is_inference
        calls = (
            lambda: tensor.is_inference(1),
            lambda: bound(1),
            lambda: descriptor(tensor, 1),
            lambda: tensor.is_inference(1, 2),
            lambda: tensor.is_inference(input=tensor),
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
            "descriptor_result": descriptor(tensor),
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

    def mode_dispatch_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "is_inference")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        recording = RecordingMode()
        with recording:
            intercepted = tensor.is_inference()
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
                forwarded = tensor.is_inference()

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
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_contract(torch),
            self.mode_dispatch_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
