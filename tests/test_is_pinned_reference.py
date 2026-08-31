import inspect
import re
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsPinnedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.is_pinned() differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
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
        channels_last = module.zeros(
            (2, 3, 4, 5), dtype=module.float32
        ).contiguous(memory_format=module.channels_last)
        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)
        return leaf, tracked, (
            module.tensor(-3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            source,
            channels_last,
            strided,
            offset,
            extreme_empty,
            leaf,
            produced,
            tracked,
            tracked.detach(),
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
        result = tensor.is_pinned()
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

    def test_pageable_cpu_tensors_match_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                actual_contract = self.contract(actual)
                self.assertEqual(actual_contract, self.contract(expected))
                self.assertIs(actual_contract["result"], False)

        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())
        self.assertEqual(
            self.contract(actual_leaf.grad), self.contract(expected_leaf.grad)
        )

    def pinned_reference_tensor(self):
        accelerator = getattr(reference_torch, "accelerator", None)
        if accelerator is None or not accelerator.is_available():
            self.skipTest(
                "PyTorch has no accelerator-backed pinned-memory allocator"
            )
        try:
            tensor = reference_torch.empty(
                (3, 4),
                dtype=reference_torch.float32,
                device="cpu",
                pin_memory=True,
            )
        except (RuntimeError, NotImplementedError) as error:
            self.skipTest(
                "PyTorch accelerator-backed pinned-memory allocator is "
                f"unavailable: {error}"
            )
        if not tensor.is_pinned():
            self.skipTest(
                "PyTorch did not provide a genuine accelerator-backed pinned "
                "CPU allocation"
            )
        return tensor

    def test_genuine_pinned_pytorch_cpu_tensor_bounds_pageable_model(self):
        pinned = self.pinned_reference_tensor()
        pinned_view = pinned.transpose(0, 1)[1]
        pageable = reference_torch.empty(
            pinned.shape, dtype=reference_torch.float32, device="cpu"
        )
        actual = torch.zeros(pinned.shape, dtype=torch.float32)

        self.assertEqual(pinned.device.type, "cpu")
        self.assertNotEqual(pinned.data_ptr(), 0)
        self.assertIs(pinned.is_pinned(), True)
        self.assertIs(pinned_view.is_pinned(), True)
        self.assertGreater(pinned_view.storage_offset(), 0)
        self.assertIs(pageable.is_pinned(), False)
        self.assertIs(actual.is_pinned(), False)

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.is_pinned unexpectedly accepted the invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "is_pinned")
        bound = tensor.is_pinned
        calls = (
            lambda: tensor.is_pinned(1, 2),
            lambda: tensor.is_pinned(unexpected=True),
            lambda: tensor.is_pinned(1, unexpected=True),
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
            "call_errors": tuple(self.error(call) for call in calls),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def test_callable_metadata_documentation_and_shared_errors_match(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_scope_excludes_pinned_allocation_and_deprecated_device_argument(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)
        for call in (
            lambda: tensor.is_pinned(None),
            lambda: tensor.is_pinned(device=None),
            lambda: torch.tensor([1.0], pin_memory=True),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        for name in ("zeros", "ones"):
            with self.subTest(factory=name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^{name}\(\): pin_memory=True is not supported; "
                    r"only unpinned CPU storage is implemented$",
                ):
                    getattr(torch, name)((2,), pin_memory=True)

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "is_pinned")
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
            intercepted = tensor.is_pinned()
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
                forwarded = tensor.is_pinned()

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.is_pinned()
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
            "forwarded": forwarded,
            "forwarded_type": type(forwarded).__name__,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(
                module.overrides._get_current_function_mode_stack()
            ),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
