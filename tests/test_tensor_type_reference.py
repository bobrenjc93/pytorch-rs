import inspect
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorTypeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.type() differentials require pinned PyTorch 2.13.0"
            )

    def tensor_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        produced = leaf * 2.0
        tracked = produced.transpose(0, 1)
        multi_output_view = produced.unbind()[1]
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
        channels_last = module.zeros(
            (2, 3, 4, 5), dtype=module.float32
        ).contiguous(memory_format=module.channels_last)
        gradient_leaf = module.tensor(
            [2.0, 3.0], dtype=module.float32, requires_grad=True
        )
        (gradient_leaf * 4.0).sum().backward()
        with module.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 1)

        return leaf, tracked, (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            source,
            channels_last,
            strided,
            offset,
            leaf,
            produced,
            tracked,
            multi_output_view,
            tracked.detach(),
            gradient_leaf.grad,
            no_grad_output,
            no_grad_view,
        )

    def query_contract(self, tensor):
        descriptor = inspect.getattr_static(type(tensor), "type")
        before = (
            tensor.tolist(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        results = (
            tensor.type(),
            tensor.type(*()),
            tensor.type(**{}),
            descriptor(tensor),
        )
        after = (
            tensor.tolist(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            str(tensor.layout),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )
        return {
            "results": tuple((type(result).__name__, result) for result in results),
            "state_unchanged": before == after,
        }

    def test_cpu_float32_query_matches_pytorch_2_13(self):
        actual_leaf, actual_tracked, actual_cases = self.tensor_cases(torch)
        expected_leaf, expected_tracked, expected_cases = self.tensor_cases(
            reference_torch
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.query_contract(actual), self.query_contract(expected)
                )

        actual_tracked.type()
        expected_tracked.type()
        actual_tracked.sum().backward()
        expected_tracked.sum().backward()
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "type")
        bound = tensor.type
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "descriptor_doc": descriptor.__doc__,
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
            "bound_self": bound.__self__ is tensor,
            "descriptor_get_on_class": (
                descriptor.__get__(None, module.Tensor) is descriptor
            ),
            "descriptor_get_result": descriptor.__get__(
                tensor, module.Tensor
            )(),
            "descriptor_result": descriptor(tensor),
            "bound_result": bound(),
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
        }

    def test_descriptor_metadata_and_bound_unbound_calls_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def error_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "type")
        calls = (
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(None),
            lambda: descriptor(self=tensor),
        )
        outcomes = []
        for call in calls:
            try:
                call()
            except Exception as error:
                outcomes.append((type(error).__name__, str(error)))
            else:
                outcomes.append(("result",))
        return tuple(outcomes)

    def test_receiver_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.error_contract(torch),
            self.error_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "type")
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
            intercepted = tensor.type()
        function, dispatch_types, args, kwargs = recording.calls[0]

        empty_kwargs_recording = RecordingMode(marker)
        with empty_kwargs_recording:
            empty_kwargs_result = tensor.type(**{})
        _, _, empty_kwargs_args, empty_kwargs = empty_kwargs_recording.calls[0]

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.type()

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.type()
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        class ClassMethodMode(module.overrides.TorchFunctionMode):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return marker

        try:
            with ClassMethodMode():
                tensor.type()
        except Exception as error:
            classmethod_error = (type(error).__name__, str(error))
        else:
            classmethod_error = None

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
            "empty_kwargs_intercepted": empty_kwargs_result is marker,
            "empty_kwargs_args": (
                len(empty_kwargs_args) == 1 and empty_kwargs_args[0] is tensor
            ),
            "empty_kwargs": empty_kwargs,
            "forwarding_order": order,
            "forwarded": forwarded,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "classmethod_error": classmethod_error,
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )

    def test_dtype_string_and_non_blocking_conversion_scope_is_excluded(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        unsupported_calls = (
            lambda: actual.type(torch.float32),
            lambda: actual.type("torch.FloatTensor"),
            lambda: actual.type(dtype=torch.float32),
            lambda: actual.type(non_blocking=False),
            lambda: actual.type(non_blocking=True),
            lambda: actual.type(torch.float32, False),
            lambda: actual.type("torch.FloatTensor", non_blocking=True),
            lambda: actual.type(**{"async": False}),
        )
        for call in unsupported_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        expected = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        for converted in (
            expected.type(reference_torch.float32),
            expected.type("torch.FloatTensor"),
            expected.type(dtype=reference_torch.float32),
            expected.type(non_blocking=False),
            expected.type(non_blocking=True),
            expected.type(reference_torch.float32, False),
            expected.type("torch.FloatTensor", non_blocking=True),
        ):
            self.assertTrue(
                converted is expected or converted == "torch.FloatTensor"
            )

    def test_h100_cuda_type_name_bounds_the_unsupported_device_surface(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")
        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        device = reference_torch.device("cuda", 0)
        cuda_tensor = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, device=device
        )
        reference_torch.cuda.synchronize(device)

        self.assertEqual(cuda_tensor.device.type, "cuda")
        self.assertIs(cuda_tensor.dtype, reference_torch.float32)
        self.assertEqual(cuda_tensor.type(), "torch.cuda.FloatTensor")
        self.assertEqual(
            reference_torch.tensor([1.0], dtype=reference_torch.float32).type(),
            "torch.FloatTensor",
        )
        self.assertEqual(torch.tensor([1.0]).type(), "torch.FloatTensor")
        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaises(RuntimeError):
            torch.tensor([1.0, 2.0], device="cuda:0")


if __name__ == "__main__":
    unittest.main()
