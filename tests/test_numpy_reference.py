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
class TensorNumpyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.numpy() differentials require pinned PyTorch 2.13.0"
            )

    def layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = module.tensor(values.tolist(), dtype=module.float32)
        strided = source.transpose(0, 2)

        channel_values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        channels_last = module.tensor(
            channel_values.tolist(), dtype=module.float32
        ).contiguous(memory_format=module.channels_last)
        return (
            module.tensor(-3.5, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
            strided[1],
            strided,
            channels_last,
        )

    def export_contract(self, tensor):
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
        result = tensor.numpy(force=True)
        result_contract = {
            "type": type(result).__name__,
            "dtype": str(result.dtype),
            "shape": result.shape,
            "values": result.tolist(),
            "writeable": result.flags.writeable,
        }
        result_contract["metadata_unchanged"] = metadata == (
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            tensor.data_ptr(),
            str(tensor.dtype),
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )
        return result_contract

    def array_contract(self, result):
        return {
            "type": type(result).__name__,
            "dtype": str(result.dtype),
            "shape": result.shape,
            "values": result.tolist(),
            "writeable": result.flags.writeable,
        }

    def test_forced_layout_exports_match_pytorch_2_13(self):
        actual_cases = self.layout_cases(torch)
        expected_cases = self.layout_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case, shape=actual.shape, stride=actual.stride()):
                self.assertEqual(
                    self.export_contract(actual), self.export_contract(expected)
                )

    def autograd_contract(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        observations = (
            self.export_contract(leaf),
            self.export_contract(tracked),
        )
        tracked.sum().backward()
        return observations, leaf.grad.tolist()

    def test_force_true_requires_grad_detachment_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_contract(torch), self.autograd_contract(reference_torch)
        )

    def error(self, action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        self.fail("Tensor.numpy unexpectedly accepted an invalid call")

    def signature_outcome(self, callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "numpy")
        bound = tensor.numpy
        invalid_calls = (
            lambda: tensor.numpy(True),
            lambda: tensor.numpy(True, False),
            lambda: descriptor(tensor, True),
            lambda: tensor.numpy(force=1),
            lambda: tensor.numpy(force=None),
            lambda: tensor.numpy(force=np.bool_(True)),
            lambda: tensor.numpy(force=1.0),
            lambda: tensor.numpy(force="yes"),
            lambda: tensor.numpy(force=object()),
            lambda: tensor.numpy(unexpected=True),
            lambda: tensor.numpy(force=True, unexpected=True),
            lambda: tensor.numpy(**{"unexpected": True, "force": 1}),
            lambda: descriptor(),
            lambda: descriptor(1),
            lambda: descriptor(self=tensor),
        )
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_is_method": type(descriptor) is types.MethodDescriptorType,
            "bound_is_builtin": type(bound) is types.BuiltinMethodType,
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
            "descriptor_result": self.array_contract(
                descriptor(tensor, force=True)
            ),
            "bound_result": self.array_contract(bound(force=True)),
            "errors": tuple(self.error(call) for call in invalid_calls),
        }

    def test_callable_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "numpy")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        intercepted = []
        for call, expected_kwargs in (
            (tensor.numpy, None),
            (lambda: tensor.numpy(force=False), {"force": False}),
            (lambda: tensor.numpy(force=True), {"force": True}),
        ):
            recording = RecordingMode(marker)
            with recording:
                result = call()
            function, dispatch_types, args, kwargs = recording.calls[0]
            intercepted.append(
                (
                    result is marker,
                    function is descriptor,
                    dispatch_types,
                    len(args),
                    args[0] is tensor,
                    kwargs == expected_kwargs,
                )
            )

        rejected = RecordingMode(marker)
        try:
            with rejected:
                tensor.numpy(force=1)
        except Exception as error:
            rejected_error = type(error).__name__, str(error)
        else:
            rejected_error = None

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.numpy(force=True)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.numpy(force=True)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted,
            "rejected_error": rejected_error,
            "rejected_calls": len(rejected.calls),
            "forward_order": order,
            "forwarded": forwarded.tolist(),
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "mode_stack": module.overrides._get_current_function_mode_stack(),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )

    def test_zero_copy_reference_path_is_deliberately_out_of_scope(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        for call in (actual.numpy, lambda: actual.numpy(force=False)):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError, "zero-copy NumPy storage sharing"
                ):
                    call()

        expected = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        shared = expected.numpy(force=False)
        shared[0] = 2.0
        self.assertEqual(expected.tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
