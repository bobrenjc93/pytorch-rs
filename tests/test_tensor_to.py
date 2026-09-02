import inspect
import re
import sys
import types
import unittest

import torch_rs as torch


class TensorToTests(unittest.TestCase):
    def assert_identity_call(self, tensor, call):
        metadata = (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.grad,
        )

        result = call(tensor)

        self.assertIs(result, tensor)
        self.assertEqual(
            (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.dtype,
                tensor.device,
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.grad,
            ),
            metadata,
        )

    def test_default_dtype_device_tensor_and_option_forms_return_exact_receiver(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        leaf.sum().backward()
        other = torch.ones((1,), requires_grad=True)
        cases = (
            torch.tensor(-0.0),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ]
            ).transpose(0, 1)[1],
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
            leaf,
            tracked,
            leaf.grad,
        )
        calls = (
            lambda tensor: tensor.to(),
            lambda tensor: tensor.to(torch.float32),
            lambda tensor: tensor.to(torch.float),
            lambda tensor: tensor.to(dtype=torch.float32),
            lambda tensor: tensor.to(dtype=torch.float),
            lambda tensor: tensor.to(dtype=None),
            lambda tensor: tensor.to("cpu"),
            lambda tensor: tensor.to(device="cpu"),
            lambda tensor: tensor.to(torch.device("cpu")),
            lambda tensor: tensor.to(torch.device("cpu", None)),
            lambda tensor: tensor.to(device=torch.device("cpu")),
            lambda tensor: tensor.to(device=None),
            lambda tensor: tensor.to(torch.device("cpu"), torch.float32),
            lambda tensor: tensor.to(None, torch.float32, False, False),
            lambda tensor: tensor.to(other),
            lambda tensor: tensor.to(tensor=other),
            lambda tensor: tensor.to(non_blocking=True),
            lambda tensor: tensor.to(copy=False),
            lambda tensor: tensor.to(memory_format=None),
            lambda tensor: tensor.to(memory_format=torch.preserve_format),
            lambda tensor: tensor.to(memory_format=torch.contiguous_format),
        )

        for case, tensor in enumerate(cases):
            for call_index, call in enumerate(calls):
                with self.subTest(case=case, call=call_index):
                    self.assert_identity_call(tensor, call)

        result = tracked.to(torch.float32)
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])
        self.assertIsNone(other.grad)

    def test_copy_and_non_identity_targets_are_rejected_without_mutating(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        detached = tensor.detach()
        cases = (
            (
                lambda: tensor.to(copy=True),
                NotImplementedError,
                "to(): copy=True requires a new tensor and is not supported",
            ),
            (
                lambda: tensor.to("cpu:0"),
                NotImplementedError,
                "to(): explicit indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: tensor.to(torch.device("cpu", 0)),
                NotImplementedError,
                "to(): explicit indexed CPU devices require a copy and are not supported",
            ),
            (
                lambda: tensor.to("cuda"),
                RuntimeError,
                "to(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: tensor.to(0),
                NotImplementedError,
                "to(): integer device ordinals target CUDA devices and are not supported",
            ),
            (
                lambda: tensor.to(memory_format=torch.channels_last),
                RuntimeError,
                "required rank 4 tensor to use channels_last format",
            ),
        )

        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()
                self.assertTrue(tensor.is_set_to(detached))
                self.assertTrue(tensor.requires_grad)
                self.assertTrue(tensor.is_leaf)

    def test_argument_binding_errors(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        cases = (
            lambda: tensor.to([]),
            lambda: tensor.to(torch.float32, device="cpu"),
            lambda: tensor.to(torch.float32, dtype=torch.float32),
            lambda: tensor.to("cpu", device="cpu"),
            lambda: tensor.to(tensor=other, dtype=torch.float32),
            lambda: tensor.to(other=other),
            lambda: tensor.to(copy=1),
            lambda: tensor.to(non_blocking=1),
            lambda: tensor.to(memory_format=1),
            lambda: tensor.to(torch.float32, None),
            lambda: tensor.to("cpu", torch.float32, None),
        )

        for call in cases:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^to\(\) received an invalid combination of arguments - got ",
                ):
                    call()

    def test_tensorbase_descriptor_metadata_and_unbound_calls(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to")
        bound = tensor.to

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'to' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__qualname__, "TensorBase.to")
        self.assertEqual(bound.__qualname__, "Tensor.to")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "to")
            self.assertIsNotNone(callable_object.__doc__)
            self.assertIn("to(*args, **kwargs) -> Tensor", callable_object.__doc__)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(descriptor(tensor, torch.float32), tensor)
        self.assertIs(descriptor(tensor, tensor=other), tensor)

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.to() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'to' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_torch_function_mode_observes_valid_calls_before_native_limits(self):
        tensor = torch.tensor([1.0])
        calls = []

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                calls.append((func.__name__, types, args, kwargs))
                return "handled"

        with RecordingMode():
            self.assertEqual(tensor.to("cuda"), "handled")
            self.assertEqual(tensor.to(copy=True), "handled")

        self.assertEqual([call[0] for call in calls], ["to", "to"])
        self.assertEqual(calls[0][1], ())
        self.assertIs(calls[0][2][0], tensor)
        self.assertEqual(calls[0][2][1], "cuda")
        self.assertIsNone(calls[0][3])
        self.assertEqual(calls[1][1], ())
        self.assertEqual(len(calls[1][2]), 1)
        self.assertIs(calls[1][2][0], tensor)
        self.assertEqual(calls[1][3], {"copy": True})


if __name__ == "__main__":
    unittest.main()
