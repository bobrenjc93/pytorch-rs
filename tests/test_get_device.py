import inspect
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\nget_device() -> Device ordinal (Integer)\n\n"
    "For CUDA tensors, this function returns the device ordinal of the GPU on which the tensor resides.\n"
    "For CPU tensors, this function returns `-1`.\n\n"
    "Example::\n\n"
    "    >>> x = torch.randn(3, 4, 5, device='cuda:0')\n"
    "    >>> x.get_device()\n"
    "    0\n"
    "    >>> x.cpu().get_device()\n"
    "    -1\n"
)


class TensorGetDeviceTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        offset_view = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            ("scalar", torch.tensor(3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", leaf.grad),
        )

    def test_cpu_tensors_use_device_metadata_without_materializing_values(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                result = tensor.get_device()
                self.assertIs(type(result), int)
                self.assertEqual(result, -1)
                self.assertIsNone(tensor.device.index)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_tensorbase_descriptor_documentation_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "get_device")
        bound = tensor.get_device

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "get_device")
            self.assertIsNone(callable_object.__text_signature__)
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor.__qualname__, "TensorBase.get_device")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor(tensor), -1)

    def test_no_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "get_device")
        bound = tensor.get_device
        calls = (
            (
                lambda: tensor.get_device(1),
                "TensorBase.get_device() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1, 2),
                "Tensor.get_device() takes no arguments (2 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.get_device() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.get_device(dim=0),
                "TensorBase.get_device() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.get_device() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.get_device() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'get_device' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
