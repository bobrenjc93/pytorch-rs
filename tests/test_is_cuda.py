import inspect
import math
import sys
import types
import unittest

import torch_rs as torch


PROPERTY_DOC = (
    "\nIs ``True`` if the Tensor is stored on the GPU, ``False`` otherwise.\n"
)


class TensorIsCudaTests(unittest.TestCase):
    def test_supported_scalars_empty_views_and_autograd_use_device_metadata(self):
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
        self.assertGreater(offset_view.storage_offset(), 0)

        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        cases = [
            (f"scalar {value!r}", torch.tensor(value))
            for value in (-math.inf, -1.0, -0.0, 0.0, 1.0, math.inf, math.nan)
        ]
        cases.extend(
            (
                ("empty", torch.zeros((2, 0, 3))),
                ("offset strided view", offset_view),
                ("extreme empty view", extreme_empty),
                ("autograd leaf", leaf),
                ("autograd non-leaf view", tracked),
                ("accumulated gradient", leaf.grad),
            )
        )

        for case, tensor in cases:
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
                result = tensor.is_cuda
                self.assertIs(type(result), bool)
                self.assertIs(result, False)
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

    def test_tensorbase_descriptor_documentation_and_receiver_behavior(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_cuda")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "is_cuda")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_cuda")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIs(torch.Tensor.is_cuda, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), False)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'is_cuda' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

    def test_property_is_read_only_with_pytorch_assignment_errors(self):
        for action in ("set", "delete"):
            tensor = torch.tensor([1.0])
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    if action == "set":
                        tensor.is_cuda = True
                    else:
                        del tensor.is_cuda
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'is_cuda' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )


if __name__ == "__main__":
    unittest.main()
