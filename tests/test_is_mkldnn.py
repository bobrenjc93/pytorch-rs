import inspect
import math
import sys
import types
import unittest

import torch_rs as torch


class TensorIsMkldnnTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        self.assertFalse(strided_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        return (
            *(
                (f"scalar {value!r}", torch.tensor(value))
                for value in (
                    -math.inf,
                    -1.0,
                    -0.0,
                    0.0,
                    1.0,
                    math.inf,
                    math.nan,
                )
            ),
            ("empty", torch.zeros((2, 0, 3))),
            ("strided view", strided_view),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", leaf.grad),
        )

    def test_every_supported_tensor_uses_non_mkldnn_storage(self):
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

                result = tensor.is_mkldnn

                self.assertIs(type(result), bool)
                self.assertIs(result, False)
                self.assertIs(tensor.layout, torch.strided)
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

    def test_tensorbase_descriptor_is_undocumented_and_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_mkldnn")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "is_mkldnn")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_mkldnn")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'is_mkldnn' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.is_mkldnn, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), False)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'is_mkldnn' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "is_mkldnn", True),
            lambda: delattr(tensor, "is_mkldnn"),
            lambda: descriptor.__set__(tensor, True),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'is_mkldnn' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_mkldnn")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.is_mkldnn
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertEqual(function, descriptor.__get__)
        self.assertIs(function.__self__, descriptor)
        self.assertEqual(function.__name__, "__get__")
        self.assertEqual(function.__qualname__, "getset_descriptor.__get__")
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.is_mkldnn
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, False)


if __name__ == "__main__":
    unittest.main()
