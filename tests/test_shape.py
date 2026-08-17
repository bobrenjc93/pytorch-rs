import inspect
import sys
import types
import unittest

import torch_rs as torch


SHAPE_DOC = (
    "\nshape() -> torch.Size\n\n"
    "Returns the size of the :attr:`self` tensor. Alias for :attr:`size`.\n\n"
    "See also :meth:`Tensor.size`.\n\n"
    "Example::\n\n"
    "    >>> t = torch.empty(3, 4, 5)\n"
    "    >>> t.size()\n"
    "    torch.Size([3, 4, 5])\n"
    "    >>> t.shape\n"
    "    torch.Size([3, 4, 5])\n\n"
)


class TensorShapeTests(unittest.TestCase):
    def metadata_cases(self):
        base = torch.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ]
        )
        offset = base[1]
        strided = base.transpose(0, 2)
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        self.assertGreater(offset.storage_offset(), 0)
        self.assertFalse(strided.is_contiguous())
        return (
            ("scalar", torch.tensor(3.0), ()),
            ("empty", torch.zeros((2, 0, 3)), (2, 0, 3)),
            ("offset", offset, (2, 4)),
            ("strided", strided, (4, 2, 3)),
            ("extreme empty", extreme_empty, (sys.maxsize, 0, 2)),
        )

    def test_shape_returns_canonical_size_from_native_metadata(self):
        for case, tensor, expected in self.metadata_cases():
            metadata = (
                tensor.stride(),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.requires_grad,
                tensor.is_leaf,
            )
            for _ in range(3):
                with self.subTest(case=case):
                    result = tensor.shape
                    self.assertIs(type(result), torch.Size)
                    self.assertEqual(tuple(result), expected)
                    self.assertEqual(repr(result), repr(torch.Size(expected)))
                    self.assertEqual(result, tensor.size())
            self.assertEqual(
                (
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.requires_grad,
                    tensor.is_leaf,
                ),
                metadata,
            )

    def test_tensorbase_descriptor_metadata_and_read_only_errors(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "shape")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "shape")
        self.assertEqual(descriptor.__qualname__, "TensorBase.shape")
        self.assertEqual(descriptor.__doc__, SHAPE_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertFalse(hasattr(descriptor, "__text_signature__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'shape' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.shape, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        value = descriptor.__get__(tensor, torch.Tensor)
        self.assertIs(type(value), torch.Size)
        self.assertEqual(value, torch.Size([2, 3]))

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'shape' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "shape", (3, 2)),
            lambda: delattr(tensor, "shape"),
            lambda: descriptor.__set__(tensor, (3, 2)),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'shape' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "shape")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.shape
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
                forwarded = tensor.shape
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(type(forwarded), torch.Size)
        self.assertEqual(forwarded, torch.Size([2, 3]))


if __name__ == "__main__":
    unittest.main()
