import inspect
import sys
import types
import unittest

import torch_rs as torch


PROPERTY_DOC = (
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
        dense = torch.tensor([float(value) for value in range(60)]).reshape(3, 4, 5)
        return (
            ("scalar", torch.tensor(2.5), ()),
            ("empty", torch.zeros((2, 0, 3)), (2, 0, 3)),
            ("offset", dense[1], (4, 5)),
            ("strided", dense.transpose(0, 2), (5, 4, 3)),
            (
                "extreme zero-element",
                torch.zeros((sys.maxsize, 0, sys.maxsize)),
                (sys.maxsize, 0, sys.maxsize),
            ),
        )

    def test_shape_returns_fresh_canonical_size_from_layout_metadata(self):
        for case, tensor, expected in self.metadata_cases():
            with self.subTest(case=case):
                metadata = (
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.numel(),
                    tensor.requires_grad,
                    tensor.is_leaf,
                )

                first = tensor.shape
                second = tensor.shape

                self.assertIs(type(first), torch.Size)
                self.assertIs(type(second), torch.Size)
                self.assertIsNot(first, second)
                self.assertEqual(tuple(first), expected)
                self.assertEqual(repr(first), f"torch.Size({list(expected)})")
                self.assertEqual(
                    (
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.numel(),
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_tensorbase_descriptor_is_documented_and_read_only(self):
        tensor = torch.zeros((2, 3))
        descriptor = inspect.getattr_static(torch.Tensor, "shape")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "shape")
        self.assertEqual(descriptor.__qualname__, "TensorBase.shape")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'shape' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.shape, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(type(descriptor.__get__(tensor, torch.Tensor)), torch.Size)
        self.assertEqual(descriptor.__get__(tensor, torch.Tensor), (2, 3))

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
        self.assertEqual(forwarded, (2, 3))

    def test_size_no_argument_overload_remains_unchanged(self):
        tensor = torch.zeros((2, 3))
        with self.assertRaises(TypeError) as raised:
            tensor.size()
        self.assertEqual(
            str(raised.exception),
            'size() missing 1 required positional arguments: "dim"',
        )
        self.assertEqual(tensor.size(0), 2)
        self.assertEqual(tensor.size(dim=-1), 3)


if __name__ == "__main__":
    unittest.main()
