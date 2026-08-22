import inspect
import types
import unittest

import torch_rs as torch


class TensorNameTests(unittest.TestCase):
    def tensor_cases(self):
        ordinary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ordinary_view = ordinary.transpose(0, 1)
        leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 0)
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty_tracked = empty_leaf * 2.0

        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_view = leaf.transpose(0, 0)

        tracked.sum().backward()

        return (
            ("scalar", torch.tensor(1.0)),
            ("empty", torch.zeros((0,))),
            ("ordinary", ordinary),
            ("ordinary view", ordinary_view),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
            ("autograd view", tracked_view),
            ("detached non-leaf", tracked.detach()),
            ("detached view", tracked_view.detach()),
            ("empty autograd leaf", empty_leaf),
            ("empty autograd non-leaf", empty_tracked),
            ("empty autograd view", empty_tracked.transpose(0, 2)),
            ("no-grad output", no_grad_output),
            ("no-grad view", no_grad_view),
            ("gradient", leaf.grad),
        )

    def test_every_supported_tensor_is_unnamed(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )

                self.assertIs(tensor.name, None)

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
                    ),
                    metadata,
                )

    def test_tensorbase_descriptor_is_undocumented_and_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "name")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "name")
        self.assertEqual(descriptor.__qualname__, "TensorBase.name")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'name' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.name, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIs(descriptor.__get__(tensor, torch.Tensor), None)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'name' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "name", "batch"),
            lambda: delattr(tensor, "name"),
            lambda: descriptor.__set__(tensor, "batch"),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'name' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(torch.Tensor, "name")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.name
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

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.name
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, None)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_named_dimension_apis_remain_unsupported(self):
        tensor = torch.tensor([[1.0]])
        for attribute in ("names", "rename", "rename_", "refine_names"):
            with self.subTest(attribute=attribute):
                self.assertFalse(hasattr(torch.Tensor, attribute))
                self.assertFalse(hasattr(tensor, attribute))


if __name__ == "__main__":
    unittest.main()
