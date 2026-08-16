import inspect
import types
import unittest

import torch_rs as torch


class TensorOutputNumberTests(unittest.TestCase):
    def tensor_cases(self):
        ordinary = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = leaf * 2.0
        tracked_view = tracked.transpose(0, 1)
        detached_tracked = tracked.detach()
        detached_view = tracked_view.detach()
        empty_leaf = torch.zeros((2, 0, 3), requires_grad=True)
        empty_tracked = empty_leaf * 2.0

        with torch.no_grad():
            no_grad_output = leaf * 3.0
            no_grad_leaf_view = leaf.transpose(0, 1)
            no_grad_non_leaf_view = tracked.transpose(0, 1)

        recorded_after_no_grad = no_grad_leaf_view + 1.0
        tracked.sum().backward()

        return (
            ("ordinary leaf", ordinary),
            ("ordinary operation", ordinary + 1.0),
            ("ordinary view", ordinary.transpose(0, 1)),
            ("autograd leaf", leaf),
            ("autograd non-leaf", tracked),
            ("autograd non-leaf view", tracked_view),
            ("detached non-leaf", detached_tracked),
            ("detached view", detached_view),
            ("empty leaf", empty_leaf),
            ("empty non-leaf", empty_tracked),
            ("empty view", empty_tracked.transpose(0, 2)),
            ("no-grad output", no_grad_output),
            ("no-grad leaf view", no_grad_leaf_view),
            ("no-grad non-leaf view", no_grad_non_leaf_view),
            ("recorded output after no-grad", recorded_after_no_grad),
            ("live leaf gradient", leaf.grad),
        )

    def test_every_supported_tensor_state_reports_output_zero(self):
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

                result = tensor.output_nr

                self.assertIs(type(result), int)
                self.assertEqual(result, 0)
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
        descriptor = inspect.getattr_static(torch.Tensor, "output_nr")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "output_nr")
        self.assertEqual(descriptor.__qualname__, "TensorBase.output_nr")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'output_nr' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.output_nr, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertEqual(descriptor.__get__(tensor, torch.Tensor), 0)

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'output_nr' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "output_nr", 1),
            lambda: delattr(tensor, "output_nr"),
            lambda: descriptor.__set__(tensor, 1),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'output_nr' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True) * 2.0
        descriptor = inspect.getattr_static(torch.Tensor, "output_nr")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.output_nr
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
                forwarded = tensor.output_nr
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded, 0)


if __name__ == "__main__":
    unittest.main()
