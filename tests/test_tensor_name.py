import inspect
import types
import unittest

import torch_rs as torch


class TensorNameTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)[1]
        offset_view = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        tracked.sum().backward()

        return (
            ("negative-zero scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset strided view", offset_view),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("gradient", leaf.grad),
            ("detached view", tracked.detach()),
        )

    def test_supported_tensors_have_exact_none_name_without_side_effects(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case):
                before = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                self.assertIsNone(tensor.name)
                self.assertIsNone(tensor.name)
                after = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                self.assertEqual(after, before)

    def test_tensorbase_descriptor_metadata_and_read_only_errors(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "name")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "name")
        self.assertEqual(descriptor.__qualname__, "TensorBase.name")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertFalse(hasattr(descriptor, "__text_signature__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'name' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.name, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertIsNone(descriptor.__get__(tensor, torch.Tensor))

        for action in (
            lambda: setattr(tensor, "name", None),
            lambda: setattr(tensor, "name", "batch"),
            lambda: delattr(tensor, "name"),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        ):
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'name' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )

        for action in (
            lambda: descriptor.__get__(1, int),
            lambda: descriptor.__set__(1, None),
            lambda: descriptor.__delete__(1),
        ):
            with self.subTest(action=action):
                with self.assertRaises(TypeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "descriptor 'name' for 'torch._C.TensorBase' objects "
                    "doesn't apply to a 'int' object",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "name")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
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

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.name
        self.assertEqual(order, ["upper", "lower"])
        self.assertIsNone(forwarded)

    def test_named_dimension_apis_remain_unsupported(self):
        for name in ("names", "rename", "rename_", "refine_names"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.Tensor, name))


if __name__ == "__main__":
    unittest.main()
