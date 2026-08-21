import gc
import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorDataTests(unittest.TestCase):
    def assert_data_alias(self, source, result):
        self.assertIsNot(result, source)
        self.assertTrue(result.is_set_to(source))
        self.assertEqual(result.shape, source.shape)
        self.assertEqual(result.stride(), source.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(source.detach()))

    def tensor_cases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        tracked = (leaf * 3.0).transpose(0, 2)[1]
        empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]

        self.assertFalse(tracked.is_contiguous())
        self.assertGreater(tracked.storage_offset(), 0)
        self.assertGreater(empty.storage_offset(), 0)
        return (
            ("negative-zero scalar", torch.tensor(-0.0, requires_grad=True)),
            ("ordinary", torch.tensor(values.tolist())),
            ("strided offset non-leaf", tracked),
            ("empty offset view", empty),
        )

    def test_getter_returns_a_fresh_detached_shared_storage_alias(self):
        for case, source in self.tensor_cases():
            with self.subTest(case=case):
                first = source.data
                second = source.data

                self.assert_data_alias(source, first)
                self.assert_data_alias(source, second)
                self.assertIsNot(first, second)
                self.assertTrue(first.is_set_to(second))
                self.assertFalse((first + 1.0).requires_grad)

        negative_zero = torch.tensor(-0.0).data
        self.assertEqual(
            np.asarray(negative_zero).view(np.uint32).item(), 0x8000_0000
        )

    def test_aliases_outlive_temporary_tensor_and_view_owners(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def offset_alias():
            return torch.tensor(values.tolist(), requires_grad=True)[1].data

        def strided_alias():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return temporary.transpose(0, 2)[1].data

        offset = offset_alias()
        strided = strided_alias()
        gc.collect()

        np.testing.assert_array_equal(np.asarray(offset), values[1])
        np.testing.assert_array_equal(
            np.asarray(strided), values.transpose(2, 1, 0)[1]
        )

    def test_data_is_an_autograd_boundary_without_changing_the_source_graph(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        source = (leaf * 3.0).transpose(0, 1)[1]
        alias = source.data

        self.assertTrue(source.requires_grad)
        self.assertFalse(source.is_leaf)
        self.assertFalse(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        detached_loss = (alias * alias).sum()
        self.assertFalse(detached_loss.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            detached_loss.backward()

        source.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0], [0.0, 3.0]])

    def test_alias_observes_later_internal_gradient_accumulation(self):
        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        (leaf * 4.0).sum().backward()
        gradient = leaf.grad
        alias = gradient.data

        self.assert_data_alias(gradient, alias)
        self.assertEqual(alias.tolist(), [4.0, 4.0])

        (leaf * 5.0).sum().backward()
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(gradient.tolist(), [9.0, 9.0])
        self.assertEqual(alias.tolist(), [9.0, 9.0])

    def test_tensorbase_descriptor_metadata_and_read_only_surface(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "data")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "data")
        self.assertEqual(descriptor.__qualname__, "TensorBase.data")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertFalse(hasattr(descriptor, "__text_signature__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'data' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.data, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assert_data_alias(tensor, descriptor.__get__(tensor, torch.Tensor))

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'data' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        replacement = torch.tensor([3.0, 4.0])
        actions = (
            lambda: setattr(tensor, "data", replacement),
            lambda: delattr(tensor, "data"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'data' of 'torch._C.TensorBase' objects "
                    "is not writable",
                )
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

        with self.assertRaises(TypeError) as raised:
            tensor.data[0] = 9.0
        self.assertEqual(
            str(raised.exception),
            "'torch_rs.Tensor' object does not support item assignment",
        )
        self.assertEqual(tensor.tolist(), [1.0, 2.0])

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "data")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.data
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
                forwarded = tensor.data
        self.assertEqual(order, ["upper", "lower"])
        self.assert_data_alias(tensor, forwarded)


if __name__ == "__main__":
    unittest.main()
