import gc
import inspect
import operator
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorDataTests(unittest.TestCase):
    def tensor_cases(self):
        bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x007F_FFFF,
                0x0080_0000,
                0x3F80_0000,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        scalar_storage = torch.tensor(memoryview(bits.view(np.float32)))
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        tracked = (leaf * 3.0).transpose(0, 1)[1]
        gradient_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        (gradient_leaf * 2.0).sum().backward()
        empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        return (
            *(
                (f"float32 bits 0x{value:08x}", scalar_storage[index])
                for index, value in enumerate(bits)
            ),
            ("ordinary tensor", base),
            ("strided view", strided),
            ("offset strided view", strided[1]),
            ("empty offset view", empty),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", gradient_leaf.grad),
            ("detached view", tracked.detach()),
        )

    def assert_data_alias(self, source, alias):
        self.assertIsNot(alias, source)
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertEqual(alias.is_contiguous(), source.is_contiguous())
        self.assertIs(alias.dtype, source.dtype)
        self.assertEqual(alias.device, source.device)
        self.assertEqual(alias.data_ptr(), source.data_ptr())
        self.assertTrue(source.is_set_to(alias))
        self.assertFalse(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertEqual(alias.output_nr, 0)
        self.assertIsNone(alias.grad)
        self.assertFalse((alias + 1.0).requires_grad)
        if source.numel() == 0:
            self.assertEqual(alias.numel(), 0)
        else:
            np.testing.assert_array_equal(
                np.asarray(alias).reshape(-1).view(np.uint32),
                np.asarray(source.detach()).reshape(-1).view(np.uint32),
            )

    def test_each_read_returns_a_fresh_detached_storage_alias(self):
        for case, source in self.tensor_cases():
            with self.subTest(case=case, shape=source.shape, stride=source.stride()):
                first = source.data
                second = source.data
                self.assertIsNot(first, second)
                self.assert_data_alias(source, first)
                self.assert_data_alias(source, second)

    def test_aliases_outlive_temporary_source_owners(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        def retain_offset_alias():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return temporary[1].data

        def retain_strided_alias():
            temporary = torch.tensor(values.tolist(), requires_grad=True)
            return temporary.transpose(0, 2)[1].data

        offset = retain_offset_alias()
        strided = retain_strided_alias()
        gc.collect()

        np.testing.assert_array_equal(np.asarray(offset), values[1])
        np.testing.assert_array_equal(
            np.asarray(strided), values.transpose(2, 1, 0)[1]
        )

    def test_data_isolates_autograd_without_changing_the_source_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        source_metadata = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
            source.requires_grad,
            source.is_leaf,
            source.output_nr,
        )

        alias = source.data
        detached_loss = (alias * alias).sum()
        self.assertFalse(detached_loss.requires_grad)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            detached_loss.backward()

        self.assertEqual(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.requires_grad,
                source.is_leaf,
                source.output_nr,
            ),
            source_metadata,
        )
        source.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0], [0.0, 3.0]])
        self.assertIsNone(alias.grad)

    def test_tensorbase_descriptor_is_undocumented_and_read_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "data")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "data")
        self.assertEqual(descriptor.__qualname__, "TensorBase.data")
        self.assertIsNone(descriptor.__doc__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
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

        actions = (
            lambda: setattr(tensor, "data", torch.tensor([2.0])),
            lambda: delattr(tensor, "data"),
            lambda: descriptor.__set__(tensor, torch.tensor([2.0])),
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
        self.assertEqual(tensor.tolist(), [1.0])

    def test_data_alias_does_not_add_a_storage_mutation_surface(self):
        source = torch.tensor([1.0, 2.0, 3.0])
        alias = source.data

        with self.assertRaisesRegex(TypeError, "does not support item assignment"):
            operator.setitem(alias, 0, -1.0)
        for name in ("copy_", "set_", "storage", "untyped_storage"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(alias, name))

        self.assertEqual(source.tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(alias.tolist(), [1.0, 2.0, 3.0])
        self.assertTrue(source.is_set_to(alias))

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "data")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
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

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.data
        self.assertEqual(order, ["upper", "lower"])
        self.assert_data_alias(tensor, forwarded)

    def test_not_implemented_reenters_the_declining_top_mode(self):
        tensor = torch.tensor([1.0])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return NotImplemented

        class AcceptingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        lower = AcceptingMode()
        upper = DecliningMode()
        recursion_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(80)
            with self.assertRaisesRegex(
                RecursionError, r"^maximum recursion depth exceeded$"
            ):
                with lower:
                    with upper:
                        tensor.data
        finally:
            sys.setrecursionlimit(recursion_limit)

        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, 0)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)
        self.assert_data_alias(tensor, tensor.data)


if __name__ == "__main__":
    unittest.main()
