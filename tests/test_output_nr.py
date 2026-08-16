import inspect
import struct
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

    def test_every_supported_single_output_tensor_state_reports_zero(self):
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

    def test_iteration_uses_multi_output_numbers_while_indexing_stays_zero(self):
        source = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
        )
        iterator = iter(source)

        self.assertEqual(type(iterator).__name__, "tuple_iterator")
        self.assertIs(iter(iterator), iterator)
        self.assertEqual(iterator.__length_hint__(), 3)
        rows = tuple(iterator)
        self.assertEqual(tuple(row.output_nr for row in rows), (0, 1, 2))
        self.assertEqual(iterator.__length_hint__(), 0)

        indexed = tuple(source[index] for index in range(len(source)))
        self.assertEqual(tuple(row.output_nr for row in indexed), (0, 0, 0))
        for row, direct in zip(rows, indexed, strict=True):
            self.assertEqual(row.shape, direct.shape)
            self.assertEqual(row.stride(), direct.stride())
            self.assertEqual(row.storage_offset(), direct.storage_offset())
            self.assertEqual(row.data_ptr(), direct.data_ptr())
            self.assertEqual(row.tolist(), direct.tolist())

        rows[1].sum().backward()
        self.assertEqual(
            source.grad.tolist(),
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
        )

        signed_source = torch.tensor([[1.0], [-0.0]], requires_grad=True)
        signed_rows = tuple(signed_source)
        (signed_rows[0] * signed_rows[1]).sum().backward()
        gradient_bits = tuple(
            struct.unpack(">I", struct.pack(">f", value))[0]
            for row in signed_source.grad.tolist()
            for value in row
        )
        self.assertEqual(gradient_bits, (0x80000000, 0x3F800000))

        ordinary = torch.tensor([[1.0], [2.0], [3.0]])
        self.assertEqual(tuple(row.output_nr for row in ordinary), (0, 0, 0))

        no_grad_source = torch.tensor(
            [[1.0], [2.0], [3.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_rows = tuple(no_grad_source)
        self.assertEqual(tuple(row.output_nr for row in no_grad_rows), (0, 0, 0))
        self.assertTrue(all(row.requires_grad for row in no_grad_rows))
        self.assertTrue(all(row.is_leaf for row in no_grad_rows))

        empty_rows = tuple(torch.zeros((2, 0), requires_grad=True))
        self.assertEqual(tuple(row.output_nr for row in empty_rows), (0, 1))
        self.assertEqual(tuple(row.numel() for row in empty_rows), (0, 0))
        self.assertEqual(tuple(torch.zeros((0, 2), requires_grad=True)), ())

        with self.assertRaises(TypeError) as raised:
            iter(torch.tensor(1.0))
        self.assertEqual(str(raised.exception), "iteration over a 0-d tensor")

    def test_iteration_routes_dim_and_unbind_through_torch_function_mode(self):
        source = torch.tensor([[1.0], [2.0]])

        class ReplacingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                if func.__name__ == "unbind":
                    return ("replacement-a", "replacement-b")
                return func(*args, **(kwargs or {}))

        mode = ReplacingMode()
        with mode:
            result = tuple(iter(source))

        self.assertEqual(result, ("replacement-a", "replacement-b"))
        self.assertEqual([call[0].__name__ for call in mode.calls], ["dim", "unbind"])
        for function, _, _, _ in mode.calls:
            self.assertIs(type(function), types.MethodDescriptorType)
            self.assertEqual(function.__objclass__.__name__, "TensorBase")
            self.assertEqual(function.__objclass__.__module__, "torch._C")

        dim_function, dim_types, dim_args, dim_kwargs = mode.calls[0]
        self.assertEqual(dim_function.__qualname__, "TensorBase.dim")
        self.assertEqual(dim_types, (torch.Tensor,))
        self.assertEqual(dim_args, (source,))
        self.assertIsNone(dim_kwargs)

        unbind_function, unbind_types, unbind_args, unbind_kwargs = mode.calls[1]
        self.assertEqual(unbind_function.__qualname__, "TensorBase.unbind")
        self.assertEqual(unbind_types, ())
        self.assertEqual(unbind_args, (source, 0))
        self.assertIsNone(unbind_kwargs)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_mode_replaced_unbind_uses_python_iterator_protocol(self):
        source = torch.tensor([[1.0], [2.0]])

        class GetItemOnly:
            def __init__(self):
                self.values = ("replacement-a", "replacement-b")

            def __getitem__(self, index):
                return self.values[index]

        class ReplacingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, replacement):
                self.replacement = replacement

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                if func.__name__ == "unbind":
                    return self.replacement
                return func(*args, **(kwargs or {}))

        with ReplacingMode(GetItemOnly()):
            iterator = iter(source)
        self.assertEqual(type(iterator).__name__, "iterator")
        self.assertEqual(tuple(iterator), ("replacement-a", "replacement-b"))

        with self.assertRaises(TypeError) as raised:
            with ReplacingMode(object()):
                iter(source)
        self.assertEqual(str(raised.exception), "'object' object is not iterable")
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

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
