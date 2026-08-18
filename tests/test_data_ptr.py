import ctypes
import inspect
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\ndata_ptr() -> int\n\n"
    "Returns the address of the first element of :attr:`self` tensor.\n\n"
    ".. note::\n\n"
    "    If the tensor is a copy-on-write tensor (e.g. created via\n"
    "    :meth:`_lazy_clone`), calling this method will materialize the\n"
    "    copy. Use :meth:`const_data_ptr` if you only need read-only access\n"
    "    to the data pointer.\n"
)

CONST_METHOD_DOC = (
    "\nconst_data_ptr() -> int\n\n"
    "Returns the address of the first element of :attr:`self` tensor.\n\n"
    "Unlike :meth:`data_ptr`, this is guaranteed to be a read-only access\n"
    "that will not trigger copy-on-write materialization. For regular\n"
    "(non-COW) tensors, the return value is identical to :meth:`data_ptr`.\n\n"
    ".. warning::\n\n"
    "    The returned pointer must not be used to mutate the tensor data.\n"
    "    Use :meth:`data_ptr` when write access is needed.\n"
)


class TensorDataPtrTests(unittest.TestCase):
    def test_pointer_round_trips_through_ctypes(self):
        source = torch.tensor(
            [
                [0.25, 1.25, 2.25, 3.25],
                [4.25, 5.25, 6.25, 7.25],
                [8.25, 9.25, 10.25, 11.25],
            ]
        )
        row = source[1]
        self.assertEqual(row.const_data_ptr(), row.data_ptr())
        row_values = (ctypes.c_float * 4).from_address(row.const_data_ptr())
        self.assertEqual(list(row_values), [4.25, 5.25, 6.25, 7.25])

        strided_row = source.transpose(0, 1)[1]
        self.assertEqual(strided_row.const_data_ptr(), strided_row.data_ptr())
        first_value = ctypes.c_float.from_address(strided_row.const_data_ptr())
        self.assertEqual(first_value.value, 1.25)

        materialized = strided_row.clone()
        self.assertEqual(materialized.const_data_ptr(), materialized.data_ptr())
        materialized_values = (ctypes.c_float * 3).from_address(
            materialized.const_data_ptr()
        )
        self.assertEqual(list(materialized_values), [1.25, 5.25, 9.25])

        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
        (leaf * 4.0).sum().backward()
        self.assertEqual(leaf.grad.const_data_ptr(), leaf.grad.data_ptr())
        gradient_values = (ctypes.c_float * 2).from_address(
            leaf.grad.const_data_ptr()
        )
        self.assertEqual(list(gradient_values), [4.0, 4.0])

    def test_pointer_offsets_empty_views_and_aliases(self):
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        source_ptr = source.data_ptr()
        self.assertIs(type(source_ptr), int)
        self.assertGreater(source_ptr, 0)
        self.assertEqual(source.data_ptr(), source_ptr)
        self.assertEqual(source.const_data_ptr(), source_ptr)

        transposed = source.transpose(0, 1)
        self.assertEqual(transposed.data_ptr(), source_ptr)
        self.assertEqual(transposed.const_data_ptr(), source_ptr)

        row = source[2]
        self.assertEqual(row.storage_offset(), 8)
        self.assertEqual(
            row.data_ptr() - source_ptr,
            row.storage_offset() * source.element_size(),
        )
        self.assertEqual(row.const_data_ptr(), row.data_ptr())

        strided_row = transposed[1]
        self.assertEqual(strided_row.storage_offset(), 1)
        self.assertEqual(strided_row.data_ptr() - source_ptr, source.element_size())
        self.assertEqual(strided_row.const_data_ptr(), strided_row.data_ptr())
        self.assertEqual(strided_row.detach().data_ptr(), strided_row.data_ptr())
        self.assertEqual(
            strided_row.detach().const_data_ptr(), strided_row.data_ptr()
        )
        self.assertEqual(torch.detach(strided_row).data_ptr(), strided_row.data_ptr())
        self.assertEqual(
            torch.detach(strided_row).const_data_ptr(), strided_row.data_ptr()
        )

        empty = torch.zeros((3, 0, 4))
        offset_empty = empty[2]
        self.assertGreater(offset_empty.storage_offset(), 0)
        for tensor in (
            empty,
            empty.transpose(0, 2),
            offset_empty,
            offset_empty.detach(),
            offset_empty.clone(),
        ):
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assertIs(type(tensor.data_ptr()), int)
                self.assertEqual(tensor.data_ptr(), 0)
                self.assertIs(type(tensor.const_data_ptr()), int)
                self.assertEqual(tensor.const_data_ptr(), tensor.data_ptr())

        extreme_empty = torch.zeros((sys.maxsize, 0))[sys.maxsize - 1]
        self.assertEqual(extreme_empty.storage_offset(), sys.maxsize - 1)
        self.assertEqual(extreme_empty.data_ptr(), 0)
        self.assertEqual(extreme_empty.const_data_ptr(), 0)

    def test_live_copies_materializations_and_autograd_are_independent(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        produced = leaf * 3.0
        view = produced.transpose(0, 1)
        metadata = (
            view.shape,
            view.stride(),
            view.storage_offset(),
            view.requires_grad,
            view.is_leaf,
        )

        view_ptr = view.data_ptr()
        self.assertEqual(view.const_data_ptr(), view_ptr)
        self.assertEqual(produced.data_ptr(), view_ptr)
        self.assertEqual(produced.const_data_ptr(), view_ptr)
        self.assertEqual(view.detach().data_ptr(), view_ptr)
        self.assertEqual(view.detach().const_data_ptr(), view_ptr)

        cloned = view.clone()
        packed = view.contiguous()
        self.assertGreater(cloned.data_ptr(), 0)
        self.assertGreater(packed.data_ptr(), 0)
        self.assertEqual(cloned.const_data_ptr(), cloned.data_ptr())
        self.assertEqual(packed.const_data_ptr(), packed.data_ptr())
        self.assertNotEqual(cloned.data_ptr(), view_ptr)
        self.assertNotEqual(packed.data_ptr(), view_ptr)
        self.assertNotEqual(cloned.data_ptr(), packed.data_ptr())
        self.assertTrue(cloned.requires_grad)
        self.assertFalse(cloned.is_leaf)
        self.assertTrue(packed.requires_grad)
        self.assertFalse(packed.is_leaf)
        self.assertEqual(
            (
                view.shape,
                view.stride(),
                view.storage_offset(),
                view.requires_grad,
                view.is_leaf,
            ),
            metadata,
        )

        cloned_ptr = cloned.data_ptr()
        cloned.sum().backward()
        self.assertEqual(cloned.data_ptr(), cloned_ptr)
        self.assertEqual(cloned.const_data_ptr(), cloned_ptr)
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])

        live_gradient = leaf.grad
        gradient_ptr = live_gradient.data_ptr()
        self.assertGreater(gradient_ptr, 0)
        self.assertIs(leaf.grad, live_gradient)
        self.assertEqual(leaf.grad.data_ptr(), gradient_ptr)
        self.assertEqual(leaf.grad.const_data_ptr(), gradient_ptr)

    def test_tensorbase_descriptor_documentation_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "data_ptr")
        bound = tensor.data_ptr

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object, python_313_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            self.assertEqual(callable_object.__name__, "data_ptr")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)),
                    python_313_signature,
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        self.assertEqual(descriptor.__qualname__, "TensorBase.data_ptr")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor(tensor), tensor.data_ptr())

    def test_no_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "data_ptr")
        bound = tensor.data_ptr
        calls = (
            (
                lambda: tensor.data_ptr(1),
                "TensorBase.data_ptr() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1, 2),
                "Tensor.data_ptr() takes no arguments (2 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.data_ptr() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.data_ptr(dim=0),
                (
                    "Tensor.data_ptr() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.data_ptr() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.data_ptr() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.data_ptr() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.data_ptr() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'data_ptr' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_const_tensorbase_descriptor_documentation_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "const_data_ptr")
        bound = tensor.const_data_ptr

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'const_data_ptr' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "const_data_ptr")
        self.assertEqual(descriptor.__qualname__, "TensorBase.const_data_ptr")
        self.assertEqual(bound.__name__, "const_data_ptr")
        self.assertEqual(bound.__qualname__, "Tensor.const_data_ptr")
        self.assertEqual(descriptor.__doc__, CONST_METHOD_DOC)
        self.assertEqual(bound.__doc__, CONST_METHOD_DOC)
        for callable_object, python_313_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)),
                    python_313_signature,
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(descriptor(tensor), tensor.data_ptr())
        self.assertEqual(bound(), tensor.data_ptr())

    def test_const_no_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "const_data_ptr")
        bound = tensor.const_data_ptr
        calls = (
            (
                lambda: tensor.const_data_ptr(1),
                "TensorBase.const_data_ptr() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1, 2),
                "Tensor.const_data_ptr() takes no arguments (2 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.const_data_ptr() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.const_data_ptr(dim=0),
                (
                    "Tensor.const_data_ptr() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.const_data_ptr() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.const_data_ptr() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.const_data_ptr() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.const_data_ptr() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'const_data_ptr' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.const_data_ptr() needs an argument",
            ),
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_const_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "const_data_ptr")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.const_data_ptr()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
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
                forwarded = tensor.const_data_ptr()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(type(forwarded), int)
        self.assertEqual(forwarded, tensor.data_ptr())


if __name__ == "__main__":
    unittest.main()
