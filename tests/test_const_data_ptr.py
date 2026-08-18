import ctypes
import inspect
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\nconst_data_ptr() -> int\n\n"
    "Returns the address of the first element of :attr:`self` tensor.\n\n"
    "Unlike :meth:`data_ptr`, this is guaranteed to be a read-only access\n"
    "that will not trigger copy-on-write materialization. For regular\n"
    "(non-COW) tensors, the return value is identical to :meth:`data_ptr`.\n\n"
    ".. warning::\n\n"
    "    The returned pointer must not be used to mutate the tensor data.\n"
    "    Use :meth:`data_ptr` when write access is needed.\n"
)


class TensorConstDataPtrTests(unittest.TestCase):
    def tensor_state(self, tensor):
        return (
            tensor.tolist(),
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

    def test_matches_data_ptr_for_all_supported_ordinary_storage(self):
        scalar = torch.tensor(2.5)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        )
        offset = source[2]
        strided = source.transpose(0, 1)[1]
        detached = strided.detach()
        empty = torch.zeros((3, 0, 4))[2]
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        autograd_view = (leaf * 3.0).transpose(0, 1)

        tensors = {
            "scalar": scalar,
            "contiguous": source,
            "offset": offset,
            "strided": strided,
            "empty": empty,
            "detached": detached,
            "autograd_leaf": leaf,
            "autograd_view": autograd_view,
        }
        for name, tensor in tensors.items():
            with self.subTest(name=name):
                state = self.tensor_state(tensor)
                pointer = tensor.const_data_ptr()
                self.assertIs(type(pointer), int)
                self.assertEqual(pointer, tensor.data_ptr())
                self.assertEqual(tensor.const_data_ptr(), pointer)
                self.assertEqual(self.tensor_state(tensor), state)

        self.assertEqual(empty.const_data_ptr(), 0)
        self.assertEqual(
            offset.const_data_ptr() - source.const_data_ptr(),
            offset.storage_offset() * offset.element_size(),
        )
        self.assertEqual(
            strided.const_data_ptr() - source.const_data_ptr(),
            strided.storage_offset() * strided.element_size(),
        )
        self.assertEqual(
            ctypes.c_float.from_address(scalar.const_data_ptr()).value,
            2.5,
        )
        self.assertEqual(
            ctypes.c_float.from_address(strided.const_data_ptr()).value,
            1.0,
        )

        autograd_view.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])
        self.assertEqual(leaf.grad.const_data_ptr(), leaf.grad.data_ptr())

    def test_tensorbase_descriptor_documentation_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "const_data_ptr")
        bound = tensor.const_data_ptr

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object, python_313_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            self.assertEqual(callable_object.__name__, "const_data_ptr")
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

        self.assertEqual(
            descriptor.__qualname__, "TensorBase.const_data_ptr"
        )
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertEqual(descriptor(tensor), tensor.const_data_ptr())

    def test_no_argument_errors_match_pytorch_2_13(self):
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
        )
        for call, message in calls:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptor_and_forward(self):
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
        self.assertEqual(forwarded, tensor.data_ptr())


if __name__ == "__main__":
    unittest.main()
