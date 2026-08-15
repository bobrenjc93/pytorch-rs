import inspect
import sys
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch


PROPERTY_DOC = (
    "\nReturns a view of a matrix (2-D tensor) conjugated and transposed.\n\n"
    "``x.H`` is equivalent to ``x.transpose(0, 1).conj()`` for complex matrices and\n"
    "``x.transpose(0, 1)`` for real matrices.\n\n"
    ".. seealso::\n\n"
    "        :attr:`~.Tensor.mH`: An attribute that also works on batches of matrices.\n"
)
SCALAR_WARNING = "Tensor.H is deprecated on 0-D tensors. Consider using x.conj()."


def stable_warning_message(message):
    return message.split(" (Triggered internally at ", 1)[0]


class TensorHermitianTransposeTests(unittest.TestCase):
    def assert_native_transpose_view(self, source):
        expected = source.transpose(0, 1)
        result = source.H

        self.assertIsNot(result, source)
        self.assertTrue(result.is_set_to(expected))
        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(result.stride(), expected.stride())
        self.assertEqual(result.storage_offset(), source.storage_offset())
        self.assertIs(result.dtype, source.dtype)
        self.assertEqual(result.device, source.device)
        self.assertEqual(result.data_ptr(), source.data_ptr())
        self.assertFalse(result.is_conj())
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

        self.assertTrue(result.is_set_to(source.mT))
        self.assertTrue(result.is_set_to(source.mH))
        restored = result.H
        self.assertIsNot(restored, source)
        self.assertTrue(restored.is_set_to(source))

    def test_00_scalar_warning_is_once_only_and_points_to_the_caller(self):
        scalar = torch.tensor(2.5, requires_grad=True)
        marker = object()

        class InterceptingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return marker

        with warnings.catch_warnings(record=True) as intercepted_warnings:
            warnings.simplefilter("always")
            with InterceptingMode():
                intercepted = scalar.H
        self.assertIs(intercepted, marker)
        self.assertEqual(intercepted_warnings, [])

        metadata = (
            scalar.shape,
            scalar.stride(),
            scalar.storage_offset(),
            scalar.dtype,
            scalar.device,
            scalar.requires_grad,
            scalar.is_leaf,
            scalar.data_ptr(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            first = scalar.H
            second = scalar.H

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(stable_warning_message(str(caught[0].message)), SCALAR_WARNING)
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(caught[0].lineno, warning_line)
        self.assertIs(first, scalar)
        self.assertIs(second, scalar)
        self.assertEqual(
            (
                first.shape,
                first.stride(),
                first.storage_offset(),
                first.dtype,
                first.device,
                first.requires_grad,
                first.is_leaf,
                first.data_ptr(),
            ),
            metadata,
        )

    def test_matrix_empty_offset_and_strided_views_reuse_transpose(self):
        values = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        dense = torch.tensor(values.tolist())
        cases = (
            ("matrix", torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
            ("empty rows", torch.zeros((0, 3))),
            ("empty columns", torch.zeros((2, 0))),
            ("offset", dense[1]),
            ("offset strided", dense.transpose(0, 2)[1]),
            ("empty offset", torch.zeros((3, 0, 2)).transpose(0, 2)[1]),
        )

        for case, source in cases:
            with self.subTest(case=case, shape=source.shape, stride=source.stride()):
                self.assert_native_transpose_view(source)

    def test_vectors_and_batched_tensors_raise_rank_specific_errors(self):
        cases = (
            (
                torch.tensor([1.0, 2.0, 3.0]),
                "tensor.H is only supported on matrices (2-D tensors). "
                "Got 1-D tensor.",
            ),
            (
                torch.zeros((2, 3, 4)),
                "tensor.H is only supported on matrices (2-D tensors). "
                "Got 3-D tensor. For batches of matrices, consider using tensor.mH",
            ),
            (
                torch.zeros((2, 3, 4, 5)),
                "tensor.H is only supported on matrices (2-D tensors). "
                "Got 4-D tensor. For batches of matrices, consider using tensor.mH",
            ),
            (
                torch.zeros((sys.maxsize, 0, sys.maxsize)),
                "tensor.H is only supported on matrices (2-D tensors). "
                "Got 3-D tensor. For batches of matrices, consider using tensor.mH",
            ),
        )
        for tensor, message in cases:
            with self.subTest(shape=tensor.shape):
                with self.assertRaises(RuntimeError) as raised:
                    tensor.H
                self.assertEqual(str(raised.exception), message)

        offset = torch.zeros((sys.maxsize, 0, 1))[sys.maxsize - 1]
        result = offset.H
        self.assertEqual(result.shape, (1, 0))
        self.assertEqual(result.stride(), (1, 1))
        self.assertEqual(result.storage_offset(), sys.maxsize - 1)
        self.assertEqual(result.data_ptr(), offset.data_ptr())
        self.assertEqual(result.tolist(), [[]])

    def test_autograd_and_no_grad_follow_the_native_view_path(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        view = leaf.H
        self.assertTrue(view.requires_grad)
        self.assertFalse(view.is_leaf)
        self.assertEqual(view.shape, (3, 2))
        self.assertEqual(view.stride(), (1, 3))
        self.assertEqual(view.data_ptr(), leaf.data_ptr())

        (view * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]], dtype=np.float32),
        )

        empty = torch.zeros((2, 0), requires_grad=True)
        empty.H.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.numel(), 0)

        no_grad_source = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            no_grad_view = no_grad_source.H
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        self.assertEqual(no_grad_view.shape, (3, 2))
        self.assertEqual(no_grad_view.stride(), (1, 3))
        self.assertEqual(no_grad_view.data_ptr(), no_grad_source.data_ptr())
        (no_grad_view * no_grad_view).sum().backward()
        self.assertIsNone(no_grad_source.grad)

        scalar = torch.tensor(3.0, requires_grad=True)
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter("ignore")
            self.assertIs(scalar.H, scalar)

    def test_tensorbase_descriptor_is_documented_and_read_only(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "H")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "H")
        self.assertEqual(descriptor.__qualname__, "TensorBase.H")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertEqual(repr(descriptor), "<attribute 'H' of 'torch._C.TensorBase' objects>")
        self.assertIs(torch.Tensor.H, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assertTrue(
            descriptor.__get__(tensor, torch.Tensor).is_set_to(tensor.transpose(0, 1))
        )

        with self.assertRaises(TypeError) as raised:
            descriptor.__get__(1, int)
        self.assertEqual(
            str(raised.exception),
            "descriptor 'H' for 'torch._C.TensorBase' objects "
            "doesn't apply to a 'int' object",
        )

        actions = (
            lambda: setattr(tensor, "H", tensor),
            lambda: delattr(tensor, "H"),
            lambda: descriptor.__set__(tensor, tensor),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(action=action):
                with self.assertRaises(AttributeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "attribute 'H' of 'torch._C.TensorBase' objects is not writable",
                )

    def test_torch_function_modes_receive_descriptor_get_and_forward(self):
        tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        descriptor = inspect.getattr_static(torch.Tensor, "H")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.H
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
                forwarded = tensor.H
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(forwarded.is_set_to(tensor.transpose(0, 1)))


if __name__ == "__main__":
    unittest.main()
