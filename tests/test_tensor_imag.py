import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


ERROR = "imag is not implemented for tensors with non-complex dtypes."
PROPERTY_DOC = (
    "\nReturns a new tensor containing imaginary values of the :attr:`self` "
    "tensor.\n"
    "The returned tensor and :attr:`self` share the same underlying storage.\n\n"
    ".. warning::\n"
    "    :func:`imag` is only supported for tensors with complex dtypes.\n\n"
    "Example::\n\n"
    "    >>> x=torch.randn(4, dtype=torch.cfloat)\n"
    "    >>> x\n"
    "    tensor([(0.3100+0.3553j), (-0.5445-0.7896j), "
    "(-1.6492-0.0633j), (-0.0638-0.8119j)])\n"
    "    >>> x.imag\n"
    "    tensor([ 0.3553, -0.7896, -0.0633, -0.8119])\n\n"
)


class TensorImagTests(unittest.TestCase):
    def tensor_cases(self):
        scalar_bits = np.asarray(
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
        scalar_storage = torch.tensor(memoryview(scalar_bits.view(np.float32)))
        base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        offset = strided[1]
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        channels_last = torch.zeros((2, 3, 4, 5)).contiguous(
            memory_format=torch.channels_last
        )
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertEqual(empty.shape, (0, 2))
        self.assertGreater(empty.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(leaf.is_leaf)
        self.assertFalse(non_leaf.is_leaf)
        return (
            *(
                (f"float32 bits 0x{bits:08x}", scalar_storage[index])
                for index, bits in enumerate(scalar_bits)
            ),
            ("empty offset view", empty),
            ("offset strided view", offset),
            ("strided view", strided),
            ("channels-last tensor", channels_last),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
        )

    def tensor_state(self, tensor):
        detached = tensor.detach()
        return (
            (
                tensor.shape,
                tensor.stride(),
                tensor.storage_offset(),
                tensor.layout,
                tensor.is_contiguous(),
                tensor.is_contiguous(memory_format=torch.channels_last),
                tensor.dtype,
                tensor.device,
                tensor.requires_grad,
                tensor.is_leaf,
                tensor.data_ptr(),
            ),
            detached,
            np.asarray(detached).reshape(-1).view(np.uint32).copy(),
        )

    def assert_imag_error(self, action):
        with self.assertRaises(RuntimeError) as raised:
            action()
        self.assertEqual(str(raised.exception), ERROR)
        self.assertEqual(raised.exception.args, (ERROR,))
        return raised.exception

    def test_every_supported_tensor_raises_fresh_errors_without_side_effects(self):
        for case, tensor in self.tensor_cases():
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata, detached, bits = self.tensor_state(tensor)
                errors = [self.assert_imag_error(lambda: tensor.imag) for _ in range(3)]

                after_metadata, after_detached, after_bits = self.tensor_state(tensor)
                self.assertEqual(after_metadata, metadata)
                self.assertTrue(tensor.is_set_to(detached))
                self.assertTrue(after_detached.is_set_to(detached))
                np.testing.assert_array_equal(after_bits, bits)
                self.assertTrue(
                    all(
                        first is not second
                        for index, first in enumerate(errors)
                        for second in errors[index + 1 :]
                    )
                )

    def test_leaf_and_non_leaf_autograd_state_survives_failed_reads(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        leaf_before = self.tensor_state(leaf)
        non_leaf_before = self.tensor_state(non_leaf)

        self.assert_imag_error(lambda: leaf.imag)
        self.assert_imag_error(lambda: non_leaf.imag)

        leaf_after = self.tensor_state(leaf)
        non_leaf_after = self.tensor_state(non_leaf)
        self.assertEqual(leaf_after[0], leaf_before[0])
        self.assertEqual(non_leaf_after[0], non_leaf_before[0])
        np.testing.assert_array_equal(leaf_after[2], leaf_before[2])
        np.testing.assert_array_equal(non_leaf_after[2], non_leaf_before[2])

        non_leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        gradient_state = self.tensor_state(gradient)

        self.assert_imag_error(lambda: leaf.imag)
        self.assert_imag_error(lambda: non_leaf.imag)
        self.assertIs(leaf.grad, gradient)
        self.assertEqual(self.tensor_state(gradient)[0], gradient_state[0])
        np.testing.assert_array_equal(self.tensor_state(gradient)[2], gradient_state[2])

    def test_tensorbase_descriptor_metadata_assignment_and_deletion(self):
        tensor = torch.tensor([1.0])
        replacement = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "imag")

        self.assertIs(type(descriptor), types.GetSetDescriptorType)
        self.assertFalse(callable(descriptor))
        self.assertEqual(descriptor.__name__, "imag")
        self.assertEqual(descriptor.__qualname__, "TensorBase.imag")
        self.assertEqual(descriptor.__doc__, PROPERTY_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertFalse(hasattr(descriptor, "__text_signature__"))
        self.assertEqual(
            repr(descriptor),
            "<attribute 'imag' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.imag, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assert_imag_error(lambda: descriptor.__get__(tensor, torch.Tensor))

        actions = (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__delete__(tensor),
        )
        errors = []
        for action in actions:
            with self.subTest(action=action):
                errors.append(self.assert_imag_error(action))
        self.assertTrue(
            all(
                first is not second
                for index, first in enumerate(errors)
                for second in errors[index + 1 :]
            )
        )

        invalid_actions = (
            lambda: descriptor.__get__(1, int),
            lambda: descriptor.__set__(1, replacement),
            lambda: descriptor.__delete__(1),
        )
        for action in invalid_actions:
            with self.subTest(action=action):
                with self.assertRaises(TypeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "descriptor 'imag' for 'torch._C.TensorBase' objects "
                    "doesn't apply to a 'int' object",
                )

        self.assertEqual(tensor.tolist(), [1.0])

    def test_torch_function_modes_intercept_reads_and_restore_after_errors(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "imag")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.imag
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

        with self.assertRaisesRegex(RuntimeError, f"^{re.escape(ERROR)}$"):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tensor.imag
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        replacement = torch.tensor([2.0])
        mutation_actions = (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__delete__(tensor),
        )
        for action in mutation_actions:
            with self.subTest(action=action):
                mode = RecordingMode()
                with mode:
                    self.assert_imag_error(action)
                self.assertEqual(mode.calls, [])

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
        with self.assertRaisesRegex(
            RecursionError, r"^maximum recursion depth exceeded$"
        ):
            with lower:
                with upper:
                    tensor.imag
        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, 0)
        self.assert_imag_error(lambda: tensor.imag)

    def test_complex_dtypes_and_top_level_imag_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "imag"))
        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
