import inspect
import types
import unittest

import numpy as np
import torch_rs as torch


ERROR = "imag is not implemented for tensors with non-complex dtypes."
PROPERTY_DOC = (
    "\nReturns a new tensor containing imaginary values of the :attr:`self` tensor.\n"
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
    def assert_imag_error(self, action):
        with self.assertRaises(RuntimeError) as raised:
            action()
        self.assertIs(type(raised.exception), RuntimeError)
        self.assertEqual(str(raised.exception), ERROR)
        self.assertEqual(raised.exception.args, (ERROR,))
        return raised.exception

    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.output_nr,
        )

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
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        )
        strided = base.transpose(0, 3)
        offset = strided[1]
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        channels_last = base.contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.zeros((2, 3, 4, 5, 6)).contiguous(
            memory_format=torch.channels_last_3d
        )
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        non_leaf = (leaf * 3.0).transpose(0, 1)[1]
        with torch.no_grad():
            no_grad_view = leaf.transpose(0, 1)

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertEqual(empty.shape, (0, 2))
        self.assertGreater(empty.storage_offset(), 0)
        self.assertTrue(
            channels_last.is_contiguous(memory_format=torch.channels_last)
        )
        self.assertTrue(
            channels_last_3d.is_contiguous(memory_format=torch.channels_last_3d)
        )
        return leaf, non_leaf, (
            *(
                (f"float32 bits 0x{bits:08x}", scalar_storage[index])
                for index, bits in enumerate(scalar_bits)
            ),
            ("contiguous", base),
            ("empty offset view", empty),
            ("offset strided view", offset),
            ("strided view", strided),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            ("autograd leaf", leaf),
            ("autograd non-leaf", non_leaf),
            ("detached non-leaf", non_leaf.detach()),
            ("no-grad leaf view", no_grad_view),
            ("high-rank empty", torch.zeros((1, 0, 1, 1, 1, 1))),
        )

    def test_every_supported_tensor_raises_fresh_errors_without_side_effects(self):
        leaf, non_leaf, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)
                alias = tensor.detach()
                bits = (
                    np.asarray(alias).reshape(-1).view(np.uint32).copy()
                )

                errors = [
                    self.assert_imag_error(lambda tensor=tensor: tensor.imag)
                    for _ in range(3)
                ]

                self.assertEqual(len({id(error) for error in errors}), len(errors))
                self.assertEqual(self.metadata(tensor), metadata)
                self.assertTrue(tensor.is_set_to(alias))
                np.testing.assert_array_equal(
                    np.asarray(tensor.detach()).reshape(-1).view(np.uint32),
                    bits,
                )

        self.assertIsNone(leaf.grad)
        non_leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        gradient = leaf.grad
        self.assert_imag_error(lambda: leaf.imag)
        self.assertIs(leaf.grad, gradient)

    def test_tensorbase_descriptor_metadata_and_mutation_errors_match(self):
        tensor = torch.tensor([1.0], requires_grad=True)
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
        self.assertEqual(
            repr(descriptor),
            "<attribute 'imag' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.imag, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        self.assert_imag_error(lambda: descriptor.__get__(tensor, torch.Tensor))

        receiver_actions = (
            lambda: descriptor.__get__(1, int),
            lambda: descriptor.__set__(1, replacement),
            lambda: descriptor.__delete__(1),
        )
        for action in receiver_actions:
            with self.subTest(receiver_action=action):
                with self.assertRaises(TypeError) as raised:
                    action()
                self.assertEqual(
                    str(raised.exception),
                    "descriptor 'imag' for 'torch._C.TensorBase' objects "
                    "doesn't apply to a 'int' object",
                )

        metadata = self.metadata(tensor)
        alias = tensor.detach()
        actions = (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__set__(tensor, None),
            lambda: descriptor.__delete__(tensor),
        )
        for action in actions:
            with self.subTest(mutation_action=action):
                errors = [self.assert_imag_error(action) for _ in range(3)]
                self.assertEqual(len({id(error) for error in errors}), len(errors))
                self.assertEqual(self.metadata(tensor), metadata)
                self.assertTrue(tensor.is_set_to(alias))

    def test_torch_function_modes_intercept_reads_and_not_mutations(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        replacement = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "imag")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
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

        with self.assertRaisesRegex(RuntimeError, f"^{ERROR}$"):
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tensor.imag
        self.assertEqual(order, ["upper", "lower"])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return NotImplemented

        lower = RecordingMode(marker)
        upper = DecliningMode()
        with self.assertRaisesRegex(
            RecursionError, r"^maximum recursion depth exceeded$"
        ):
            with lower:
                with upper:
                    tensor.imag
        self.assertGreater(upper.calls, 1)
        self.assertEqual(lower.calls, [])

        mutation_actions = (
            lambda: setattr(tensor, "imag", replacement),
            lambda: delattr(tensor, "imag"),
            lambda: descriptor.__set__(tensor, replacement),
            lambda: descriptor.__delete__(tensor),
        )
        for action in mutation_actions:
            with self.subTest(mutation_action=action):
                mode = RecordingMode(marker)
                with mode:
                    self.assert_imag_error(action)
                self.assertEqual(mode.calls, [])

        self.assert_imag_error(lambda: tensor.imag)

    def test_complex_dtypes_and_top_level_imag_remain_unsupported(self):
        self.assertTrue(hasattr(torch.Tensor, "imag"))
        for name in (
            "complex32",
            "complex64",
            "complex128",
            "chalf",
            "cfloat",
            "cdouble",
        ):
            with self.subTest(dtype=name):
                self.assertFalse(hasattr(torch, name))

        self.assertFalse(hasattr(torch, "imag"))
        self.assertFalse(hasattr(torch._C, "imag"))
        self.assertNotIn("imag", torch.__all__)


if __name__ == "__main__":
    unittest.main()
