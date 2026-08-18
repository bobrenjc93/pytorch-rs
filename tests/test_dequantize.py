import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


METHOD_DOC = (
    "\ndequantize() -> Tensor\n\n"
    "Given a quantized Tensor, dequantize it and return the dequantized float "
    "Tensor.\n"
)


class TensorDequantizeTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        multi_output = tracked.unbind()[1]
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        strided_view = source.transpose(0, 1)
        offset_view = strided_view[1]
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )

        self.assertFalse(strided_view.is_contiguous())
        self.assertGreater(offset_view.storage_offset(), 0)
        self.assertEqual(multi_output.output_nr, 1)
        return (
            leaf,
            tracked,
            (
                ("scalar", torch.tensor(-3.5)),
                ("empty", torch.zeros((2, 0, 3))),
                ("contiguous", source),
                ("strided view", strided_view),
                ("offset strided view", offset_view),
                ("extreme empty view", extreme_empty),
                (
                    "signed zeros and non-finites",
                    torch.tensor(memoryview(special_bits.view(np.float32))),
                ),
                ("autograd leaf", leaf),
                ("autograd non-leaf view", tracked),
                ("multi-output autograd view", multi_output),
                ("detached autograd view", tracked.detach()),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

    def test_non_quantized_tensors_return_exact_receiver_without_changes(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.device,
                    tensor.layout,
                    tensor.is_quantized,
                    tensor.requires_grad,
                    tensor.is_leaf,
                    tensor.retains_grad,
                    tensor.output_nr,
                )
                bits = self.value_bits(tensor)

                result = tensor.dequantize()

                self.assertIs(result, tensor)
                self.assertIs(result.layout, torch.strided)
                self.assertFalse(result.is_quantized)
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr(),
                        result.dtype,
                        result.device,
                        result.layout,
                        result.is_quantized,
                        result.requires_grad,
                        result.is_leaf,
                        result.retains_grad,
                        result.output_nr,
                    ),
                    metadata,
                )
                if bits is not None:
                    np.testing.assert_array_equal(self.value_bits(result), bits)

        tracked.dequantize().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        gradient = leaf.grad
        self.assertIs(leaf.dequantize(), leaf)
        self.assertIs(leaf.grad, gradient)

    def test_no_grad_preserves_the_existing_autograd_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)

        with torch.no_grad():
            result = tracked.dequantize()

        self.assertIs(result, tracked)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        result.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 3.0, dtype=np.float32)
        )

    def test_tensorbase_descriptor_documentation_and_signature(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "dequantize")
        bound = tensor.dequantize

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'dequantize' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "dequantize")
        self.assertEqual(descriptor.__qualname__, "TensorBase.dequantize")
        self.assertEqual(bound.__name__, "dequantize")
        self.assertEqual(bound.__qualname__, "Tensor.dequantize")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(torch.Tensor.dequantize, descriptor)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(), tensor)

    def test_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "dequantize")
        bound = tensor.dequantize
        cases = (
            (
                lambda: tensor.dequantize(1),
                "TensorBase.dequantize() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.dequantize() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.dequantize() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.dequantize(1, 2),
                "TensorBase.dequantize() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.dequantize(input=tensor),
                "TensorBase.dequantize() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.dequantize() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.dequantize() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.dequantize() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'dequantize' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.dequantize() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_method_descriptor_and_forward(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "dequantize")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.dequantize()
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
                forwarded = tensor.dequantize()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

    def test_surface_excludes_quantized_storage_and_top_level_function(self):
        tensor = torch.tensor([1.0], dtype=torch.float32)

        self.assertFalse(tensor.is_quantized)
        self.assertIs(tensor.dtype, torch.float32)
        for name in (
            "dequantize",
            "quantize_per_tensor",
            "quantize_per_channel",
            "qint8",
            "quint8",
            "qint32",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
