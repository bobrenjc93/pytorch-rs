import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nto_dense(dtype=None, *, masked_grad=True) -> Tensor\n\n"
    "Creates a strided copy of :attr:`self` if :attr:`self` is not a strided "
    "tensor, otherwise returns :attr:`self`.\n\n"
    "Keyword args:\n"
    "    {dtype}\n"
    "    masked_grad (bool, optional): If set to ``True`` (default) and\n"
    "      :attr:`self` has a sparse layout then the backward of\n"
    "      :meth:`to_dense` returns ``grad.sparse_mask(self)``.\n\n"
    "Example::\n\n"
    "    >>> s = torch.sparse_coo_tensor(\n"
    "    ...        torch.tensor([[1, 1],\n"
    "    ...                      [0, 2]]),\n"
    "    ...        torch.tensor([9, 10]),\n"
    "    ...        size=(3, 3))\n"
    "    >>> s.to_dense()\n"
    "    tensor([[ 0,  0,  0],\n"
    "            [ 9,  0, 10],\n"
    "            [ 0,  0,  0]])\n"
)


class TensorToDenseTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        strided = source.transpose(0, 1)
        offset = strided[1]

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        return (
            leaf,
            tracked,
            (
                ("scalar", torch.tensor(-3.5)),
                ("empty", torch.zeros((2, 0, 3))),
                ("contiguous", source),
                ("noncontiguous", strided),
                ("offset noncontiguous", offset),
                ("autograd leaf", leaf),
                ("autograd non-leaf", tracked),
                ("detached autograd view", tracked.detach()),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

    def test_strided_tensors_return_exact_receiver_without_changes(self):
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
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                bits = self.value_bits(tensor)

                result = tensor.to_dense()

                self.assertIs(result, tensor)
                self.assertIs(result.layout, torch.strided)
                self.assertFalse(result.is_sparse)
                self.assertEqual(
                    (
                        result.shape,
                        result.stride(),
                        result.storage_offset(),
                        result.data_ptr(),
                        result.dtype,
                        result.device,
                        result.layout,
                        result.requires_grad,
                        result.is_leaf,
                    ),
                    metadata,
                )
                if bits is not None:
                    np.testing.assert_array_equal(self.value_bits(result), bits)

        tracked.to_dense().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )

    def test_no_grad_returns_same_autograd_tensor_and_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 3.0).transpose(0, 1)

        with torch.no_grad():
            result = tracked.to_dense()

        self.assertIs(result, tracked)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        result.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 3.0, dtype=np.float32)
        )

    def test_descriptor_ownership_and_documentation_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to_dense")
        bound = tensor.to_dense

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'to_dense' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "to_dense")
        self.assertEqual(descriptor.__qualname__, "TensorBase.to_dense")
        self.assertEqual(bound.__name__, "to_dense")
        self.assertEqual(bound.__qualname__, "Tensor.to_dense")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(descriptor)
        with self.assertRaises(ValueError):
            inspect.signature(bound)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(bound(), tensor)

    def test_surface_remains_no_argument_and_strided_only(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to_dense")
        calls = (
            lambda: tensor.to_dense(torch.float32),
            lambda: tensor.to_dense(dtype=torch.float32),
            lambda: tensor.to_dense(masked_grad=False),
            lambda: descriptor(tensor, torch.float32),
            lambda: descriptor(tensor, masked_grad=True),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        self.assertFalse(hasattr(torch, "to_dense"))
        self.assertFalse(hasattr(torch, "sparse_coo_tensor"))
        self.assertFalse(tensor.is_sparse)
        self.assertIs(tensor.layout, torch.strided)

    def test_torch_function_modes_match_tensorbase_method_dispatch(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "to_dense")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            result = tensor.to_dense()
        self.assertIs(result, marker)
        self.assertEqual(len(recording.calls), 1)
        function, dispatch_types, args, kwargs = recording.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, ())
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
                forwarded = tensor.to_dense()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        with self.assertRaises(TypeError) as raised:
            with lower:
                with declining:
                    tensor.to_dense()
        self.assertRegex(
            str(raised.exception),
            re.compile(
                r"^Multiple dispatch failed for 'torch\.Tensor\.to_dense'; all "
                r"__torch_function__ handlers returned NotImplemented:\n\n"
                r"  - mode object <.*RecordingMode object at 0x[0-9a-f]+>\n\n"
                r"For more information, try re-running with "
                r"TORCH_LOGS=not_implemented$"
            ),
        )
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(lower.calls, [])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])


if __name__ == "__main__":
    unittest.main()
