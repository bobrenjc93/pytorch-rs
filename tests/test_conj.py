import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


METHOD_DOC = "\nconj() -> Tensor\n\nSee :func:`torch.conj`\n"


class TensorConjTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 2.0).transpose(0, 1)
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
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
        return (
            leaf,
            tracked,
            (
                ("scalar", torch.tensor(-3.5)),
                ("empty", torch.zeros((2, 0, 3))),
                ("eager negative", source.neg()),
                ("strided view", strided_view),
                ("offset strided view", offset_view),
                ("extreme empty view", extreme_empty),
                (
                    "signed zeros and non-finites",
                    torch.tensor(memoryview(special_bits.view(np.float32))),
                ),
                ("autograd leaf", leaf),
                ("autograd non-leaf view", tracked),
                ("detached autograd view", tracked.detach()),
            ),
        )

    def value_bits(self, tensor):
        if 0 in tensor.shape:
            return None
        return np.asarray(tensor.detach()).reshape(-1).view(np.uint32).copy()

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
        )

    def test_real_float32_tensors_return_the_exact_receiver(self):
        leaf, tracked, cases = self.tensor_cases()
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = self.metadata(tensor)
                bits = self.value_bits(tensor)

                result = tensor.conj()

                self.assertIs(result, tensor)
                self.assertIs(result.is_conj(), False)
                self.assertEqual(self.metadata(result), metadata)
                if bits is not None:
                    np.testing.assert_array_equal(self.value_bits(result), bits)

        tracked.conj().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad), np.full((2, 2), 2.0, dtype=np.float32)
        )
        gradient = leaf.grad
        self.assertIs(leaf.conj(), leaf)
        self.assertIs(leaf.grad, gradient)

    def test_no_grad_preserves_the_existing_autograd_graph(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 3.0).transpose(0, 1)[1]
        metadata = self.metadata(tracked)

        with torch.no_grad():
            result = tracked.conj()

        self.assertIs(result, tracked)
        self.assertEqual(self.metadata(result), metadata)
        self.assertTrue(result.requires_grad)
        self.assertFalse(result.is_leaf)
        result.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[0.0, 3.0], [0.0, 3.0]], dtype=np.float32),
        )

    def test_descriptor_documentation_and_signature_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "conj")
        bound = tensor.conj

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'conj' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "conj")
        self.assertEqual(descriptor.__qualname__, "TensorBase.conj")
        self.assertEqual(bound.__name__, "conj")
        self.assertEqual(bound.__qualname__, "Tensor.conj")
        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertIs(descriptor(tensor), tensor)

    def test_invalid_calls_and_receivers_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "conj")
        bound = tensor.conj
        cases = (
            (
                lambda: tensor.conj(1),
                "TensorBase.conj() takes no arguments (1 given)",
            ),
            (lambda: bound(1), "Tensor.conj() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.conj() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.conj(1, 2),
                "TensorBase.conj() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.conj(input=tensor),
                "TensorBase.conj() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.conj() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.conj() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.conj() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'conj' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.conj() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_torch_function_modes_receive_the_descriptor_and_can_forward(self):
        tensor = torch.tensor([1.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "conj")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.conj()
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
                forwarded = tensor.conj()
        self.assertEqual(order, ["upper", "lower"])
        self.assertIs(forwarded, tensor)
        self.assertTrue(forwarded.requires_grad)

    def test_scope_remains_real_method_only(self):
        self.assertTrue(hasattr(torch.Tensor, "conj"))
        self.assertFalse(hasattr(torch, "conj"))
        self.assertFalse(hasattr(torch.Tensor, "conj_physical"))
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
