import inspect
import sys
import types
import unittest

import torch_rs as torch


IS_COALESCED_DOC = (
    "\nis_coalesced() -> bool\n\n"
    "Returns ``True`` if :attr:`self` is a :ref:`sparse COO tensor\n"
    "<sparse-coo-docs>` that is coalesced, ``False`` otherwise.\n\n"
    ".. warning::\n"
    "  Throws an error if :attr:`self` is not a sparse COO tensor.\n\n"
    "See :meth:`coalesce` and :ref:`uncoalesced tensors "
    "<sparse-uncoalesced-coo-docs>`.\n"
)
STRIDED_ERROR = (
    "is_coalesced expected sparse coordinate tensor layout but got Strided"
)


class TensorIsCoalescedTests(unittest.TestCase):
    def tensor_cases(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        gradient = leaf.grad
        source = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ],
            dtype=torch.float32,
        )
        noncontiguous = source.transpose(0, 1)
        offset = noncontiguous[1]

        self.assertFalse(noncontiguous.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        self.assertIsNotNone(gradient)
        return leaf, gradient, (
            ("scalar", torch.tensor(3.5, dtype=torch.float32)),
            ("empty", torch.zeros((2, 0, 3), dtype=torch.float32)),
            ("noncontiguous view", noncontiguous),
            ("offset view", offset),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", gradient),
        )

    def metadata(self, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            tensor.tolist(),
            tensor.dtype,
            tensor.device,
            tensor.requires_grad,
            tensor.is_leaf,
            repr(tensor),
        )

    def test_every_supported_strided_tensor_raises_without_mutation(self):
        leaf, gradient, cases = self.tensor_cases()
        gradient_metadata = self.metadata(gradient)

        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                before = self.metadata(tensor)
                with self.assertRaises(RuntimeError) as raised:
                    tensor.is_coalesced()
                self.assertEqual(str(raised.exception), STRIDED_ERROR)
                self.assertEqual(self.metadata(tensor), before)
                self.assertIs(tensor.layout, torch.strided)

        self.assertIs(leaf.grad, gradient)
        self.assertEqual(self.metadata(gradient), gradient_metadata)

    def test_tensorbase_descriptor_and_documentation_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_coalesced")
        bound = tensor.is_coalesced

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'is_coalesced' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__name__, "is_coalesced")
        self.assertEqual(descriptor.__qualname__, "TensorBase.is_coalesced")
        self.assertEqual(bound.__name__, "is_coalesced")
        self.assertEqual(bound.__qualname__, "Tensor.is_coalesced")
        self.assertEqual(descriptor.__doc__, IS_COALESCED_DOC)
        self.assertEqual(bound.__doc__, IS_COALESCED_DOC)
        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)

        for call in (bound, lambda: descriptor(tensor)):
            with self.assertRaises(RuntimeError) as raised:
                call()
            self.assertEqual(str(raised.exception), STRIDED_ERROR)

    def test_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_coalesced")
        bound = tensor.is_coalesced
        cases = (
            (
                lambda: tensor.is_coalesced(1),
                "TensorBase.is_coalesced() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.is_coalesced() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.is_coalesced() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_coalesced(1, 2),
                "TensorBase.is_coalesced() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_coalesced(input=tensor),
                (
                    "Tensor.is_coalesced() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.is_coalesced() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.is_coalesced() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.is_coalesced() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.is_coalesced() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'is_coalesced' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.is_coalesced() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_intercept_and_forward_to_layout_error(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_coalesced")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.is_coalesced()
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

        with self.assertRaises(RuntimeError) as raised:
            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    tensor.is_coalesced()
        self.assertEqual(str(raised.exception), STRIDED_ERROR)
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(
            len(torch.overrides._get_current_function_mode_stack()), 0
        )

    def test_sparse_coo_construction_and_operations_remain_unsupported(self):
        for name in ("sparse_coo_tensor", "sparse_coo"):
            self.assertFalse(hasattr(torch, name))
        for name in ("coalesce", "_coalesced_", "indices", "_indices"):
            self.assertFalse(hasattr(torch.Tensor, name))


if __name__ == "__main__":
    unittest.main()
