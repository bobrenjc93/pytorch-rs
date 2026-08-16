import inspect
import sys
import types
import unittest

import torch_rs as torch


DENSE_DIM_DOC = (
    "\ndense_dim() -> int\n\n"
    "Return the number of dense dimensions in a :ref:`sparse tensor "
    "<sparse-docs>` :attr:`self`.\n\n"
    ".. note::\n"
    "  Returns ``len(self.shape)`` if :attr:`self` is not a sparse tensor.\n\n"
    "See also :meth:`Tensor.sparse_dim` and :ref:`hybrid tensors "
    "<sparse-hybrid-coo-docs>`.\n"
)
SPARSE_DIM_DOC = (
    "\nsparse_dim() -> int\n\n"
    "Return the number of sparse dimensions in a :ref:`sparse tensor "
    "<sparse-docs>` :attr:`self`.\n\n"
    ".. note::\n"
    "  Returns ``0`` if :attr:`self` is not a sparse tensor.\n\n"
    "See also :meth:`Tensor.dense_dim` and :ref:`hybrid tensors "
    "<sparse-hybrid-coo-docs>`.\n"
)


class TensorDenseSparseDimTests(unittest.TestCase):
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
        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )

        self.assertFalse(strided.is_contiguous())
        self.assertGreater(offset.storage_offset(), 0)
        return leaf, tracked, (
            ("scalar", torch.tensor(3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("strided view", strided),
            ("offset strided view", offset),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
        )

    def test_strided_dimension_metadata_is_constant_and_non_mutating(self):
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
                    tensor.requires_grad,
                    tensor.is_leaf,
                    tensor.is_sparse,
                    tensor.is_sparse_csr,
                )

                dense_dimensions = tensor.dense_dim()
                sparse_dimensions = tensor.sparse_dim()

                self.assertIs(type(dense_dimensions), int)
                self.assertIs(type(sparse_dimensions), int)
                self.assertEqual(dense_dimensions, tensor.ndim)
                self.assertEqual(sparse_dimensions, 0)
                self.assertEqual(dense_dimensions + sparse_dimensions, tensor.ndim)
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
                        tensor.is_sparse,
                        tensor.is_sparse_csr,
                    ),
                    metadata,
                )

        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0], [2.0, 2.0]])
        self.assertEqual(leaf.dense_dim(), leaf.ndim)
        self.assertEqual(tracked.sparse_dim(), 0)

    def test_tensorbase_descriptors_and_documentation_match_pytorch_2_13(self):
        tensor = torch.zeros((2, 0, 3))
        for name, expected, doc in (
            ("dense_dim", 3, DENSE_DIM_DOC),
            ("sparse_dim", 0, SPARSE_DIM_DOC),
        ):
            with self.subTest(name=name):
                descriptor = inspect.getattr_static(torch.Tensor, name)
                bound = getattr(tensor, name)

                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(
                    repr(descriptor),
                    f"<method '{name}' of 'torch._C.TensorBase' objects>",
                )
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(descriptor.__qualname__, f"TensorBase.{name}")
                self.assertEqual(bound.__name__, name)
                self.assertEqual(bound.__qualname__, f"Tensor.{name}")
                self.assertEqual(descriptor.__doc__, doc)
                self.assertEqual(bound.__doc__, doc)
                for callable_object, expected_signature in (
                    (descriptor, "(self, /)"),
                    (bound, "()"),
                ):
                    if sys.version_info >= (3, 13):
                        self.assertEqual(
                            callable_object.__text_signature__, "($self, /)"
                        )
                        self.assertEqual(
                            str(inspect.signature(callable_object)),
                            expected_signature,
                        )
                    else:
                        self.assertIsNone(callable_object.__text_signature__)
                        with self.assertRaises(ValueError):
                            inspect.signature(callable_object)
                self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
                self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
                self.assertFalse(hasattr(descriptor, "__module__"))
                self.assertIsNone(bound.__module__)
                self.assertEqual(descriptor(tensor), expected)
                self.assertEqual(bound(), expected)

    def test_invalid_calls_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        for name in ("dense_dim", "sparse_dim"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            bound = getattr(tensor, name)
            if name == "dense_dim":
                inline_positional = lambda: tensor.dense_dim(1)
                inline_multiple = lambda: tensor.dense_dim(1, 2)
                inline_keyword = lambda: tensor.dense_dim(input=tensor)
            else:
                inline_positional = lambda: tensor.sparse_dim(1)
                inline_multiple = lambda: tensor.sparse_dim(1, 2)
                inline_keyword = lambda: tensor.sparse_dim(input=tensor)

            cases = (
                (
                    inline_positional,
                    f"TensorBase.{name}() takes no arguments (1 given)",
                ),
                (
                    lambda bound=bound: bound(1),
                    f"Tensor.{name}() takes no arguments (1 given)",
                ),
                (
                    lambda descriptor=descriptor: descriptor(tensor, 1),
                    f"TensorBase.{name}() takes no arguments (1 given)",
                ),
                (
                    inline_multiple,
                    f"TensorBase.{name}() takes no arguments (2 given)",
                ),
                (
                    inline_keyword,
                    (
                        f"Tensor.{name}() takes no keyword arguments"
                        if sys.version_info < (3, 11)
                        else f"TensorBase.{name}() takes no keyword arguments"
                    ),
                ),
                (
                    lambda bound=bound: bound(unexpected=True),
                    f"Tensor.{name}() takes no keyword arguments",
                ),
                (
                    lambda descriptor=descriptor: descriptor(
                        tensor, unexpected=True
                    ),
                    f"TensorBase.{name}() takes no keyword arguments",
                ),
                (
                    lambda descriptor=descriptor: descriptor(),
                    f"unbound method TensorBase.{name}() needs an argument",
                ),
                (
                    lambda descriptor=descriptor: descriptor(1),
                    f"descriptor '{name}' for 'torch._C.TensorBase' objects "
                    "doesn't apply to a 'int' object",
                ),
                (
                    lambda descriptor=descriptor: descriptor(self=tensor),
                    f"unbound method TensorBase.{name}() needs an argument",
                ),
            )
            for case, (call, message) in enumerate(cases):
                with self.subTest(name=name, case=case):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_receive_descriptors_and_forward(self):
        tensor = torch.tensor([1.0])
        for name, expected in (("dense_dim", 1), ("sparse_dim", 0)):
            with self.subTest(name=name):
                descriptor = inspect.getattr_static(torch.Tensor, name)
                marker = object()

                class RecordingMode(torch.overrides.TorchFunctionMode):
                    def __init__(self):
                        self.calls = []

                    def __torch_function__(
                        self, func, types, args=(), kwargs=None
                    ):
                        self.calls.append((func, types, args, kwargs))
                        return marker

                mode = RecordingMode()
                with mode:
                    result = getattr(tensor, name)()
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

                    def __torch_function__(
                        self, func, types, args=(), kwargs=None
                    ):
                        order.append(self.label)
                        return func(*args, **(kwargs or {}))

                with ForwardingMode("lower"):
                    with ForwardingMode("upper"):
                        forwarded = getattr(tensor, name)()
                self.assertEqual(order, ["upper", "lower"])
                self.assertIs(type(forwarded), int)
                self.assertEqual(forwarded, expected)


if __name__ == "__main__":
    unittest.main()
