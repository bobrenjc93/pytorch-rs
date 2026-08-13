import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\ntype_as(tensor) -> Tensor\n\n"
    "Returns this tensor cast to the type of the given tensor.\n\n"
    "This is a no-op if the tensor is already of the correct type. This is\n"
    "equivalent to ``self.type(tensor.type())``\n\n"
    "Args:\n"
    "    tensor (Tensor): the tensor which has the desired type\n"
)


class TensorTypeAsTests(unittest.TestCase):
    def assert_identity_call(self, source, other):
        metadata = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.dtype,
            source.device,
            source.requires_grad,
            source.is_leaf,
            source.grad,
        )
        detached = source.detach()

        positional = source.type_as(other)
        keyword = source.type_as(other=other)

        self.assertIs(positional, source)
        self.assertIs(keyword, source)
        self.assertTrue(source.is_set_to(detached))
        self.assertEqual(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.dtype,
                source.device,
                source.requires_grad,
                source.is_leaf,
                source.grad,
            ),
            metadata,
        )

    def test_scalar_empty_and_strided_tensors_return_the_exact_receiver(self):
        scalar = torch.tensor(-0.0)
        empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1]
        strided = torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        ).transpose(0, 1)[1]
        other = torch.tensor([8.0], requires_grad=True)

        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(strided.stride(), (3,))
        self.assertGreater(strided.storage_offset(), 0)
        for case, source in enumerate((scalar, empty, strided)):
            with self.subTest(case=case):
                self.assert_identity_call(source, other)

    def test_autograd_tensor_keeps_the_same_graph(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = (leaf * 3.0).transpose(0, 1)[1]
        other = torch.zeros((2,), requires_grad=True)
        graph_before = (
            source.requires_grad,
            source.is_leaf,
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad,
            other.grad,
        )

        result = source.type_as(other=other)

        self.assertIs(result, source)
        self.assertEqual(
            (
                source.requires_grad,
                source.is_leaf,
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                other.grad,
            ),
            graph_before,
        )
        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[0.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        self.assertIsNone(other.grad)

    def test_descriptor_documentation_and_unbound_call(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type_as")
        bound = tensor.type_as

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "type_as")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor, other), tensor)
        self.assertIs(descriptor(tensor, other=other), tensor)

    def test_binding_and_tensor_type_error_precedence(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "type_as")
        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.type_as() needs an argument",
            ),
            (
                lambda: descriptor(1, other),
                "descriptor 'type_as' for 'torch._C.TensorBase' objects doesn't apply to a 'int' object",
            ),
            (
                lambda: tensor.type_as(),
                'type_as() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.type_as(other, other),
                "type_as() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.type_as(other, other=other),
                "type_as() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.type_as(foo=other),
                'type_as() missing 1 required positional arguments: "other"',
            ),
            (
                lambda: tensor.type_as(other, extra=True),
                "type_as() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: tensor.type_as(1),
                "type_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.type_as(None),
                "type_as(): argument 'other' (position 1) must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.type_as([]),
                "type_as(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.type_as(
                    np.zeros((2, 3), dtype=np.float32)
                ),
                "type_as(): argument 'other' (position 1) must be Tensor, not numpy.ndarray",
            ),
            (
                lambda: tensor.type_as(other=1),
                "type_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.type_as(other=None),
                "type_as(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.type_as(other=[]),
                "type_as(): argument 'other' must be Tensor, not list",
            ),
            (
                lambda: tensor.type_as(**{"other": 1, "extra": True}),
                "type_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.type_as(**{"extra": True, "other": 1}),
                "type_as(): argument 'other' must be Tensor, not int",
            ),
            (
                lambda: tensor.type_as(1, other=other),
                "type_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
            (
                lambda: tensor.type_as(1, extra=True),
                "type_as(): argument 'other' (position 1) must be Tensor, not int",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
