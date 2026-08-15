import inspect
import re
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\nfloat(memory_format=torch.preserve_format) -> Tensor\n\n"
    "``self.float()`` is equivalent to ``self.to(torch.float32)``. See :func:`to`.\n\n"
    "Args:\n"
    "    memory_format (:class:`torch.memory_format`, optional): the desired memory format of\n"
    "        returned Tensor. Default: ``torch.preserve_format``.\n"
)


class TensorFloatMethodTests(unittest.TestCase):
    def test_float32_identity_requests_return_exact_receiver(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        leaf.sum().backward()
        cases = (
            torch.tensor(-0.0),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ]
            ).transpose(0, 1)[1],
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
            leaf,
            tracked,
            leaf.grad,
        )

        for case, tensor in enumerate(cases):
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                gradient = tensor.grad
                results = (
                    tensor.float(),
                    tensor.float(memory_format=None),
                    tensor.float(memory_format=torch.preserve_format),
                    tensor.float(memory_format=torch.contiguous_format),
                )
                for result in results:
                    self.assertIs(result, tensor)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.data_ptr(),
                        tensor.dtype,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )
                self.assertIs(tensor.grad, gradient)

        self.assertFalse(cases[2].is_contiguous())
        self.assertIs(
            cases[2].float(memory_format=torch.contiguous_format), cases[2]
        )

    def test_channels_last_requests_materialize_and_preserve_autograd(self):
        cases = (
            (
                (2, 3, 2, 4),
                (0, 3),
                torch.channels_last,
                (12, 1, 6, 3),
            ),
            (
                (2, 3, 2, 4, 5),
                (0, 4),
                torch.channels_last_3d,
                (48, 1, 24, 6, 3),
            ),
        )
        for shape, dimensions, memory_format, expected_stride in cases:
            with self.subTest(memory_format=memory_format):
                leaf = torch.ones(shape, requires_grad=True)
                source = (leaf * 3.0).transpose(*dimensions)
                result = source.float(memory_format=memory_format)

                self.assertIsNot(result, source)
                self.assertIs(result.dtype, torch.float32)
                self.assertEqual(result.tolist(), source.tolist())
                self.assertEqual(result.shape, source.shape)
                self.assertEqual(result.stride(), expected_stride)
                self.assertEqual(result.storage_offset(), 0)
                self.assertNotEqual(result.data_ptr(), source.data_ptr())
                self.assertTrue(
                    result.is_contiguous(memory_format=memory_format)
                )
                self.assertTrue(result.requires_grad)
                self.assertFalse(result.is_leaf)
                self.assertIs(
                    result.float(memory_format=memory_format), result
                )
                self.assertIs(
                    result.float(memory_format=torch.preserve_format), result
                )

                row_major = result.float(
                    memory_format=torch.contiguous_format
                )
                self.assertIsNot(row_major, result)
                self.assertTrue(row_major.is_contiguous())
                self.assertEqual(row_major.tolist(), result.tolist())
                self.assertIs(
                    row_major.float(memory_format=torch.contiguous_format),
                    row_major,
                )

                result.sum().backward()
                self.assertEqual(
                    leaf.grad.tolist(), torch.full(shape, 3.0).tolist()
                )

        singleton = torch.zeros((2, 1, 4, 5))
        self.assertTrue(
            singleton.is_contiguous(memory_format=torch.channels_last)
        )
        singleton_result = singleton.float(
            memory_format=torch.channels_last
        )
        self.assertIsNot(singleton_result, singleton)
        self.assertEqual(singleton_result.stride(), (20, 1, 5, 1))
        self.assertIs(
            singleton_result.float(memory_format=torch.channels_last),
            singleton_result,
        )

        empty = torch.zeros((0, 1, 4, 5))
        first_empty_result = empty.float(memory_format=torch.channels_last)
        second_empty_result = first_empty_result.float(
            memory_format=torch.channels_last
        )
        self.assertIsNot(first_empty_result, empty)
        self.assertIsNot(second_empty_result, first_empty_result)
        self.assertEqual(first_empty_result.stride(), (20, 1, 5, 1))
        self.assertEqual(second_empty_result.stride(), (20, 1, 5, 1))

    def test_channels_last_respects_no_grad(self):
        leaf = torch.ones((2, 3, 4, 5), requires_grad=True)
        with torch.no_grad():
            result = leaf.transpose(0, 3).float(
                memory_format=torch.channels_last
            )
        self.assertFalse(result.requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertTrue(leaf.requires_grad)
        self.assertIsNone(leaf.grad)

    def test_rank_binding_and_memory_format_errors(self):
        tensor = torch.zeros((2, 3))
        cases = (
            (
                lambda: tensor.float(torch.preserve_format),
                "float() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: tensor.float(
                    torch.preserve_format, torch.contiguous_format
                ),
                "float() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: tensor.float(memory_format=1),
                "float(): argument 'memory_format' must be torch.memory_format, not int",
            ),
            (
                lambda: tensor.float(unexpected=True),
                "float() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.float(
                    **{"unexpected": True, "memory_format": 1}
                ),
                "float(): argument 'memory_format' must be torch.memory_format, not int",
            ),
            (
                lambda: tensor.float(memory_format=torch.channels_last),
                "required rank 4 tensor to use channels_last format",
            ),
            (
                lambda: tensor.float(memory_format=torch.channels_last_3d),
                "required rank 5 tensor to use channels_last_3d format",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(
                    (TypeError, RuntimeError), f"^{re.escape(message)}$"
                ):
                    call()

        extreme = torch.zeros((3, 0, 1, sys.maxsize)).transpose(0, 1)
        for memory_format in (torch.channels_last, torch.channels_last_3d):
            with self.subTest(extreme_memory_format=memory_format):
                with self.assertRaisesRegex(
                    RuntimeError, "^Stride calculation overflowed$"
                ):
                    extreme.float(memory_format=memory_format)

    def test_tensorbase_descriptor_metadata_documentation_and_unbound_calls(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "float")
        bound = tensor.float

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor),
            "<method 'float' of 'torch._C.TensorBase' objects>",
        )
        self.assertEqual(descriptor.__qualname__, "TensorBase.float")
        self.assertEqual(bound.__qualname__, "Tensor.float")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "float")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertIs(descriptor(tensor), tensor)
        self.assertIs(
            descriptor(tensor, memory_format=torch.contiguous_format), tensor
        )

        cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.float() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'float' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
