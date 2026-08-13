import inspect
import sys
import types
import unittest

import torch_rs as torch


METHOD_DOC = (
    "\nis_complex() -> bool\n\n"
    "Returns True if the data type of :attr:`self` is a complex data type.\n"
)


class TensorIsComplexTests(unittest.TestCase):
    def test_float32_scalar_empty_strided_and_autograd_tensors_are_not_complex(self):
        leaf = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()

        offset_view = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
            ]
        ).transpose(0, 1)[1]
        self.assertGreater(offset_view.storage_offset(), 0)

        extreme_empty = (
            torch.zeros((0,))
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        cases = (
            ("scalar", torch.tensor(3.5)),
            ("empty", torch.zeros((2, 0, 3))),
            ("offset strided view", offset_view),
            ("extreme empty view", extreme_empty),
            ("autograd leaf", leaf),
            ("autograd non-leaf view", tracked),
            ("accumulated gradient", leaf.grad),
        )
        for case, tensor in cases:
            with self.subTest(case=case, shape=tensor.shape, stride=tensor.stride()):
                metadata = (
                    tensor.shape,
                    tensor.stride(),
                    tensor.storage_offset(),
                    tensor.dtype,
                    tensor.device,
                    tensor.requires_grad,
                    tensor.is_leaf,
                )
                result = tensor.is_complex()
                self.assertIs(result, False)
                self.assertEqual(
                    (
                        tensor.shape,
                        tensor.stride(),
                        tensor.storage_offset(),
                        tensor.dtype,
                        tensor.device,
                        tensor.requires_grad,
                        tensor.is_leaf,
                    ),
                    metadata,
                )

    def test_callable_metadata_matches_the_public_builtin_surface(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_complex")
        bound = tensor.is_complex

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertTrue(callable(callable_object))
            self.assertEqual(callable_object.__name__, "is_complex")
            self.assertIsNone(callable_object.__text_signature__)
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIs(descriptor(tensor), False)

    def test_positional_and_keyword_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_complex")
        bound = tensor.is_complex
        cases = (
            (
                lambda: tensor.is_complex(1),
                "TensorBase.is_complex() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.is_complex() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.is_complex() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_complex(1, 2),
                "TensorBase.is_complex() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_complex(input=tensor),
                "TensorBase.is_complex() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.is_complex() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.is_complex() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        for call in (lambda: descriptor(), lambda: descriptor(1)):
            with self.assertRaises(TypeError):
                call()


if __name__ == "__main__":
    unittest.main()
