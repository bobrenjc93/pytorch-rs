import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nis_signed() -> bool\n\n"
    "Returns True if the data type of :attr:`self` is a signed data type.\n"
)


class TensorIsSignedTests(unittest.TestCase):
    def test_float32_scalar_empty_strided_and_autograd_tensors_are_signed(self):
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
                results = (
                    tensor.is_signed(),
                    torch.is_signed(tensor),
                    torch.is_signed(input=tensor),
                    torch.is_signed(x=tensor),
                    torch.is_signed(a=tensor),
                )
                self.assertEqual(results, (True, True, True, True, True))
                self.assertTrue(all(type(result) is bool for result in results))
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
        descriptor = inspect.getattr_static(torch.Tensor, "is_signed")
        bound = tensor.is_signed
        function = torch.is_signed

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertIs(type(function), types.BuiltinFunctionType)
        for callable_object in (descriptor, bound, function):
            self.assertTrue(callable(callable_object))
            self.assertEqual(callable_object.__name__, "is_signed")
        for callable_object, python_313_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)),
                    python_313_signature,
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertIsNone(function.__doc__)
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIs(descriptor(tensor), True)
        self.assertIn("is_signed", torch.__all__)

    def test_method_and_top_level_binding_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_signed")
        bound = tensor.is_signed
        cases = (
            (
                lambda: tensor.is_signed(1),
                "TensorBase.is_signed() takes no arguments (1 given)",
            ),
            (
                lambda: bound(1),
                "Tensor.is_signed() takes no arguments (1 given)",
            ),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.is_signed() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_signed(1, 2),
                "TensorBase.is_signed() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.is_signed(input=tensor),
                "TensorBase.is_signed() takes no keyword arguments",
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.is_signed() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.is_signed() takes no keyword arguments",
            ),
            (
                lambda: descriptor(),
                "unbound method TensorBase.is_signed() needs an argument",
            ),
            (
                lambda: descriptor(1),
                "descriptor 'is_signed' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.is_signed() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        top_level_cases = (
            (
                lambda: torch.is_signed(),
                'is_signed() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_signed(tensor, tensor),
                "is_signed() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_signed(tensor, input=tensor),
                "is_signed() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_signed(tensor, extra=True, input=tensor),
                "is_signed() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_signed(tensor, input=tensor, extra=True),
                "is_signed() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_signed(tensor, extra=True),
                "is_signed() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_signed(input=tensor, a=tensor),
                "is_signed() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.is_signed(foo=tensor),
                'is_signed() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_signed(1),
                "is_signed(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.is_signed(input=[]),
                "is_signed(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_signed(a=1),
                "is_signed(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.is_signed(x=[]),
                "is_signed(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_signed(np.zeros((2, 3), dtype=np.float32)),
                "is_signed(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in top_level_cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
