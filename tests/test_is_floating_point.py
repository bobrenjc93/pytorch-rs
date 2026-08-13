import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = (
    "\nis_floating_point() -> bool\n\n"
    "Returns True if the data type of :attr:`self` is a floating point data type.\n"
)
FUNCTION_DOC = (
    "\nis_floating_point(input: Tensor) -> bool\n\n"
    "Returns True if the data type of :attr:`input` is a floating point data type i.e.,\n"
    "one of ``torch.float64``, ``torch.float32``, ``torch.float16``, and ``torch.bfloat16``.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> torch.is_floating_point(torch.tensor([1.0, 2.0, 3.0]))\n"
    "    True\n"
    "    >>> torch.is_floating_point(torch.tensor([1, 2, 3], dtype=torch.int32))\n"
    "    False\n"
    "    >>> torch.is_floating_point(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))\n"
    "    True\n"
    "    >>> torch.is_floating_point(torch.tensor([1, 2, 3], dtype=torch.complex64))\n"
    "    False\n"
)


class TensorIsFloatingPointTests(unittest.TestCase):
    def assert_floating_metadata_only(self, tensor):
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
            tensor.is_floating_point(),
            torch.is_floating_point(tensor),
            torch.is_floating_point(input=tensor),
        )
        self.assertEqual(results, (True, True, True))
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

    def test_scalar_empty_strided_and_autograd_tensors_use_dtype_metadata(self):
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
                self.assert_floating_metadata_only(tensor)

    def test_callable_metadata_matches_the_public_builtin_surface(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_floating_point")
        bound = tensor.is_floating_point
        function = torch.is_floating_point

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertIs(type(function), types.BuiltinFunctionType)
        for callable_object in (descriptor, bound, function):
            self.assertTrue(callable(callable_object))
            self.assertEqual(callable_object.__name__, "is_floating_point")
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIs(descriptor(tensor), True)
        self.assertIn("is_floating_point", torch.__all__)

    def test_method_and_top_level_binding_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: tensor.is_floating_point(1),
                "Tensor.is_floating_point() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.is_floating_point(input=tensor),
                "Tensor.is_floating_point() takes no keyword arguments",
            ),
            (
                lambda: torch.is_floating_point(),
                'is_floating_point() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_floating_point(tensor, tensor),
                "is_floating_point() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_floating_point(tensor, input=tensor),
                "is_floating_point() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_floating_point(tensor, extra=True),
                "is_floating_point() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_floating_point(1),
                "is_floating_point(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.is_floating_point(input=[]),
                "is_floating_point(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_floating_point(
                    np.zeros((2, 3), dtype=np.float32)
                ),
                "is_floating_point(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        descriptor = inspect.getattr_static(torch.Tensor, "is_floating_point")
        with self.assertRaises(TypeError):
            descriptor()
        with self.assertRaises(TypeError):
            descriptor(1)


if __name__ == "__main__":
    unittest.main()
