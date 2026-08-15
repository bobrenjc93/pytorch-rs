import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
from tests.signature_utils import assert_no_argument_signature


METHOD_DOC = (
    "\nis_complex() -> bool\n\n"
    "Returns True if the data type of :attr:`self` is a complex data type.\n"
)
FUNCTION_DOC = (
    "\nis_complex(input: Tensor) -> bool\n\n"
    "Returns True if the data type of :attr:`input` is a complex data type i.e.,\n"
    "one of ``torch.complex64``, and ``torch.complex128``.\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Example::\n\n"
    "    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.complex64))\n"
    "    True\n"
    "    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.complex128))\n"
    "    True\n"
    "    >>> torch.is_complex(torch.tensor([1, 2, 3], dtype=torch.int32))\n"
    "    False\n"
    "    >>> torch.is_complex(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16))\n"
    "    False\n"
)


class TensorIsComplexTests(unittest.TestCase):
    def assert_complex_metadata_only(self, tensor):
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
            tensor.is_complex(),
            torch.is_complex(tensor),
            torch.is_complex(input=tensor),
            torch.is_complex(a=tensor),
            torch.is_complex(x=tensor),
        )
        self.assertEqual(results, (False, False, False, False, False))
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
                self.assert_complex_metadata_only(tensor)

    def test_callable_metadata_matches_the_public_builtin_surface(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "is_complex")
        bound = tensor.is_complex
        function = torch.is_complex

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertIs(type(function), types.BuiltinFunctionType)
        for callable_object in (descriptor, bound, function):
            self.assertTrue(callable(callable_object))
            self.assertEqual(callable_object.__name__, "is_complex")
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        self.assertEqual(descriptor.__doc__, METHOD_DOC)
        self.assertEqual(bound.__doc__, METHOD_DOC)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertIs(descriptor(tensor), False)
        self.assertIn("is_complex", torch.__all__)

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

        top_level_cases = (
            (
                lambda: torch.is_complex(),
                'is_complex() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_complex(tensor, tensor),
                "is_complex() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_complex(tensor, input=tensor),
                "is_complex() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_complex(tensor, extra=True, input=tensor),
                "is_complex() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_complex(tensor, input=tensor, extra=True),
                "is_complex() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_complex(tensor, extra=True),
                "is_complex() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_complex(1),
                "is_complex(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.is_complex(input=[]),
                "is_complex(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_complex(a=1),
                "is_complex(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.is_complex(x=[]),
                "is_complex(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_complex(np.zeros((2, 3), dtype=np.float32)),
                "is_complex(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in top_level_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
