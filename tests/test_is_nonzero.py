import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = (
    "\nis_nonzero(input) -> (bool)\n\n"
    "Returns True if the :attr:`input` is a single element tensor which is not equal to zero\n"
    "after type conversions.\n"
    "i.e. not equal to ``torch.tensor([0.])`` or ``torch.tensor([0])`` or\n"
    "``torch.tensor([False])``.\n"
    "Throws a ``RuntimeError`` if ``torch.numel() != 1`` (even in case\n"
    "of sparse tensors).\n\n"
    "Args:\n"
    "    input (Tensor): the input tensor.\n\n"
    "Examples::\n\n"
    "    >>> torch.is_nonzero(torch.tensor([0.]))\n"
    "    False\n"
    "    >>> torch.is_nonzero(torch.tensor([1.5]))\n"
    "    True\n"
    "    >>> torch.is_nonzero(torch.tensor([False]))\n"
    "    False\n"
    "    >>> torch.is_nonzero(torch.tensor([3]))\n"
    "    True\n"
    "    >>> torch.is_nonzero(torch.tensor([1, 3, 5]))\n"
    "    Traceback (most recent call last):\n"
    "    ...\n"
    "    RuntimeError: Boolean value of Tensor with more than one value is ambiguous\n"
    "    >>> torch.is_nonzero(torch.tensor([]))\n"
    "    Traceback (most recent call last):\n"
    "    ...\n"
    "    RuntimeError: Boolean value of Tensor with no values is ambiguous\n"
)


class TorchIsNonzeroTests(unittest.TestCase):
    def assert_is_nonzero(self, tensor, expected):
        results = (
            torch.is_nonzero(tensor),
            torch.is_nonzero(input=tensor),
            torch.is_nonzero(a=tensor),
            torch.is_nonzero(x=tensor),
        )
        self.assertEqual(results, (expected,) * len(results))
        self.assertTrue(all(type(result) is bool for result in results))

    def test_scalar_signed_zero_and_non_finite_values(self):
        cases = (
            (0.0, False),
            (-0.0, False),
            (1.0, True),
            (-2.5, True),
            (float("nan"), True),
            (float("inf"), True),
            (-float("inf"), True),
        )
        for value, expected in cases:
            tensor = torch.tensor(value)
            with self.subTest(value=value):
                self.assertEqual(tensor.shape, ())
                self.assert_is_nonzero(tensor, expected)

    def test_one_element_offset_views_use_their_strided_value(self):
        values = [0.0, -0.0, 3.0, -4.0, float("nan"), float("inf"), -float("inf")]
        expected = [False, False, True, True, True, True, True]
        source = torch.tensor([values]).transpose(0, 1)

        for index, truth in enumerate(expected):
            tensor = source[index]
            with self.subTest(index=index, value=values[index]):
                self.assertEqual(tensor.shape, (1,))
                self.assertEqual(tensor.stride(), (len(values),))
                self.assertEqual(tensor.storage_offset(), index)
                self.assert_is_nonzero(tensor, truth)

    def test_empty_and_multi_element_tensors_are_ambiguous(self):
        cases = (
            (
                torch.zeros((0,)),
                "Boolean value of Tensor with no values is ambiguous",
            ),
            (
                torch.zeros((2, 0, 3)).transpose(0, 2),
                "Boolean value of Tensor with no values is ambiguous",
            ),
            (
                torch.tensor([0.0, 0.0]),
                "Boolean value of Tensor with more than one value is ambiguous",
            ),
            (
                torch.tensor([[0.0, 0.0], [0.0, 0.0]]).transpose(0, 1),
                "Boolean value of Tensor with more than one value is ambiguous",
            ),
        )
        for tensor, message in cases:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                for call in (
                    lambda tensor=tensor: torch.is_nonzero(tensor),
                    lambda tensor=tensor: torch.is_nonzero(input=tensor),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, f"^{re.escape(message)}$"
                    ):
                        call()

    def test_callable_metadata_matches_the_public_builtin_surface(self):
        function = torch.is_nonzero

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "is_nonzero")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__text_signature__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIn("is_nonzero", torch.__all__)

    def test_binding_and_tensor_type_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.is_nonzero(),
                'is_nonzero() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_nonzero(tensor, tensor),
                "is_nonzero() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_nonzero(tensor, input=tensor),
                "is_nonzero() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_nonzero(tensor, extra=True, input=tensor),
                "is_nonzero() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_nonzero(tensor, input=tensor, extra=True),
                "is_nonzero() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.is_nonzero(extra=tensor),
                'is_nonzero() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.is_nonzero(1, extra=True),
                "is_nonzero(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.is_nonzero(input=[]),
                "is_nonzero(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_nonzero(a=1),
                "is_nonzero(): argument 'input' must be Tensor, not int",
            ),
            (
                lambda: torch.is_nonzero(x=[]),
                "is_nonzero(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.is_nonzero(np.zeros((2, 3), dtype=np.float32)),
                "is_nonzero(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
