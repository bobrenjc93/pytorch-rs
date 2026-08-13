import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

ELEMENT_SIZE_DOC = """
element_size() -> int

Returns the size in bytes of an individual element.

Example::

    >>> torch.tensor([]).element_size()
    4
    >>> torch.tensor([], dtype=torch.uint8).element_size()
    1

"""


class TensorIntrospectionTests(unittest.TestCase):
    def assert_introspection(self, tensor, *, rank, elements):
        rank_values = (
            tensor.ndim,
            tensor.dim(),
            tensor.ndimension(),
        )
        element_values = (
            tensor.nelement(),
            tensor.numel(),
            torch.numel(tensor),
            torch.numel(input=tensor),
        )
        for value in (
            *rank_values,
            *element_values,
            tensor.element_size(),
            tensor.nbytes,
        ):
            self.assertIs(type(value), int)
        self.assertEqual(rank_values, (rank,) * len(rank_values))
        self.assertEqual(element_values, (elements,) * len(element_values))
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.element_size(), 4)
        self.assertEqual(tensor.nbytes, elements * tensor.element_size())

    def test_scalar_empty_and_metadata_only_views(self):
        cases = (
            (torch.tensor(3.5), 0, 1),
            (torch.zeros((0,)), 1, 0),
            (torch.zeros((2, 3, 4)), 3, 24),
            (torch.zeros((2, 3, 4)).transpose(0, 2), 3, 24),
            (
                torch.tensor(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                        [8.0, 9.0, 10.0, 11.0],
                    ]
                ).transpose(0, 1)[1],
                1,
                3,
            ),
            (torch.zeros((1,) * 65), 65, 1),
            (torch.zeros((2, 0, 3)).transpose(0, 2), 3, 0),
            (torch.zeros((4, 2, 0, 3)).transpose(0, 3)[1], 3, 0),
            (torch.zeros((1, 2, 1, 0)).squeeze((0, 2)), 2, 0),
            (
                torch.zeros((0,))
                .reshape((2, 0, sys.maxsize))
                .transpose(0, 2),
                3,
                0,
            ),
            (torch.zeros((sys.maxsize, 0))[sys.maxsize - 1], 1, 0),
        )
        for tensor, rank, elements in cases:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assert_introspection(tensor, rank=rank, elements=elements)

    def test_descriptor_kinds_signatures_and_unbound_calls(self):
        tensor = torch.zeros((2, 0, 3))
        ndim = inspect.getattr_static(torch.Tensor, "ndim")
        self.assertIs(type(ndim), types.GetSetDescriptorType)
        self.assertFalse(callable(ndim))
        self.assertEqual(ndim.__name__, "ndim")
        self.assertEqual(ndim.__get__(tensor, torch.Tensor), 3)
        self.assertIs(ndim.__get__(None, torch.Tensor), ndim)

        nbytes = inspect.getattr_static(torch.Tensor, "nbytes")
        self.assertIs(type(nbytes), types.GetSetDescriptorType)
        self.assertFalse(callable(nbytes))
        self.assertEqual(nbytes.__name__, "nbytes")
        self.assertEqual(nbytes.__get__(tensor, torch.Tensor), 0)
        self.assertIs(nbytes.__get__(None, torch.Tensor), nbytes)

        for name, expected in (
            ("dim", 3),
            ("ndimension", 3),
            ("nelement", 0),
            ("numel", 0),
            ("element_size", 4),
        ):
            with self.subTest(name=name):
                descriptor = inspect.getattr_static(torch.Tensor, name)
                bound = getattr(tensor, name)
                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(descriptor.__name__, name)
                self.assertIsNone(descriptor.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)
                with self.assertRaises(ValueError):
                    inspect.signature(bound)
                self.assertEqual(descriptor(tensor), expected)
                if name == "element_size":
                    self.assertEqual(descriptor.__doc__, ELEMENT_SIZE_DOC)
                    self.assertEqual(bound.__doc__, ELEMENT_SIZE_DOC)
                    self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
                    self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
                    self.assertEqual(bound(), 4)

        self.assertIs(type(torch.numel), types.BuiltinFunctionType)
        self.assertIsNone(torch.numel.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(torch.numel)

    def test_metadata_properties_are_read_only_and_methods_reject_arguments(self):
        tensor = torch.zeros((2, 3))
        for name, value in (("ndim", 9), ("nbytes", 99)):
            with self.subTest(name=name, action="set"):
                with self.assertRaisesRegex(
                    AttributeError, rf"attribute '{name}'.*not writable"
                ):
                    setattr(tensor, name, value)
            with self.subTest(name=name, action="delete"):
                with self.assertRaisesRegex(
                    AttributeError, rf"attribute '{name}'.*not writable"
                ):
                    delattr(tensor, name)

        for name in ("dim", "ndimension", "nelement", "numel"):
            with self.subTest(name=name, kind="positional"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^Tensor\.{name}\(\) takes no arguments \(1 given\)$",
                ):
                    getattr(tensor, name)(1)
            with self.subTest(name=name, kind="keyword"):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^Tensor\.{name}\(\) takes no keyword arguments$",
                ):
                    getattr(tensor, name)(input=tensor)
            with self.subTest(name=name, kind="unbound"):
                with self.assertRaises(TypeError):
                    getattr(torch.Tensor, name)()
                with self.assertRaises(TypeError):
                    getattr(torch.Tensor, name)(1)

        descriptor = inspect.getattr_static(torch.Tensor, "element_size")
        bound = tensor.element_size
        invalid_element_size_calls = (
            (
                "inline positional",
                lambda: tensor.element_size(1),
                "TensorBase.element_size() takes no arguments (1 given)",
            ),
            (
                "stored positional",
                lambda: bound(1),
                "Tensor.element_size() takes no arguments (1 given)",
            ),
            (
                "descriptor positional",
                lambda: descriptor(tensor, 1),
                "TensorBase.element_size() takes no arguments (1 given)",
            ),
            (
                "inline keyword",
                lambda: tensor.element_size(unexpected=True),
                "TensorBase.element_size() takes no keyword arguments",
            ),
            (
                "stored keyword",
                lambda: bound(unexpected=True),
                "Tensor.element_size() takes no keyword arguments",
            ),
            (
                "descriptor keyword",
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.element_size() takes no keyword arguments",
            ),
        )
        for case, call, message in invalid_element_size_calls:
            with self.subTest(method="element_size", case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                actual_message = str(raised.exception)
                if case == "inline keyword":
                    # CPython 3.10 materializes this keyword call as a bound
                    # method, while newer runtimes retain the descriptor fast
                    # path. The reference suite checks the exact message
                    # against PyTorch for the active runtime.
                    self.assertIn(
                        actual_message,
                        {
                            message,
                            "Tensor.element_size() takes no keyword arguments",
                        },
                    )
                else:
                    self.assertEqual(actual_message, message)

        for call in (lambda: descriptor(), lambda: descriptor(1)):
            with self.assertRaises(TypeError):
                call()

    def test_top_level_numel_binding_errors_match_pytorch(self):
        tensor = torch.zeros((2, 3))
        cases = (
            (
                lambda: torch.numel(),
                'numel() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.numel(tensor, tensor),
                "numel() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.numel(tensor, input=tensor),
                "numel() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.numel(tensor, extra=True),
                "numel() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.numel(1),
                "numel(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.numel(input=[]),
                "numel(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.numel(np.zeros((2, 3), dtype=np.float32)),
                "numel(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
