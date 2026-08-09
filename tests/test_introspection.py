import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorIntrospectionTests(unittest.TestCase):
    def assert_introspection(self, tensor, *, rank, elements):
        calls = (
            tensor.ndim,
            tensor.dim(),
            tensor.ndimension(),
            tensor.nelement(),
            tensor.numel(),
            torch.numel(tensor),
            torch.numel(input=tensor),
        )
        for value in calls:
            self.assertIs(type(value), int)
        self.assertEqual(calls[:3], (rank,) * 3)
        self.assertEqual(calls[3:], (elements,) * 4)

    def test_scalar_empty_and_metadata_only_views(self):
        cases = (
            (torch.tensor(3.5), 0, 1),
            (torch.zeros((0,)), 1, 0),
            (torch.zeros((2, 3, 4)), 3, 24),
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
        )
        for tensor, rank, elements in cases:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                self.assert_introspection(tensor, rank=rank, elements=elements)

    def test_dtype_introspection_uses_tensor_metadata_for_all_layouts(self):
        tensors = (
            torch.tensor(3.5),
            torch.zeros((0,)),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2),
            torch.zeros((4, 3, 2)).transpose(0, 2)[1],
        )
        for tensor in tensors:
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                floating_values = (
                    tensor.is_floating_point(),
                    torch.is_floating_point(tensor),
                    torch.is_floating_point(input=tensor),
                )
                complex_values = (
                    tensor.is_complex(),
                    torch.is_complex(tensor),
                    torch.is_complex(input=tensor),
                )
                self.assertEqual(floating_values, (True,) * 3)
                self.assertEqual(complex_values, (False,) * 3)
                self.assertTrue(all(type(value) is bool for value in floating_values))
                self.assertTrue(all(type(value) is bool for value in complex_values))
                self.assertIs(type(tensor.element_size()), int)
                self.assertEqual(tensor.element_size(), 4)

    def test_is_tensor_recognizes_tensors_and_rejects_arbitrary_objects(self):
        tensors = (
            torch.tensor(1.0),
            torch.zeros((0,)),
            torch.zeros((2, 3)).transpose(0, 1),
        )
        for value in tensors:
            with self.subTest(kind="tensor", shape=value.shape):
                self.assertIs(torch.is_tensor(value), True)

        non_tensors = (
            None,
            object(),
            1,
            1.5,
            True,
            [],
            {"value": 1},
            np.zeros((2, 3), dtype=np.float32),
            torch.Tensor,
            torch.float32,
            torch.device("cpu"),
        )
        for value in non_tensors:
            with self.subTest(kind="non-tensor", type=type(value).__name__):
                self.assertIs(torch.is_tensor(value), False)

    def test_descriptor_kinds_signatures_and_unbound_calls(self):
        tensor = torch.zeros((2, 0, 3))
        ndim = inspect.getattr_static(torch.Tensor, "ndim")
        self.assertIs(type(ndim), types.GetSetDescriptorType)
        self.assertFalse(callable(ndim))
        self.assertEqual(ndim.__name__, "ndim")
        self.assertEqual(ndim.__get__(tensor, torch.Tensor), 3)
        self.assertIs(ndim.__get__(None, torch.Tensor), ndim)

        for name, expected in (
            ("dim", 3),
            ("ndimension", 3),
            ("nelement", 0),
            ("numel", 0),
            ("is_floating_point", True),
            ("is_complex", False),
            ("element_size", 4),
        ):
            with self.subTest(name=name):
                descriptor = inspect.getattr_static(torch.Tensor, name)
                bound = getattr(tensor, name)
                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(descriptor.__text_signature__, "($self, /)")
                self.assertEqual(str(inspect.signature(descriptor)), "(self, /)")
                self.assertEqual(str(inspect.signature(bound)), "()")
                self.assertEqual(descriptor(tensor), expected)

        self.assertIs(type(torch.numel), types.BuiltinFunctionType)
        self.assertIsNone(torch.numel.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(torch.numel)

        self.assertIs(type(torch.is_tensor), types.BuiltinFunctionType)
        self.assertEqual(torch.is_tensor.__text_signature__, "(obj, /)")
        self.assertEqual(str(inspect.signature(torch.is_tensor)), "(obj, /)")

        for function in (torch.is_floating_point, torch.is_complex):
            self.assertIs(type(function), types.BuiltinFunctionType)
            self.assertIsNone(function.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def test_ndim_is_read_only_and_methods_reject_arguments(self):
        tensor = torch.zeros((2, 3))
        with self.assertRaisesRegex(
            AttributeError, r"attribute 'ndim'.*not writable"
        ):
            tensor.ndim = 9
        with self.assertRaisesRegex(
            AttributeError, r"attribute 'ndim'.*not writable"
        ):
            del tensor.ndim

        for name in (
            "dim",
            "ndimension",
            "nelement",
            "numel",
            "is_floating_point",
            "is_complex",
            "element_size",
        ):
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

    def test_is_tensor_positional_only_binding_errors(self):
        tensor = torch.tensor(1.0)
        cases = (
            (
                lambda: torch.is_tensor(),
                "is_tensor() missing 1 required positional argument: 'obj'",
            ),
            (
                lambda: torch.is_tensor(tensor, tensor),
                "is_tensor() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_tensor(obj=tensor),
                "is_tensor() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.is_tensor(tensor, obj=tensor),
                "is_tensor() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.is_tensor(extra=tensor),
                "is_tensor() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_top_level_dtype_predicate_binding_errors(self):
        tensor = torch.tensor(1.0)
        for name in ("is_floating_point", "is_complex"):
            function = getattr(torch, name)
            cases = (
                (
                    lambda function=function: function(),
                    f'{name}() missing 1 required positional arguments: "input"',
                ),
                (
                    lambda function=function: function(tensor, tensor),
                    f"{name}() takes 1 positional argument but 2 were given",
                ),
                (
                    lambda function=function: function(tensor, input=tensor),
                    f"{name}() got multiple values for argument 'input'",
                ),
                (
                    lambda function=function: function(tensor, extra=True),
                    f"{name}() got an unexpected keyword argument 'extra'",
                ),
                (
                    lambda function=function: function(1),
                    f"{name}(): argument 'input' (position 1) must be Tensor, not int",
                ),
                (
                    lambda function=function: function(input=[]),
                    f"{name}(): argument 'input' must be Tensor, not list",
                ),
                (
                    lambda function=function: function(
                        np.zeros((2, 3), dtype=np.float32)
                    ),
                    f"{name}(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
                ),
            )
            for call, message in cases:
                with self.subTest(name=name, message=message):
                    with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
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
