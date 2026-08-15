import inspect
import re
import sys
import types
import typing
import unittest

import numpy as np
import torch_rs as torch
from typing_extensions import TypeIs


FUNCTION_DOC = """Returns True if `obj` is a PyTorch tensor.

    Args:
        obj (object): Object to test
    Example::

        >>> x = torch.tensor([1, 2, 3])
        >>> torch.is_tensor(x)
        True

    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "Returns True if `obj` is a PyTorch tensor.\n\n"
        "Args:\n"
        "    obj (object): Object to test\n"
        "Example::\n\n"
        "    >>> x = torch.tensor([1, 2, 3])\n"
        "    >>> torch.is_tensor(x)\n"
        "    True\n\n"
    )


class ConversionTrap:
    def __init__(self):
        self.calls = []

    def called(self, name):
        self.calls.append(name)
        raise AssertionError(f"is_tensor invoked {name}")

    def __torch_function__(self, *args, **kwargs):
        self.called("__torch_function__")

    def __torch_dispatch__(self, *args, **kwargs):
        self.called("__torch_dispatch__")

    def __array__(self, *args, **kwargs):
        self.called("__array__")

    @property
    def __array_interface__(self):
        self.called("__array_interface__")

    @property
    def __array_struct__(self):
        self.called("__array_struct__")

    @property
    def __cuda_array_interface__(self):
        self.called("__cuda_array_interface__")

    def __dlpack__(self, *args, **kwargs):
        self.called("__dlpack__")

    def __dlpack_device__(self):
        self.called("__dlpack_device__")

    def __iter__(self):
        self.called("__iter__")

    def __len__(self):
        self.called("__len__")

    def __index__(self):
        self.called("__index__")

    def __float__(self):
        self.called("__float__")


class IsTensorTests(unittest.TestCase):
    def test_native_tensor_instances_return_exact_bool_true(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        cases = (
            torch.tensor(3.5),
            torch.tensor([1.0, 2.0]),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
        )
        for case, tensor in enumerate(cases):
            with self.subTest(case=case, shape=tensor.shape):
                result = torch.is_tensor(tensor)
                self.assertIs(type(result), bool)
                self.assertIs(result, True)

    def test_non_tensors_return_false_without_conversion(self):
        conversion_trap = ConversionTrap()
        cases = (
            None,
            True,
            1,
            1.5,
            2.0j,
            "tensor",
            b"tensor",
            [],
            [1.0, 2.0],
            (1.0, 2.0),
            range(3),
            np.float32(1.0),
            np.array(1.0, dtype=np.float32),
            np.arange(4, dtype=np.float32),
            np.dtype(np.float32),
            torch.float32,
            torch.device("cpu"),
            torch.contiguous_format,
            torch.preserve_format,
            inspect.getattr_static(torch.Tensor, "dtype"),
            inspect.getattr_static(torch.Tensor, "device"),
            torch.Tensor,
            conversion_trap,
        )
        for case, value in enumerate(cases):
            with self.subTest(case=case, value_type=type(value).__name__):
                result = torch.is_tensor(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, False)
        self.assertEqual(conversion_trap.calls, [])

    def test_predicate_reads_the_live_public_tensor_binding(self):
        native_tensor_type = torch.Tensor
        tensor = torch.tensor([1.0])
        try:
            torch.Tensor = int
            self.assertIs(torch.is_tensor(1), True)
            self.assertIs(torch.is_tensor(tensor), False)

            torch.Tensor = (int, str)
            self.assertIs(torch.is_tensor("tensor"), True)

            torch.Tensor = 42
            with self.assertRaisesRegex(
                TypeError,
                r"^isinstance\(\) arg 2 must be a type, a tuple of types, or a union$",
            ):
                torch.is_tensor(1)
        finally:
            torch.Tensor = native_tensor_type

        self.assertIs(torch.is_tensor(tensor), True)

    def test_callable_metadata_matches_pytorch_2_13(self):
        function = torch.is_tensor
        return_annotation = TypeIs[typing.ForwardRef("torch.Tensor")]
        expected_signature = inspect.Signature(
            parameters=(
                inspect.Parameter(
                    "obj",
                    inspect.Parameter.POSITIONAL_ONLY,
                    annotation=typing.Any,
                ),
            ),
            return_annotation=return_annotation,
        )

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_tensor")
        self.assertEqual(function.__qualname__, "is_tensor")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            function.__annotations__,
            {"obj": typing.Any, "return": return_annotation},
        )
        self.assertEqual(inspect.get_annotations(function), function.__annotations__)
        resolved_annotations = typing.get_type_hints(function)
        self.assertIs(resolved_annotations["obj"], typing.Any)
        self.assertIs(typing.get_origin(resolved_annotations["return"]), TypeIs)
        self.assertEqual(
            typing.get_args(resolved_annotations["return"]), (torch.Tensor,)
        )
        self.assertEqual(inspect.signature(function), expected_signature)
        expected_signature_text = (
            "(obj: Any, /) -> TypeIs[ForwardRef('torch.Tensor')]"
            if sys.version_info >= (3, 13)
            else "(obj: Any, /) -> typing_extensions.TypeIs[ForwardRef('torch.Tensor')]"
        )
        self.assertEqual(str(inspect.signature(function)), expected_signature_text)
        self.assertIn("is_tensor", torch.__all__)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
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
                lambda: torch.is_tensor(input=tensor),
                "is_tensor() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: torch.is_tensor(extra=tensor),
                "is_tensor() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_tensor(tensor, extra=tensor),
                "is_tensor() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_tensor(obj=tensor, extra=tensor),
                "is_tensor() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
