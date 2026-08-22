import copy
import inspect
import pickle
import re
import sys
import types
import typing
import unittest

import numpy as np
import torch_rs as torch


FUNCTION_DOC = """Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    Example::

        >>> import torch
        >>> # UntypedStorage (recommended)
        >>> tensor = torch.tensor([1, 2, 3])
        >>> storage = tensor.untyped_storage()
        >>> torch.is_storage(storage)
        True
        >>>
        >>> # TypedStorage (legacy)
        >>> typed_storage = torch.TypedStorage(5, dtype=torch.float32)
        >>> torch.is_storage(typed_storage)
        True
        >>>
        >>> # regular tensor (should return False)
        >>> torch.is_storage(tensor)
        False
        >>>
        >>> # non-storage object
        >>> torch.is_storage([1, 2, 3])
        False
    """

if sys.version_info >= (3, 13):
    FUNCTION_DOC = (
        "Returns True if `obj` is a PyTorch storage object.\n\n"
        "Args:\n"
        "    obj (Object): Object to test\n"
        "Example::\n\n"
        "    >>> import torch\n"
        "    >>> # UntypedStorage (recommended)\n"
        "    >>> tensor = torch.tensor([1, 2, 3])\n"
        "    >>> storage = tensor.untyped_storage()\n"
        "    >>> torch.is_storage(storage)\n"
        "    True\n"
        "    >>>\n"
        "    >>> # TypedStorage (legacy)\n"
        "    >>> typed_storage = torch.TypedStorage(5, dtype=torch.float32)\n"
        "    >>> torch.is_storage(typed_storage)\n"
        "    True\n"
        "    >>>\n"
        "    >>> # regular tensor (should return False)\n"
        "    >>> torch.is_storage(tensor)\n"
        "    False\n"
        "    >>>\n"
        "    >>> # non-storage object\n"
        "    >>> torch.is_storage([1, 2, 3])\n"
        "    False\n"
    )


class ConversionTrap:
    def __init__(self):
        self.calls = []

    def called(self, name):
        self.calls.append(name)
        raise AssertionError(f"is_storage invoked {name}")

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

    def __int__(self):
        self.called("__int__")

    def __float__(self):
        self.called("__float__")

    def __complex__(self):
        self.called("__complex__")

    def __bool__(self):
        self.called("__bool__")


class AttributeTrap:
    def __getattribute__(self, name):
        raise AssertionError(f"is_storage read attribute {name}")


class IsStorageTests(unittest.TestCase):
    def test_every_reachable_kind_returns_exact_false_without_hooks(self):
        conversion_trap = ConversionTrap()
        attribute_trap = AttributeTrap()
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        cases = (
            None,
            True,
            1,
            1.5,
            2.0j,
            "storage",
            b"storage",
            [],
            [1.0, 2.0],
            (1.0, 2.0),
            range(3),
            {},
            {1.0, 2.0},
            object(),
            np.float32(1.0),
            np.array(1.0, dtype=np.float32),
            np.arange(4, dtype=np.float32),
            np.dtype(np.float32),
            torch.tensor(3.5),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
            torch.float32,
            torch.device("cpu"),
            torch.strided,
            torch.contiguous_format,
            torch.preserve_format,
            inspect.getattr_static(torch.Tensor, "dtype"),
            inspect.getattr_static(torch.Tensor, "device"),
            torch.Tensor,
            torch.tensor,
            torch,
            conversion_trap,
            attribute_trap,
            *vars(torch).values(),
        )
        for case, value in enumerate(cases):
            with self.subTest(case=case, value_type=type(value).__name__):
                result = torch.is_storage(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, False)
        self.assertEqual(conversion_trap.calls, [])

    def test_callable_metadata_matches_pytorch_2_13(self):
        function = torch.is_storage
        return_annotation = typing.TypeGuard[
            typing.ForwardRef("TypedStorage | UntypedStorage")
        ]
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
        self.assertEqual(function.__name__, "is_storage")
        self.assertEqual(function.__qualname__, "is_storage")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            function.__annotations__,
            {"obj": typing.Any, "return": return_annotation},
        )
        self.assertEqual(inspect.get_annotations(function), function.__annotations__)
        self.assertIs(
            typing.get_origin(function.__annotations__["return"]),
            typing.TypeGuard,
        )
        self.assertEqual(
            typing.get_args(function.__annotations__["return"]),
            (typing.ForwardRef("TypedStorage | UntypedStorage"),),
        )
        self.assertEqual(inspect.signature(function), expected_signature)
        self.assertEqual(
            str(inspect.signature(function)),
            "(obj: Any, /) -> TypeGuard[ForwardRef('TypedStorage | UntypedStorage')]",
        )

    def test_exports_copying_and_pickling(self):
        function = torch.is_storage
        self.assertEqual(torch.__all__.count("is_storage"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_storage"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_storage_object_surfaces_remain_unsupported(self):
        for name in (
            "storage",
            "Storage",
            "TypedStorage",
            "UntypedStorage",
            "FloatStorage",
        ):
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

        for name in ("storage", "storage_type", "untyped_storage", "_typed_storage"):
            with self.subTest(owner="Tensor", name=name):
                self.assertFalse(hasattr(torch.Tensor, name))

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        value = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.is_storage(),
                "is_storage() missing 1 required positional argument: 'obj'",
            ),
            (
                lambda: torch.is_storage(value, value),
                "is_storage() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_storage(obj=value),
                "is_storage() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.is_storage(value, obj=value),
                "is_storage() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.is_storage(input=value),
                "is_storage() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: torch.is_storage(extra=value),
                "is_storage() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_storage(value, extra=value),
                "is_storage() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_storage(obj=value, extra=value),
                "is_storage() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

        self.assertIs(torch.is_storage(value, **{}), False)


if __name__ == "__main__":
    unittest.main()
