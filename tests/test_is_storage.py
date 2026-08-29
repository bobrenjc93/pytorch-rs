import copy
import importlib
import inspect
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    from typing import TypeGuard
except ImportError:
    from typing_extensions import TypeGuard


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
        "    False\n\n"
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

    def __float__(self):
        self.called("__float__")


class IsStorageTests(unittest.TestCase):
    def test_supported_boundary_returns_exact_false_without_conversion(self):
        conversion_trap = ConversionTrap()
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
            torch.tensor(3.5),
            torch.tensor([1.0, 2.0]),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
            torch.float32,
            torch.device("cpu"),
            torch.layout,
            torch.strided,
            torch.memory_format,
            torch.contiguous_format,
            inspect.getattr_static(torch.Tensor, "dtype"),
            inspect.getattr_static(torch.Tensor, "device"),
            torch.Tensor,
            conversion_trap,
        )
        for case, value in enumerate(cases):
            with self.subTest(case=case, value_type=type(value).__name__):
                result = torch.is_storage(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, False)
        self.assertEqual(conversion_trap.calls, [])

    def test_callable_metadata_imports_copying_and_pickle_match_pytorch_shape(self):
        function = torch.is_storage
        return_annotation = TypeGuard[
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
        self.assertIs(inspect.getmodule(function), torch)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertEqual(
            function.__annotations__,
            {"obj": typing.Any, "return": return_annotation},
        )
        self.assertEqual(inspect.get_annotations(function), function.__annotations__)
        resolved_annotations = typing.get_type_hints(function)
        self.assertIs(resolved_annotations["obj"], typing.Any)
        self.assertIs(
            typing.get_origin(resolved_annotations["return"]),
            TypeGuard,
        )
        storage_union = typing.get_args(resolved_annotations["return"])[0]
        storage_classes = typing.get_args(storage_union)
        self.assertEqual(
            tuple(cls.__name__ for cls in storage_classes),
            ("TypedStorage", "UntypedStorage"),
        )
        self.assertEqual(
            tuple(cls.__module__ for cls in storage_classes),
            ("torch.storage", "torch.storage"),
        )
        self.assertEqual(inspect.signature(function), expected_signature)
        self.assertEqual(
            str(inspect.signature(function)),
            str(expected_signature),
        )

        package_import = {}
        package_wildcard = {}
        exec("from torch_rs import is_storage", package_import)
        exec("from torch_rs import *", package_wildcard)
        self.assertIs(package_import["is_storage"], function)
        self.assertIs(package_wildcard["is_storage"], function)
        self.assertEqual(torch.__all__.count("is_storage"), 1)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"is_storage", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_positional_only_binding_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.is_storage(),
                "is_storage() missing 1 required positional argument: 'obj'",
            ),
            (
                lambda: torch.is_storage(tensor, tensor),
                "is_storage() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.is_storage(obj=tensor),
                "is_storage() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.is_storage(tensor, obj=tensor),
                "is_storage() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.is_storage(input=tensor),
                "is_storage() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: torch.is_storage(extra=tensor),
                "is_storage() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_storage(tensor, extra=tensor),
                "is_storage() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.is_storage(obj=tensor, extra=tensor),
                "is_storage() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_storage_objects_and_mutation_apis_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        native = importlib.import_module("torch_rs._C")

        self.assertNotIn("is_storage", vars(native))
        self.assertNotIn("is_storage", native.__all__)
        for name in ("Storage", "TypedStorage", "UntypedStorage", "storage"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.storage")

        for name in (
            "share_memory_",
            "storage",
            "storage_type",
            "untyped_storage",
            "_typed_storage",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))

    def test_import_is_probe_free_and_does_not_import_pytorch(self):
        script = r'''
import importlib
import sys

class RejectExternalRuntimeImport:
    blocked = {"torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())

import torch_rs as torch
from torch_rs import is_storage

assert is_storage is torch.is_storage
assert is_storage(torch.tensor([1.0])) is False
for value in (
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
    torch.float32,
    torch.device("cpu"),
    torch.layout,
    torch.strided,
    torch.memory_format,
    torch.contiguous_format,
    object(),
):
    assert is_storage(value) is False
package_wildcard = {}
exec("from torch_rs import *", package_wildcard)
assert package_wildcard["is_storage"] is is_storage
for name in ("Storage", "TypedStorage", "UntypedStorage", "storage"):
    assert not hasattr(torch, name)
try:
    importlib.import_module("torch_rs.storage")
except ModuleNotFoundError:
    pass
else:
    raise AssertionError("torch_rs.storage unexpectedly imported")
assert not any(
    name == "torch" or name.startswith("torch.")
    for name in sys.modules
)
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
