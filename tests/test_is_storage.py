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

    def __float__(self):
        self.called("__float__")


class IsStorageTests(unittest.TestCase):
    def test_supported_boundary_inputs_return_exact_bool_false(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        conversion_trap = ConversionTrap()
        cases = (
            tensor,
            tensor.transpose(0, 1),
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
            torch.contiguous_format,
            torch.preserve_format,
            inspect.getattr_static(torch.Tensor, "dtype"),
            inspect.getattr_static(torch.Tensor, "device"),
            torch.Tensor,
            object(),
            conversion_trap,
        )
        for case, value in enumerate(cases):
            with self.subTest(case=case, value_type=type(value).__name__):
                result = torch.is_storage(value)
                self.assertIs(type(result), bool)
                self.assertIs(result, False)
        self.assertEqual(conversion_trap.calls, [])

    def test_callable_metadata_matches_pytorch_2_13_shape(self):
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
        self.assertIs(inspect.getmodule(function), torch)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            function.__annotations__,
            {"obj": typing.Any, "return": return_annotation},
        )
        self.assertEqual(inspect.get_annotations(function), function.__annotations__)
        self.assertEqual(inspect.signature(function), expected_signature)
        self.assertEqual(
            str(inspect.signature(function)),
            "(obj: Any, /) -> TypeGuard[ForwardRef('TypedStorage | UntypedStorage')]",
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertIn("is_storage", torch.__all__)

    def test_direct_wildcard_copy_and_pickle_use_the_canonical_package_function(self):
        function = torch.is_storage
        native = importlib.import_module("torch_rs._C")
        direct_namespace = {}
        wildcard_namespace = {}
        native_wildcard_namespace = {}

        exec("from torch_rs import is_storage", direct_namespace)
        exec("from torch_rs import *", wildcard_namespace)
        exec("from torch_rs._C import *", native_wildcard_namespace)

        self.assertIs(direct_namespace["is_storage"], function)
        self.assertIs(wildcard_namespace["is_storage"], function)
        self.assertEqual(torch.__all__.count("is_storage"), 1)
        self.assertNotIn("is_storage", vars(native))
        self.assertNotIn("is_storage", native.__all__)
        self.assertNotIn("is_storage", native_wildcard_namespace)

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
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_storage_objects_and_mutation_apis_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        for name in ("Storage", "TypedStorage", "UntypedStorage", "storage"):
            with self.subTest(name=name, owner="torch"):
                self.assertFalse(hasattr(torch, name))
        for name in (
            "share_memory_",
            "storage",
            "storage_type",
            "untyped_storage",
            "_typed_storage",
        ):
            with self.subTest(name=name, owner="Tensor"):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.storage")

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import inspect
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

values = (
    torch.tensor([1.0]),
    torch.float32,
    torch.device("cpu"),
    torch.layout,
    torch.strided,
    torch.contiguous_format,
    inspect.getattr_static(torch.Tensor, "dtype"),
    None,
    True,
    1,
    1.5,
    [1.0, 2.0],
    object(),
)
for value in values:
    assert torch.is_storage(value) is False
assert "is_storage" in torch.__all__
assert not hasattr(torch, "Storage")
assert not hasattr(torch, "TypedStorage")
assert not hasattr(torch, "UntypedStorage")
assert not hasattr(torch, "storage")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
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
