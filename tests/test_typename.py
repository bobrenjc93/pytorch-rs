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


FUNCTION_DOC = """
    String representation of the type of an object.

    This function returns a fully qualified string representation of an object's type.
    Args:
        obj (object): The object whose type to represent
    Returns:
        str: the type of the object `o`
    Example:
        >>> x = torch.tensor([1, 2, 3])
        >>> torch.typename(x)
        'torch.LongTensor'
        >>> torch.typename(torch.nn.Parameter)
        'torch.nn.parameter.Parameter'
    """


class ExampleClass:
    class NestedClass:
        pass

    def method(self):
        return None


def example_function():
    return None


class TypenameTests(unittest.TestCase):
    def test_supported_tensors_use_the_float_cpu_legacy_name(self):
        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = (leaf * 2.0).transpose(0, 1)
        tracked.sum().backward()
        tensors = (
            torch.tensor(1.0),
            torch.tensor([1.0, 2.0]),
            torch.zeros((2, 0, 3)),
            torch.zeros((2, 3, 4)).transpose(0, 2)[1],
            leaf,
            tracked,
            leaf.grad,
        )

        for case, tensor in enumerate(tensors):
            with self.subTest(case=case, shape=tensor.shape):
                result = torch.typename(tensor)
                self.assertIs(type(result), str)
                self.assertEqual(result, "torch.FloatTensor")

    def test_builtins_classes_functions_and_instances_are_qualified(self):
        instance = ExampleClass()
        cases = (
            (None, "NoneType"),
            (True, "bool"),
            (1, "int"),
            (1.5, "float"),
            (2.0j, "complex"),
            ("value", "str"),
            (b"value", "bytes"),
            ([], "list"),
            ((), "tuple"),
            ({}, "dict"),
            (object, "object"),
            (object(), "object"),
            (len, "len"),
            (int, "int"),
            (sys, "sys"),
            (ExampleClass, f"{__name__}.ExampleClass"),
            (ExampleClass.NestedClass, f"{__name__}.ExampleClass.NestedClass"),
            (example_function, f"{__name__}.example_function"),
            (ExampleClass.method, f"{__name__}.ExampleClass.method"),
            (instance.method, f"{__name__}.ExampleClass.method"),
            (instance, f"{__name__}.ExampleClass"),
            (torch.typename, "torch_rs.typename"),
            (torch.Tensor, "torch_rs.Tensor"),
        )

        for value, expected in cases:
            with self.subTest(value=repr(value), expected=expected):
                result = torch.typename(value)
                self.assertIs(type(result), str)
                self.assertEqual(result, expected)

    def test_empty_and_builtin_module_names_are_omitted(self):
        class EmptyModule:
            pass

        class BuiltinModule:
            pass

        EmptyModule.__module__ = None
        BuiltinModule.__module__ = "builtins"
        empty_qualname = (
            "TypenameTests.test_empty_and_builtin_module_names_are_omitted."
            "<locals>.EmptyModule"
        )
        builtin_qualname = (
            "TypenameTests.test_empty_and_builtin_module_names_are_omitted."
            "<locals>.BuiltinModule"
        )

        for value, expected in (
            (EmptyModule, empty_qualname),
            (EmptyModule(), empty_qualname),
            (BuiltinModule, builtin_qualname),
            (BuiltinModule(), builtin_qualname),
        ):
            with self.subTest(value=repr(value)):
                self.assertEqual(torch.typename(value), expected)

    def test_callable_metadata_matches_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.typename
        expected_signature = inspect.Signature(
            parameters=(
                inspect.Parameter(
                    "obj",
                    inspect.Parameter.POSITIONAL_ONLY,
                    annotation=typing.Any,
                ),
            ),
            return_annotation=str,
        )

        self.assertIs(torch, package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "typename")
        self.assertEqual(function.__qualname__, "typename")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"obj": typing.Any, "return": str})
        self.assertEqual(inspect.get_annotations(function), function.__annotations__)
        self.assertEqual(typing.get_type_hints(function), function.__annotations__)
        self.assertEqual(inspect.signature(function), expected_signature)
        self.assertEqual(str(inspect.signature(function)), "(obj: Any, /) -> str")
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_function(self):
        function = torch.typename

        self.assertEqual(torch.__all__.count("typename"), 1)
        self.assertNotIn("typename", torch._C.__all__)
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["typename"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_positional_only_argument_errors_match_pytorch_2_13(self):
        tensor = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.typename(),
                "typename() missing 1 required positional argument: 'obj'",
            ),
            (
                lambda: torch.typename(tensor, tensor),
                "typename() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.typename(obj=tensor),
                "typename() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.typename(tensor, obj=tensor),
                "typename() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
            (
                lambda: torch.typename(input=tensor),
                "typename() got an unexpected keyword argument 'input'",
            ),
            (
                lambda: torch.typename(extra=tensor),
                "typename() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.typename(tensor, extra=tensor),
                "typename() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.typename(obj=tensor, extra=tensor),
                "typename() got some positional-only arguments passed as keyword arguments: 'obj'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$") as raised:
                    call()
                self.assertEqual(raised.exception.args, (message,))

    def test_legacy_tensor_types_and_type_conversion_remain_unsupported(self):
        legacy_types = (
            "ByteTensor",
            "CharTensor",
            "ShortTensor",
            "IntTensor",
            "LongTensor",
            "HalfTensor",
            "FloatTensor",
            "DoubleTensor",
            "BoolTensor",
            "BFloat16Tensor",
        )
        for name in legacy_types:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "type"))

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.typename(torch.tensor([1.0])) == "torch.FloatTensor"
assert torch.typename(int) == "int"
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
