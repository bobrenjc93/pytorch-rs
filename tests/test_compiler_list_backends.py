import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """


class CompilerListBackendsTests(unittest.TestCase):
    def test_returns_fresh_empty_lists_for_the_backend_free_runtime(self):
        function = torch.compiler.list_backends
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        excluded = ["debug", "experimental"]
        results = [
            function(),
            function(()),
            function([]),
            function(set()),
            function(("debug",)),
            function(("experimental",)),
            function(("custom",)),
            function(exclude_tags=excluded),
            function(None),
            function(0),
        ]

        self.assertEqual(excluded, ["debug", "experimental"])
        for result in results:
            self.assertIs(type(result), list)
            self.assertEqual(result, [])
        for index, result in enumerate(results):
            for other in results[index + 1 :]:
                self.assertIsNot(result, other)

        results[0].append("invented")
        self.assertEqual(results[0], ["invented"])
        self.assertEqual(function(), [])
        self.assertEqual(results[1:], [[] for _ in results[1:]])

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.list_backends
        parameter = inspect.Parameter(
            "exclude_tags",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=("debug", "experimental"),
        )

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            inspect.signature(function),
            inspect.Signature(
                parameters=(parameter,),
                return_annotation=list[str],
            ),
        )
        self.assertEqual(function.__annotations__, {"return": list[str]})
        self.assertEqual(typing.get_type_hints(function), {"return": list[str]})
        self.assertEqual(function.__name__, "list_backends")
        self.assertEqual(function.__qualname__, "list_backends")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__defaults__, (("debug", "experimental"),))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_imports_copy_and_pickle_use_the_canonical_function(self):
        compiler = torch.compiler
        function = compiler.list_backends

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "list_backends",
                "get_default_backend",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
            ],
        )
        package_namespace = {}
        exec("from torch_rs import compiler", package_namespace)
        self.assertIs(package_namespace["compiler"], compiler)

        direct_namespace = {}
        exec("from torch_rs.compiler import list_backends", direct_namespace)
        self.assertIs(direct_namespace["list_backends"], function)

        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("list_backends", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("list_backends", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_call_shape_errors_match_pytorch_2_13(self):
        function = torch.compiler.list_backends
        cases = (
            (
                lambda: function(None, None),
                "list_backends() takes from 0 to 1 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function((), exclude_tags=()),
                "list_backends() got multiple values for argument 'exclude_tags'",
            ),
            (
                lambda: function(tags=()),
                "list_backends() got an unexpected keyword argument 'tags'",
            ),
            (
                lambda: function((), extra=True),
                "list_backends() got an unexpected keyword argument 'extra'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_calls_do_not_import_pytorch_dynamo_or_a_backend_registry(self):
        script = r'''
import sys

class RejectCompilerDependencyImport:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
            or fullname == "torch_rs.compiler.backends"
            or fullname.startswith("torch_rs.compiler.backends.")
            or fullname == "torch_rs.compiler.registry"
        ):
            raise RuntimeError(f"compiler dependency import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerDependencyImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
first = torch.compiler.list_backends()
second = torch.compiler.list_backends(exclude_tags=())
assert first == []
assert second == []
assert first is not second
assert set(sys.modules) == modules_before_call
assert not hasattr(torch, "_dynamo")
assert not hasattr(torch.compiler, "register_backend")
assert not hasattr(torch.compiler, "set_default_backend")
assert not hasattr(torch.compiler, "compile")
assert not hasattr(torch, "compile")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name == "torch_rs.compiler.backends"
    or name.startswith("torch_rs.compiler.backends.")
    or name == "torch_rs.compiler.registry"
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
