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

COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
]


class _UninspectableTags:
    def _fail(self, operation):
        raise AssertionError(f"exclude_tags was inspected through {operation}")

    def __bool__(self):
        self._fail("bool")

    def __contains__(self, value):
        self._fail("contains")

    def __iter__(self):
        self._fail("iteration")

    def __len__(self):
        self._fail("length")

    def __repr__(self):
        self._fail("repr")

    def __str__(self):
        self._fail("str")


class CompilerListBackendsTests(unittest.TestCase):
    def test_empty_registry_returns_fresh_empty_lists_for_supported_argument_forms(self):
        function = torch.compiler.list_backends
        default = function()
        second_default = function()
        tuple_tags = function(("debug", "experimental"))
        empty_tuple_tags = function(())
        list_tags = function([])
        none_tags = function(None)
        string_tags = function("debug")
        opaque_tags = function(_UninspectableTags())

        results = (
            default,
            second_default,
            tuple_tags,
            empty_tuple_tags,
            list_tags,
            none_tags,
            string_tags,
            opaque_tags,
        )
        for result in results:
            self.assertIs(type(result), list)
            self.assertEqual(result, [])

        for left, right in zip(results, results[1:]):
            self.assertIsNot(left, right)

    def test_query_preserves_grad_and_compiler_state(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()

        def backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend)
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            self.assertIs(torch.is_grad_enabled(), True)
            self.assertEqual(compiler.list_backends(), [])
            self.assertIs(torch.is_grad_enabled(), True)
            with torch.no_grad():
                self.assertIs(torch.is_grad_enabled(), False)
                self.assertEqual(compiler.list_backends(exclude_tags=()), [])
                self.assertIs(torch.is_grad_enabled(), False)

            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(
                (
                    compiler.is_compiling(),
                    compiler.is_dynamo_compiling(),
                    compiler.is_exporting(),
                ),
                expected_queries,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.list_backends
        return_annotation = list[str]

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(exclude_tags=('debug', 'experimental')) -> list[str]",
        )
        self.assertEqual(function.__annotations__, {"return": return_annotation})
        self.assertEqual(typing.get_type_hints(function), {"return": return_annotation})
        self.assertEqual(function.__name__, "list_backends")
        self.assertEqual(function.__qualname__, "list_backends")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (("debug", "experimental"),))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_direct_wildcard_copy_pickle_and_reload_use_canonical_function(self):
        compiler = torch.compiler
        function = compiler.list_backends

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        from torch_rs.compiler import list_backends

        self.assertIs(list_backends, function)

        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("list_backends", torch.__all__)
        self.assertFalse(hasattr(torch, "list_backends"))
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

        old_function = function
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.list_backends

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertEqual(new_function(), [])
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.list_backends
        cases = (
            (
                lambda: function([], []),
                "list_backends() takes from 0 to 1 positional arguments but 2 "
                "were given",
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
                lambda: function(exclude_tags=(), unexpected=()),
                "list_backends() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compile_registration_and_execution_paths_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "list_backends"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "register_backend"))

        unsupported_compiler_names = (
            "allow_in_graph",
            "substitute_in_graph",
            "cudagraph_mark_step_begin",
            "load_compiled_function",
            "wrap_numpy",
            "nested_compile_region",
        )
        for name in unsupported_compiler_names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.compiler, name))

    def test_importing_and_calling_does_not_import_pytorch_or_a_registry(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
first = torch.compiler.list_backends()
second = torch.compiler.list_backends(exclude_tags=object())
assert first == []
assert second == []
assert first is not second
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs._dynamo")
    or name.startswith("torch_rs.compiler.backends")
    or name == "torch_rs.compiler.registry"
    for name in sys.modules
)
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
