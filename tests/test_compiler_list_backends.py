import contextlib
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

ALL_BACKENDS = [
    "aot_eager",
    "aot_eager_decomp_partition",
    "aot_eager_decomp_partition_crossref",
    "aot_eager_decomp_partition_with_mode",
    "aot_eager_default_partitioner",
    "aot_ts",
    "cudagraphs",
    "dynamo_accuracy_minifier_backend",
    "dynamo_minifier_backend",
    "eager",
    "eager_debug",
    "eager_noexcept",
    "inductor",
    "invoke_subgraph",
    "non_leaf_compile_error_TESTING_ONLY",
    "openxla",
    "openxla_eval",
    "pre_dispatch_eager",
    "relu_accuracy_error_TESTING_ONLY",
    "relu_compile_error_TESTING_ONLY",
    "relu_runtime_error_TESTING_ONLY",
    "ts",
    "tvm",
]

DEFAULT_BACKENDS = ["cudagraphs", "inductor", "openxla", "tvm"]
EXCLUDE_DEBUG_BACKENDS = [
    "cudagraphs",
    "inductor",
    "openxla",
    "openxla_eval",
    "tvm",
]
EXCLUDE_EXPERIMENTAL_BACKENDS = [
    name for name in ALL_BACKENDS if name != "openxla_eval"
]


class CompilerListBackendsTests(unittest.TestCase):
    def test_default_and_tag_filtered_backend_names(self):
        self.assertEqual(torch.compiler.list_backends(), DEFAULT_BACKENDS)
        self.assertEqual(
            torch.compiler.list_backends(exclude_tags=("debug", "experimental")),
            DEFAULT_BACKENDS,
        )
        self.assertEqual(torch.compiler.list_backends(()), ALL_BACKENDS)
        self.assertEqual(torch.compiler.list_backends([]), ALL_BACKENDS)
        self.assertEqual(torch.compiler.list_backends(("debug",)), EXCLUDE_DEBUG_BACKENDS)
        self.assertEqual(torch.compiler.list_backends(["debug"]), EXCLUDE_DEBUG_BACKENDS)
        self.assertEqual(torch.compiler.list_backends({"debug"}), EXCLUDE_DEBUG_BACKENDS)
        self.assertEqual(
            torch.compiler.list_backends(("experimental",)),
            EXCLUDE_EXPERIMENTAL_BACKENDS,
        )
        self.assertEqual(torch.compiler.list_backends(None), ALL_BACKENDS)
        self.assertEqual(torch.compiler.list_backends("debug"), ALL_BACKENDS)
        self.assertEqual(torch.compiler.list_backends((b"debug",)), ALL_BACKENDS)

    def test_returns_fresh_sorted_lists_without_shared_mutable_state(self):
        first = torch.compiler.list_backends(())
        second = torch.compiler.list_backends(())

        self.assertIs(type(first), list)
        self.assertIs(type(second), list)
        self.assertIsNot(first, second)
        self.assertEqual(first, ALL_BACKENDS)
        self.assertEqual(second, ALL_BACKENDS)
        self.assertEqual(first, sorted(first))
        first.append("custom")
        first.remove("aot_eager")
        self.assertEqual(second, ALL_BACKENDS)
        self.assertEqual(torch.compiler.list_backends(()), ALL_BACKENDS)

    def test_signature_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.list_backends

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(exclude_tags=('debug', 'experimental')) -> list[str]",
        )
        self.assertEqual(function.__annotations__, {"return": list[str]})
        self.assertEqual(typing.get_type_hints(function), {"return": list[str]})
        self.assertEqual(function.__name__, "list_backends")
        self.assertEqual(function.__qualname__, "list_backends")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertEqual(function.__defaults__, (("debug", "experimental"),))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_direct_import_copy_and_pickle_use_canonical_objects(self):
        compiler = torch.compiler
        function = compiler.list_backends

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)

        from torch_rs.compiler import list_backends

        self.assertIs(list_backends, function)
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

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.compiler.list_backends
        cases = (
            (
                lambda: function(None, None),
                "list_backends() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: function((), exclude_tags=()),
                "list_backends() got multiple values for argument 'exclude_tags'",
            ),
            (
                lambda: function(extra=()),
                "list_backends() got an unexpected keyword argument 'extra'",
            ),
            (lambda: function(1), "'int' object is not iterable"),
            (lambda: function(True), "'bool' object is not iterable"),
            (lambda: function([[]]), "unhashable type: 'list'"),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_calls_preserve_compiler_and_grad_state_across_reload(self):
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
            for context in (contextlib.nullcontext(), torch.no_grad()):
                with context:
                    expected_grad_state = torch.is_grad_enabled()
                    self.assertEqual(
                        compiler.list_backends(),
                        DEFAULT_BACKENDS,
                    )
                    self.assertIs(torch.is_grad_enabled(), expected_grad_state)
                    self.assertIs(compiler.get_default_backend(), backend)
                    self.assertEqual(
                        (
                            compiler.is_compiling(),
                            compiler.is_dynamo_compiling(),
                            compiler.is_exporting(),
                        ),
                        expected_queries,
                    )

            old_function = compiler.list_backends
            old_exports = compiler.__all__
            reloaded = importlib.reload(compiler)
            new_function = reloaded.list_backends

            self.assertIs(reloaded, compiler)
            self.assertIs(torch.compiler, compiler)
            self.assertIs(sys.modules["torch_rs.compiler"], compiler)
            self.assertIsNot(new_function, old_function)
            self.assertIsNot(compiler.__all__, old_exports)
            self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
            self.assertIs(compiler.get_default_backend(), backend)
            self.assertEqual(new_function(), DEFAULT_BACKENDS)
            self.assertIs(copy.copy(old_function), old_function)
            self.assertIs(copy.deepcopy(old_function), old_function)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(new_function, protocol)),
                        new_function,
                    )
        finally:
            compiler.set_default_backend(original_backend)

    def test_compile_registration_and_graph_apis_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch.compiler, "allow_in_graph"))
        self.assertFalse(hasattr(torch.compiler, "substitute_in_graph"))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.compiler.list_backends() == ["cudagraphs", "inductor", "openxla", "tvm"]
assert torch.compiler.list_backends(())[:2] == [
    "aot_eager",
    "aot_eager_decomp_partition",
]
assert not hasattr(torch, "compile")
assert not hasattr(torch.compiler, "compile")
assert not hasattr(torch.compiler, "allow_in_graph")
assert not hasattr(torch.compiler, "substitute_in_graph")
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
