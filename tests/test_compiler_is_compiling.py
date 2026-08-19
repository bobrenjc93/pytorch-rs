import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


FUNCTION_DOC = """
    Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().

    Note that there are 2 other related flags that should deprecated eventually:
      * torch._dynamo.external_utils.is_compiling()
      * torch._utils.is_compiling()

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_compiling():
        >>>        pass # ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     # ...rest of the function...
    """


UNSUPPORTED_COMPILER_EXPORTS = (
    "compile",
    "config",
    "assume_constant_result",
    "reset",
    "allow_in_graph",
    "substitute_in_graph",
    "list_backends",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_stance",
    "set_enable_guard_collectives",
    "cudagraph_mark_step_begin",
    "load_compiled_function",
    "wrap_numpy",
    "is_dynamo_compiling",
    "is_exporting",
    "save_cache_artifacts",
    "load_cache_artifacts",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
    "nested_compile_region",
)


class CompilerIsCompilingTests(unittest.TestCase):
    def test_eager_false_is_exact_and_does_not_change_grad_mode(self):
        function = torch.compiler.is_compiling

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), False)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_eager_false_is_stable_across_threads_and_no_grad(self):
        function = torch.compiler.is_compiling
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                        function(),
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertIs(result[0], expected_grad_state)
            self.assertIs(result[1], False)
            self.assertIs(result[2], expected_grad_state)
            self.assertIs(result[3], False)
            self.assertIs(result[4], expected_grad_state)

    def test_canonical_imports_wildcards_and_unsupported_surface(self):
        compiler = importlib.import_module("torch_rs.compiler")
        from torch_rs import compiler as package_compiler
        from torch_rs.compiler import is_compiling

        self.assertIs(torch.compiler, compiler)
        self.assertIs(package_compiler, compiler)
        self.assertIs(is_compiling, compiler.is_compiling)
        self.assertEqual(compiler.__all__, ["is_compiling"])
        self.assertNotIn("compiler", torch.__all__)

        compiler_namespace = {}
        package_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        exec("from torch_rs import *", package_namespace)
        self.assertIs(compiler_namespace["is_compiling"], is_compiling)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("_")},
            {"is_compiling"},
        )
        self.assertNotIn("compiler", package_namespace)

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        for name in UNSUPPORTED_COMPILER_EXPORTS:
            with self.subTest(name=name):
                self.assertFalse(hasattr(compiler, name))
                self.assertNotIn(name, compiler_namespace)

    def test_metadata_signature_documentation_and_pickling(self):
        compiler = torch.compiler
        function = compiler.is_compiling

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_compiling")
        self.assertEqual(function.__qualname__, "is_compiling")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertFalse(hasattr(function, "__wrapped__"))
        self.assertFalse(hasattr(function, "__signature__"))
        self.assertEqual(function.__dict__, {})
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiler.is_compiling
        cases = (
            (
                lambda: function(None),
                "is_compiling() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_compiling() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(unexpected=True),
                "is_compiling() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(None, unexpected=True),
                "is_compiling() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_import_and_calls_do_not_import_pytorch(self):
        script = """
import builtins
import importlib
import sys

original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise AssertionError(f"unexpected PyTorch import: {name}")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
assert "torch" not in sys.modules
import torch_rs
compiler = importlib.import_module("torch_rs.compiler")
assert torch_rs.compiler is compiler
assert compiler.is_compiling() is False
assert compiler.is_compiling() is False
assert "torch" not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
