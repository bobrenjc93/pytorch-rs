import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import textwrap
import threading
import types
import unittest

import torch_rs as torch
import torch_rs.compiler as compiler


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

if sys.version_info >= (3, 13):
    # CPython 3.13+ cleans function docstring indentation while preserving
    # the initial and terminating newlines.
    FUNCTION_DOC = "\n" + inspect.cleandoc(FUNCTION_DOC) + "\n"


UNSUPPORTED_COMPILER_APIS = (
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
    def query_outcome(self):
        before = torch.is_grad_enabled()
        result = compiler.is_compiling()
        after = torch.is_grad_enabled()
        return before, result, after

    def test_eager_and_grad_mode_states_are_always_exact_false(self):
        self.assertEqual(self.query_outcome(), (True, False, True))
        with torch.no_grad():
            self.assertEqual(self.query_outcome(), (False, False, False))
            with torch.no_grad():
                self.assertEqual(self.query_outcome(), (False, False, False))
            self.assertEqual(self.query_outcome(), (False, False, False))
        self.assertEqual(self.query_outcome(), (True, False, True))

    def test_threaded_eager_and_grad_mode_states_are_always_false(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (self.query_outcome(), self.query_outcome())
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
            grad_enabled = index % 2 == 0
            expected = (grad_enabled, False, grad_enabled)
            self.assertEqual(result, (expected, expected))

    def test_imports_exports_and_unsupported_compiler_surface(self):
        imported = importlib.import_module("torch_rs.compiler")
        from torch_rs import compiler as from_package
        from torch_rs.compiler import is_compiling

        self.assertIs(torch.compiler, compiler)
        self.assertIs(imported, compiler)
        self.assertIs(from_package, compiler)
        self.assertIs(is_compiling, compiler.is_compiling)
        self.assertIsNone(compiler.__doc__)
        self.assertEqual(compiler.__all__, ["is_compiling"])
        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("is_compiling", torch.__all__)
        self.assertFalse(hasattr(torch, "is_compiling"))

        wildcard_namespace = {}
        exec("from torch_rs.compiler import *", wildcard_namespace)
        self.assertEqual(
            {name for name in wildcard_namespace if name != "__builtins__"},
            {"is_compiling"},
        )
        self.assertIs(wildcard_namespace["is_compiling"], compiler.is_compiling)

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        for name in UNSUPPORTED_COMPILER_APIS:
            with self.subTest(name=name):
                self.assertFalse(hasattr(compiler, name))
                self.assertNotIn(name, compiler.__all__)

    def test_signature_annotation_documentation_module_and_pickling(self):
        function = compiler.is_compiling
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "is_compiling")
        self.assertEqual(function.__qualname__, "is_compiling")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        signature = inspect.signature(function)
        self.assertEqual(str(signature), "() -> bool")
        self.assertEqual(tuple(signature.parameters), ())
        self.assertIs(signature.return_annotation, bool)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = compiler.is_compiling
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
                lambda: function(enabled=True),
                "is_compiling() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_compiling() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(value=None),
                "is_compiling() got an unexpected keyword argument 'value'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_import_and_calls_do_not_import_pytorch(self):
        script = textwrap.dedent(
            """\
            import sys

            assert not any(
                name == "torch" or name.startswith("torch.") for name in sys.modules
            )

            class RejectPyTorchImport:
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "torch" or fullname.startswith("torch."):
                        raise RuntimeError("PyTorch import was attempted")
                    return None

            sys.meta_path.insert(0, RejectPyTorchImport())

            import torch_rs
            import torch_rs.compiler as compiler
            from torch_rs.compiler import is_compiling

            assert torch_rs.compiler is compiler
            assert is_compiling is compiler.is_compiling
            assert is_compiling() is False
            assert not any(
                name == "torch" or name.startswith("torch.") for name in sys.modules
            )
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
