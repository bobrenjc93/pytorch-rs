import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest
from collections.abc import Callable

import torch_rs as torch
import torch_rs._compiler_state as compiler_state


FUNCTION_DOC = """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """


class CompilerRegisterBackendTests(unittest.TestCase):
    def setUp(self):
        self._backends = compiler_state.backends.copy()
        self._compiler_fns = compiler_state.compiler_fns.copy()
        compiler_state.backends.clear()
        compiler_state.compiler_fns.clear()

    def tearDown(self):
        compiler_state.backends.clear()
        compiler_state.backends.update(self._backends)
        compiler_state.compiler_fns.clear()
        compiler_state.compiler_fns.update(self._compiler_fns)

    def test_direct_and_decorator_forms_register_without_invoking_backend(self):
        events = []

        def default_name_backend(graph_module, example_inputs):
            events.append("default")
            raise AssertionError("backend should not be invoked")

        def explicit_backend(graph_module, example_inputs):
            events.append("explicit")
            raise AssertionError("backend should not be invoked")

        returned = torch.compiler.register_backend(default_name_backend)
        explicit_returned = torch.compiler.register_backend(
            explicit_backend,
            "explicit_backend_name",
        )

        @torch.compiler.register_backend()
        def factory_backend(graph_module, example_inputs):
            events.append("factory")
            raise AssertionError("backend should not be invoked")

        @torch.compiler.register_backend(name="debug_backend", tags=("debug",))
        def debug_backend(graph_module, example_inputs):
            events.append("debug")
            raise AssertionError("backend should not be invoked")

        @torch.compiler.register_backend(
            name="experimental_backend",
            tags=("experimental",),
        )
        def experimental_backend(graph_module, example_inputs):
            events.append("experimental")
            raise AssertionError("backend should not be invoked")

        self.assertIs(returned, default_name_backend)
        self.assertIs(explicit_returned, explicit_backend)
        self.assertEqual(events, [])
        self.assertEqual(default_name_backend._tags, ())
        self.assertEqual(explicit_backend._tags, ())
        self.assertEqual(factory_backend._tags, ())
        self.assertEqual(debug_backend._tags, ("debug",))
        self.assertEqual(experimental_backend._tags, ("experimental",))
        self.assertIs(compiler_state.compiler_fns["default_name_backend"], returned)
        self.assertIs(
            compiler_state.compiler_fns["explicit_backend_name"],
            explicit_returned,
        )

        self.assertEqual(
            torch.compiler.list_backends(),
            [
                "default_name_backend",
                "explicit_backend_name",
                "factory_backend",
            ],
        )
        self.assertEqual(
            torch.compiler.list_backends(()),
            [
                "debug_backend",
                "default_name_backend",
                "experimental_backend",
                "explicit_backend_name",
                "factory_backend",
            ],
        )
        self.assertEqual(torch.compiler.list_backends(None), torch.compiler.list_backends(()))
        self.assertEqual(
            torch.compiler.list_backends(("debug",)),
            [
                "default_name_backend",
                "experimental_backend",
                "explicit_backend_name",
                "factory_backend",
            ],
        )
        self.assertEqual(
            torch.compiler.list_backends(("experimental",)),
            [
                "debug_backend",
                "default_name_backend",
                "explicit_backend_name",
                "factory_backend",
            ],
        )
        self.assertIn("debug_backend", torch.compiler.list_backends("debug"))

    def test_duplicate_and_invalid_registrations_preserve_registry(self):
        def backend(graph_module, example_inputs):
            return graph_module.forward

        class NamelessBackend:
            def __call__(self, graph_module, example_inputs):
                return graph_module.forward

        torch.compiler.register_backend(backend, "unique_backend")
        before = torch.compiler.list_backends(())

        with self.assertRaises(AssertionError) as duplicate_error:
            torch.compiler.register_backend(backend, "unique_backend")
        self.assertEqual(str(duplicate_error.exception), "duplicate name: unique_backend")
        self.assertEqual(torch.compiler.list_backends(()), before)

        with self.assertRaises(AssertionError) as callable_error:
            torch.compiler.register_backend(42, "numeric_backend")
        self.assertEqual(
            str(callable_error.exception),
            "compiler_fn must be callable, got <class 'int'>",
        )
        self.assertEqual(torch.compiler.list_backends(()), before)

        with self.assertRaises(AttributeError):
            torch.compiler.register_backend(NamelessBackend())
        self.assertEqual(torch.compiler.list_backends(()), before)

        with self.assertRaises(TypeError) as name_error:
            torch.compiler.register_backend(backend, object())
        self.assertEqual(
            str(name_error.exception),
            "name must be a string, got <class 'object'>",
        )
        self.assertEqual(torch.compiler.list_backends(()), before)

        with self.assertRaises(TypeError):
            torch.compiler.register_backend(backend, "bad_tags", tags=None)
        self.assertEqual(torch.compiler.list_backends(()), before)

    def test_query_returns_fresh_sorted_lists_and_preserves_eager_state(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()

        def backend_z(graph_module, example_inputs):
            return graph_module.forward

        def backend_a(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(backend_z)
            expected_compiler_flags = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )
            self.assertIs(torch.is_grad_enabled(), True)
            compiler.register_backend(backend_z, "z_backend")

            with torch.no_grad():
                self.assertIs(torch.is_grad_enabled(), False)
                compiler.register_backend(backend_a, "a_backend")
                first = compiler.list_backends(())
                second = compiler.list_backends(())
                self.assertEqual(first, ["a_backend", "z_backend"])
                self.assertEqual(second, first)
                self.assertIsNot(second, first)
                self.assertIs(torch.is_grad_enabled(), False)

            self.assertIs(torch.is_grad_enabled(), True)
            self.assertIs(compiler.get_default_backend(), backend_z)
            self.assertEqual(
                (
                    compiler.is_compiling(),
                    compiler.is_dynamo_compiling(),
                    compiler.is_exporting(),
                ),
                expected_compiler_flags,
            )
        finally:
            compiler.set_default_backend(original_backend)

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.register_backend
        return_annotation = Callable[..., typing.Any]

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(compiler_fn: collections.abc.Callable[[typing.Any, list[typing.Any]], torch_rs.compiler.CompiledFn] | None = None, name: str | None = None, tags: collections.abc.Sequence[str] = ()) -> collections.abc.Callable[..., typing.Any]",
        )
        self.assertEqual(function.__annotations__["return"], return_annotation)
        self.assertEqual(typing.get_type_hints(function)["return"], return_annotation)
        self.assertEqual(function.__name__, "register_backend")
        self.assertEqual(function.__qualname__, "register_backend")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (None, None, ()))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_direct_import_copy_pickle_and_reload_preserve_registry(self):
        compiler = torch.compiler
        function = compiler.register_backend

        from torch_rs.compiler import register_backend

        self.assertIs(register_backend, function)
        self.assertNotIn("register_backend", compiler.__all__)
        self.assertNotIn("register_backend", torch.__all__)
        self.assertFalse(hasattr(torch, "register_backend"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

        def backend(graph_module, example_inputs):
            return graph_module.forward

        compiler.register_backend(backend, "reload_backend")
        old_function = function
        reloaded = importlib.reload(compiler)
        new_function = reloaded.register_backend

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertEqual(compiler.list_backends(()), ["reload_backend"])
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

    def test_subprocess_registry_is_isolated_and_does_not_import_pytorch(self):
        def backend(graph_module, example_inputs):
            return graph_module.forward

        torch.compiler.register_backend(backend, "parent_backend")

        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.compiler.list_backends(()) == []
events = []

def child_backend(graph_module, example_inputs):
    events.append("called")
    raise AssertionError("backend should not be invoked")

registered = torch.compiler.register_backend(
    child_backend,
    "child_backend",
    tags=("debug",),
)
assert registered is child_backend
assert events == []
assert torch.compiler.list_backends() == []
assert torch.compiler.list_backends(()) == ["child_backend"]
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
        self.assertEqual(torch.compiler.list_backends(()), ["parent_backend"])


if __name__ == "__main__":
    unittest.main()
