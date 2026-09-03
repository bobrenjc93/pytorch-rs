from collections.abc import Callable, Sequence
import copy
import functools
import importlib
import inspect
import pickle
import subprocess
import sys
import types
import typing
import unittest

import torch_rs as torch
from torch_rs import _compiler_state


REGISTER_BACKEND_DOC = """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    This compatibility layer records backend names for discovery only; it does
    not add `torch.compile` execution.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        raise AssertionError("backend should not be invoked")


class _NamedString(str):
    pass


class CompilerRegisterBackendTests(unittest.TestCase):
    def setUp(self):
        self._registry = _compiler_state.registered_backends
        self._saved_registry = dict(self._registry)
        self._registry.clear()
        self._original_backend = torch.compiler.get_default_backend()
        self._original_guard_collectives = (
            torch.compiler.set_enable_guard_collectives(False)
        )

    def tearDown(self):
        self._registry.clear()
        self._registry.update(self._saved_registry)
        torch.compiler.set_default_backend(self._original_backend)
        torch.compiler.set_enable_guard_collectives(self._original_guard_collectives)

    def test_register_backend_direct_and_decorator_forms(self):
        compiler = torch.compiler
        calls = []

        def direct_backend(graph_module, example_inputs):
            calls.append((graph_module, example_inputs))
            return graph_module.forward

        self.assertIs(compiler.register_backend(direct_backend), direct_backend)
        self.assertEqual(direct_backend._tags, ())
        self.assertEqual(compiler.list_backends(), ["direct_backend"])
        self.assertEqual(calls, [])

        @compiler.register_backend(
            name="debug_backend", tags=("debug", "torch_rs")
        )
        def decorated_backend(graph_module, example_inputs):
            calls.append((graph_module, example_inputs))
            return graph_module.forward

        self.assertEqual(decorated_backend._tags, ("debug", "torch_rs"))
        self.assertEqual(
            compiler.list_backends(exclude_tags=()),
            ["debug_backend", "direct_backend"],
        )
        self.assertEqual(compiler.list_backends(), ["direct_backend"])
        self.assertEqual(
            compiler.list_backends(exclude_tags=("experimental",)),
            ["debug_backend", "direct_backend"],
        )
        self.assertEqual(
            compiler.list_backends(exclude_tags=None),
            ["debug_backend", "direct_backend"],
        )
        self.assertEqual(calls, [])

    def test_default_name_empty_name_and_string_subclass_are_preserved(self):
        compiler = torch.compiler

        def named_backend(graph_module, example_inputs):
            return graph_module.forward

        def empty_name_backend(graph_module, example_inputs):
            return graph_module.forward

        subclass_name = _NamedString("subclass_backend")

        def subclass_backend(graph_module, example_inputs):
            return graph_module.forward

        self.assertIs(compiler.register_backend(named_backend), named_backend)
        self.assertIs(
            compiler.register_backend(empty_name_backend, name=""),
            empty_name_backend,
        )
        self.assertIs(
            compiler.register_backend(subclass_backend, name=subclass_name),
            subclass_backend,
        )

        names = compiler.list_backends(exclude_tags=())
        self.assertEqual(
            names, ["empty_name_backend", "named_backend", "subclass_backend"]
        )
        self.assertIs(
            next(name for name in names if name == "subclass_backend"),
            subclass_name,
        )

    def test_duplicate_and_invalid_registration_preserve_registry(self):
        compiler = torch.compiler

        def backend(graph_module, example_inputs):
            return graph_module.forward

        def other_backend(graph_module, example_inputs):
            return graph_module.forward

        self.assertIs(compiler.register_backend(backend, name="duplicate"), backend)
        before = dict(self._registry)

        duplicate_cases = (
            lambda: compiler.register_backend(backend, name="duplicate"),
            lambda: compiler.register_backend(other_backend, name="duplicate"),
        )
        for case, call in enumerate(duplicate_cases):
            with self.subTest(case=case):
                with self.assertRaises(AssertionError) as raised:
                    call()
                self.assertEqual(str(raised.exception), "duplicate name: duplicate")
                self.assertEqual(self._registry, before)

        invalid_values = (123, object(), ["bad"])
        for value in invalid_values:
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(AssertionError) as raised:
                    compiler.register_backend(other_backend, name=value)
                self.assertEqual(
                    str(raised.exception),
                    f"name must be a string, got {type(value)}",
                )
                self.assertEqual(self._registry, before)

        with self.assertRaises(AssertionError) as raised:
            compiler.register_backend("not callable", name="not_callable")
        self.assertEqual(
            str(raised.exception),
            "compiler_fn must be callable, got <class 'str'>",
        )
        self.assertEqual(self._registry, before)

        with self.assertRaises(TypeError) as raised:
            compiler.register_backend(other_backend, name="bad_tags", tags=None)
        self.assertEqual(str(raised.exception), "'NoneType' object is not iterable")
        self.assertEqual(self._registry, before)

        with self.assertRaises(AttributeError) as raised:
            compiler.register_backend(_CallableBackend())
        self.assertEqual(
            str(raised.exception),
            "'_CallableBackend' object has no attribute '__name__'",
        )
        self.assertEqual(self._registry, before)

        with self.assertRaises(AttributeError) as raised:
            compiler.register_backend(len, name="builtin_len")
        self.assertEqual(
            str(raised.exception),
            "'builtin_function_or_method' object has no attribute '_tags'",
        )
        self.assertEqual(self._registry, before)

    def test_argument_errors_match_python_function_shape(self):
        function = torch.compiler.register_backend
        cases = (
            (
                lambda: function(None, None, (), None),
                "register_backend() takes from 0 to 3 positional arguments but 4 "
                "were given",
            ),
            (
                lambda: function(lambda graph_module, example_inputs: None, bad=True),
                "register_backend() got an unexpected keyword argument 'bad'",
            ),
            (
                lambda: function(None, None, tags=(), name="x"),
                "register_backend() got multiple values for argument 'name'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.register_backend
        compiler_fn_annotation = Callable[..., typing.Any] | None
        return_annotation = Callable[..., typing.Any]

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(compiler_fn: collections.abc.Callable[..., typing.Any] | None = "
            "None, name: str | None = None, tags: collections.abc.Sequence[str] "
            "= ()) -> collections.abc.Callable[..., typing.Any]",
        )
        self.assertEqual(
            function.__annotations__,
            {
                "compiler_fn": compiler_fn_annotation,
                "name": str | None,
                "tags": Sequence[str],
                "return": return_annotation,
            },
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {
                "compiler_fn": compiler_fn_annotation,
                "name": str | None,
                "tags": Sequence[str],
                "return": return_annotation,
            },
        )
        self.assertEqual(function.__name__, "register_backend")
        self.assertEqual(function.__qualname__, "register_backend")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(REGISTER_BACKEND_DOC),
        )
        self.assertEqual(function.__defaults__, (None, None, ()))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertNotIn("torch", function.__code__.co_names)
        self.assertNotIn("_dynamo", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_explicit_import_copy_pickle_and_reload_use_canonical_function(self):
        compiler = torch.compiler
        function = compiler.register_backend

        from torch_rs.compiler import register_backend

        self.assertIs(register_backend, function)
        self.assertNotIn("register_backend", compiler.__all__)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

        factory = function(name="factory_backend", tags=("debug",))
        self.assertIs(type(factory), functools.partial)
        self.assertIs(factory.func, function)
        self.assertEqual(factory.args, ())
        self.assertEqual(
            factory.keywords,
            {"name": "factory_backend", "tags": ("debug",)},
        )
        for copied_factory in (copy.copy(factory), copy.deepcopy(factory)):
            self.assertIs(type(copied_factory), functools.partial)
            self.assertIs(copied_factory.func, function)
            self.assertEqual(copied_factory.args, ())
            self.assertEqual(
                copied_factory.keywords,
                {"name": "factory_backend", "tags": ("debug",)},
            )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(factory_protocol=protocol):
                round_trip = pickle.loads(pickle.dumps(factory, protocol=protocol))
                self.assertIs(type(round_trip), functools.partial)
                self.assertIs(round_trip.func, function)
                self.assertEqual(round_trip.args, ())
                self.assertEqual(
                    round_trip.keywords,
                    {"name": "factory_backend", "tags": ("debug",)},
                )

        def reloaded_backend(graph_module, example_inputs):
            return graph_module.forward

        self.assertIs(
            function(reloaded_backend, name="reloaded_backend"),
            reloaded_backend,
        )
        old_function = function
        reloaded = importlib.reload(compiler)
        new_function = reloaded.register_backend

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertEqual(compiler.list_backends(), ["reloaded_backend"])
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )

        module_name = compiler.__name__
        try:
            self.assertIs(sys.modules.pop(module_name), compiler)
            replacement = importlib.import_module(module_name)
            self.assertIsNot(replacement, compiler)
            self.assertIs(torch.compiler, replacement)
            self.assertEqual(replacement.list_backends(), ["reloaded_backend"])

            def replacement_backend(graph_module, example_inputs):
                return graph_module.forward

            self.assertIs(
                old_function(replacement_backend, name="replacement_backend"),
                replacement_backend,
            )
            self.assertEqual(
                replacement.list_backends(),
                ["reloaded_backend", "replacement_backend"],
            )
        finally:
            sys.modules[module_name] = compiler
            torch.compiler = compiler

    def test_registration_preserves_eager_helper_state(self):
        compiler = torch.compiler

        def default_backend(graph_module, example_inputs):
            return graph_module.forward

        def registry_backend(graph_module, example_inputs):
            raise AssertionError("backend should not be invoked")

        compiler.set_default_backend(default_backend)
        self.assertIs(compiler.set_enable_guard_collectives(True), False)
        before_compile_flags = (
            compiler.is_compiling(),
            compiler.is_dynamo_compiling(),
            compiler.is_exporting(),
        )

        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(
            compiler.register_backend(registry_backend, name="state_backend"),
            registry_backend,
        )
        self.assertEqual(compiler.list_backends(), ["state_backend"])
        self.assertIs(compiler.reset(), None)
        self.assertEqual(compiler.list_backends(), ["state_backend"])
        self.assertIs(torch.is_grad_enabled(), True)

        with torch.no_grad():
            self.assertIs(torch.is_grad_enabled(), False)
            self.assertEqual(
                compiler.list_backends(exclude_tags=()),
                ["state_backend"],
            )
            self.assertIs(torch.is_grad_enabled(), False)

        self.assertIs(compiler.get_default_backend(), default_backend)
        self.assertEqual(
            (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            ),
            before_compile_flags,
        )
        self.assertIs(compiler.set_enable_guard_collectives(False), True)

    def test_subprocesses_have_isolated_registries_without_pytorch_imports(self):
        register_script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

modules_before_registration = set(sys.modules)
called = False

def backend(graph_module, example_inputs):
    global called
    called = True
    raise AssertionError("backend should not be invoked")

assert torch.compiler.list_backends() == []
assert torch.compiler.register_backend(backend, name="child_backend") is backend
assert torch.compiler.list_backends() == ["child_backend"]
assert torch.compiler.list_backends(exclude_tags=()) == ["child_backend"]
assert called is False
assert set(sys.modules) == modules_before_registration
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not hasattr(torch, "compile")
assert not hasattr(torch.compiler, "compile")
"""
        isolated_script = r"""
import torch_rs as torch

assert torch.compiler.list_backends() == []
assert not hasattr(torch, "compile")
assert not hasattr(torch.compiler, "compile")
"""

        for script in (register_script, isolated_script):
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
