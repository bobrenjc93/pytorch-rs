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
from torch_rs import _compiler_state as _state


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

COMPILER_EXPORTS = [
    "assume_constant_result",
    "reset",
    "list_backends",
    "register_backend",
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


def _picklable_backend(graph_module, example_inputs):
    return graph_module.forward


def _secondary_picklable_backend(graph_module, example_inputs):
    return graph_module.forward


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


class _SlotCallableBackend:
    __slots__ = ()

    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


class _BackendName(str):
    pass


class CompilerRegisterBackendTests(unittest.TestCase):
    def setUp(self):
        self._registered_backends = dict(_state.registered_backends)
        self._registered_backend_fns = dict(_state.registered_backend_fns)
        self._picklable_tags_present = hasattr(_picklable_backend, "_tags")
        self._picklable_tags = getattr(_picklable_backend, "_tags", None)
        self._secondary_tags_present = hasattr(_secondary_picklable_backend, "_tags")
        self._secondary_tags = getattr(_secondary_picklable_backend, "_tags", None)
        _state.registered_backends.clear()
        _state.registered_backend_fns.clear()

    def tearDown(self):
        _state.registered_backends.clear()
        _state.registered_backends.update(self._registered_backends)
        _state.registered_backend_fns.clear()
        _state.registered_backend_fns.update(self._registered_backend_fns)
        if self._picklable_tags_present:
            _picklable_backend._tags = self._picklable_tags
        elif hasattr(_picklable_backend, "_tags"):
            del _picklable_backend._tags
        if self._secondary_tags_present:
            _secondary_picklable_backend._tags = self._secondary_tags
        elif hasattr(_secondary_picklable_backend, "_tags"):
            del _secondary_picklable_backend._tags

    def test_direct_decorator_and_factory_forms_register_without_invocation(self):
        calls = []

        def direct_backend(graph_module, example_inputs):
            calls.append(("direct", graph_module, example_inputs))
            return graph_module.forward

        self.assertIs(
            torch.compiler.register_backend(
                direct_backend,
                name="zz_direct",
                tags=("debug",),
            ),
            direct_backend,
        )

        @torch.compiler.register_backend(name="zz_decorated")
        def decorated_backend(graph_module, example_inputs):
            calls.append(("decorated", graph_module, example_inputs))
            return graph_module.forward

        @torch.compiler.register_backend
        def implicit_backend(graph_module, example_inputs):
            calls.append(("implicit", graph_module, example_inputs))
            return graph_module.forward

        factory = torch.compiler.register_backend(
            name="zz_factory",
            tags=["experimental"],
        )
        self.assertIs(type(factory), functools.partial)

        @factory
        def factory_backend(graph_module, example_inputs):
            calls.append(("factory", graph_module, example_inputs))
            return graph_module.forward

        self.assertEqual(calls, [])
        self.assertEqual(direct_backend._tags, ("debug",))
        self.assertEqual(decorated_backend._tags, ())
        self.assertEqual(implicit_backend._tags, ())
        self.assertEqual(factory_backend._tags, ("experimental",))
        self.assertIs(_state.registered_backends["zz_direct"], None)
        self.assertIs(_state.registered_backend_fns["zz_direct"], direct_backend)
        self.assertIs(
            _state.registered_backend_fns["zz_decorated"],
            decorated_backend,
        )
        self.assertIs(
            _state.registered_backend_fns["implicit_backend"],
            implicit_backend,
        )
        self.assertIs(
            _state.registered_backend_fns["zz_factory"],
            factory_backend,
        )
        self.assertEqual(
            torch.compiler.list_backends(exclude_tags=()),
            ["implicit_backend", "zz_decorated", "zz_direct", "zz_factory"],
        )
        self.assertEqual(
            torch.compiler.list_backends(),
            ["implicit_backend", "zz_decorated"],
        )

    def test_string_subclass_names_are_preserved_and_empty_names_use_function_name(self):
        subclass_name = _BackendName("zz_string_subclass")

        def empty_name_backend(graph_module, example_inputs):
            return graph_module.forward

        def subclass_backend(graph_module, example_inputs):
            return graph_module.forward

        self.assertIs(
            torch.compiler.register_backend(
                empty_name_backend,
                name="",
            ),
            empty_name_backend,
        )
        self.assertIs(
            torch.compiler.register_backend(
                subclass_backend,
                name=subclass_name,
            ),
            subclass_backend,
        )
        all_backends = torch.compiler.list_backends(exclude_tags=())
        self.assertEqual(all_backends, ["empty_name_backend", subclass_name])
        self.assertIs(all_backends[1], subclass_name)

    def test_duplicate_non_callable_and_invalid_name_errors_preserve_registry(self):
        def backend(graph_module, example_inputs):
            return graph_module.forward

        def other_backend(graph_module, example_inputs):
            return graph_module.forward

        self.assertIs(
            torch.compiler.register_backend(backend, name="zz_backend"),
            backend,
        )
        before_backends = dict(_state.registered_backends)
        before_functions = dict(_state.registered_backend_fns)

        invalid_cases = (
            (
                lambda: torch.compiler.register_backend(
                    other_backend,
                    name="zz_backend",
                ),
                AssertionError,
                "duplicate name: zz_backend",
            ),
            (
                lambda: torch.compiler.register_backend(42, name="zz_int"),
                AssertionError,
                "compiler_fn must be callable, got <class 'int'>",
            ),
            (
                lambda: torch.compiler.register_backend(other_backend, name=1),
                AssertionError,
                "name must be str or None, got <class 'int'>",
            ),
            (
                lambda: torch.compiler.register_backend(name=object()),
                AssertionError,
                "name must be str or None, got <class 'object'>",
            ),
            (
                lambda: torch.compiler.register_backend(_CallableBackend()),
                AttributeError,
                "'_CallableBackend' object has no attribute '__name__'",
            ),
            (
                lambda: torch.compiler.register_backend(
                    _SlotCallableBackend(),
                    name="zz_slots",
                ),
                AttributeError,
                None,
            ),
        )
        for call, error_type, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                if message is not None:
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                self.assertEqual(_state.registered_backends, before_backends)
                self.assertEqual(_state.registered_backend_fns, before_functions)

        callable_backend = _CallableBackend()
        self.assertIs(
            torch.compiler.register_backend(callable_backend, name="zz_callable"),
            callable_backend,
        )
        self.assertEqual(callable_backend._tags, ())
        self.assertIs(
            _state.registered_backend_fns["zz_callable"],
            callable_backend,
        )

    def test_list_backends_filters_tags_and_returns_fresh_sorted_lists(self):
        def visible_backend(graph_module, example_inputs):
            return graph_module.forward

        def debug_backend(graph_module, example_inputs):
            return graph_module.forward

        def experimental_backend(graph_module, example_inputs):
            return graph_module.forward

        def custom_backend(graph_module, example_inputs):
            return graph_module.forward

        torch.compiler.register_backend(visible_backend, name="zz_visible")
        torch.compiler.register_backend(
            debug_backend,
            name="zz_debug",
            tags=("debug",),
        )
        torch.compiler.register_backend(
            experimental_backend,
            name="zz_experimental",
            tags=("experimental",),
        )
        torch.compiler.register_backend(
            custom_backend,
            name="zz_custom",
            tags=("custom", "debug"),
        )

        default = torch.compiler.list_backends()
        second_default = torch.compiler.list_backends()
        all_backends = torch.compiler.list_backends(exclude_tags=())
        none_excluded = torch.compiler.list_backends(exclude_tags=None)
        debug_excluded = torch.compiler.list_backends(exclude_tags=("debug",))
        experimental_excluded = torch.compiler.list_backends(
            exclude_tags=("experimental",)
        )
        string_excluded = torch.compiler.list_backends(exclude_tags="debug")

        self.assertEqual(default, ["zz_visible"])
        self.assertEqual(second_default, ["zz_visible"])
        self.assertIsNot(default, second_default)
        self.assertEqual(
            all_backends,
            ["zz_custom", "zz_debug", "zz_experimental", "zz_visible"],
        )
        self.assertEqual(none_excluded, all_backends)
        self.assertEqual(debug_excluded, ["zz_experimental", "zz_visible"])
        self.assertEqual(
            experimental_excluded,
            ["zz_custom", "zz_debug", "zz_visible"],
        )
        self.assertEqual(string_excluded, all_backends)

        debug_backend._tags = ("custom",)
        self.assertEqual(
            torch.compiler.list_backends(exclude_tags=("custom",)),
            ["zz_experimental", "zz_visible"],
        )

    def test_signature_documentation_and_function_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.register_backend
        compiler_fn_annotation = Callable[..., typing.Any] | None
        name_annotation = str | None
        tags_annotation = Sequence[str]
        return_annotation = Callable[..., typing.Any]

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            inspect.signature(function),
            inspect.Signature(
                [
                    inspect.Parameter(
                        "compiler_fn",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation=compiler_fn_annotation,
                    ),
                    inspect.Parameter(
                        "name",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation=name_annotation,
                    ),
                    inspect.Parameter(
                        "tags",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=(),
                        annotation=tags_annotation,
                    ),
                ],
                return_annotation=return_annotation,
            ),
        )
        self.assertEqual(
            function.__annotations__,
            {
                "compiler_fn": compiler_fn_annotation,
                "name": name_annotation,
                "tags": tags_annotation,
                "return": return_annotation,
            },
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {
                "compiler_fn": compiler_fn_annotation,
                "name": name_annotation,
                "tags": tags_annotation,
                "return": return_annotation,
            },
        )
        self.assertEqual(function.__name__, "register_backend")
        self.assertEqual(function.__qualname__, "register_backend")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertEqual(function.__defaults__, (None, None, ()))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_exports_copy_pickle_and_reload_use_canonical_objects(self):
        compiler = torch.compiler
        register_backend = compiler.register_backend

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        from torch_rs.compiler import register_backend as imported_register_backend

        self.assertIs(imported_register_backend, register_backend)

        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(COMPILER_EXPORTS),
        )
        for name in COMPILER_EXPORTS:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("register_backend", torch.__all__)
        self.assertFalse(hasattr(torch, "register_backend"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("register_backend", top_level_namespace)

        factory = register_backend(name="zz_factory", tags=("debug",))
        self.assertIs(type(factory), functools.partial)
        self.assertIs(factory.func, register_backend)
        self.assertEqual(factory.keywords, {"name": "zz_factory", "tags": ("debug",)})

        self.assertIs(copy.copy(register_backend), register_backend)
        self.assertIs(copy.deepcopy(register_backend), register_backend)
        self.assertIs(copy.copy(_picklable_backend), _picklable_backend)
        self.assertIs(copy.deepcopy(_picklable_backend), _picklable_backend)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(register_backend, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), register_backend)
                loaded_factory = pickle.loads(pickle.dumps(factory, protocol))
                self.assertIs(loaded_factory.func, register_backend)
                self.assertEqual(loaded_factory.keywords, factory.keywords)

        self.assertIs(
            register_backend(_picklable_backend, name="zz_picklable"),
            _picklable_backend,
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(_picklable_backend, protocol)),
                _picklable_backend,
            )

        old_register_backend = register_backend
        old_list_backends = compiler.list_backends
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_register_backend = reloaded.register_backend

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_register_backend, old_register_backend)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertEqual(compiler.list_backends(exclude_tags=()), ["zz_picklable"])
        self.assertEqual(old_list_backends(exclude_tags=()), ["zz_picklable"])
        self.assertIs(
            old_register_backend(
                _secondary_picklable_backend,
                name="zz_secondary",
            ),
            _secondary_picklable_backend,
        )
        self.assertEqual(
            new_register_backend.__name__,
            "register_backend",
        )
        self.assertEqual(
            compiler.list_backends(exclude_tags=()),
            ["zz_picklable", "zz_secondary"],
        )
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_register_backend)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_register_backend, protocol)),
                new_register_backend,
            )

    def test_subprocess_registry_is_isolated_and_does_not_import_pytorch(self):
        torch.compiler.register_backend(_picklable_backend, name="zz_parent")

        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

calls = []

def backend(graph_module, example_inputs):
    calls.append((graph_module, example_inputs))
    return graph_module.forward

modules_before_call = set(sys.modules)
assert torch.compiler.list_backends(exclude_tags=()) == []
assert torch.compiler.register_backend(backend, name="zz_child") is backend
assert torch.compiler.list_backends(exclude_tags=()) == ["zz_child"]
assert calls == []
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
        self.assertEqual(torch.compiler.list_backends(exclude_tags=()), ["zz_parent"])

    def test_registration_and_listing_preserve_eager_helper_state(self):
        compiler = torch.compiler
        original_backend = compiler.get_default_backend()
        original_guard_collectives = compiler.set_enable_guard_collectives(False)
        compiler.set_enable_guard_collectives(True)

        def default_backend(graph_module, example_inputs):
            return graph_module.forward

        def registry_backend(graph_module, example_inputs):
            return graph_module.forward

        try:
            compiler.set_default_backend(default_backend)
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            self.assertIs(torch.is_grad_enabled(), True)
            self.assertIs(
                compiler.register_backend(registry_backend, name="zz_state"),
                registry_backend,
            )
            self.assertEqual(compiler.list_backends(exclude_tags=()), ["zz_state"])
            self.assertIs(torch.is_grad_enabled(), True)
            with torch.no_grad():
                self.assertIs(torch.is_grad_enabled(), False)
                self.assertEqual(compiler.list_backends(), ["zz_state"])
                self.assertIs(torch.is_grad_enabled(), False)

            self.assertIs(compiler.get_default_backend(), default_backend)
            self.assertEqual(
                (
                    compiler.is_compiling(),
                    compiler.is_dynamo_compiling(),
                    compiler.is_exporting(),
                ),
                expected_queries,
            )
            self.assertIs(compiler.reset(), None)
            self.assertEqual(compiler.list_backends(exclude_tags=()), ["zz_state"])
            self.assertIs(compiler.get_default_backend(), default_backend)
            self.assertIs(compiler.set_enable_guard_collectives(False), True)
        finally:
            compiler.set_default_backend(original_backend)
            compiler.set_enable_guard_collectives(original_guard_collectives)

    def test_compile_graph_capture_and_execution_paths_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "list_backends"))
        self.assertFalse(hasattr(torch, "register_backend"))
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertTrue(callable(torch.compiler.register_backend))

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


if __name__ == "__main__":
    unittest.main()
