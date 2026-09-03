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


FUNCTION_DOC = """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """

REGISTER_DOC = """
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

TEST_BACKEND_PREFIX = "_torch_rs_test_compiler_backend_"


def _clear_test_backends():
    state = importlib.import_module("torch_rs._compiler_state")
    for name in tuple(state.registered_backends):
        if type(name) is str and name.startswith(TEST_BACKEND_PREFIX):
            state.registered_backends.pop(name, None)
            state.registered_backend_tags.pop(name, None)


def _torch_rs_test_compiler_backend_default(graph_module, example_inputs):
    raise AssertionError("registered backend should not be invoked")


def _torch_rs_test_compiler_backend_pickle(graph_module, example_inputs):
    raise AssertionError("registered backend should not be invoked")


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        raise AssertionError("registered backend should not be invoked")


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
    def setUp(self):
        _clear_test_backends()

    def tearDown(self):
        _clear_test_backends()

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
            compiler.register_backend(
                backend,
                name=f"{TEST_BACKEND_PREFIX}state",
                tags=("debug",),
            )
            expected_queries = (
                compiler.is_compiling(),
                compiler.is_dynamo_compiling(),
                compiler.is_exporting(),
            )

            self.assertIs(torch.is_grad_enabled(), True)
            self.assertEqual(compiler.list_backends(), [])
            self.assertEqual(
                compiler.list_backends(exclude_tags=()),
                [f"{TEST_BACKEND_PREFIX}state"],
            )
            self.assertIs(torch.is_grad_enabled(), True)
            with torch.no_grad():
                self.assertIs(torch.is_grad_enabled(), False)
                self.assertEqual(
                    compiler.list_backends(exclude_tags=("debug",)),
                    [],
                )
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
        self.assertNotIn("torch", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_register_backend_signature_documentation_and_metadata(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.register_backend
        callable_annotation = Callable[..., typing.Any]

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
                        annotation=callable_annotation | None,
                    ),
                    inspect.Parameter(
                        "name",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=None,
                        annotation=str | None,
                    ),
                    inspect.Parameter(
                        "tags",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        default=(),
                        annotation=Sequence[str],
                    ),
                ],
                return_annotation=callable_annotation,
            ),
        )
        self.assertEqual(
            function.__annotations__,
            {
                "compiler_fn": callable_annotation | None,
                "name": str | None,
                "tags": Sequence[str],
                "return": callable_annotation,
            },
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {
                "compiler_fn": callable_annotation | None,
                "name": str | None,
                "tags": Sequence[str],
                "return": callable_annotation,
            },
        )
        self.assertEqual(function.__name__, "register_backend")
        self.assertEqual(function.__qualname__, "register_backend")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(REGISTER_DOC)
        )
        self.assertEqual(function.__defaults__, (None, None, ()))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertNotIn("torch", function.__code__.co_names)
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_register_backend_direct_decorator_and_tag_filtering(self):
        compiler = torch.compiler
        invoked = []

        def stable_backend(graph_module, example_inputs):
            invoked.append("stable")
            return graph_module.forward

        def custom_backend(graph_module, example_inputs):
            invoked.append("custom")
            return graph_module.forward

        def experimental_backend(graph_module, example_inputs):
            invoked.append("experimental")
            return graph_module.forward

        experimental_backend.__name__ = f"{TEST_BACKEND_PREFIX}experimental"

        registered_stable = compiler.register_backend(
            stable_backend,
            name=f"{TEST_BACKEND_PREFIX}stable",
        )

        @compiler.register_backend(
            name=f"{TEST_BACKEND_PREFIX}debug",
            tags=("debug", "custom"),
        )
        def decorated_debug_backend(graph_module, example_inputs):
            invoked.append("decorated_debug")
            return graph_module.forward

        registered_default_name = compiler.register_backend(
            _torch_rs_test_compiler_backend_default
        )
        registered_custom = compiler.register_backend(
            compiler_fn=custom_backend,
            name=f"{TEST_BACKEND_PREFIX}custom",
            tags=("custom",),
        )
        decorator = compiler.register_backend(tags=("experimental",))
        registered_experimental = decorator(experimental_backend)

        self.assertIs(registered_stable, stable_backend)
        self.assertIs(registered_default_name, _torch_rs_test_compiler_backend_default)
        self.assertIs(registered_custom, custom_backend)
        self.assertIs(registered_experimental, experimental_backend)
        self.assertIsInstance(decorator, functools.partial)
        self.assertEqual(stable_backend._tags, ())
        self.assertEqual(decorated_debug_backend._tags, ("debug", "custom"))
        self.assertEqual(custom_backend._tags, ("custom",))
        self.assertEqual(experimental_backend._tags, ("experimental",))
        self.assertEqual(invoked, [])

        default_visible = [
            name
            for name in compiler.list_backends()
            if name.startswith(TEST_BACKEND_PREFIX)
        ]
        all_visible = [
            name
            for name in compiler.list_backends(exclude_tags=())
            if name.startswith(TEST_BACKEND_PREFIX)
        ]
        custom_visible = [
            name
            for name in compiler.list_backends(exclude_tags=("custom",))
            if name.startswith(TEST_BACKEND_PREFIX)
        ]

        self.assertEqual(
            default_visible,
            [
                f"{TEST_BACKEND_PREFIX}custom",
                _torch_rs_test_compiler_backend_default.__name__,
                f"{TEST_BACKEND_PREFIX}stable",
            ],
        )
        self.assertEqual(
            all_visible,
            [
                f"{TEST_BACKEND_PREFIX}custom",
                f"{TEST_BACKEND_PREFIX}debug",
                _torch_rs_test_compiler_backend_default.__name__,
                f"{TEST_BACKEND_PREFIX}experimental",
                f"{TEST_BACKEND_PREFIX}stable",
            ],
        )
        self.assertEqual(
            custom_visible,
            [
                _torch_rs_test_compiler_backend_default.__name__,
                f"{TEST_BACKEND_PREFIX}experimental",
                f"{TEST_BACKEND_PREFIX}stable",
            ],
        )
        self.assertEqual(invoked, [])

    def test_register_backend_rejects_duplicate_noncallable_and_invalid_names(self):
        compiler = torch.compiler

        def duplicate_backend(graph_module, example_inputs):
            return graph_module.forward

        name = f"{TEST_BACKEND_PREFIX}duplicate"
        self.assertIs(
            compiler.register_backend(duplicate_backend, name=name),
            duplicate_backend,
        )
        with self.assertRaises(AssertionError) as duplicate:
            compiler.register_backend(
                lambda graph_module, example_inputs: graph_module.forward,
                name=name,
            )
        self.assertEqual(str(duplicate.exception), f"duplicate name: {name}")

        with self.assertRaises(AssertionError) as noncallable:
            compiler.register_backend(object(), name=f"{TEST_BACKEND_PREFIX}object")
        self.assertEqual(
            str(noncallable.exception),
            "compiler_fn must be callable, got <class 'object'>",
        )

        with self.assertRaises(TypeError) as invalid_name:
            compiler.register_backend(duplicate_backend, name=object())
        self.assertEqual(
            str(invalid_name.exception),
            "backend name must be a string, got <class 'object'>",
        )

        nameless = _CallableBackend()
        with self.assertRaises(AttributeError):
            compiler.register_backend(nameless)

        class EmptyNameBackend:
            __name__ = ""

            def __call__(self, graph_module, example_inputs):
                return graph_module.forward

        with self.assertRaises(ValueError) as empty_name:
            compiler.register_backend(EmptyNameBackend())
        self.assertEqual(str(empty_name.exception), "backend name must be non-empty")

        self.assertEqual(
            [
                registered_name
                for registered_name in compiler.list_backends(exclude_tags=())
                if registered_name.startswith(TEST_BACKEND_PREFIX)
            ],
            [name],
        )

    def test_register_backend_supports_attribute_less_callables_by_name(self):
        name = f"{TEST_BACKEND_PREFIX}builtin"

        self.assertIs(
            torch.compiler.register_backend(len, name=name, tags=("debug",)),
            len,
        )
        self.assertEqual(torch.compiler.list_backends(), [])
        self.assertEqual(torch.compiler.list_backends(exclude_tags=()), [name])

    def test_direct_wildcard_copy_pickle_and_reload_use_canonical_function(self):
        compiler = torch.compiler
        function = compiler.list_backends
        register = compiler.register_backend

        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        from torch_rs.compiler import list_backends, register_backend

        self.assertIs(list_backends, function)
        self.assertIs(register_backend, register)

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
        self.assertNotIn("register_backend", torch.__all__)
        self.assertFalse(hasattr(torch, "list_backends"))
        self.assertFalse(hasattr(torch, "register_backend"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("list_backends", top_level_namespace)
        self.assertNotIn("register_backend", top_level_namespace)

        for exported_function in (function, register):
            self.assertIs(copy.copy(exported_function), exported_function)
            self.assertIs(copy.deepcopy(exported_function), exported_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)
                payload = pickle.dumps(register, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), register)

        registered_backend = register(
            _torch_rs_test_compiler_backend_pickle,
            name=f"{TEST_BACKEND_PREFIX}pickle",
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(backend_protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(registered_backend, protocol)),
                    registered_backend,
                )

        old_function = function
        old_register = register
        old_exports = compiler.__all__
        reloaded = importlib.reload(compiler)
        new_function = reloaded.list_backends
        new_register = reloaded.register_backend

        self.assertIs(reloaded, compiler)
        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(new_register, old_register)
        self.assertIsNot(compiler.__all__, old_exports)
        self.assertEqual(compiler.__all__, COMPILER_EXPORTS)
        self.assertEqual(new_function(exclude_tags=()), [f"{TEST_BACKEND_PREFIX}pickle"])
        self.assertEqual(old_function(exclude_tags=()), [f"{TEST_BACKEND_PREFIX}pickle"])
        self.assertIs(copy.copy(old_function), old_function)
        self.assertIs(copy.deepcopy(old_function), old_function)
        self.assertIs(copy.copy(old_register), old_register)
        self.assertIs(copy.deepcopy(old_register), old_register)
        for old_exported_function in (old_function, old_register):
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_exported_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            self.assertIs(
                pickle.loads(pickle.dumps(new_function, protocol)),
                new_function,
            )
            self.assertIs(
                pickle.loads(pickle.dumps(new_register, protocol)),
                new_register,
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

    def test_register_backend_argument_errors(self):
        function = torch.compiler.register_backend
        cases = (
            (
                lambda: function(None, None, None, None),
                "register_backend() takes from 0 to 3 positional arguments but 4 "
                "were given",
            ),
            (
                lambda: function(None, backend_name="name"),
                "register_backend() got an unexpected keyword argument "
                "'backend_name'",
            ),
            (
                lambda: function(None, None, (), tags=()),
                "register_backend() got multiple values for argument 'tags'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compile_execution_paths_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "list_backends"))
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

    def test_importing_registering_and_calling_does_not_import_pytorch_or_invoke_backend(self):
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
def backend(graph_module, example_inputs):
    raise AssertionError("backend was invoked")

registered = torch.compiler.register_backend(
    backend,
    name="_torch_rs_test_compiler_backend_subprocess",
    tags=("debug",),
)
second = torch.compiler.list_backends()
third = torch.compiler.list_backends(exclude_tags=())
assert first == []
assert second == []
assert third == ["_torch_rs_test_compiler_backend_subprocess"]
assert registered is backend
assert backend._tags == ("debug",)
assert first is not second
assert second is not third
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

    def test_registered_backends_are_process_local(self):
        name = f"{TEST_BACKEND_PREFIX}parent"

        def backend(graph_module, example_inputs):
            raise AssertionError("registered backend should not be invoked")

        torch.compiler.register_backend(backend, name=name)
        script = rf"""
import torch_rs as torch

parent_name = {name!r}
child_name = {f"{TEST_BACKEND_PREFIX}child"!r}
assert parent_name not in torch.compiler.list_backends(exclude_tags=())

def backend(graph_module, example_inputs):
    raise AssertionError("registered backend should not be invoked")

torch.compiler.register_backend(backend, name=child_name)
assert torch.compiler.list_backends(exclude_tags=()) == [child_name]
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
        self.assertIn(name, torch.compiler.list_backends(exclude_tags=()))
        self.assertNotIn(
            f"{TEST_BACKEND_PREFIX}child",
            torch.compiler.list_backends(exclude_tags=()),
        )


if __name__ == "__main__":
    unittest.main()
