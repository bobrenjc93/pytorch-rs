from collections.abc import Callable
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch


SETTER_DOC = '''Set the default backend for ``torch.compile`` when no ``backend`` argument is specified.

    Passing ``None`` resets the default back to ``"inductor"``.

    Args:
        backend: A backend name (string), a callable backend, or ``None``.

    Example::

        >>> torch.compiler.set_default_backend("eager")
        >>> torch.compiler.get_default_backend()
        'eager'
        >>> torch.compiler.set_default_backend(None)  # reset
        >>> torch.compiler.get_default_backend()
        'inductor'
    '''


RESET_DOC = """
    Reset the in-process compiler state.

    This function clears Dynamo's in-memory compilation caches and related
    process-local state used by :func:`torch.compile`. It does not delete
    filesystem caches, such as Inductor's disk cache.
    """


class _CallableBackend:
    def __call__(self, graph_module, example_inputs):
        return graph_module.forward


class _StringBackend(str):
    pass


class CompilerSetDefaultBackendTests(unittest.TestCase):
    def setUp(self):
        self.original_backend = torch.compiler.get_default_backend()
        torch.compiler.set_default_backend(None)

    def tearDown(self):
        torch.compiler.set_default_backend(self.original_backend)

    def test_accepts_strings_callables_and_none_without_coercion(self):
        def function_backend(graph_module, example_inputs):
            return graph_module.forward

        callable_backend = _CallableBackend()
        string_backend = _StringBackend("custom")
        values = (
            "",
            "eager",
            string_backend,
            function_backend,
            callable_backend,
            _CallableBackend,
            len,
        )

        for backend in values:
            with self.subTest(backend=backend):
                self.assertIs(torch.compiler.set_default_backend(backend), None)
                self.assertIs(torch.compiler.get_default_backend(), backend)

        callable_backend.attribute = "updated"
        torch.compiler.set_default_backend(callable_backend)
        self.assertIs(torch.compiler.get_default_backend(), callable_backend)
        self.assertEqual(torch.compiler.get_default_backend().attribute, "updated")

        self.assertIs(torch.compiler.set_default_backend(None), None)
        reset_backend = torch.compiler.get_default_backend()
        self.assertIs(type(reset_backend), str)
        self.assertEqual(reset_backend, "inductor")

    def test_keyword_call_and_invalid_values_match_pytorch_errors(self):
        backend = "".join(("ea", "ger"))
        self.assertIs(torch.compiler.set_default_backend(backend=backend), None)
        self.assertIs(torch.compiler.get_default_backend(), backend)

        invalid_values = (False, 0, 1.5, [], {}, object(), _CallableBackend())
        for value in invalid_values[:-1]:
            with self.subTest(value=value):
                before = torch.compiler.get_default_backend()
                message = (
                    "backend must be a string or callable, got "
                    f"{type(value)}"
                )
                with self.assertRaises(TypeError) as raised:
                    torch.compiler.set_default_backend(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.compiler.get_default_backend(), before)

        callable_backend = invalid_values[-1]
        self.assertIs(torch.compiler.set_default_backend(callable_backend), None)
        self.assertIs(torch.compiler.get_default_backend(), callable_backend)

    def test_updates_are_process_global_across_threads(self):
        compiler = torch.compiler
        initial = _CallableBackend()
        updated = _CallableBackend()
        worker_ready = threading.Event()
        read_updated = threading.Event()
        observations = []
        errors = []

        compiler.set_default_backend(initial)

        def observer():
            try:
                observations.append(compiler.get_default_backend() is initial)
                worker_ready.set()
                if not read_updated.wait(timeout=10):
                    raise RuntimeError("timed out waiting for the updated backend")
                observations.append(compiler.get_default_backend() is updated)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=observer)
        thread.start()
        self.assertTrue(worker_ready.wait(timeout=10))
        compiler.set_default_backend(updated)
        read_updated.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])

        worker_backend = _CallableBackend()
        worker_results = []

        def writer():
            worker_results.append(compiler.set_default_backend(worker_backend))
            worker_results.append(compiler.get_default_backend() is worker_backend)

        thread = threading.Thread(target=writer)
        thread.start()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(worker_results, [None, True])
        self.assertIs(compiler.get_default_backend(), worker_backend)

    def test_reset_preserves_backend_and_grad_mode(self):
        backend = _CallableBackend()
        torch.compiler.set_default_backend(backend)

        def assert_reset_preserves_state(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(torch.compiler.reset(), None)
            self.assertIs(torch.compiler.get_default_backend(), backend)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_reset_preserves_state(True)
        with torch.no_grad():
            assert_reset_preserves_state(False)
            with torch.no_grad():
                assert_reset_preserves_state(False)
        assert_reset_preserves_state(True)

    def test_reload_and_reimport_share_process_state_with_old_functions(self):
        original_module = torch.compiler
        original_getter = original_module.get_default_backend
        original_setter = original_module.set_default_backend
        first_backend = _CallableBackend()
        second_backend = _CallableBackend()
        third_backend = _CallableBackend()

        original_setter(first_backend)
        self.assertIs(importlib.reload(original_module), original_module)
        self.assertIs(torch.compiler, original_module)
        self.assertIs(original_getter(), first_backend)
        self.assertIs(original_module.get_default_backend(), first_backend)

        self.assertIs(
            original_module.set_default_backend(second_backend),
            None,
        )
        self.assertIs(original_getter(), second_backend)

        module_name = original_module.__name__
        try:
            self.assertIs(sys.modules.pop(module_name), original_module)
            replacement_module = importlib.import_module(module_name)
            self.assertIsNot(replacement_module, original_module)
            self.assertIs(torch.compiler, replacement_module)
            self.assertIs(replacement_module.get_default_backend(), second_backend)
            self.assertIs(original_getter(), second_backend)

            self.assertIs(
                replacement_module.set_default_backend(third_backend),
                None,
            )
            self.assertIs(original_getter(), third_backend)
            self.assertIs(replacement_module.get_default_backend(), third_backend)
            self.assertIs(original_module.reset(), None)
            self.assertIs(replacement_module.get_default_backend(), third_backend)

            self.assertIs(original_setter(None), None)
            self.assertEqual(replacement_module.get_default_backend(), "inductor")
        finally:
            sys.modules[module_name] = original_module
            torch.compiler = original_module

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        setter = compiler.set_default_backend
        reset = compiler.reset
        backend_annotation = str | Callable[..., typing.Any] | None

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        for function in (setter, reset):
            self.assertIs(type(function), types.FunctionType)
            self.assertEqual(function.__module__, "torch_rs.compiler")
            self.assertIs(inspect.getmodule(function), compiler)
            self.assertIsNone(function.__defaults__)
            self.assertIsNone(function.__kwdefaults__)
            self.assertEqual(function.__dict__, {})
            self.assertFalse(hasattr(function, "__text_signature__"))

        self.assertEqual(
            inspect.signature(setter),
            inspect.Signature(
                [
                    inspect.Parameter(
                        "backend",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=backend_annotation,
                    )
                ],
                return_annotation=None,
            ),
        )
        self.assertEqual(
            setter.__annotations__,
            {"backend": backend_annotation, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(setter),
            {"backend": backend_annotation, "return": type(None)},
        )
        self.assertEqual(setter.__name__, "set_default_backend")
        self.assertEqual(setter.__qualname__, "set_default_backend")
        self.assertEqual(
            inspect.cleandoc(setter.__doc__),
            inspect.cleandoc(SETTER_DOC),
        )

        self.assertEqual(str(inspect.signature(reset)), "() -> None")
        self.assertEqual(reset.__annotations__, {"return": None})
        self.assertEqual(typing.get_type_hints(reset), {"return": type(None)})
        self.assertEqual(reset.__name__, "reset")
        self.assertEqual(reset.__qualname__, "reset")
        self.assertEqual(
            inspect.cleandoc(reset.__doc__),
            inspect.cleandoc(RESET_DOC),
        )

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
                "disable",
                "set_default_backend",
                "get_default_backend",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
                "keep_portable_guards_unsafe",
                "skip_guard_on_all_nn_modules_unsafe",
                "skip_guard_on_globals_unsafe",
                "skip_all_guards_unsafe",
            ],
        )

        namespace = {}
        exec("from torch_rs.compiler import *", namespace)
        self.assertEqual(
            {name for name in namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
            self.assertIs(namespace[name], getattr(compiler, name))

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        for name in ("compiler", "reset", "set_default_backend"):
            self.assertNotIn(name, torch.__all__)
            self.assertNotIn(name, top_level_namespace)

        for function in (compiler.reset, compiler.set_default_backend):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(function=function.__name__, protocol=protocol):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.compiler", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_call_shape_errors_preserve_backend(self):
        setter = torch.compiler.set_default_backend
        reset = torch.compiler.reset
        backend = _CallableBackend()
        setter(backend)
        cases = (
            (
                lambda: setter(),
                "set_default_backend() missing 1 required positional argument: "
                "'backend'",
            ),
            (
                lambda: setter(None, None),
                "set_default_backend() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: setter(value=None),
                "set_default_backend() got an unexpected keyword argument 'value'",
            ),
            (
                lambda: setter(None, backend=None),
                "set_default_backend() got multiple values for argument 'backend'",
            ),
            (
                lambda: reset(None),
                "reset() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: reset(enabled=True),
                "reset() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))
                self.assertIs(torch.compiler.get_default_backend(), backend)

    def test_importing_reloading_and_calling_does_not_import_compilers(self):
        script = r'''
import importlib
import sys

class RejectCompilerImport:
    def find_spec(self, fullname, path=None, target=None):
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "torch_rs._dynamo"
            or fullname.startswith("torch_rs._dynamo.")
            or fullname.startswith("torch_rs.compiler.backends")
            or fullname == "torch_rs.compiler.registry"
        ):
            raise RuntimeError(f"compiler import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectCompilerImport())
import torch_rs as torch

compiler = torch.compiler
modules_before_call = set(sys.modules)
backend = lambda graph_module, example_inputs: graph_module.forward
assert compiler.set_default_backend(backend) is None
assert compiler.get_default_backend() is backend
assert compiler.reset() is None
assert compiler.get_default_backend() is backend
assert importlib.reload(compiler) is compiler
assert compiler.get_default_backend() is backend
old_getter = compiler.get_default_backend
old_setter = compiler.set_default_backend
del sys.modules["torch_rs.compiler"]
replacement = importlib.import_module("torch_rs.compiler")
assert replacement is not compiler
assert torch.compiler is replacement
assert replacement.get_default_backend() is backend
assert old_getter() is backend
assert replacement.set_default_backend(None) is None
assert old_getter() == "inductor"
assert old_setter("eager") is None
assert replacement.get_default_backend() == "eager"
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name == "torch_rs._dynamo"
    or name.startswith("torch_rs._dynamo.")
    or name.startswith("torch_rs.compiler.backends")
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
