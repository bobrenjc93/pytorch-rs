from collections.abc import Callable
import contextlib
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


FUNCTION_DOC = '''Return the current default backend for ``torch.compile``.

    Returns:
        The current default backend (string or callable). Initially ``"inductor"``.
    '''


class CompilerGetDefaultBackendTests(unittest.TestCase):
    def setUp(self):
        self.original_backend = torch.compiler.get_default_backend()
        torch.compiler.set_default_backend(None)

    def tearDown(self):
        torch.compiler.set_default_backend(self.original_backend)

    def test_returns_exact_inductor_without_registry_lookups(self):
        function = torch.compiler.get_default_backend
        self.assertEqual(function.__code__.co_names, ("_state", "default_backend"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        for _ in range(4):
            result = function()
            self.assertIs(type(result), str)
            self.assertEqual(result, "inductor")

    def test_query_preserves_grad_mode(self):
        function = torch.compiler.get_default_backend

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertEqual(function(), "inductor")
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_inductor_is_stable_across_threads_and_grad_modes(self):
        function = torch.compiler.get_default_backend
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    first = function()
                    middle_grad_state = torch.is_grad_enabled()
                    second = function()
                    results[index] = (
                        torch.is_grad_enabled(),
                        type(first) is str,
                        first,
                        middle_grad_state,
                        type(second) is str,
                        second,
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
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    True,
                    "inductor",
                    expected_grad_state,
                    True,
                    "inductor",
                    expected_grad_state,
                ),
            )

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.get_default_backend
        return_annotation = str | Callable[..., typing.Any]

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            inspect.signature(function),
            inspect.Signature(return_annotation=return_annotation),
        )
        self.assertEqual(function.__annotations__, {"return": return_annotation})
        self.assertEqual(
            typing.get_type_hints(function),
            {"return": return_annotation},
        )
        self.assertEqual(function.__name__, "get_default_backend")
        self.assertEqual(function.__qualname__, "get_default_backend")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.get_default_backend

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
                "disable",
                "set_default_backend",
                "get_default_backend",
                "set_enable_guard_collectives",
                "cudagraph_mark_step_begin",
                "is_compiling",
                "is_dynamo_compiling",
                "is_exporting",
                "keep_portable_guards_unsafe",
                "skip_guard_on_inbuilt_nn_modules_unsafe",
                "skip_guard_on_all_nn_modules_unsafe",
                "keep_tensor_guards_unsafe",
                "skip_guard_on_globals_unsafe",
                "skip_all_guards_unsafe",
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            set(compiler.__all__),
        )
        for name in compiler.__all__:
            self.assertIs(compiler_namespace[name], getattr(compiler, name))

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("get_default_backend", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("get_default_backend", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiler.get_default_backend
        cases = (
            (
                lambda: function(None),
                "get_default_backend() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_default_backend() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(backend=None),
                "get_default_backend() got an unexpected keyword argument 'backend'",
            ),
            (
                lambda: function(None, backend=None),
                "get_default_backend() got an unexpected keyword argument 'backend'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_setter_and_reset_are_available_without_compilation(self):
        self.assertTrue(callable(torch.compiler.reset))
        self.assertTrue(callable(torch.compiler.set_default_backend))
        self.assertIn("reset", torch.compiler.__all__)
        self.assertIn("set_default_backend", torch.compiler.__all__)
        self.assertFalse(hasattr(torch.compiler, "compile"))
        self.assertFalse(hasattr(torch, "compile"))

    def test_importing_and_calling_does_not_import_pytorch_or_a_registry(self):
        script = r'''
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

modules_before_call = set(sys.modules)
assert torch.compiler.get_default_backend() == "inductor"
backend = "".join(("ea", "ger"))
assert torch.compiler.set_default_backend(backend) is None
assert torch.compiler.get_default_backend() is backend
assert torch.compiler.reset() is None
assert torch.compiler.get_default_backend() is backend
assert torch.compiler.set_default_backend(None) is None
assert torch.compiler.get_default_backend() == "inductor"
assert set(sys.modules) == modules_before_call
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
assert not any(
    name.startswith("torch_rs._dynamo")
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
