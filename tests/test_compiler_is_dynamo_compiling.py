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


FUNCTION_DOC = """
    Indicates whether a graph is traced via TorchDynamo.

    It's stricter than is_compiling() flag, as it would only be set to True when
    TorchDynamo is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_dynamo_compiling():
        >>>        pass # ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     # ...rest of the function...
    """


class CompilerIsDynamoCompilingTests(unittest.TestCase):
    def test_eager_false_is_exact_and_preserves_grad_mode(self):
        function = torch.compiler.is_dynamo_compiling

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

    def test_eager_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.compiler.is_dynamo_compiling
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
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    False,
                    expected_grad_state,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[3], False)

    def test_signature_annotations_documentation_and_module_identity(self):
        compiler = importlib.import_module("torch_rs.compiler")
        function = compiler.is_dynamo_compiling

        self.assertIs(torch.compiler, compiler)
        self.assertIs(sys.modules["torch_rs.compiler"], compiler)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_dynamo_compiling")
        self.assertEqual(function.__qualname__, "is_dynamo_compiling")
        self.assertEqual(function.__module__, "torch_rs.compiler")
        self.assertIs(inspect.getmodule(function), compiler)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        compiler = torch.compiler
        function = compiler.is_dynamo_compiling

        self.assertEqual(
            compiler.__all__,
            [
                "assume_constant_result",
                "reset",
                "allow_in_graph",
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
            ],
        )
        compiler_namespace = {}
        exec("from torch_rs.compiler import *", compiler_namespace)
        self.assertEqual(
            {name for name in compiler_namespace if not name.startswith("__")},
            {
                "assume_constant_result",
                "reset",
                "allow_in_graph",
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
            },
        )
        self.assertIs(
            compiler_namespace["assume_constant_result"],
            compiler.assume_constant_result,
        )
        self.assertIs(compiler_namespace["is_dynamo_compiling"], function)
        self.assertIs(compiler_namespace["is_exporting"], compiler.is_exporting)

        self.assertNotIn("compiler", torch.__all__)
        self.assertNotIn("assume_constant_result", torch.__all__)
        self.assertNotIn("is_dynamo_compiling", torch.__all__)
        self.assertNotIn("is_exporting", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("compiler", top_level_namespace)
        self.assertNotIn("assume_constant_result", top_level_namespace)
        self.assertNotIn("is_dynamo_compiling", top_level_namespace)
        self.assertNotIn("is_exporting", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.compiler", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiler.is_dynamo_compiling
        cases = (
            (
                lambda: function(None),
                "is_dynamo_compiling() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_dynamo_compiling() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_dynamo_compiling() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_dynamo_compiling() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_compilation_remains_unsupported(self):
        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch, "export"))
        self.assertFalse(hasattr(torch, "is_dynamo_compiling"))

    def test_importing_the_package_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

assert torch.compiler.is_dynamo_compiling() is False
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
