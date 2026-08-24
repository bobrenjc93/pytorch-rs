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


FUNCTION_DOC = """Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """


class AreDeterministicAlgorithmsEnabledTests(unittest.TestCase):
    def setUp(self):
        self.original_enabled = torch.are_deterministic_algorithms_enabled()
        self.original_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        torch.use_deterministic_algorithms(False)

    def tearDown(self):
        torch.use_deterministic_algorithms(
            self.original_enabled,
            warn_only=self.original_warn_only,
        )

    def test_default_false_is_exact_and_preserves_grad_mode(self):
        function = torch.are_deterministic_algorithms_enabled
        self.assertEqual(
            function.__code__.co_names,
            ("_C", "_get_deterministic_algorithms"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

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

    def test_default_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.are_deterministic_algorithms_enabled
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
        package = importlib.import_module("torch_rs")
        function = package.are_deterministic_algorithms_enabled

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "are_deterministic_algorithms_enabled")
        self.assertEqual(function.__qualname__, "are_deterministic_algorithms_enabled")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_the_canonical_module(self):
        function = torch.are_deterministic_algorithms_enabled

        self.assertEqual(
            torch.__all__.count("are_deterministic_algorithms_enabled"),
            1,
        )
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(namespace["are_deterministic_algorithms_enabled"], function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.are_deterministic_algorithms_enabled
        cases = (
            (
                lambda: function(None),
                "are_deterministic_algorithms_enabled() takes 0 positional "
                "arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "are_deterministic_algorithms_enabled() takes 0 positional "
                "arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "are_deterministic_algorithms_enabled() got an unexpected "
                "keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "are_deterministic_algorithms_enabled() got an unexpected "
                "keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_deterministic_configuration_surface(self):
        self.assertTrue(hasattr(torch, "use_deterministic_algorithms"))
        self.assertEqual(torch.__all__.count("use_deterministic_algorithms"), 1)
        self.assertFalse(hasattr(torch, "set_deterministic_debug_mode"))
        self.assertNotIn("set_deterministic_debug_mode", torch.__all__)
        names = (
            "_set_deterministic_algorithms",
            "_get_deterministic_algorithms",
            "_get_deterministic_algorithms_warn_only",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(torch._C, name))
                self.assertNotIn(name, torch._C.__all__)

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

assert torch.are_deterministic_algorithms_enabled() is False
assert torch.use_deterministic_algorithms(True) is None
assert torch.are_deterministic_algorithms_enabled() is True
assert torch.use_deterministic_algorithms(False) is None
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
