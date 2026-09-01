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


FUNCTION_DOC = """Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """


class IsDeterministicAlgorithmsWarnOnlyEnabledTests(unittest.TestCase):
    def test_default_false_is_exact_and_preserves_grad_mode(self):
        function = torch.is_deterministic_algorithms_warn_only_enabled
        self.assertEqual(function.__code__.co_names, ())
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
        function = torch.is_deterministic_algorithms_warn_only_enabled
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
        function = package.is_deterministic_algorithms_warn_only_enabled

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(
            function.__name__,
            "is_deterministic_algorithms_warn_only_enabled",
        )
        self.assertEqual(
            function.__qualname__,
            "is_deterministic_algorithms_warn_only_enabled",
        )
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
        function = torch.is_deterministic_algorithms_warn_only_enabled

        self.assertEqual(
            torch.__all__.count("is_deterministic_algorithms_warn_only_enabled"),
            1,
        )
        namespace = {}
        exec("from torch_rs import *", namespace)
        self.assertIs(
            namespace["is_deterministic_algorithms_warn_only_enabled"],
            function,
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.is_deterministic_algorithms_warn_only_enabled
        cases = (
            (
                lambda: function(None),
                "is_deterministic_algorithms_warn_only_enabled() takes 0 "
                "positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_deterministic_algorithms_warn_only_enabled() takes 0 "
                "positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_deterministic_algorithms_warn_only_enabled() got an "
                "unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_deterministic_algorithms_warn_only_enabled() got an "
                "unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_default_only_deterministic_algorithm_apis_are_exposed(self):
        self.assertTrue(hasattr(torch, "use_deterministic_algorithms"))
        self.assertEqual(torch.__all__.count("use_deterministic_algorithms"), 1)
        self.assertTrue(hasattr(torch, "set_deterministic_debug_mode"))
        self.assertEqual(torch.__all__.count("set_deterministic_debug_mode"), 1)

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

assert torch.is_deterministic_algorithms_warn_only_enabled() is False
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
