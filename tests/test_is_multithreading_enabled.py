import contextlib
import copy
import importlib
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


FUNCTION_DOC = "Returns True if multithreading is currently enabled."


class IsMultithreadingEnabledTests(unittest.TestCase):
    def test_true_is_exact_and_does_not_change_grad_mode(self):
        function = torch.autograd.is_multithreading_enabled

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), True)
            self.assertIs(torch._C._is_multithreading_enabled(), True)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_true_is_stable_across_threads_and_grad_modes(self):
        function = torch.autograd.is_multithreading_enabled
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
                        torch._C._is_multithreading_enabled(),
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
                (expected_grad_state, True, True, expected_grad_state),
            )
            self.assertIs(result[1], True)
            self.assertIs(result[2], True)

    def test_builtin_metadata_imports_copying_and_pickling(self):
        function = torch.autograd.is_multithreading_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "_is_multithreading_enabled")
        self.assertEqual(function.__qualname__, "_is_multithreading_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(
            repr(function),
            "<built-in function _is_multithreading_enabled>",
        )
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C._is_multithreading_enabled, function)
        assert_no_argument_signature(self, function, "()")

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertIs(importlib.import_module("torch_rs.autograd"), torch.autograd)
        self.assertIs(importlib.import_module("torch_rs._C"), torch._C)
        explicit_namespace = {}
        exec(
            "from torch_rs.autograd import is_multithreading_enabled",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["is_multithreading_enabled"], function)
        native_namespace = {}
        exec(
            "from torch_rs._C import _is_multithreading_enabled",
            native_namespace,
        )
        self.assertIs(native_namespace["_is_multithreading_enabled"], function)

    def test_query_is_excluded_from_wildcard_and_top_level_exports(self):
        function = torch.autograd.is_multithreading_enabled
        self.assertNotIn("is_multithreading_enabled", torch.autograd.__all__)
        self.assertNotIn("_is_multithreading_enabled", torch._C.__all__)
        self.assertNotIn("is_multithreading_enabled", torch.__all__)
        self.assertNotIn("_is_multithreading_enabled", torch.__all__)
        self.assertFalse(hasattr(torch, "is_multithreading_enabled"))
        self.assertFalse(hasattr(torch, "_is_multithreading_enabled"))

        autograd_namespace = {}
        exec("from torch_rs.autograd import *", autograd_namespace)
        self.assertNotIn("is_multithreading_enabled", autograd_namespace)
        self.assertNotIn("_is_multithreading_enabled", autograd_namespace)

        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("is_multithreading_enabled", top_level_namespace)
        self.assertNotIn("_is_multithreading_enabled", top_level_namespace)
        self.assertIs(torch.autograd.is_multithreading_enabled, function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.autograd.is_multithreading_enabled
        cases = (
            (
                lambda: function(None),
                "torch._C._is_multithreading_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch._C._is_multithreading_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch._C._is_multithreading_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch._C._is_multithreading_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), True)

    def test_multithreading_mutation_remains_unsupported(self):
        self.assertFalse(hasattr(torch._C, "_set_multithreading_enabled"))
        self.assertNotIn("_set_multithreading_enabled", torch._C.__all__)
        self.assertFalse(hasattr(torch.autograd, "set_multithreading_enabled"))
        self.assertFalse(
            hasattr(torch.autograd.grad_mode, "set_multithreading_enabled")
        )

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs.autograd import is_multithreading_enabled

assert is_multithreading_enabled is torch._C._is_multithreading_enabled
assert is_multithreading_enabled() is True
assert not hasattr(torch._C, "_set_multithreading_enabled")
assert not hasattr(torch.autograd, "set_multithreading_enabled")
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
