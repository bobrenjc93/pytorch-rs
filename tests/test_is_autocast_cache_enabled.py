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


class IsAutocastCacheEnabledTests(unittest.TestCase):
    def query_around_cpu_execution(self):
        before_grad = torch.is_grad_enabled()
        first = torch.is_autocast_cache_enabled()
        values = torch.relu(
            torch.tensor([[-1.0, 2.0], [3.0, -4.0]]) + torch.ones((2, 2))
        ).tolist()
        second = torch._C.is_autocast_cache_enabled()
        after_grad = torch.is_grad_enabled()
        return before_grad, first, values, second, after_grad

    def test_default_true_is_exact_across_grad_modes_and_cpu_execution(self):
        self.assertEqual(
            self.query_around_cpu_execution(),
            (True, True, [[0.0, 3.0], [4.0, 0.0]], True, True),
        )
        with torch.no_grad():
            self.assertEqual(
                self.query_around_cpu_execution(),
                (False, True, [[0.0, 3.0], [4.0, 0.0]], True, False),
            )
            with torch.no_grad():
                self.assertEqual(
                    self.query_around_cpu_execution(),
                    (False, True, [[0.0, 3.0], [4.0, 0.0]], True, False),
                )
        self.assertEqual(
            self.query_around_cpu_execution(),
            (True, True, [[0.0, 3.0], [4.0, 0.0]], True, True),
        )

    def test_default_true_is_stable_across_threads_and_grad_modes(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = self.query_around_cpu_execution()
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
                    [[0.0, 3.0], [4.0, 0.0]],
                    True,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], True)
            self.assertIs(result[3], True)

    def test_builtin_ownership_null_documentation_exports_and_pickling(self):
        function = torch.is_autocast_cache_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_autocast_cache_enabled")
        self.assertEqual(function.__qualname__, "is_autocast_cache_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__doc__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(
            repr(function),
            "<built-in function is_autocast_cache_enabled>",
        )
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.is_autocast_cache_enabled, function)
        assert_no_argument_signature(self, function, "()")

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertEqual(torch.__all__.count("is_autocast_cache_enabled"), 1)
        self.assertEqual(torch._C.__all__.count("is_autocast_cache_enabled"), 1)

        self.assertIs(importlib.import_module("torch_rs._C"), torch._C)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertIs(top_level_namespace["is_autocast_cache_enabled"], function)
        native_namespace = {}
        exec(
            "from torch_rs._C import is_autocast_cache_enabled",
            native_namespace,
        )
        self.assertIs(native_namespace["is_autocast_cache_enabled"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.is_autocast_cache_enabled
        cases = (
            (
                lambda: function(None),
                "torch.is_autocast_cache_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.is_autocast_cache_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.is_autocast_cache_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.is_autocast_cache_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), True)

    def test_autocast_mutation_and_execution_remain_unsupported(self):
        for owner in (torch, torch._C):
            with self.subTest(owner=owner.__name__):
                self.assertFalse(hasattr(owner, "set_autocast_cache_enabled"))

        for name in ("autocast", "is_autocast_enabled"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)

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

assert torch.is_autocast_cache_enabled is torch._C.is_autocast_cache_enabled
assert torch.is_autocast_cache_enabled() is True
values = torch.relu(torch.tensor([-1.0, 2.0]) + torch.ones(2)).tolist()
assert values == [0.0, 3.0]
assert torch.is_autocast_cache_enabled() is True
assert not hasattr(torch, "set_autocast_cache_enabled")
assert not hasattr(torch, "autocast")
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
