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


class ClearAutocastCacheTests(unittest.TestCase):
    def preserve_cache_state(self):
        previous = torch.is_autocast_cache_enabled()
        self.addCleanup(torch.set_autocast_cache_enabled, previous)
        return previous

    def test_returns_none_and_preserves_cache_grad_and_execution(self):
        self.preserve_cache_state()
        function = torch.clear_autocast_cache

        def assert_preserved(expected_cache, expected_grad):
            before = (
                torch.is_autocast_cache_enabled(),
                torch.is_grad_enabled(),
            )
            returned = function()
            total = (torch.tensor([1.0, -2.0, 3.0]) * 2.0).sum().item()
            after = (
                torch.is_autocast_cache_enabled(),
                torch.is_grad_enabled(),
            )
            self.assertEqual(before, (expected_cache, expected_grad))
            self.assertIs(returned, None)
            self.assertEqual(total, 4.0)
            self.assertEqual(after, before)

        for cache_enabled in (True, False):
            with self.subTest(cache_enabled=cache_enabled, grad_enabled=True):
                torch.set_autocast_cache_enabled(cache_enabled)
                assert_preserved(cache_enabled, True)
            with self.subTest(cache_enabled=cache_enabled, grad_enabled=False):
                with torch.no_grad():
                    assert_preserved(cache_enabled, False)
                self.assertIs(torch.is_grad_enabled(), True)

        torch.set_autocast_cache_enabled(False)
        values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        result = (values * 2.0).sum()
        self.assertIs(function(), None)
        result.backward()
        self.assertEqual(values.grad.tolist(), [2.0, 2.0, 2.0])
        self.assertIs(torch.is_autocast_cache_enabled(), False)
        self.assertIs(torch.is_grad_enabled(), True)

    def test_thread_local_states_remain_isolated_while_clearing(self):
        self.preserve_cache_state()
        torch.set_autocast_cache_enabled(False)

        worker_count = 8
        ready = threading.Barrier(worker_count + 1)
        release = threading.Barrier(worker_count + 1)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                expected_grad = index % 2 == 0
                selected_cache = index % 3 != 0
                context = (
                    contextlib.nullcontext() if expected_grad else torch.no_grad()
                )
                with context:
                    initial_cache = torch.is_autocast_cache_enabled()
                    torch.set_autocast_cache_enabled(selected_cache)
                    before = (
                        torch.is_autocast_cache_enabled(),
                        torch.is_grad_enabled(),
                    )
                    returned = torch.clear_autocast_cache()
                    after = (
                        torch.is_autocast_cache_enabled(),
                        torch.is_grad_enabled(),
                    )
                    ready.wait(timeout=10)
                    release.wait(timeout=10)
                    final = (
                        torch.is_autocast_cache_enabled(),
                        torch.is_grad_enabled(),
                    )
                    results[index] = (initial_cache, before, returned, after, final)
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()

        ready.wait(timeout=10)
        self.assertEqual(
            (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()),
            (False, True),
        )
        self.assertIs(torch.clear_autocast_cache(), None)
        self.assertIs(torch.is_autocast_cache_enabled(), False)
        torch.set_autocast_cache_enabled(True)
        self.assertIs(torch.clear_autocast_cache(), None)
        release.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(
            (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()),
            (True, True),
        )
        for index, result in enumerate(results):
            expected_grad = index % 2 == 0
            selected_cache = index % 3 != 0
            expected_state = (selected_cache, expected_grad)
            self.assertEqual(
                result,
                (True, expected_state, None, expected_state, expected_state),
            )

    def test_subprocess_has_independent_default_states(self):
        self.preserve_cache_state()
        torch.set_autocast_cache_enabled(False)
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.clear_autocast_cache
assert function is torch._C.clear_autocast_cache
assert (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()) == (True, True)
assert function() is None
assert (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()) == (True, True)
torch.set_autocast_cache_enabled(False)
with torch.no_grad():
    assert function() is None
    assert (
        torch.is_autocast_cache_enabled(),
        torch.is_grad_enabled(),
    ) == (False, False)
assert (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()) == (False, True)
assert not hasattr(torch, "autocast")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""

        with torch.no_grad():
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()),
                (False, False),
            )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(
            (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()),
            (False, True),
        )

    def test_native_and_package_reload_preserve_identity_and_state(self):
        self.preserve_cache_state()
        torch.set_autocast_cache_enabled(False)
        function = torch.clear_autocast_cache
        native = torch._C

        with torch.no_grad():
            self.assertIs(importlib.reload(native), native)
            self.assertIs(native.clear_autocast_cache, function)
            self.assertIs(torch.clear_autocast_cache, function)
            self.assertIs(function(), None)
            self.assertEqual(
                (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()),
                (False, False),
            )

        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch._C, native)
        self.assertIs(torch.clear_autocast_cache, function)
        self.assertIs(torch._C.clear_autocast_cache, function)
        self.assertEqual(
            (torch.is_autocast_cache_enabled(), torch.is_grad_enabled()),
            (False, True),
        )

    def test_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.clear_autocast_cache
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "clear_autocast_cache")
        self.assertEqual(function.__qualname__, "clear_autocast_cache")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__doc__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(repr(function), "<built-in function clear_autocast_cache>")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.clear_autocast_cache, function)
        self.assertEqual(function.__reduce__(), "clear_autocast_cache")
        assert_no_argument_signature(self, function, "()")

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertEqual(torch.__all__.count("clear_autocast_cache"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["clear_autocast_cache"], function)

        native_module = importlib.import_module("torch_rs._C")
        self.assertIs(native_module, torch._C)
        explicit_namespace = {}
        exec(
            "from torch_rs._C import clear_autocast_cache",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["clear_autocast_cache"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_argument_errors_are_exact_and_preserve_state(self):
        self.preserve_cache_state()
        function = torch.clear_autocast_cache
        cases = (
            (
                lambda: function(None),
                "torch.clear_autocast_cache() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.clear_autocast_cache() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.clear_autocast_cache() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.clear_autocast_cache() takes no keyword arguments",
            ),
        )

        for cache_enabled in (True, False):
            torch.set_autocast_cache_enabled(cache_enabled)
            for grad_enabled in (True, False):
                context = (
                    contextlib.nullcontext() if grad_enabled else torch.no_grad()
                )
                with context:
                    before = (
                        torch.is_autocast_cache_enabled(),
                        torch.is_grad_enabled(),
                    )
                    for call, message in cases:
                        with self.subTest(
                            cache_enabled=cache_enabled,
                            grad_enabled=grad_enabled,
                            message=message,
                        ):
                            with self.assertRaises(TypeError) as raised:
                                call()
                            self.assertEqual(str(raised.exception), message)
                            self.assertEqual(raised.exception.args, (message,))
                            self.assertEqual(
                                (
                                    torch.is_autocast_cache_enabled(),
                                    torch.is_grad_enabled(),
                                ),
                                before,
                            )
                    self.assertIs(function(**{}), None)
                    self.assertEqual(
                        (
                            torch.is_autocast_cache_enabled(),
                            torch.is_grad_enabled(),
                        ),
                        before,
                    )

    def test_autocast_contexts_and_mixed_precision_remain_unsupported(self):
        self.assertIs(torch.clear_autocast_cache, torch._C.clear_autocast_cache)
        self.assertIn("clear_autocast_cache", torch.__all__)
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))


if __name__ == "__main__":
    unittest.main()
