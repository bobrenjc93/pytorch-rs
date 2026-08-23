import contextlib
import copy
import importlib
import inspect
import json
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ClearAutocastCacheReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "clear_autocast_cache differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def state_outcome(self, module):
        getter = module.is_autocast_cache_enabled
        setter = module.set_autocast_cache_enabled
        function = module.clear_autocast_cache
        previous = getter()
        outcomes = []
        try:
            for cache_enabled in (True, False):
                setter(cache_enabled)
                before = (getter(), module.is_grad_enabled())
                returned = function()
                total = (module.tensor([1.0, -2.0, 3.0]) * 2.0).sum().item()
                after = (getter(), module.is_grad_enabled())
                outcomes.append((before, returned is None, total, after))
                with module.no_grad():
                    before = (getter(), module.is_grad_enabled())
                    returned = function()
                    after = (getter(), module.is_grad_enabled())
                    outcomes.append((before, returned is None, after))

            setter(False)
            values = module.tensor([1.0, -2.0, 3.0], requires_grad=True)
            result = (values * 2.0).sum()
            returned = function()
            result.backward()
            outcomes.append(
                (
                    returned is None,
                    values.grad.tolist(),
                    getter(),
                    module.is_grad_enabled(),
                )
            )
            return outcomes
        finally:
            setter(previous)

    def thread_outcome(self, module):
        getter = module.is_autocast_cache_enabled
        setter = module.set_autocast_cache_enabled
        function = module.clear_autocast_cache
        previous = getter()
        setter(False)
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
                    contextlib.nullcontext() if expected_grad else module.no_grad()
                )
                with context:
                    initial_cache = getter()
                    setter(selected_cache)
                    before = (getter(), module.is_grad_enabled())
                    returned = function()
                    after = (getter(), module.is_grad_enabled())
                    ready.wait(timeout=10)
                    release.wait(timeout=10)
                    final = (getter(), module.is_grad_enabled())
                    results[index] = (
                        initial_cache,
                        before,
                        returned is None,
                        after,
                        final,
                    )
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        try:
            for thread in threads:
                thread.start()
            ready.wait(timeout=10)
            main_before = (getter(), module.is_grad_enabled())
            main_returned = function()
            setter(True)
            second_returned = function()
            main_after = (getter(), module.is_grad_enabled())
            release.wait(timeout=10)
            for thread in threads:
                thread.join(timeout=10)
            return (
                main_before,
                main_returned is None,
                second_returned is None,
                main_after,
                tuple(results),
                tuple(errors),
                tuple(thread.is_alive() for thread in threads),
            )
        finally:
            setter(previous)

    def subprocess_outcome(self, module_name):
        script = f"""
import importlib
import json

module = importlib.import_module({module_name!r})
function = module.clear_autocast_cache
states = [(module.is_autocast_cache_enabled(), module.is_grad_enabled())]
states.append(
    (
        function() is None,
        module.is_autocast_cache_enabled(),
        module.is_grad_enabled(),
    )
)
module.set_autocast_cache_enabled(False)
with module.no_grad():
    states.append(
        (
            function() is None,
            module.is_autocast_cache_enabled(),
            module.is_grad_enabled(),
        )
    )
states.append((module.is_autocast_cache_enabled(), module.is_grad_enabled()))
print(json.dumps(states))
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
        return json.loads(completed.stdout)

    def reload_outcome(self, module):
        getter = module.is_autocast_cache_enabled
        setter = module.set_autocast_cache_enabled
        function = module.clear_autocast_cache
        previous = getter()
        setter(False)
        native = module._C
        try:
            with module.no_grad():
                reloaded = importlib.reload(native)
                returned = function()
                return (
                    reloaded is native,
                    native.clear_autocast_cache is function,
                    module.clear_autocast_cache is function,
                    returned is None,
                    getter(),
                    module.is_grad_enabled(),
                )
        finally:
            setter(previous)

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, str(error)

    def test_return_value_state_and_execution_match_pytorch_2_13(self):
        self.assertEqual(self.state_outcome(torch), self.state_outcome(reference_torch))

    def test_thread_isolation_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_outcome(torch),
            self.thread_outcome(reference_torch),
        )

    def test_subprocess_defaults_and_isolation_match_pytorch_2_13(self):
        self.assertEqual(
            self.subprocess_outcome("torch_rs"),
            self.subprocess_outcome("torch"),
        )

    def test_native_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_builtin_metadata_exports_copying_and_pickling_match(self):
        actual = torch.clear_autocast_cache
        expected = reference_torch.clear_autocast_cache
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch"),
            expected.__module__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C.clear_autocast_cache, actual)
        self.assertIs(reference_torch._C.clear_autocast_cache, expected)
        self.assertEqual(actual.__reduce__(), expected.__reduce__())
        self.assertEqual(
            self.signature_outcome(actual),
            self.signature_outcome(expected),
        )

        for function in (actual, expected):
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    module=function.__module__,
                    protocol=protocol,
                ):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

        self.assertEqual(
            torch.__all__.count("clear_autocast_cache"),
            reference_torch.__all__.count("clear_autocast_cache"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["clear_autocast_cache"], function)

    def test_argument_errors_match_and_preserve_state(self):
        cases = (
            lambda function: function(None),
            lambda function: function(None, None),
            lambda function: function(enabled=True),
            lambda function: function(None, enabled=True),
        )
        actual_previous = torch.is_autocast_cache_enabled()
        expected_previous = reference_torch.is_autocast_cache_enabled()
        try:
            for cache_enabled in (True, False):
                torch.set_autocast_cache_enabled(cache_enabled)
                reference_torch.set_autocast_cache_enabled(cache_enabled)
                with torch.no_grad(), reference_torch.no_grad():
                    actual_before = (
                        torch.is_autocast_cache_enabled(),
                        torch.is_grad_enabled(),
                    )
                    expected_before = (
                        reference_torch.is_autocast_cache_enabled(),
                        reference_torch.is_grad_enabled(),
                    )
                    for case, call in enumerate(cases):
                        with self.subTest(cache_enabled=cache_enabled, case=case):
                            self.assert_error_matches(
                                lambda: call(torch.clear_autocast_cache),
                                lambda: call(reference_torch.clear_autocast_cache),
                            )
                            self.assertEqual(
                                (
                                    torch.is_autocast_cache_enabled(),
                                    torch.is_grad_enabled(),
                                ),
                                actual_before,
                            )
                            self.assertEqual(
                                (
                                    reference_torch.is_autocast_cache_enabled(),
                                    reference_torch.is_grad_enabled(),
                                ),
                                expected_before,
                            )

                    self.assertIs(torch.clear_autocast_cache(**{}), None)
                    self.assertIs(reference_torch.clear_autocast_cache(**{}), None)
        finally:
            torch.set_autocast_cache_enabled(actual_previous)
            reference_torch.set_autocast_cache_enabled(expected_previous)

    def test_autocast_execution_remains_outside_the_supported_surface(self):
        self.assertIs(torch.clear_autocast_cache, torch._C.clear_autocast_cache)
        self.assertIs(
            reference_torch.clear_autocast_cache,
            reference_torch._C.clear_autocast_cache,
        )
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertTrue(hasattr(reference_torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertTrue(hasattr(reference_torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))
        self.assertTrue(hasattr(reference_torch.cpu, "amp"))


if __name__ == "__main__":
    unittest.main()
