import contextlib
import copy
import importlib
import inspect
import pickle
import threading
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsAutocastCacheEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "autocast cache state differentials require pinned PyTorch 2.13.0"
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
        previous = getter()
        try:
            states = [(getter(), module.is_grad_enabled())]
            for enabled in (False, True, False):
                returned = setter(enabled)
                values = module.tensor([1.0, -2.0, 3.0])
                states.append(
                    (
                        returned is None,
                        getter(),
                        (values * 2.0).sum().item(),
                        module.is_grad_enabled(),
                    )
                )
            with module.no_grad():
                states.append((getter(), module.is_grad_enabled()))
                returned = setter(True)
                states.append((returned is None, getter(), module.is_grad_enabled()))
            states.append((getter(), module.is_grad_enabled()))
            return states
        finally:
            setter(previous)

    def thread_outcome(self, module):
        getter = module.is_autocast_cache_enabled
        setter = module.set_autocast_cache_enabled
        previous = getter()
        setter(False)
        worker_count = 8
        ready = threading.Barrier(worker_count + 1)
        release = threading.Barrier(worker_count + 1)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    initial = getter()
                    selected = index % 3 != 0
                    returned = setter(selected)
                    after_set = getter()
                    ready.wait(timeout=10)
                    release.wait(timeout=10)
                    results[index] = (
                        initial,
                        returned is None,
                        after_set,
                        getter(),
                        module.is_grad_enabled(),
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
            main_before = getter()
            setter(True)
            main_after = getter()
            release.wait(timeout=10)
            for thread in threads:
                thread.join(timeout=10)
            return (
                main_before,
                main_after,
                tuple(results),
                tuple(errors),
                tuple(thread.is_alive() for thread in threads),
            )
        finally:
            setter(previous)

    def reload_outcome(self, module):
        getter = module.is_autocast_cache_enabled
        setter = module.set_autocast_cache_enabled
        previous = getter()
        setter(False)
        native = module._C
        old_getter = native.is_autocast_cache_enabled
        old_setter = native.set_autocast_cache_enabled
        try:
            reloaded = importlib.reload(native)
            return (
                reloaded is native,
                native.is_autocast_cache_enabled is old_getter,
                native.set_autocast_cache_enabled is old_setter,
                getter(),
            )
        finally:
            setter(previous)

    def signature_outcome(self, function):
        try:
            return "return", str(inspect.signature(function))
        except BaseException as error:
            return "error", type(error).__name__, str(error)

    def test_default_mutation_grad_and_cpu_states_match_pytorch_2_13(self):
        self.assertIs(torch.is_autocast_cache_enabled(), True)
        self.assertIs(reference_torch.is_autocast_cache_enabled(), True)
        self.assertEqual(
            self.state_outcome(torch),
            self.state_outcome(reference_torch),
        )

    def test_thread_local_defaults_and_isolation_match_pytorch_2_13(self):
        self.assertEqual(
            self.thread_outcome(torch),
            self.thread_outcome(reference_torch),
        )

    def test_native_reload_persistence_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_builtin_metadata_exports_copying_and_pickling_match(self):
        for name in (
            "is_autocast_cache_enabled",
            "set_autocast_cache_enabled",
        ):
            actual = getattr(torch, name)
            expected = getattr(reference_torch, name)
            with self.subTest(name=name):
                self.assertIs(type(actual), types.BuiltinFunctionType)
                self.assertIs(type(expected), types.BuiltinFunctionType)
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs.torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(
                    actual.__text_signature__, expected.__text_signature__
                )
                self.assertEqual(
                    hasattr(actual, "__annotations__"),
                    hasattr(expected, "__annotations__"),
                )
                self.assertEqual(repr(actual), repr(expected))
                self.assertIs(actual.__self__, torch._C)
                self.assertIs(expected.__self__, reference_torch._C)
                self.assertIs(getattr(torch._C, name), actual)
                self.assertIs(getattr(reference_torch._C, name), expected)
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
                            name=name,
                            module=function.__module__,
                            protocol=protocol,
                        ):
                            restored = pickle.loads(
                                pickle.dumps(function, protocol=protocol)
                            )
                            self.assertIs(restored, function)

                self.assertEqual(
                    torch.__all__.count(name),
                    reference_torch.__all__.count(name),
                )
                for module, function in (
                    (torch, actual),
                    (reference_torch, expected),
                ):
                    wildcard_namespace = {}
                    exec(f"from {module.__name__} import *", wildcard_namespace)
                    self.assertIs(wildcard_namespace[name], function)

    def test_getter_and_setter_argument_errors_match_without_mutation(self):
        getter_cases = (
            lambda function: function(None),
            lambda function: function(None, None),
            lambda function: function(enabled=True),
            lambda function: function(None, enabled=True),
        )
        for case, call in enumerate(getter_cases):
            with self.subTest(api="getter", case=case):
                self.assert_error_matches(
                    lambda: call(torch.is_autocast_cache_enabled),
                    lambda: call(reference_torch.is_autocast_cache_enabled),
                )

        class BoolLike:
            calls = 0

            def __bool__(self):
                type(self).calls += 1
                return True

        setter_cases = (
            lambda function: function(),
            lambda function: function(True, False),
            lambda function: function(enabled=True),
            lambda function: function(True, enabled=False),
            lambda function: function(1),
            lambda function: function(0),
            lambda function: function(None),
            lambda function: function("true"),
            lambda function: function([]),
            lambda function: function(np.bool_(True)),
            lambda function: function(np.int64(1)),
            lambda function: function(BoolLike()),
        )
        actual_previous = torch.is_autocast_cache_enabled()
        expected_previous = reference_torch.is_autocast_cache_enabled()
        try:
            for initial in (True, False):
                for case, call in enumerate(setter_cases):
                    with self.subTest(api="setter", initial=initial, case=case):
                        torch.set_autocast_cache_enabled(initial)
                        reference_torch.set_autocast_cache_enabled(initial)
                        self.assert_error_matches(
                            lambda: call(torch.set_autocast_cache_enabled),
                            lambda: call(
                                reference_torch.set_autocast_cache_enabled
                            ),
                        )
                        self.assertIs(torch.is_autocast_cache_enabled(), initial)
                        self.assertIs(
                            reference_torch.is_autocast_cache_enabled(), initial
                        )
        finally:
            torch.set_autocast_cache_enabled(actual_previous)
            reference_torch.set_autocast_cache_enabled(expected_previous)

        self.assertEqual(BoolLike.calls, 0)
        self.assertIs(
            torch.is_autocast_cache_enabled(**{}),
            reference_torch.is_autocast_cache_enabled(**{}),
        )
        actual_previous = torch.is_autocast_cache_enabled()
        expected_previous = reference_torch.is_autocast_cache_enabled()
        try:
            self.assertIs(torch.set_autocast_cache_enabled(False, **{}), None)
            self.assertIs(
                reference_torch.set_autocast_cache_enabled(False, **{}), None
            )
            self.assertIs(torch.is_autocast_cache_enabled(), False)
            self.assertIs(reference_torch.is_autocast_cache_enabled(), False)
        finally:
            torch.set_autocast_cache_enabled(actual_previous)
            reference_torch.set_autocast_cache_enabled(expected_previous)

    def test_only_state_control_is_added_while_autocast_stays_unsupported(self):
        self.assertTrue(hasattr(torch, "set_autocast_cache_enabled"))
        self.assertTrue(hasattr(reference_torch, "set_autocast_cache_enabled"))
        self.assertTrue(hasattr(torch._C, "set_autocast_cache_enabled"))
        self.assertTrue(hasattr(reference_torch._C, "set_autocast_cache_enabled"))
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertTrue(hasattr(reference_torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertTrue(hasattr(reference_torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))
        self.assertTrue(hasattr(reference_torch.cpu, "amp"))


if __name__ == "__main__":
    unittest.main()
