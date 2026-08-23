import contextlib
import copy
import importlib
import inspect
import pickle
import threading
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("set_autocast_cache_enabled must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AutocastCacheEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "autocast-cache state differentials require pinned PyTorch 2.13.0"
            )

    def setUp(self):
        self.original_actual = torch.is_autocast_cache_enabled()
        self.original_expected = reference_torch.is_autocast_cache_enabled()
        torch.set_autocast_cache_enabled(True)
        reference_torch.set_autocast_cache_enabled(True)

    def tearDown(self):
        torch.set_autocast_cache_enabled(self.original_actual)
        reference_torch.set_autocast_cache_enabled(self.original_expected)

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
        outcomes = []
        for state in (False, True, False):
            before_grad = module.is_grad_enabled()
            result = setter(state)
            values = module.tensor([1.0, -2.0, 3.0])
            outcomes.append(
                (
                    result is None,
                    getter(),
                    (values * 2.0).sum().item(),
                    module.is_grad_enabled(),
                    before_grad,
                )
            )
            with module.no_grad():
                outcomes.append(
                    (
                        setter(not state) is None,
                        getter(),
                        module.is_grad_enabled(),
                    )
                )
            outcomes.append((getter(), module.is_grad_enabled()))
        return outcomes

    def thread_outcome(self, module):
        getter = module.is_autocast_cache_enabled
        setter = module.set_autocast_cache_enabled
        previous = getter()
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        try:
            setter(False)

            def worker(index):
                try:
                    target = index % 2 == 0
                    context = (
                        module.no_grad()
                        if index % 3 == 0
                        else contextlib.nullcontext()
                    )
                    with context:
                        before = getter()
                        result = setter(target)
                        barrier.wait(timeout=10)
                        values = module.tensor([float(index), 1.0])
                        results[index] = (
                            before,
                            result is None,
                            getter(),
                            module.is_grad_enabled(),
                            (values + 1.0).sum().item(),
                        )
                except BaseException as error:
                    errors.append((type(error).__name__, str(error)))

            threads = [
                threading.Thread(target=worker, args=(index,))
                for index in range(worker_count)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            alive = tuple(thread.is_alive() for thread in threads)
            return alive, errors, results, getter()
        finally:
            setter(previous)

    def reload_outcome(self, module):
        native = module._C
        old_getter = module.is_autocast_cache_enabled
        old_setter = module.set_autocast_cache_enabled
        previous = old_getter()
        try:
            old_setter(False)
            reloaded = importlib.reload(native)
            after_reload = old_getter()
            new_getter = native.is_autocast_cache_enabled
            new_setter = native.set_autocast_cache_enabled
            new_setter(True)
            old_observes_new = old_getter()
            old_setter(False)
            new_observes_old = new_getter()
            return (
                reloaded is native,
                new_getter is old_getter,
                new_setter is old_setter,
                after_reload,
                old_observes_new,
                new_observes_old,
            )
        finally:
            old_setter(previous)

    def test_state_updates_and_cpu_execution_match_pytorch_2_13(self):
        self.assertEqual(self.state_outcome(torch), self.state_outcome(reference_torch))

    def test_thread_local_defaults_and_isolation_match_pytorch_2_13(self):
        self.assertEqual(
            self.thread_outcome(torch),
            self.thread_outcome(reference_torch),
        )

    def test_same_thread_native_reload_persistence_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_strict_bool_errors_are_non_mutating_and_match_pytorch_2_13(self):
        invalid_values = (
            (None, None),
            (0, 0),
            (1, 1),
            (0.0, 0.0),
            ("", ""),
            ([], []),
            (object(), object()),
            (_RejectTruthiness(), _RejectTruthiness()),
            (torch.tensor(True), reference_torch.tensor(True)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
        )

        for state in (False, True):
            for actual_value, expected_value in invalid_values:
                torch.set_autocast_cache_enabled(state)
                reference_torch.set_autocast_cache_enabled(state)
                with self.subTest(state=state, value=type(actual_value).__name__):
                    self.assert_error_matches(
                        lambda: torch.set_autocast_cache_enabled(actual_value),
                        lambda: reference_torch.set_autocast_cache_enabled(
                            expected_value
                        ),
                    )
                    self.assertIs(torch.is_autocast_cache_enabled(), state)
                    self.assertIs(
                        reference_torch.is_autocast_cache_enabled(), state
                    )

    def test_builtin_contracts_match_pytorch_2_13(self):
        for name, expected_signature in (
            ("is_autocast_cache_enabled", "()"),
            ("set_autocast_cache_enabled", "(object, /)"),
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

                for function in (actual, expected):
                    if function.__text_signature__ is None:
                        with self.assertRaises(ValueError):
                            inspect.signature(function)
                    else:
                        self.assertEqual(
                            str(inspect.signature(function)), expected_signature
                        )
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                        with self.subTest(
                            name=name,
                            function=function,
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
                    namespace = {}
                    exec(f"from {module.__name__} import *", namespace)
                    self.assertIs(namespace[name], function)

    def test_getter_argument_errors_match_pytorch_2_13(self):
        cases = (
            lambda function: function(None),
            lambda function: function(None, None),
            lambda function: function(enabled=True),
            lambda function: function(None, enabled=True),
        )
        for case, call in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(torch.is_autocast_cache_enabled),
                    lambda: call(reference_torch.is_autocast_cache_enabled),
                )

        self.assertIs(torch.is_autocast_cache_enabled(**{}), True)
        self.assertIs(reference_torch.is_autocast_cache_enabled(**{}), True)

    def test_setter_binding_errors_match_and_do_not_mutate(self):
        cases = (
            lambda function: function(),
            lambda function: function(False, True),
            lambda function: function(enabled=True),
            lambda function: function(True, enabled=False),
        )
        for state in (False, True):
            torch.set_autocast_cache_enabled(state)
            reference_torch.set_autocast_cache_enabled(state)
            for case, call in enumerate(cases):
                with self.subTest(state=state, case=case):
                    self.assert_error_matches(
                        lambda: call(torch.set_autocast_cache_enabled),
                        lambda: call(reference_torch.set_autocast_cache_enabled),
                    )
                    self.assertIs(torch.is_autocast_cache_enabled(), state)
                    self.assertIs(
                        reference_torch.is_autocast_cache_enabled(), state
                    )

        self.assertIs(torch.set_autocast_cache_enabled(False, **{}), None)
        self.assertIs(
            reference_torch.set_autocast_cache_enabled(False, **{}), None
        )

    def assert_reference_autocast_context_is_not_emulated(self, device_type):
        torch.set_autocast_cache_enabled(True)
        reference_torch.set_autocast_cache_enabled(True)
        with reference_torch.autocast(
            device_type=device_type,
            enabled=True,
            cache_enabled=False,
        ):
            self.assertIs(torch.is_autocast_cache_enabled(), True)
            self.assertIs(reference_torch.is_autocast_cache_enabled(), False)
        self.assertIs(torch.is_autocast_cache_enabled(), True)
        self.assertIs(reference_torch.is_autocast_cache_enabled(), True)

    def test_reference_cpu_autocast_context_remains_unsupported(self):
        self.assert_reference_autocast_context_is_not_emulated("cpu")

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "PyTorch CUDA is required for the CUDA autocast boundary",
    )
    def test_reference_cuda_autocast_context_remains_unsupported(self):
        self.assert_reference_autocast_context_is_not_emulated("cuda")

    def test_context_and_mixed_precision_surfaces_stay_deliberately_absent(self):
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
