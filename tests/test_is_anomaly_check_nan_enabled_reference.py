import contextlib
import copy
import pickle
import threading
import types
import unittest

import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class IsAnomalyCheckNanEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "is_anomaly_check_nan_enabled differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def supported_state_outcome(self, module):
        function = module.is_anomaly_check_nan_enabled

        def query_outcome():
            before = module.is_grad_enabled()
            first = function()
            middle = module.is_grad_enabled()
            second = function()
            after = module.is_grad_enabled()
            return (
                before,
                first is True,
                middle,
                second is True,
                after,
            )

        states = [query_outcome()]
        with module.no_grad():
            states.append(query_outcome())
            with module.no_grad():
                states.append(query_outcome())
            states.append(query_outcome())
        states.append(query_outcome())

        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = query_outcome()
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

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        return states, worker_states

    def test_supported_default_and_grad_states_match_pytorch_2_13(self):
        self.assertEqual(
            self.supported_state_outcome(torch),
            self.supported_state_outcome(reference_torch),
        )

    def test_reference_only_mutation_bounds_the_unsupported_false_state(self):
        actual_function = torch.is_anomaly_check_nan_enabled
        expected_function = reference_torch.is_anomaly_check_nan_enabled

        actual_states = [actual_function()]
        expected_states = [expected_function()]
        expected_anomaly_states = [reference_torch.is_anomaly_enabled()]
        with reference_torch.autograd.set_detect_anomaly(True, check_nan=False):
            actual_states.append(actual_function())
            expected_states.append(expected_function())
            expected_anomaly_states.append(reference_torch.is_anomaly_enabled())
            self.assertIs(torch.is_grad_enabled(), True)
            self.assertIs(reference_torch.is_grad_enabled(), True)
        actual_states.append(actual_function())
        expected_states.append(expected_function())
        expected_anomaly_states.append(reference_torch.is_anomaly_enabled())

        for state in actual_states:
            self.assertIs(state, True)
        self.assertEqual(expected_states, [True, False, True])
        self.assertEqual(expected_anomaly_states, [False, True, False])
        self.assertIs(torch.is_grad_enabled(), True)
        self.assertIs(reference_torch.is_grad_enabled(), True)

    def test_builtin_contract_matches_pytorch_2_13(self):
        actual = torch.is_anomaly_check_nan_enabled
        expected = reference_torch.is_anomaly_check_nan_enabled

        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs.torch_rs", "torch"),
            expected.__module__,
        )
        self.assertIsNone(actual.__doc__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(
            hasattr(actual, "__annotations__"),
            hasattr(expected, "__annotations__"),
        )
        self.assertFalse(hasattr(actual, "__annotations__"))
        self.assertEqual(repr(actual), repr(expected))
        self.assertIs(actual.__self__, torch._C)
        self.assertIs(expected.__self__, reference_torch._C)
        self.assertIs(torch._C.is_anomaly_check_nan_enabled, actual)
        self.assertIs(reference_torch._C.is_anomaly_check_nan_enabled, expected)
        for function in (actual, expected):
            assert_no_argument_signature(self, function, "()")
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)

        self.assertEqual(
            torch.__all__.count("is_anomaly_check_nan_enabled"),
            reference_torch.__all__.count("is_anomaly_check_nan_enabled"),
        )
        for module, function in ((torch, actual), (reference_torch, expected)):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(
                wildcard_namespace["is_anomaly_check_nan_enabled"], function
            )
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                    self.assertIs(restored, function)

    def test_no_argument_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.is_anomaly_check_nan_enabled(None),
                lambda: reference_torch.is_anomaly_check_nan_enabled(None),
            ),
            (
                lambda: torch.is_anomaly_check_nan_enabled(None, None),
                lambda: reference_torch.is_anomaly_check_nan_enabled(None, None),
            ),
            (
                lambda: torch.is_anomaly_check_nan_enabled(enabled=True),
                lambda: reference_torch.is_anomaly_check_nan_enabled(enabled=True),
            ),
            (
                lambda: torch.is_anomaly_check_nan_enabled(None, enabled=True),
                lambda: reference_torch.is_anomaly_check_nan_enabled(
                    None, enabled=True
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assertIs(torch.is_anomaly_check_nan_enabled(**{}), True)
        self.assertIs(reference_torch.is_anomaly_check_nan_enabled(**{}), True)

    def test_mutation_and_detection_surfaces_stay_deliberately_absent(self):
        top_level_names = ("set_anomaly_enabled",)
        for name in top_level_names:
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertTrue(hasattr(reference_torch, name))
            with self.subTest(owner="torch._C", name=name):
                self.assertFalse(hasattr(torch._C, name))
                self.assertTrue(hasattr(reference_torch._C, name))

        autograd_names = ("anomaly_mode", "detect_anomaly", "set_detect_anomaly")
        for name in autograd_names:
            with self.subTest(owner="torch.autograd", name=name):
                self.assertFalse(hasattr(torch.autograd, name))
                self.assertTrue(hasattr(reference_torch.autograd, name))


if __name__ == "__main__":
    unittest.main()
