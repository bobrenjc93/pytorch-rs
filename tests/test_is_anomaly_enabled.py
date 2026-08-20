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


class IsAnomalyEnabledTests(unittest.TestCase):
    def test_default_false_is_exact_and_does_not_change_grad_mode(self):
        function = torch.is_anomaly_enabled

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
        function = torch.is_anomaly_enabled
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

    def test_builtin_ownership_null_metadata_exports_copying_and_pickling(self):
        function = torch.is_anomaly_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_anomaly_enabled")
        self.assertEqual(function.__qualname__, "is_anomaly_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__doc__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(repr(function), "<built-in function is_anomaly_enabled>")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.is_anomaly_enabled, function)
        assert_no_argument_signature(self, function, "()")

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertEqual(torch.__all__.count("is_anomaly_enabled"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_anomaly_enabled"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.is_anomaly_enabled
        cases = (
            (
                lambda: function(None),
                "torch.is_anomaly_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.is_anomaly_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.is_anomaly_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.is_anomaly_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), False)

    def test_anomaly_mutation_detection_and_check_nan_apis_remain_unsupported(self):
        top_level_names = (
            "set_anomaly_enabled",
            "is_anomaly_check_nan_enabled",
        )
        for name in top_level_names:
            with self.subTest(owner="torch", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
            with self.subTest(owner="torch._C", name=name):
                self.assertFalse(hasattr(torch._C, name))

        autograd_names = ("anomaly_mode", "detect_anomaly", "set_detect_anomaly")
        for name in autograd_names:
            with self.subTest(owner="torch.autograd", name=name):
                self.assertFalse(hasattr(torch.autograd, name))
                self.assertNotIn(name, torch.autograd.__all__)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertTrue(set(top_level_names).isdisjoint(wildcard_namespace))


if __name__ == "__main__":
    unittest.main()
