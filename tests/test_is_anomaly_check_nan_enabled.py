import contextlib
import copy
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


class IsAnomalyCheckNanEnabledTests(unittest.TestCase):
    def test_default_true_is_exact_and_does_not_change_grad_mode(self):
        function = torch.is_anomaly_check_nan_enabled

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), True)
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)

        assert_query_preserves_grad_mode(True)
        with torch.no_grad():
            assert_query_preserves_grad_mode(False)
            with torch.no_grad():
                assert_query_preserves_grad_mode(False)
            assert_query_preserves_grad_mode(False)
        assert_query_preserves_grad_mode(True)

    def test_default_true_is_stable_across_threads_and_grad_modes(self):
        function = torch.is_anomaly_check_nan_enabled
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
                    True,
                    expected_grad_state,
                    True,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], True)
            self.assertIs(result[3], True)

    def test_builtin_ownership_null_metadata_exports_copying_and_pickling(self):
        function = torch.is_anomaly_check_nan_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_anomaly_check_nan_enabled")
        self.assertEqual(function.__qualname__, "is_anomaly_check_nan_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__doc__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(
            repr(function), "<built-in function is_anomaly_check_nan_enabled>"
        )
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.is_anomaly_check_nan_enabled, function)
        assert_no_argument_signature(self, function, "()")

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertEqual(torch.__all__.count("is_anomaly_check_nan_enabled"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_anomaly_check_nan_enabled"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

    def test_rejects_all_arguments_with_pytorch_2_13_errors(self):
        function = torch.is_anomaly_check_nan_enabled
        cases = (
            (
                lambda: function(None),
                "torch.is_anomaly_check_nan_enabled() takes no arguments (1 given)",
            ),
            (
                lambda: function(None, None),
                "torch.is_anomaly_check_nan_enabled() takes no arguments (2 given)",
            ),
            (
                lambda: function(enabled=True),
                "torch.is_anomaly_check_nan_enabled() takes no keyword arguments",
            ),
            (
                lambda: function(None, enabled=True),
                "torch.is_anomaly_check_nan_enabled() takes no keyword arguments",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(**{}), True)

    def test_anomaly_setters_and_detection_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "set_anomaly_enabled"))
        self.assertNotIn("set_anomaly_enabled", torch.__all__)
        self.assertFalse(hasattr(torch._C, "set_anomaly_enabled"))

        for name in ("anomaly_mode", "detect_anomaly", "set_detect_anomaly"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.autograd, name))
                self.assertNotIn(name, torch.autograd.__all__)

        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertNotIn("set_anomaly_enabled", wildcard_namespace)

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

function = torch.is_anomaly_check_nan_enabled
assert function() is True
assert function.__self__ is torch._C
assert torch._C.is_anomaly_check_nan_enabled is function
assert torch.__all__.count("is_anomaly_check_nan_enabled") == 1
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
