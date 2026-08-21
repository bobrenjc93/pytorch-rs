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
    def test_default_true_is_exact_and_does_not_change_grad_mode(self):
        function = torch.is_autocast_cache_enabled

        def assert_query_preserves_grad_mode(expected_grad_state):
            self.assertIs(torch.is_grad_enabled(), expected_grad_state)
            self.assertIs(function(), True)
            values = torch.tensor([1.0, -2.0, 3.0])
            self.assertEqual((values * 2.0).sum().item(), 4.0)
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
        function = torch.is_autocast_cache_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    values = torch.tensor([float(index), 1.0])
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(),
                        (values + 1.0).sum().item(),
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
                    float(index + 3),
                    True,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], True)
            self.assertIs(result[3], True)

    def test_true_before_and_after_supported_cpu_autograd_execution(self):
        function = torch.is_autocast_cache_enabled
        values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

        self.assertIs(function(), True)
        result = (values * 2.0).sum()
        self.assertIs(function(), True)
        result.backward()
        self.assertEqual(values.grad.tolist(), [2.0, 2.0, 2.0])
        self.assertIs(function(), True)

    def test_builtin_ownership_null_metadata_exports_copying_and_pickling(self):
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
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_autocast_cache_enabled"], function)

        native_module = importlib.import_module("torch_rs._C")
        self.assertIs(native_module, torch._C)
        explicit_namespace = {}
        exec(
            "from torch_rs._C import is_autocast_cache_enabled",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["is_autocast_cache_enabled"], function)

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

    def test_autocast_mutation_and_execution_surfaces_remain_unsupported(self):
        self.assertFalse(hasattr(torch, "set_autocast_cache_enabled"))
        self.assertFalse(hasattr(torch._C, "set_autocast_cache_enabled"))
        self.assertNotIn("set_autocast_cache_enabled", torch.__all__)
        self.assertFalse(hasattr(torch, "autocast"))
        self.assertFalse(hasattr(torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))

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

function = torch.is_autocast_cache_enabled
assert function is torch._C.is_autocast_cache_enabled
assert function() is True
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
