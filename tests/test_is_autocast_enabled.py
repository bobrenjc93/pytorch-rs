import contextlib
import copy
import importlib
import inspect
import pickle
import subprocess
import sys
import threading
import types
import unittest

import torch_rs as torch


class IsAutocastEnabledTests(unittest.TestCase):
    def test_default_cpu_and_cuda_states_are_exact_false(self):
        function = torch.is_autocast_enabled
        previous_cache_state = torch.is_autocast_cache_enabled()
        self.addCleanup(
            torch.set_autocast_cache_enabled,
            previous_cache_state,
        )

        for cache_enabled in (True, False):
            torch.set_autocast_cache_enabled(cache_enabled)
            for expected_grad, context in (
                (True, contextlib.nullcontext()),
                (False, torch.no_grad()),
            ):
                with context:
                    for name, call in (
                        ("default", lambda: function()),
                        ("cpu", lambda: function("cpu")),
                        ("cuda", lambda: function("cuda")),
                        ("indexed_cpu", lambda: function("cpu:0")),
                        ("indexed_cuda", lambda: function("cuda:0")),
                        (
                            "cpu_keyword",
                            lambda: function(device_type="cpu"),
                        ),
                        (
                            "cuda_keyword",
                            lambda: function(device_type="cuda"),
                        ),
                    ):
                        with self.subTest(
                            cache_enabled=cache_enabled,
                            grad_enabled=expected_grad,
                            call=name,
                        ):
                            self.assertIs(call(), False)
                            self.assertIs(torch.is_grad_enabled(), expected_grad)
                            self.assertIs(
                                torch.is_autocast_cache_enabled(),
                                cache_enabled,
                            )

    def test_default_state_is_stable_across_threads(self):
        function = torch.is_autocast_enabled
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
                        function(),
                        function("cpu"),
                        function(device_type="cuda"),
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
            self.assertEqual(result, (False, False, False, index % 2 == 0))
            for state in result[:3]:
                self.assertIs(state, False)

    def test_builtin_metadata_exports_copying_pickling_and_reload(self):
        function = torch.is_autocast_enabled
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_autocast_enabled")
        self.assertEqual(function.__qualname__, "is_autocast_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(repr(function), "<built-in function is_autocast_enabled>")
        self.assertIs(function.__self__, torch._C)
        self.assertIs(torch._C.is_autocast_enabled, function)
        self.assertEqual(function.__reduce__(), "is_autocast_enabled")
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        self.assertEqual(torch.__all__.count("is_autocast_enabled"), 1)
        self.assertEqual(torch._C.__all__.count("is_autocast_enabled"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_autocast_enabled"], function)

        explicit_namespace = {}
        exec(
            "from torch_rs._C import is_autocast_enabled",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["is_autocast_enabled"], function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertIs(importlib.reload(torch._C), torch._C)
        self.assertIs(torch._C.is_autocast_enabled, function)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.is_autocast_enabled, function)

    def test_positional_and_keyword_binding_errors_match_pytorch_2_13(self):
        function = torch.is_autocast_enabled
        cases = (
            (
                lambda: function(None),
                TypeError,
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not NoneType",
            ),
            (
                lambda: function(1),
                TypeError,
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not int",
            ),
            (
                lambda: function(torch.device("cpu")),
                TypeError,
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not torch.device",
            ),
            (
                lambda: function(device_type=None),
                TypeError,
                "is_autocast_enabled(): argument 'device_type' must be str, "
                "not NoneType",
            ),
            (
                lambda: function(foo="cpu"),
                TypeError,
                'is_autocast_enabled() missing 1 required positional arguments: '
                '"device_type"',
            ),
            (
                lambda: function("cpu", "cuda"),
                TypeError,
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, str), but expected one of:\n"
                " * (str device_type)\n"
                " * ()\n",
            ),
            (
                lambda: function("cpu", device_type="cuda"),
                TypeError,
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, device_type=str), but expected one of:\n"
                " * (str device_type)\n"
                " * ()\n",
            ),
            (
                lambda: function(""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: function("cuda:"),
                RuntimeError,
                "Invalid device string: 'cuda:'",
            ),
            (
                lambda: function("cpu:01"),
                RuntimeError,
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: function("cpu:2147483648"),
                RuntimeError,
                "Could not parse device index '2147483648' in device string "
                "'cpu:2147483648'",
            ),
            (
                lambda: function("CUDA"),
                RuntimeError,
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, "
                "ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, "
                "mtia, privateuseone device type at start of device string: CUDA",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        class StringSubclass(str):
            pass

        self.assertIs(function(**{}), False)
        self.assertIs(function(StringSubclass("cpu")), False)
        self.assertIs(function(device_type=StringSubclass("cuda")), False)

    def test_other_autocast_controls_remain_unsupported(self):
        self.assertTrue(hasattr(torch, "is_autocast_cache_enabled"))
        for name in (
            "autocast",
            "get_autocast_dtype",
            "get_autocast_cpu_dtype",
            "get_autocast_gpu_dtype",
            "is_autocast_cpu_enabled",
            "set_autocast_enabled",
            "set_autocast_cpu_enabled",
            "set_autocast_cpu_dtype",
            "set_autocast_dtype",
            "set_autocast_gpu_dtype",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        self.assertFalse(hasattr(torch, "amp"))
        self.assertFalse(hasattr(torch.cpu, "amp"))
        self.assertFalse(hasattr(torch, "cuda"))

    def test_import_and_queries_do_not_import_pytorch_or_discover_accelerators(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

def unexpected_discovery():
    raise AssertionError("accelerator discovery was attempted")

torch.accelerator._discover_accelerator = unexpected_discovery
function = torch.is_autocast_enabled
assert function is torch._C.is_autocast_enabled
assert function() is False
assert function("cpu") is False
assert function("cuda") is False
assert function(device_type="cpu") is False
assert function(device_type="cuda") is False
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
