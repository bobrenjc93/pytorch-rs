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


AUTOCAST_DEVICE_TYPES = (
    "cpu",
    "cuda",
    "ipu",
    "xpu",
    "maia",
    "xla",
    "mps",
    "hpu",
    "mtia",
    "privateuseone",
)

NON_AUTOCAST_DEVICE_TYPES = (
    "mkldnn",
    "opengl",
    "opencl",
    "ideep",
    "hip",
    "ve",
    "fpga",
    "lazy",
    "vulkan",
    "meta",
)

UNKNOWN_AUTOCAST_DEVICE_ERROR = (
    "unknown device type for autocast in "
    "get_autocast_dispatch_key_from_device_type"
)
EXPECTED_DEVICE_ERROR = (
    "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, "
    "hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, "
    "privateuseone device type at start of device string: banana"
)


class IsAutocastEnabledTests(unittest.TestCase):
    def test_default_and_autocast_device_types_return_exact_false(self):
        function = torch.is_autocast_enabled

        self.assertIs(function(), False)
        self.assertIs(function(**{}), False)
        for device_type in AUTOCAST_DEVICE_TYPES:
            with self.subTest(device_type=device_type):
                self.assertIs(function(device_type), False)
                self.assertIs(function(device_type=device_type), False)
                self.assertIs(function(f"{device_type}:0"), False)
                self.assertIs(function(device_type.encode()), False)

    def test_default_state_is_stable_across_threads(self):
        function = torch.is_autocast_enabled
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    function(),
                    function("cpu"),
                    function(device_type="cuda"),
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
        self.assertEqual(results, [(False, False, False)] * worker_count)
        for result in results:
            for value in result:
                self.assertIs(value, False)

    def test_builtin_metadata_reload_copying_and_pickling(self):
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
        with self.assertRaises(ValueError):
            inspect.signature(function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                restored = pickle.loads(pickle.dumps(function, protocol=protocol))
                self.assertIs(restored, function)

        self.assertEqual(torch.__all__.count("is_autocast_enabled"), 1)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["is_autocast_enabled"], function)

        native_module = importlib.import_module("torch_rs._C")
        self.assertIs(native_module, torch._C)
        explicit_namespace = {}
        exec("from torch_rs._C import is_autocast_enabled", explicit_namespace)
        self.assertIs(explicit_namespace["is_autocast_enabled"], function)
        self.assertIs(importlib.reload(native_module), native_module)
        self.assertIs(torch._C.is_autocast_enabled, function)
        self.assertIs(torch.is_autocast_enabled, function)

    def test_recognized_non_autocast_devices_are_rejected(self):
        for device_type in NON_AUTOCAST_DEVICE_TYPES:
            for value in (device_type, f"{device_type}:0"):
                with self.subTest(device_type=value):
                    with self.assertRaises(RuntimeError) as raised:
                        torch.is_autocast_enabled(value)
                    self.assertEqual(
                        str(raised.exception), UNKNOWN_AUTOCAST_DEVICE_ERROR
                    )
                    self.assertEqual(
                        raised.exception.args, (UNKNOWN_AUTOCAST_DEVICE_ERROR,)
                    )

    def test_invalid_device_and_type_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.is_autocast_enabled(None),
                TypeError,
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not NoneType",
            ),
            (
                lambda: torch.is_autocast_enabled(device_type=torch.device("cpu")),
                TypeError,
                "is_autocast_enabled(): argument 'device_type' must be str, "
                "not torch.device",
            ),
            (
                lambda: torch.is_autocast_enabled(""),
                RuntimeError,
                "Device string must not be empty",
            ),
            (
                lambda: torch.is_autocast_enabled("banana"),
                RuntimeError,
                EXPECTED_DEVICE_ERROR,
            ),
            (
                lambda: torch.is_autocast_enabled("CPU"),
                RuntimeError,
                EXPECTED_DEVICE_ERROR.replace("banana", "CPU"),
            ),
            (
                lambda: torch.is_autocast_enabled("cpu:01"),
                RuntimeError,
                "Invalid device string: 'cpu:01'",
            ),
            (
                lambda: torch.is_autocast_enabled("cpu:2147483648"),
                RuntimeError,
                "Could not parse device index '2147483648' in device string "
                "'cpu:2147483648'",
            ),
            (
                lambda: torch.is_autocast_enabled("cpu\x00tail"),
                RuntimeError,
                "Invalid device string: 'cpu",
            ),
            (
                lambda: torch.is_autocast_enabled("\udcff"),
                RuntimeError,
                "error unpacking string as utf-8",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        with self.assertRaises(UnicodeDecodeError) as raised:
            torch.is_autocast_enabled(b"\xff")
        self.assertEqual(raised.exception.encoding, "utf-8")
        self.assertEqual(raised.exception.object, b"Invalid device string: '\xff'")
        self.assertEqual((raised.exception.start, raised.exception.end), (24, 25))
        self.assertEqual(raised.exception.reason, "invalid start byte")

    def test_arity_and_keyword_errors_match_pytorch_2_13(self):
        cases = (
            (
                lambda: torch.is_autocast_enabled("cpu", "cuda"),
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, str), but expected one of:\n"
                " * (str device_type)\n * ()\n",
            ),
            (
                lambda: torch.is_autocast_enabled("cpu", device_type="cuda"),
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, device_type=str), but expected one of:\n"
                " * (str device_type)\n * ()\n",
            ),
            (
                lambda: torch.is_autocast_enabled(foo="cpu"),
                'is_autocast_enabled() missing 1 required positional arguments: "device_type"',
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_autocast_contexts_controls_and_transitions_remain_unsupported(self):
        top_level_names = (
            "set_autocast_enabled",
            "get_autocast_dtype",
            "set_autocast_dtype",
            "get_autocast_gpu_dtype",
            "set_autocast_gpu_dtype",
            "get_autocast_cpu_dtype",
            "set_autocast_cpu_dtype",
            "is_autocast_cpu_enabled",
            "autocast",
            "amp",
        )
        native_names = (
            "set_autocast_enabled",
            "get_autocast_dtype",
            "set_autocast_dtype",
            "set_autocast_cache_enabled",
        )
        for name in top_level_names:
            with self.subTest(owner="torch_rs", name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertNotIn(name, torch.__all__)
        for name in native_names:
            with self.subTest(owner="torch_rs._C", name=name):
                self.assertFalse(hasattr(torch._C, name))
        self.assertFalse(hasattr(torch.cpu, "amp"))

    def test_reload_and_query_do_not_import_pytorch(self):
        script = r"""
import importlib
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.is_autocast_enabled
assert function() is False
assert function("cpu") is False
assert importlib.reload(torch._C) is torch._C
assert torch._C.is_autocast_enabled is function
assert importlib.reload(torch) is torch
assert torch.is_autocast_enabled is function
assert torch.__all__.count("is_autocast_enabled") == 1
assert not hasattr(torch, "set_autocast_enabled")
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
