import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import textwrap
import threading
import types
import unittest
from unittest import mock

import numpy as np
import torch_rs as torch


INVALID_COMBINATION_SUFFIX = (
    ", but expected one of:\n * (str device_type)\n * ()\n"
)


class IsAutocastEnabledTests(unittest.TestCase):
    def test_default_cpu_and_cuda_states_are_exact_false_without_probes(self):
        function = torch.is_autocast_enabled
        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        calls = (
            lambda: function(),
            lambda: function(**{}),
            lambda: function("cpu"),
            lambda: function("cpu:0"),
            lambda: function("cpu:2147483647"),
            lambda: function(device_type="cpu"),
            lambda: function("cuda"),
            lambda: function("cuda:0"),
            lambda: function("cuda:7"),
            lambda: function(device_type="cuda"),
            lambda: function(b"cpu"),
            lambda: function(device_type=b"cuda:0"),
        )

        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch.object(
                        torch.accelerator,
                        "_discover_accelerator",
                        side_effect=AssertionError("accelerator was probed"),
                    ) as discovery:
                        with mock.patch(
                            "os.cpu_count",
                            side_effect=AssertionError("hardware was probed"),
                        ):
                            for call in calls:
                                self.assertIs(call(), False)
                    discovery.assert_not_called()

    def test_state_is_constant_across_threads_grad_and_cache_state(self):
        previous_cache = torch.is_autocast_cache_enabled()
        self.addCleanup(torch.set_autocast_cache_enabled, previous_cache)
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    torch.set_autocast_cache_enabled(index % 3 == 0)
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        torch.is_autocast_enabled(),
                        torch.is_autocast_enabled("cpu"),
                        torch.is_autocast_enabled("cuda"),
                        torch.is_autocast_cache_enabled(),
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
            self.assertEqual(
                result,
                (
                    index % 2 == 0,
                    False,
                    False,
                    False,
                    index % 3 == 0,
                ),
            )
            for value in result[1:4]:
                self.assertIs(value, False)

    def test_builtin_metadata_exports_reload_copying_and_pickling(self):
        function = torch.is_autocast_enabled
        native = torch._C

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "is_autocast_enabled")
        self.assertEqual(function.__qualname__, "is_autocast_enabled")
        self.assertEqual(function.__module__, torch.tensor.__module__)
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertFalse(hasattr(function, "__annotations__"))
        self.assertEqual(repr(function), "<built-in function is_autocast_enabled>")
        self.assertIs(function.__self__, native)
        self.assertIs(native.is_autocast_enabled, function)
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

        imported_native = importlib.import_module("torch_rs._C")
        self.assertIs(imported_native, native)
        explicit_namespace = {}
        exec(
            "from torch_rs._C import is_autocast_enabled",
            explicit_namespace,
        )
        self.assertIs(explicit_namespace["is_autocast_enabled"], function)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.is_autocast_enabled, function)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.is_autocast_enabled, function)

    def test_positional_keyword_and_legacy_string_binding(self):
        function = torch.is_autocast_enabled

        class StringSubclass(str):
            calls = 0

            def __str__(self):
                type(self).calls += 1
                return "xpu"

        for value in (
            "cpu",
            "cpu:0",
            "cuda",
            "cuda:7",
            b"cpu",
            b"cuda:0",
            np.str_("cpu"),
            StringSubclass("cuda"),
        ):
            with self.subTest(value=value, binding="positional"):
                self.assertIs(function(value), False)
            with self.subTest(value=value, binding="keyword"):
                self.assertIs(function(device_type=value), False)
        self.assertEqual(StringSubclass.calls, 0)

    def test_binding_errors_match_pytorch_2_13(self):
        function = torch.is_autocast_enabled

        class HostileKeyword(str):
            __hash__ = str.__hash__

            def __eq__(self, other):
                raise RuntimeError("keyword equality trap")

        cases = (
            (
                lambda: function(None),
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not NoneType",
            ),
            (
                lambda: function(1),
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not int",
            ),
            (
                lambda: function(torch.device("cpu")),
                "is_autocast_enabled(): argument 'device_type' (position 1) "
                "must be str, not torch.device",
            ),
            (
                lambda: function(device_type=None),
                "is_autocast_enabled(): argument 'device_type' must be str, "
                "not NoneType",
            ),
            (
                lambda: function(device_type=bytearray(b"cpu")),
                "is_autocast_enabled(): argument 'device_type' must be str, "
                "not bytearray",
            ),
            (
                lambda: function(enabled=True),
                'is_autocast_enabled() missing 1 required positional arguments: "device_type"',
            ),
            (
                lambda: function("cpu", "cuda"),
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, str)" + INVALID_COMBINATION_SUFFIX,
            ),
            (
                lambda: function("cpu", device_type="cuda"),
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, device_type=str)"
                + INVALID_COMBINATION_SUFFIX,
            ),
            (
                lambda: function("cpu", enabled=True),
                "is_autocast_enabled() received an invalid combination of "
                "arguments - got (str, enabled=bool)"
                + INVALID_COMBINATION_SUFFIX,
            ),
            (
                lambda: function(
                    **{HostileKeyword("device_type"): "cpu"}
                ),
                'is_autocast_enabled() missing 1 required positional arguments: "device_type"',
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        message = (
            "is_autocast_enabled() received an invalid combination of "
            "arguments - got (str, bad"
        )
        with self.assertRaises(TypeError) as raised:
            function("cpu", **{"bad\0tail": 1})
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

    def test_cpu_cuda_device_string_errors_match_pytorch_2_13(self):
        function = torch.is_autocast_enabled
        cases = (
            ("", "Device string must not be empty"),
            (b"", "Device string must not be empty"),
            ("cpu:-1", "Invalid device string: 'cpu:-1'"),
            ("cpu:00", "Invalid device string: 'cpu:00'"),
            ("cuda:", "Invalid device string: 'cuda:'"),
            ("cuda:abc", "Invalid device string: 'cuda:abc'"),
            (
                "cuda:2147483648",
                "Could not parse device index '2147483648' in device string "
                "'cuda:2147483648'",
            ),
            (
                "CPU",
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, "
                "ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, "
                "mtia, privateuseone device type at start of device string: CPU",
            ),
            (
                "banana",
                "Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, "
                "ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, "
                "mtia, privateuseone device type at start of device string: banana",
            ),
        )
        for value, message in cases:
            with self.subTest(value=value):
                with self.assertRaises(RuntimeError) as raised:
                    function(value)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        message = (
            "is_autocast_enabled(): device 'xpu' is not supported; only 'cpu' "
            "and 'cuda' are implemented"
        )
        with self.assertRaises(RuntimeError) as raised:
            function("xpu")
        self.assertEqual(str(raised.exception), message)
        self.assertEqual(raised.exception.args, (message,))

        for call in (
            lambda: function("\ud800"),
            lambda: function(device_type="\ud800"),
        ):
            with self.subTest(value="surrogate"):
                with self.assertRaises(RuntimeError) as raised:
                    call()
                self.assertEqual(
                    str(raised.exception), "error unpacking string as utf-8"
                )
                self.assertEqual(
                    raised.exception.args,
                    ("error unpacking string as utf-8",),
                )

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_diagnostics_raise_bad_alloc_instead_of_aborting(self):
        script = textwrap.dedent(
            """\
            import os
            import resource
            import sys

            import torch_rs as torch

            if sys.argv[1] == "arguments":
                payload = (None,) * 500_000
                call = lambda: torch.is_autocast_enabled(*payload)
            elif sys.argv[1] == "device":
                payload = "z" * (16 * 1024 * 1024)
                call = lambda: torch.is_autocast_enabled(payload)
            else:
                raise AssertionError(sys.argv[1])

            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            limit = current_virtual_size + 4 * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            try:
                call()
            except RuntimeError as error:
                assert str(error) == "std::bad_alloc", repr(error)
            else:
                raise AssertionError("the constrained call unexpectedly succeeded")
            """
        )
        for case in ("arguments", "device"):
            with self.subTest(case=case):
                completed = subprocess.run(
                    [sys.executable, "-c", script, case],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                if completed.returncode == 77:
                    self.skipTest("process hard address-space limit is too low")
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )

    def test_contexts_setters_and_dtype_controls_remain_unsupported(self):
        self.assertIs(torch.is_autocast_enabled, torch._C.is_autocast_enabled)
        self.assertIn("is_autocast_enabled", torch.__all__)
        self.assertTrue(hasattr(torch, "is_autocast_cache_enabled"))
        self.assertTrue(hasattr(torch, "set_autocast_cache_enabled"))
        for name in (
            "autocast",
            "amp",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "get_autocast_cpu_dtype",
            "get_autocast_dtype",
            "get_autocast_gpu_dtype",
            "set_autocast_cpu_dtype",
            "set_autocast_cpu_enabled",
            "set_autocast_dtype",
            "set_autocast_enabled",
            "set_autocast_gpu_dtype",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))
                self.assertFalse(hasattr(torch._C, name))
        self.assertFalse(hasattr(torch.cpu, "amp"))

    def test_importing_and_calling_does_not_import_or_probe_pytorch(self):
        script = r"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch

def reject_discovery():
    raise AssertionError("accelerator discovery was attempted")

torch.accelerator._discover_accelerator = reject_discovery
function = torch.is_autocast_enabled
assert function is torch._C.is_autocast_enabled
for arguments in ((), ("cpu",), ("cpu:0",), ("cuda",), ("cuda:0",)):
    assert function(*arguments) is False
assert function(device_type="cpu") is False
assert function(device_type="cuda") is False
assert not hasattr(torch, "cuda")
assert not hasattr(torch, "autocast")
assert not hasattr(torch, "set_autocast_enabled")
assert not hasattr(torch, "get_autocast_dtype")
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
