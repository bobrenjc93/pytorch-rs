import contextlib
import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """Returns number of CPU devices (not cores). Always 1.

    N.B. This function only exists to facilitate device-agnostic code
    """


class CpuDeviceCountTests(unittest.TestCase):
    def test_returns_exact_one_without_runtime_probes(self):
        function = torch.cpu.device_count

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    with mock.patch(
                        "os.cpu_count",
                        side_effect=AssertionError("CPU hardware was probed"),
                    ):
                        result = function()
                self.assertIs(type(result), int)
                self.assertEqual(result, 1)

    def test_one_is_stable_across_threads_and_grad_modes(self):
        function = torch.cpu.device_count
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
                    1,
                    expected_grad_state,
                    1,
                    expected_grad_state,
                ),
            )
            self.assertIs(type(result[1]), int)
            self.assertIs(type(result[3]), int)

    def test_signature_annotations_documentation_and_module_identity(self):
        cpu = importlib.import_module("torch_rs.cpu")
        function = cpu.device_count

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> int")
        self.assertEqual(function.__annotations__, {"return": int})
        self.assertEqual(typing.get_type_hints(function), {"return": int})
        self.assertEqual(function.__name__, "device_count")
        self.assertEqual(function.__qualname__, "device_count")
        self.assertEqual(function.__module__, "torch_rs.cpu")
        self.assertIs(inspect.getmodule(function), cpu)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        cpu = torch.cpu
        function = cpu.device_count

        self.assertEqual(cpu.__all__, ["is_available", "device_count"])

        package_import = {}
        exec("from torch_rs import cpu", package_import)
        self.assertIs(package_import["cpu"], cpu)

        direct_import = {}
        exec("from torch_rs.cpu import device_count", direct_import)
        self.assertIs(direct_import["device_count"], function)

        cpu_namespace = {}
        exec("from torch_rs.cpu import *", cpu_namespace)
        self.assertEqual(
            {name for name in cpu_namespace if not name.startswith("__")},
            {"device_count", "is_available"},
        )
        self.assertIs(cpu_namespace["device_count"], function)
        self.assertIs(cpu_namespace["is_available"], cpu.is_available)

        self.assertNotIn("cpu", torch.__all__)
        self.assertNotIn("device_count", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("cpu", top_level_namespace)
        self.assertNotIn("device_count", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.cpu", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.cpu.device_count
        cases = (
            (
                lambda: function(None),
                "device_count() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "device_count() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(device=True),
                "device_count() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: function(None, device=True),
                "device_count() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_streams_synchronization_amp_and_other_cpu_apis_remain_unsupported(self):
        cpu = torch.cpu

        self.assertEqual(
            {name for name in vars(cpu) if not name.startswith("_")},
            {"device_count", "is_available"},
        )
        for name in (
            "amp",
            "current_device",
            "current_stream",
            "Event",
            "get_capabilities",
            "is_initialized",
            "set_device",
            "Stream",
            "StreamContext",
            "stream",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cpu, name))

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.cpu.amp")
        self.assertFalse(hasattr(torch, "device_count"))

    def test_importing_and_calling_does_not_import_pytorch(self):
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

function = torch.cpu.device_count
assert function.__code__.co_names == ()
result = function()
assert type(result) is int
assert result == 1
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
