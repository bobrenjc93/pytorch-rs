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


FUNCTION_DOC = """Waits for all kernels in all streams on the CPU device to complete.

    Args:
        device (torch.device or int, optional): ignored, there's only one CPU device.

    N.B. This function only exists to facilitate device-agnostic code.
    """


class HostileDevice:
    def __getattribute__(self, name):
        raise AssertionError(f"synchronize inspected device attribute {name!r}")

    def __repr__(self):
        raise AssertionError("synchronize represented the device")

    def __str__(self):
        raise AssertionError("synchronize converted the device to text")

    def __bool__(self):
        raise AssertionError("synchronize tested the device's truth value")

    def __index__(self):
        raise AssertionError("synchronize converted the device to an index")


class CpuSynchronizeTests(unittest.TestCase):
    def test_exact_none_noop_accepts_every_ignored_device_value(self):
        function = torch.cpu.synchronize

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        tracked = leaf * 3.0
        metadata = (
            tracked.tolist(),
            tracked.shape,
            tracked.stride(),
            tracked.storage_offset(),
            tracked.data_ptr(),
            tracked.requires_grad,
            tracked.is_leaf,
        )
        devices = (
            None,
            True,
            False,
            0,
            -1,
            2**1000,
            1.5,
            float("nan"),
            "",
            "cpu",
            "cpu:0",
            "cuda:0",
            torch.device("cpu"),
            torch.device("cpu", 0),
            torch.tensor([1]),
            object(),
            HostileDevice(),
            ["device"],
            {"device": "cpu"},
            lambda: None,
            Ellipsis,
            NotImplemented,
        )

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
                        self.assertIs(function(), None)
                        self.assertIs(function(device=None), None)
                        for device in devices:
                            self.assertIs(function(device), None)

        self.assertEqual(
            (
                tracked.tolist(),
                tracked.shape,
                tracked.stride(),
                tracked.storage_offset(),
                tracked.data_ptr(),
                tracked.requires_grad,
                tracked.is_leaf,
            ),
            metadata,
        )
        tracked.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0], [3.0, 3.0]])

    def test_noop_is_stable_across_threads_and_grad_modes(self):
        function = torch.cpu.synchronize
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        devices = (
            None,
            "cpu",
            "cuda:7",
            0,
            -1,
            torch.device("cpu"),
            torch.tensor([1]),
            HostileDevice(),
        )
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        function(devices[index]),
                        torch.is_grad_enabled(),
                        function(device=devices[-index - 1]),
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
                    None,
                    expected_grad_state,
                    None,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], None)
            self.assertIs(result[3], None)

    def test_signature_annotations_documentation_and_module_identity(self):
        cpu = importlib.import_module("torch_rs.cpu")
        function = cpu.synchronize
        device_annotation = torch.device | str | int | None

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(device: torch_rs.device | str | int | None = None) -> None",
        )
        self.assertEqual(
            function.__annotations__,
            {"device": device_annotation, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"device": device_annotation, "return": type(None)},
        )
        self.assertEqual(function.__name__, "synchronize")
        self.assertEqual(function.__qualname__, "synchronize")
        self.assertEqual(function.__module__, "torch_rs.cpu")
        self.assertIs(inspect.getmodule(function), cpu)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

        parameter = inspect.signature(function).parameters["device"]
        self.assertEqual(parameter.kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertIsNone(parameter.default)
        self.assertEqual(parameter.annotation, device_annotation)
        self.assertIs(inspect.signature(function).return_annotation, None)

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        cpu = torch.cpu
        function = cpu.synchronize

        self.assertEqual(
            cpu.__all__, ["is_available", "synchronize", "device_count"]
        )

        package_import = {}
        exec("from torch_rs import cpu", package_import)
        self.assertIs(package_import["cpu"], cpu)

        direct_import = {}
        exec("from torch_rs.cpu import synchronize", direct_import)
        self.assertIs(direct_import["synchronize"], function)

        cpu_namespace = {}
        exec("from torch_rs.cpu import *", cpu_namespace)
        self.assertEqual(
            {name for name in cpu_namespace if not name.startswith("__")},
            {"device_count", "is_available", "synchronize"},
        )
        self.assertIs(cpu_namespace["synchronize"], function)
        self.assertIs(cpu_namespace["is_available"], cpu.is_available)
        self.assertIs(cpu_namespace["device_count"], cpu.device_count)

        self.assertNotIn("cpu", torch.__all__)
        self.assertNotIn("synchronize", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("cpu", top_level_namespace)
        self.assertNotIn("synchronize", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.cpu", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_argument_errors_match_pytorch_2_13_python_binding(self):
        function = torch.cpu.synchronize
        cases = (
            (
                lambda: function(None, None),
                "synchronize() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: function(None, None, None),
                "synchronize() takes from 0 to 1 positional arguments but 3 were given",
            ),
            (
                lambda: function(unexpected=True),
                "synchronize() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(None, unexpected=True),
                "synchronize() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(device=None, unexpected=True),
                "synchronize() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(None, device=None),
                "synchronize() got multiple values for argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_streams_events_and_device_mutation_remain_unsupported(self):
        cpu = torch.cpu

        self.assertEqual(
            {name for name in vars(cpu) if not name.startswith("_")},
            {"device_count", "is_available", "synchronize"},
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
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cpu, name))

        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.cpu.amp")
        self.assertFalse(hasattr(torch, "synchronize"))

        before = torch.device("cpu", 0)
        self.assertIs(cpu.synchronize(before), None)
        self.assertEqual(before, torch.device("cpu", 0))
        self.assertEqual(before.type, "cpu")
        self.assertEqual(before.index, 0)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class HostileDevice:
    def __getattribute__(self, name):
        raise AssertionError(name)
    def __repr__(self):
        raise AssertionError("repr")
    def __bool__(self):
        raise AssertionError("bool")
    def __index__(self):
        raise AssertionError("index")

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

function = torch.cpu.synchronize
assert function.__code__.co_names == ()
assert function() is None
assert function(HostileDevice()) is None
assert function(device="cuda:0") is None
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
