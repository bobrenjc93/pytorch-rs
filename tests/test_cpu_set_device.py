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


FUNCTION_DOC = """Sets the current device, in CPU we do nothing.

    N.B. This function only exists to facilitate device-agnostic code
    """


class UnusableDevice:
    def __getattribute__(self, name):
        raise AssertionError(f"device attribute was inspected: {name}")

    def __repr__(self):
        raise AssertionError("device representation was inspected")

    def __str__(self):
        raise AssertionError("device string was inspected")

    def __index__(self):
        raise AssertionError("device index was inspected")

    def __bool__(self):
        raise AssertionError("device truthiness was inspected")


class CpuSetDeviceTests(unittest.TestCase):
    def test_returns_exact_none_for_every_ignored_device_without_runtime_probes(self):
        function = torch.cpu.set_device

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        devices = (
            None,
            0,
            -1,
            sys.maxsize,
            True,
            "cpu",
            "cpu:127",
            "cuda:0",
            "mps",
            "",
            torch.device("cpu"),
            torch.device("cpu", 0),
            torch.tensor([1.0]),
            [],
            {"device": "cuda:0"},
            UnusableDevice(),
        )
        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
                "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        for environment in environments:
            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch(
                    "os.cpu_count",
                    side_effect=AssertionError("CPU hardware was probed"),
                ):
                    for case, device in enumerate(devices):
                        with self.subTest(environment=environment, case=case):
                            self.assertIs(function(device), None)
                            self.assertIs(function(device=device), None)

    def test_noop_preserves_tensor_autograd_and_cpu_state(self):
        function = torch.cpu.set_device
        leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        result = (leaf * 3.0).transpose(0, 1)
        stream = torch.cpu.Stream()
        default_stream = torch.cpu.current_stream()
        metadata = (
            result.shape,
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        )

        with torch.cpu.stream(stream):
            self.assertIs(torch.cpu.current_stream(), stream)
            self.assertEqual(torch.cpu.current_device(), "cpu")
            self.assertIs(function(result), None)
            self.assertEqual(torch.cpu.current_device(), "cpu")
            self.assertIs(torch.cpu.current_stream(), stream)

        self.assertIs(torch.cpu.current_stream(), default_stream)
        self.assertEqual(
            (
                result.shape,
                result.stride(),
                result.storage_offset(),
                result.data_ptr(),
                result.requires_grad,
                result.is_leaf,
            ),
            metadata,
        )
        self.assertEqual(result.tolist(), [[3.0], [6.0]])
        self.assertIsNone(leaf.grad)

        result.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[3.0, 3.0]])

    def test_none_is_stable_across_threads_and_grad_modes(self):
        function = torch.cpu.set_device
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        devices = (
            None,
            0,
            -1,
            "cpu",
            "cuda:0",
            torch.device("cpu"),
            torch.tensor(1.0),
            UnusableDevice(),
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
        function = cpu.set_device

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(device: torch_rs.device | str | int | None) -> None",
        )
        device_annotation = torch.device | str | int | None
        self.assertEqual(
            function.__annotations__,
            {"device": device_annotation, "return": None},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"device": device_annotation, "return": type(None)},
        )
        self.assertEqual(function.__name__, "set_device")
        self.assertEqual(function.__qualname__, "set_device")
        self.assertEqual(function.__module__, "torch_rs.cpu")
        self.assertIs(inspect.getmodule(function), cpu)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_pickle_and_reload_use_canonical_module(self):
        cpu = torch.cpu
        function = cpu.set_device

        self.assertEqual(
            cpu.__all__,
            [
                "is_available",
                "is_initialized",
                "synchronize",
                "current_device",
                "current_stream",
                "stream",
                "set_device",
                "device_count",
                "Stream",
                "StreamContext",
                "Event",
            ],
        )

        package_import = {}
        exec("from torch_rs import cpu", package_import)
        self.assertIs(package_import["cpu"], cpu)

        direct_import = {}
        exec("from torch_rs.cpu import set_device", direct_import)
        self.assertIs(direct_import["set_device"], function)

        cpu_namespace = {}
        exec("from torch_rs.cpu import *", cpu_namespace)
        self.assertEqual(
            {name for name in cpu_namespace if not name.startswith("__")},
            set(cpu.__all__),
        )
        for name in cpu.__all__:
            with self.subTest(cpu_export=name):
                self.assertIs(cpu_namespace[name], getattr(cpu, name))

        self.assertNotIn("set_device", torch.__all__)
        self.assertFalse(hasattr(torch, "set_device"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("set_device", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.cpu", payload)
                self.assertIs(pickle.loads(payload), function)

        old_function = function
        payload = pickle.dumps(old_function)
        self.assertIs(importlib.reload(cpu), cpu)
        new_function = cpu.set_device
        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIsNot(new_function, old_function)
        self.assertIs(old_function(object()), None)
        self.assertIs(new_function(object()), None)
        self.assertIs(pickle.loads(payload), new_function)
        self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_function)

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.cpu.set_device
        cases = (
            (
                lambda: function(),
                "set_device() missing 1 required positional argument: 'device'",
            ),
            (
                lambda: function(None, None),
                "set_device() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: function(None, None, None),
                "set_device() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: function(unexpected=True),
                "set_device() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(None, device=None),
                "set_device() got multiple values for argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(device=None), None)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import os
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class UnusableDevice:
    def __getattribute__(self, name):
        raise AssertionError(f"device attribute was inspected: {name}")

sys.meta_path.insert(0, RejectPytorchImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
    PYTORCH_ENABLE_MPS_FALLBACK="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)
import torch_rs as torch

function = torch.cpu.set_device
assert function.__code__.co_names == ()
assert function(UnusableDevice()) is None
assert function(device="cuda:0") is None
assert torch.cpu.current_device() == "cpu"
assert not hasattr(torch, "set_device")
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
