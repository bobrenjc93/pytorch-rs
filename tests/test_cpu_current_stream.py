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


FUNCTION_DOC = """Returns the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): Ignored.

    N.B. This function only exists to facilitate device-agnostic code

    """


class UnusableDevice:
    def __getattribute__(self, name):
        raise AssertionError(f"device attribute was inspected: {name}")

    def __repr__(self):
        raise AssertionError("device representation was inspected")


class CpuCurrentStreamTests(unittest.TestCase):
    def test_returns_one_canonical_stream_without_runtime_probes(self):
        function = torch.cpu.current_stream
        stream = function()

        self.assertEqual(function.__code__.co_names, ("_current_stream",))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertIs(type(stream), torch.cpu.Stream)
        self.assertIs(stream, torch.cpu._default_cpu_stream)
        self.assertIs(stream, torch.cpu._current_stream)
        self.assertEqual(vars(stream), {})

        devices = (
            None,
            0,
            -1,
            sys.maxsize,
            True,
            "cpu",
            "cpu:127",
            "cuda:0",
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
                "OMP_NUM_THREADS": "1",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with mock.patch.dict(os.environ, environment, clear=True):
                self.assertIs(function(), stream)
                for case, device in enumerate(devices):
                    with self.subTest(environment=environment, case=case):
                        self.assertIs(function(device), stream)
                        self.assertIs(function(device=device), stream)

    def test_identity_is_shared_across_threads(self):
        function = torch.cpu.current_stream
        stream = function()
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    function(UnusableDevice()),
                    function(device=object()),
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
        for result in results:
            self.assertIs(result[0], stream)
            self.assertIs(result[1], stream)

    def test_tensor_execution_and_autograd_are_unchanged(self):
        leaf = torch.tensor([[1.0, 2.0]], requires_grad=True)
        result = (leaf * 3.0).transpose(0, 1)
        metadata = (
            result.shape,
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        )

        self.assertIs(torch.cpu.current_stream(result), torch.cpu.current_stream())
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

    def test_signature_annotations_documentation_and_module_identity(self):
        cpu = importlib.import_module("torch_rs.cpu")
        function = cpu.current_stream

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(
            str(inspect.signature(function)),
            "(device: torch_rs.device | str | int | None = None) "
            "-> torch_rs.cpu.Stream",
        )
        device_annotation = torch.device | str | int | None
        self.assertEqual(
            function.__annotations__,
            {"device": device_annotation, "return": cpu.Stream},
        )
        self.assertEqual(
            typing.get_type_hints(function),
            {"device": device_annotation, "return": cpu.Stream},
        )
        self.assertEqual(function.__name__, "current_stream")
        self.assertEqual(function.__qualname__, "current_stream")
        self.assertEqual(function.__module__, "torch_rs.cpu")
        self.assertIs(inspect.getmodule(function), cpu)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertEqual(function.__defaults__, (None,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_exports_copy_and_pickle_use_canonical_objects(self):
        cpu = torch.cpu
        function = cpu.current_stream
        stream = function()

        self.assertEqual(
            cpu.__all__,
            [
                "is_available",
                "is_initialized",
                "synchronize",
                "current_device",
                "current_stream",
                "device_count",
                "Stream",
                "Event",
            ],
        )

        direct_import = {}
        exec("from torch_rs.cpu import current_stream", direct_import)
        self.assertIs(direct_import["current_stream"], function)

        cpu_namespace = {}
        exec("from torch_rs.cpu import *", cpu_namespace)
        self.assertEqual(
            {name for name in cpu_namespace if not name.startswith("__")},
            set(cpu.__all__),
        )
        self.assertIs(cpu_namespace["current_stream"], function)

        self.assertNotIn("current_stream", torch.__all__)
        self.assertFalse(hasattr(torch, "current_stream"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("current_stream", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        stream.payload = [1, 2, 3]
        try:
            shallow = copy.copy(stream)
            deep = copy.deepcopy(stream)
            self.assertIsNot(shallow, stream)
            self.assertIs(type(shallow), cpu.Stream)
            self.assertIs(shallow.payload, stream.payload)
            self.assertIsNot(deep, stream)
            self.assertIs(type(deep), cpu.Stream)
            self.assertEqual(deep.payload, stream.payload)
            self.assertIsNot(deep.payload, stream.payload)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(protocol=protocol):
                    function_payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.cpu", function_payload)
                    self.assertIs(pickle.loads(function_payload), function)

                    stream_payload = pickle.dumps(stream, protocol=protocol)
                    self.assertIn(b"torch_rs.cpu", stream_payload)
                    restored = pickle.loads(stream_payload)
                    self.assertIsNot(restored, stream)
                    self.assertIsNot(restored, function())
                    self.assertIs(type(restored), cpu.Stream)
                    self.assertEqual(restored.payload, stream.payload)
                    self.assertIsNot(restored.payload, stream.payload)
        finally:
            del stream.payload

    def test_argument_errors_match_pytorch_2_13(self):
        function = torch.cpu.current_stream
        cases = (
            (
                lambda: function(None, None),
                "current_stream() takes from 0 to 1 positional arguments "
                "but 2 were given",
            ),
            (
                lambda: function(None, None, None),
                "current_stream() takes from 0 to 1 positional arguments "
                "but 3 were given",
            ),
            (
                lambda: function(unexpected=True),
                "current_stream() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: function(None, device=None),
                "current_stream() got multiple values for argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_module_reload_replaces_function_class_and_default_stream(self):
        cpu = torch.cpu
        old_function = cpu.current_stream
        old_type = cpu.Stream
        old_stream = old_function()
        old_stream.payload = [1, 2, 3]
        function_payload = pickle.dumps(old_function)
        type_payload = pickle.dumps(old_type)
        stream_payload = pickle.dumps(old_stream)

        self.assertIs(importlib.reload(cpu), cpu)
        new_function = cpu.current_stream
        new_type = cpu.Stream
        new_stream = new_function()
        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIsNot(new_function, old_function)
        self.assertIsNot(new_type, old_type)
        self.assertIsNot(new_stream, old_stream)
        self.assertIs(type(new_stream), new_type)
        self.assertEqual(vars(new_stream), {})
        self.assertIs(new_stream, cpu._default_cpu_stream)
        self.assertIs(new_stream, cpu._current_stream)
        self.assertIs(old_function(), new_stream)
        self.assertIs(type(old_stream), old_type)
        self.assertNotIsInstance(old_stream, new_type)
        self.assertIs(old_stream.record_event(), None)

        self.assertIs(pickle.loads(function_payload), new_function)
        self.assertIs(pickle.loads(type_payload), new_type)
        restored = pickle.loads(stream_payload)
        self.assertIsNot(restored, new_stream)
        self.assertIs(type(restored), new_type)
        self.assertEqual(restored.payload, old_stream.payload)

        self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
        new_restored = pickle.loads(pickle.dumps(new_stream))
        self.assertIsNot(new_restored, new_stream)
        self.assertIs(type(new_restored), new_type)
        for value in (old_function, old_type, old_stream):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(value)

    def test_stream_context_apis_remain_unsupported(self):
        self.assertTrue(hasattr(torch.cpu, "current_stream"))
        for name in ("stream", "StreamContext"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.cpu, name))

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
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
import torch_rs as torch

function = torch.cpu.current_stream
stream = function()
assert function.__code__.co_names == ("_current_stream",)
assert type(stream) is torch.cpu.Stream
assert function(UnusableDevice()) is stream
assert function(device="cuda:0") is stream
assert not hasattr(torch.cpu, "stream")
assert not hasattr(torch.cpu, "StreamContext")
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
