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


class CpuStreamTests(unittest.TestCase):
    def test_stateless_operations_are_probe_free(self):
        stream_type = torch.cpu.Stream

        for method in (
            stream_type.__init__,
            stream_type.record_event,
            stream_type.wait_event,
            stream_type.wait_stream,
        ):
            with self.subTest(method=method.__name__):
                self.assertEqual(method.__code__.co_names, ())
                self.assertEqual(method.__code__.co_freevars, ())
                self.assertEqual(method.__code__.co_cellvars, ())

        class UnusableArgument:
            def __getattribute__(self, name):
                raise AssertionError(f"argument attribute was inspected: {name}")

            def __repr__(self):
                raise AssertionError("argument representation was inspected")

        argument = UnusableArgument()
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
                    stream = stream_type(argument)
                    keyword_stream = stream_type(priority=argument)
                    self.assertIs(stream.record_event(), None)
                    self.assertIs(stream.wait_event(argument), None)
                    self.assertIs(stream.wait_event(event=argument), None)
                    self.assertIs(stream.wait_stream(argument), None)
                    self.assertIs(stream.wait_stream(stream=argument), None)
                    self.assertEqual(vars(stream), {})
                    self.assertEqual(vars(keyword_stream), {})

        for priority in (-1, 0, 1, None, True, 1.5, "high", object()):
            with self.subTest(priority_type=type(priority).__name__):
                self.assertEqual(vars(stream_type(priority)), {})

    def test_shared_stream_is_thread_safe(self):
        stream = torch.cpu.Stream()
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                argument = object()
                local_stream = torch.cpu.Stream(priority=argument)
                barrier.wait(timeout=10)
                results[index] = (
                    stream.record_event(),
                    stream.wait_event(argument),
                    stream.wait_stream(local_stream),
                    local_stream.wait_stream(stream),
                    vars(local_stream),
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
        self.assertEqual(results, [(None, None, None, None, {})] * worker_count)
        self.assertEqual(vars(stream), {})

    def test_class_constructor_and_method_metadata(self):
        cpu = importlib.import_module("torch_rs.cpu")
        stream_type = cpu.Stream

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIs(type(stream_type), type)
        self.assertEqual(stream_type.__name__, "Stream")
        self.assertEqual(stream_type.__qualname__, "Stream")
        self.assertEqual(stream_type.__module__, "torch_rs.cpu")
        self.assertEqual(
            inspect.getdoc(stream_type),
            "N.B. This class only exists to facilitate device-agnostic code",
        )
        self.assertEqual(stream_type.__annotations__, {})
        self.assertEqual(stream_type.__bases__, (object,))
        self.assertEqual(
            str(inspect.signature(stream_type)), "(priority: int = -1) -> None"
        )
        self.assertIs(inspect.getmodule(stream_type), cpu)
        self.assertEqual(
            {name for name in vars(stream_type) if not name.startswith("_")},
            {"record_event", "wait_event", "wait_stream"},
        )

        expected = {
            "__init__": (
                "(self, priority: int = -1) -> None",
                {"priority": int, "return": None},
                (-1,),
            ),
            "record_event": ("(self) -> None", {"return": None}, None),
            "wait_event": ("(self, event) -> None", {"return": None}, None),
            "wait_stream": ("(self, stream) -> None", {"return": None}, None),
        }
        for name, (signature, annotations, defaults) in expected.items():
            with self.subTest(method=name):
                method = vars(stream_type)[name]
                self.assertIs(type(method), types.FunctionType)
                self.assertEqual(str(inspect.signature(method)), signature)
                self.assertEqual(method.__annotations__, annotations)
                self.assertEqual(
                    typing.get_type_hints(method),
                    {
                        key: type(None) if value is None else value
                        for key, value in annotations.items()
                    },
                )
                self.assertEqual(method.__name__, name)
                self.assertEqual(method.__qualname__, f"Stream.{name}")
                self.assertEqual(method.__module__, "torch_rs.cpu")
                self.assertIs(inspect.getmodule(method), cpu)
                self.assertIsNone(method.__doc__)
                self.assertEqual(method.__defaults__, defaults)
                self.assertIsNone(method.__kwdefaults__)
                self.assertEqual(method.__dict__, {})
                self.assertFalse(hasattr(method, "__text_signature__"))

        stream = stream_type()
        self.assertIs(type(stream), stream_type)
        self.assertEqual(vars(stream), {})
        self.assertTrue(hasattr(stream, "__weakref__"))

    def test_constructor_and_method_argument_errors(self):
        stream_type = torch.cpu.Stream
        stream = stream_type()
        cases = (
            (
                lambda: stream_type(None, None),
                "Stream.__init__() takes from 1 to 2 positional arguments "
                "but 3 were given",
            ),
            (
                lambda: stream_type(1, priority=2),
                "Stream.__init__() got multiple values for argument 'priority'",
            ),
            (
                lambda: stream_type(unexpected=1),
                "Stream.__init__() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: stream.record_event(None),
                "Stream.record_event() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: stream.record_event(event=None),
                "Stream.record_event() got an unexpected keyword argument 'event'",
            ),
            (
                lambda: stream.wait_event(),
                "Stream.wait_event() missing 1 required positional argument: 'event'",
            ),
            (
                lambda: stream.wait_event(None, event=None),
                "Stream.wait_event() got multiple values for argument 'event'",
            ),
            (
                lambda: stream.wait_stream(unexpected=None),
                "Stream.wait_stream() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_exports_copy_and_pickle_use_the_canonical_class(self):
        cpu = torch.cpu
        stream_type = cpu.Stream

        self.assertEqual(
            cpu.__all__,
            [
                "is_available",
                "is_initialized",
                "synchronize",
                "current_device",
                "current_stream",
                "stream",
                "device_count",
                "Stream",
                "StreamContext",
                "Event",
            ],
        )

        direct_import = {}
        exec("from torch_rs.cpu import Stream", direct_import)
        self.assertIs(direct_import["Stream"], stream_type)

        cpu_namespace = {}
        exec("from torch_rs.cpu import *", cpu_namespace)
        self.assertEqual(
            {name for name in cpu_namespace if not name.startswith("__")},
            {
                "current_device",
                "current_stream",
                "stream",
                "device_count",
                "Event",
                "is_available",
                "is_initialized",
                "Stream",
                "StreamContext",
                "synchronize",
            },
        )
        self.assertIs(cpu_namespace["Stream"], stream_type)

        self.assertNotIn("Stream", torch.__all__)
        self.assertFalse(hasattr(torch, "Stream"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("Stream", top_level_namespace)

        self.assertIs(copy.copy(stream_type), stream_type)
        self.assertIs(copy.deepcopy(stream_type), stream_type)
        for method_name in (
            "__init__",
            "record_event",
            "wait_event",
            "wait_stream",
        ):
            method = getattr(stream_type, method_name)
            self.assertIs(copy.copy(method), method)
            self.assertIs(copy.deepcopy(method), method)

        stream = stream_type(priority=object())
        stream.payload = [1, 2, 3]
        shallow = copy.copy(stream)
        deep = copy.deepcopy(stream)
        self.assertIsNot(shallow, stream)
        self.assertIs(type(shallow), stream_type)
        self.assertIs(shallow.payload, stream.payload)
        self.assertIsNot(deep, stream)
        self.assertIs(type(deep), stream_type)
        self.assertEqual(deep.payload, stream.payload)
        self.assertIsNot(deep.payload, stream.payload)

        objects = (
            stream_type,
            *(getattr(stream_type, name) for name in (
                "__init__",
                "record_event",
                "wait_event",
                "wait_stream",
            )),
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for value in objects:
                with self.subTest(protocol=protocol, value=value):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.cpu", payload)
                    self.assertIs(pickle.loads(payload), value)

            with self.subTest(protocol=protocol, value="instance"):
                payload = pickle.dumps(stream, protocol=protocol)
                self.assertIn(b"torch_rs.cpu", payload)
                restored = pickle.loads(payload)
                self.assertIs(type(restored), stream_type)
                self.assertEqual(restored.payload, stream.payload)
                self.assertIsNot(restored.payload, stream.payload)
                self.assertIs(restored.record_event(), None)

    def test_cpu_module_reload_replaces_the_canonical_class(self):
        cpu = torch.cpu
        old_type = cpu.Stream
        old_stream = old_type()
        type_payload = pickle.dumps(old_type)
        stream_payload = pickle.dumps(old_stream)

        self.assertIs(importlib.reload(cpu), cpu)
        new_type = cpu.Stream
        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIsNot(new_type, old_type)
        self.assertIs(type(old_stream), old_type)
        self.assertNotIsInstance(old_stream, new_type)
        self.assertIs(old_stream.record_event(), None)
        self.assertIs(old_stream.wait_event(object()), None)
        self.assertIs(old_stream.wait_stream(object()), None)

        self.assertIs(pickle.loads(type_payload), new_type)
        restored = pickle.loads(stream_payload)
        self.assertIs(type(restored), new_type)
        self.assertIs(restored.record_event(), None)

        new_stream = new_type()
        self.assertIs(pickle.loads(pickle.dumps(new_type)), new_type)
        self.assertIs(type(pickle.loads(pickle.dumps(new_stream))), new_type)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_type)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_stream)

    def test_stream_context_apis_are_available(self):
        for name in ("stream", "StreamContext"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(torch.cpu, name))

    def test_importing_and_using_stream_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

class UnusableArgument:
    def __getattribute__(self, name):
        raise AssertionError(f"argument attribute was inspected: {name}")

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

argument = UnusableArgument()
stream = torch.cpu.Stream(priority=argument)
assert stream.record_event() is None
assert stream.wait_event(argument) is None
assert stream.wait_stream(stream=argument) is None
assert vars(stream) == {}
assert not hasattr(torch, "Stream")
assert torch.cpu.current_stream(argument) is torch.cpu.current_stream()
with torch.cpu.stream(stream):
    assert torch.cpu.current_stream() is stream
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
