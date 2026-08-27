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


class CpuEventTests(unittest.TestCase):
    def test_stateless_operations_are_probe_free(self):
        event = torch.cpu.Event()

        for method in (
            torch.cpu.Event.query,
            torch.cpu.Event.record,
            torch.cpu.Event.synchronize,
            torch.cpu.Event.wait,
        ):
            with self.subTest(method=method.__name__):
                self.assertEqual(method.__code__.co_names, ())
                self.assertEqual(method.__code__.co_freevars, ())
                self.assertEqual(method.__code__.co_cellvars, ())

        class UnusableStream:
            def __getattribute__(self, name):
                raise AssertionError(f"stream attribute was inspected: {name}")

        stream = UnusableStream()
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
                    self.assertIs(event.query(), True)
                    self.assertIs(event.record(), None)
                    self.assertIs(event.record(None), None)
                    self.assertIs(event.record(stream=stream), None)
                    self.assertIs(event.wait(), None)
                    self.assertIs(event.wait(None), None)
                    self.assertIs(event.wait(stream=stream), None)
                    self.assertIs(event.synchronize(), None)
                    self.assertEqual(vars(event), {})

    def test_shared_event_is_thread_safe(self):
        event = torch.cpu.Event()
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                stream = object()
                barrier.wait(timeout=10)
                results[index] = (
                    event.query(),
                    event.record(stream),
                    event.wait(stream=stream),
                    event.synchronize(),
                    event.query(),
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
        self.assertEqual(results, [(True, None, None, None, True)] * worker_count)
        self.assertEqual(vars(event), {})

    def test_class_and_method_metadata(self):
        cpu = importlib.import_module("torch_rs.cpu")
        event_type = cpu.Event

        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIs(type(event_type), type)
        self.assertEqual(event_type.__name__, "Event")
        self.assertEqual(event_type.__qualname__, "Event")
        self.assertEqual(event_type.__module__, "torch_rs.cpu")
        self.assertIsNone(event_type.__doc__)
        self.assertEqual(event_type.__annotations__, {})
        self.assertEqual(event_type.__bases__, (object,))
        self.assertEqual(str(inspect.signature(event_type)), "()")
        self.assertIs(inspect.getmodule(event_type), cpu)

        expected = {
            "query": ("(self) -> bool", {"return": bool}, None),
            "record": ("(self, stream=None) -> None", {"return": None}, (None,)),
            "synchronize": ("(self) -> None", {"return": None}, None),
            "wait": ("(self, stream=None) -> None", {"return": None}, (None,)),
        }
        for name, (signature, annotations, defaults) in expected.items():
            with self.subTest(method=name):
                method = vars(event_type)[name]
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
                self.assertEqual(method.__qualname__, f"Event.{name}")
                self.assertEqual(method.__module__, "torch_rs.cpu")
                self.assertIs(inspect.getmodule(method), cpu)
                self.assertIsNone(method.__doc__)
                self.assertEqual(method.__defaults__, defaults)
                self.assertIsNone(method.__kwdefaults__)
                self.assertEqual(method.__dict__, {})
                self.assertFalse(hasattr(method, "__text_signature__"))

        event = event_type()
        self.assertIs(type(event), event_type)
        self.assertEqual(vars(event), {})
        self.assertTrue(hasattr(event, "__weakref__"))

    def test_constructor_and_method_argument_errors(self):
        event_type = torch.cpu.Event
        event = event_type()
        cases = (
            (lambda: event_type(None), "Event() takes no arguments"),
            (lambda: event_type(stream=None), "Event() takes no arguments"),
            (
                lambda: event.query(None),
                "Event.query() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: event.query(stream=None),
                "Event.query() got an unexpected keyword argument 'stream'",
            ),
            (
                lambda: event.record(None, None),
                "Event.record() takes from 1 to 2 positional arguments but 3 were given",
            ),
            (
                lambda: event.record(None, stream=None),
                "Event.record() got multiple values for argument 'stream'",
            ),
            (
                lambda: event.wait(unexpected=None),
                "Event.wait() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: event.synchronize(None),
                "Event.synchronize() takes 1 positional argument but 2 were given",
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
        event_type = cpu.Event

        self.assertEqual(
            cpu.__all__,
            [
                "is_available",
                "is_initialized",
                "synchronize",
                "current_device",
                "device_count",
                "Stream",
                "Event",
            ],
        )

        direct_import = {}
        exec("from torch_rs.cpu import Event", direct_import)
        self.assertIs(direct_import["Event"], event_type)

        cpu_namespace = {}
        exec("from torch_rs.cpu import *", cpu_namespace)
        self.assertIs(cpu_namespace["Event"], event_type)

        self.assertNotIn("Event", torch.__all__)
        self.assertFalse(hasattr(torch, "Event"))
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("Event", top_level_namespace)

        self.assertIs(copy.copy(event_type), event_type)
        self.assertIs(copy.deepcopy(event_type), event_type)
        for method_name in ("query", "record", "synchronize", "wait"):
            method = getattr(event_type, method_name)
            self.assertIs(copy.copy(method), method)
            self.assertIs(copy.deepcopy(method), method)

        event = event_type()
        event.payload = [1, 2, 3]
        shallow = copy.copy(event)
        deep = copy.deepcopy(event)
        self.assertIsNot(shallow, event)
        self.assertIs(type(shallow), event_type)
        self.assertIs(shallow.payload, event.payload)
        self.assertIsNot(deep, event)
        self.assertIs(type(deep), event_type)
        self.assertEqual(deep.payload, event.payload)
        self.assertIsNot(deep.payload, event.payload)

        objects = (event_type, *(getattr(event_type, name) for name in (
            "query",
            "record",
            "synchronize",
            "wait",
        )))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for value in objects:
                with self.subTest(protocol=protocol, value=value):
                    payload = pickle.dumps(value, protocol=protocol)
                    self.assertIn(b"torch_rs.cpu", payload)
                    self.assertIs(pickle.loads(payload), value)

            with self.subTest(protocol=protocol, value="instance"):
                payload = pickle.dumps(event, protocol=protocol)
                self.assertIn(b"torch_rs.cpu", payload)
                restored = pickle.loads(payload)
                self.assertIs(type(restored), event_type)
                self.assertEqual(restored.payload, event.payload)
                self.assertIsNot(restored.payload, event.payload)
                self.assertIs(restored.query(), True)

    def test_cpu_module_reload_replaces_the_canonical_class(self):
        cpu = torch.cpu
        old_type = cpu.Event
        old_event = old_type()
        type_payload = pickle.dumps(old_type)
        event_payload = pickle.dumps(old_event)

        self.assertIs(importlib.reload(cpu), cpu)
        new_type = cpu.Event
        self.assertIs(torch.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], cpu)
        self.assertIsNot(new_type, old_type)
        self.assertIs(type(old_event), old_type)
        self.assertNotIsInstance(old_event, new_type)
        self.assertIs(old_event.query(), True)
        self.assertIs(old_event.record(), None)
        self.assertIs(old_event.wait(), None)
        self.assertIs(old_event.synchronize(), None)

        self.assertIs(pickle.loads(type_payload), new_type)
        restored = pickle.loads(event_payload)
        self.assertIs(type(restored), new_type)
        self.assertIs(restored.query(), True)

        new_event = new_type()
        self.assertIs(pickle.loads(pickle.dumps(new_type)), new_type)
        self.assertIs(type(pickle.loads(pickle.dumps(new_event))), new_type)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_type)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(old_event)

    def test_importing_and_using_event_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch

event = torch.cpu.Event()
assert event.query() is True
assert event.record(object()) is None
assert event.wait(stream=object()) is None
assert event.synchronize() is None
assert vars(event) == {}
assert hasattr(torch.cpu, "Stream")
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
