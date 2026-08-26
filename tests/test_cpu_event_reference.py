import copy
import importlib
import inspect
import pickle
import pickletools
import sys
import threading
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuEventReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("cpu.Event differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def pickle_shape(self, value, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(value, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def event_outcome(self, module):
        event = module.cpu.Event()
        stream = object()
        return (
            event.query(),
            event.record(),
            event.record(None),
            event.record(stream=stream),
            event.wait(),
            event.wait(None),
            event.wait(stream=stream),
            event.synchronize(),
            vars(event),
        )

    def threaded_outcome(self, module):
        event = module.cpu.Event()
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
                errors.append((type(error).__name__, str(error)))

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        return results, errors, vars(event)

    def test_stateless_operations_and_threading_match_pytorch_2_13(self):
        actual = self.event_outcome(torch)
        expected = self.event_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, (True, None, None, None, None, None, None, None, {}))
        self.assertIs(actual[0], True)

        actual_threads = self.threaded_outcome(torch)
        expected_threads = self.threaded_outcome(reference_torch)
        self.assertEqual(actual_threads, expected_threads)
        self.assertEqual(actual_threads[1:], ([], {}))

    def test_class_and_method_metadata_match_pytorch_2_13(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual = actual_cpu.Event
        expected = expected_cpu.Event

        self.assertIs(type(actual), type(expected))
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__annotations__, expected.__annotations__)
        self.assertEqual(actual.__bases__, expected.__bases__)
        self.assertIs(inspect.getmodule(actual), actual_cpu)
        self.assertIs(inspect.getmodule(expected), expected_cpu)

        public_class_attributes = {
            name for name in vars(actual) if not name.startswith("_")
        }
        self.assertEqual(
            public_class_attributes,
            {name for name in vars(expected) if not name.startswith("_")},
        )
        self.assertEqual(
            public_class_attributes,
            {"query", "record", "synchronize", "wait"},
        )

        for name in sorted(public_class_attributes):
            with self.subTest(method=name):
                actual_method = vars(actual)[name]
                expected_method = vars(expected)[name]
                self.assertIs(type(actual_method), types.FunctionType)
                self.assertIs(type(expected_method), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual_method)),
                    str(inspect.signature(expected_method)),
                )
                self.assertEqual(
                    actual_method.__annotations__, expected_method.__annotations__
                )
                self.assertEqual(
                    typing.get_type_hints(actual_method),
                    typing.get_type_hints(expected_method),
                )
                self.assertEqual(actual_method.__name__, expected_method.__name__)
                self.assertEqual(actual_method.__qualname__, expected_method.__qualname__)
                self.assertEqual(
                    actual_method.__module__.replace("torch_rs", "torch"),
                    expected_method.__module__,
                )
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
                self.assertEqual(actual_method.__defaults__, expected_method.__defaults__)
                self.assertEqual(
                    actual_method.__kwdefaults__, expected_method.__kwdefaults__
                )
                self.assertEqual(actual_method.__dict__, expected_method.__dict__)
                self.assertEqual(
                    hasattr(actual_method, "__text_signature__"),
                    hasattr(expected_method, "__text_signature__"),
                )

    def test_exports_and_canonical_identity_match_the_cpu_scope(self):
        actual_cpu = torch.cpu
        expected_cpu = reference_torch.cpu
        supported = {
            "current_device",
            "device_count",
            "Event",
            "is_available",
            "is_initialized",
            "synchronize",
        }
        self.assertEqual(
            actual_cpu.__all__,
            [name for name in expected_cpu.__all__ if name in supported],
        )

        actual_direct = {}
        expected_direct = {}
        exec("from torch_rs.cpu import Event", actual_direct)
        exec("from torch.cpu import Event", expected_direct)
        self.assertIs(actual_direct["Event"], actual_cpu.Event)
        self.assertIs(expected_direct["Event"], expected_cpu.Event)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.cpu import *", actual_namespace)
        exec("from torch.cpu import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            supported,
        )
        self.assertIs(actual_namespace["Event"], actual_cpu.Event)
        self.assertIs(expected_namespace["Event"], expected_cpu.Event)

        self.assertNotIn("Event", torch.__all__)
        self.assertFalse(hasattr(torch, "Event"))
        self.assertIn("Event", reference_torch.__all__)
        self.assertIsNot(reference_torch.Event, expected_cpu.Event)

    def test_copy_and_pickle_match_pytorch_2_13(self):
        actual_type = torch.cpu.Event
        expected_type = reference_torch.cpu.Event

        for event_type in (actual_type, expected_type):
            self.assertIs(copy.copy(event_type), event_type)
            self.assertIs(copy.deepcopy(event_type), event_type)
            for name in ("query", "record", "synchronize", "wait"):
                method = getattr(event_type, name)
                self.assertIs(copy.copy(method), method)
                self.assertIs(copy.deepcopy(method), method)

        actual = actual_type()
        expected = expected_type()
        actual.payload = [1, 2, 3]
        expected.payload = [1, 2, 3]
        for copier in (copy.copy, copy.deepcopy):
            with self.subTest(copier=copier.__name__):
                actual_copy = copier(actual)
                expected_copy = copier(expected)
                self.assertEqual(vars(actual_copy), vars(expected_copy))
                self.assertIs(type(actual_copy), actual_type)
                self.assertIs(type(expected_copy), expected_type)
                self.assertEqual(
                    actual_copy.payload is actual.payload,
                    expected_copy.payload is expected.payload,
                )

        actual_objects = (
            actual_type,
            actual,
            *(getattr(actual_type, name) for name in (
                "query",
                "record",
                "synchronize",
                "wait",
            )),
        )
        expected_objects = (
            expected_type,
            expected,
            *(getattr(expected_type, name) for name in (
                "query",
                "record",
                "synchronize",
                "wait",
            )),
        )
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            for index, (actual_value, expected_value) in enumerate(
                zip(actual_objects, expected_objects, strict=True)
            ):
                with self.subTest(protocol=protocol, value=index):
                    self.assertEqual(
                        self.pickle_shape(actual_value, protocol),
                        self.pickle_shape(expected_value, protocol),
                    )
                    actual_restored = pickle.loads(
                        pickle.dumps(actual_value, protocol=protocol)
                    )
                    expected_restored = pickle.loads(
                        pickle.dumps(expected_value, protocol=protocol)
                    )
                    if isinstance(actual_value, type) or isinstance(
                        actual_value, types.FunctionType
                    ):
                        self.assertIs(actual_restored, actual_value)
                        self.assertIs(expected_restored, expected_value)
                    else:
                        self.assertIs(type(actual_restored), actual_type)
                        self.assertIs(type(expected_restored), expected_type)
                        self.assertEqual(
                            vars(actual_restored), vars(expected_restored)
                        )

    def test_argument_errors_match_pytorch_2_13(self):
        actual_type = torch.cpu.Event
        expected_type = reference_torch.cpu.Event
        actual = actual_type()
        expected = expected_type()
        cases = (
            (lambda: actual_type(None), lambda: expected_type(None)),
            (lambda: actual_type(stream=None), lambda: expected_type(stream=None)),
            (lambda: actual.query(None), lambda: expected.query(None)),
            (
                lambda: actual.query(stream=None),
                lambda: expected.query(stream=None),
            ),
            (
                lambda: actual.record(None, None),
                lambda: expected.record(None, None),
            ),
            (
                lambda: actual.record(None, stream=None),
                lambda: expected.record(None, stream=None),
            ),
            (
                lambda: actual.wait(unexpected=None),
                lambda: expected.wait(unexpected=None),
            ),
            (
                lambda: actual.synchronize(None),
                lambda: expected.synchronize(None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def reload_outcome(self, module):
        cpu = module.cpu
        old_type = cpu.Event
        old_event = old_type()
        type_payload = pickle.dumps(old_type)
        event_payload = pickle.dumps(old_event)

        reloaded = importlib.reload(cpu)
        new_type = reloaded.Event
        restored_event = pickle.loads(event_payload)
        outcome = (
            reloaded is cpu,
            module.cpu is cpu,
            sys.modules[cpu.__name__] is cpu,
            new_type is old_type,
            type(old_event) is old_type,
            isinstance(old_event, new_type),
            old_event.query(),
            old_event.record(),
            old_event.wait(),
            old_event.synchronize(),
            pickle.loads(type_payload) is new_type,
            type(restored_event) is new_type,
            restored_event.query(),
            pickle.loads(pickle.dumps(new_type)) is new_type,
            type(pickle.loads(pickle.dumps(new_type()))) is new_type,
        )

        errors = []
        for value in (old_type, old_event):
            try:
                pickle.dumps(value)
            except BaseException as error:
                errors.append(
                    (
                        type(error).__name__,
                        str(error).replace("torch_rs", "torch"),
                        tuple(
                            item.replace("torch_rs", "torch")
                            if isinstance(item, str)
                            else item
                            for item in error.args
                        ),
                    )
                )
            else:
                errors.append(None)
        return outcome, errors

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_outcome(torch)
        expected = self.reload_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual[0],
            (
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                None,
                None,
                None,
                True,
                True,
                True,
                True,
                True,
            ),
        )
        self.assertTrue(all(error[0] == "PicklingError" for error in actual[1]))


if __name__ == "__main__":
    unittest.main()
