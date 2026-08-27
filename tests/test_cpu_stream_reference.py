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


class UnusableArgument:
    def __getattribute__(self, name):
        raise AssertionError(f"argument attribute was inspected: {name}")

    def __repr__(self):
        raise AssertionError("argument representation was inspected")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuStreamReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cpu.Stream differentials require pinned PyTorch 2.13.0"
            )

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

    def stream_outcome(self, module):
        argument = UnusableArgument()
        stream = module.cpu.Stream(argument)
        keyword_stream = module.cpu.Stream(priority=argument)
        return (
            stream.record_event(),
            stream.wait_event(argument),
            stream.wait_event(event=argument),
            stream.wait_stream(argument),
            stream.wait_stream(stream=argument),
            vars(stream),
            vars(keyword_stream),
        )

    def threaded_outcome(self, module):
        stream = module.cpu.Stream()
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                argument = object()
                local_stream = module.cpu.Stream(priority=argument)
                barrier.wait(timeout=10)
                results[index] = (
                    stream.record_event(),
                    stream.wait_event(argument),
                    stream.wait_stream(local_stream),
                    local_stream.wait_stream(stream),
                    vars(local_stream),
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
        return results, errors, vars(stream)

    def test_stateless_probe_free_operations_and_threading_match_pytorch_2_13(self):
        actual = self.stream_outcome(torch)
        expected = self.stream_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual, (None, None, None, None, None, {}, {}))

        actual_threads = self.threaded_outcome(torch)
        expected_threads = self.threaded_outcome(reference_torch)
        self.assertEqual(actual_threads, expected_threads)
        self.assertEqual(actual_threads[1:], ([], {}))

        for module in (torch, reference_torch):
            stream_type = module.cpu.Stream
            for method in (
                stream_type.__init__,
                stream_type.record_event,
                stream_type.wait_event,
                stream_type.wait_stream,
            ):
                with self.subTest(module=module.__name__, method=method.__name__):
                    self.assertEqual(method.__code__.co_names, ())
                    self.assertEqual(method.__code__.co_freevars, ())
                    self.assertEqual(method.__code__.co_cellvars, ())

    def test_class_constructor_and_method_metadata_match_pytorch_2_13(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual = actual_cpu.Stream
        expected = expected_cpu.Stream

        self.assertIs(type(actual), type(expected))
        self.assertEqual(
            str(inspect.signature(actual)), str(inspect.signature(expected))
        )
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
            {"record_event", "wait_event", "wait_stream"},
        )

        for name in ("__init__", *sorted(public_class_attributes)):
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
                self.assertEqual(
                    actual_method.__qualname__, expected_method.__qualname__
                )
                self.assertEqual(
                    actual_method.__module__.replace("torch_rs", "torch"),
                    expected_method.__module__,
                )
                self.assertEqual(actual_method.__doc__, expected_method.__doc__)
                self.assertEqual(
                    actual_method.__defaults__, expected_method.__defaults__
                )
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
            "Stream",
            "synchronize",
        }
        self.assertEqual(
            actual_cpu.__all__,
            [name for name in expected_cpu.__all__ if name in supported],
        )

        actual_direct = {}
        expected_direct = {}
        exec("from torch_rs.cpu import Stream", actual_direct)
        exec("from torch.cpu import Stream", expected_direct)
        self.assertIs(actual_direct["Stream"], actual_cpu.Stream)
        self.assertIs(expected_direct["Stream"], expected_cpu.Stream)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.cpu import *", actual_namespace)
        exec("from torch.cpu import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            supported,
        )
        self.assertIs(actual_namespace["Stream"], actual_cpu.Stream)
        self.assertIs(expected_namespace["Stream"], expected_cpu.Stream)

        self.assertNotIn("Stream", torch.__all__)
        self.assertFalse(hasattr(torch, "Stream"))
        self.assertIn("Stream", reference_torch.__all__)
        self.assertIsNot(reference_torch.Stream, expected_cpu.Stream)

    def test_copy_and_pickle_match_pytorch_2_13(self):
        actual_type = torch.cpu.Stream
        expected_type = reference_torch.cpu.Stream

        for stream_type in (actual_type, expected_type):
            self.assertIs(copy.copy(stream_type), stream_type)
            self.assertIs(copy.deepcopy(stream_type), stream_type)
            for name in (
                "__init__",
                "record_event",
                "wait_event",
                "wait_stream",
            ):
                method = getattr(stream_type, name)
                self.assertIs(copy.copy(method), method)
                self.assertIs(copy.deepcopy(method), method)

        actual = actual_type(priority=object())
        expected = expected_type(priority=object())
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

        method_names = ("__init__", "record_event", "wait_event", "wait_stream")
        actual_objects = (
            actual_type,
            actual,
            *(getattr(actual_type, name) for name in method_names),
        )
        expected_objects = (
            expected_type,
            expected,
            *(getattr(expected_type, name) for name in method_names),
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
        actual_type = torch.cpu.Stream
        expected_type = reference_torch.cpu.Stream
        actual = actual_type()
        expected = expected_type()
        cases = (
            (lambda: actual_type(None, None), lambda: expected_type(None, None)),
            (
                lambda: actual_type(1, priority=2),
                lambda: expected_type(1, priority=2),
            ),
            (
                lambda: actual_type(unexpected=1),
                lambda: expected_type(unexpected=1),
            ),
            (
                lambda: actual.record_event(None),
                lambda: expected.record_event(None),
            ),
            (
                lambda: actual.record_event(event=None),
                lambda: expected.record_event(event=None),
            ),
            (lambda: actual.wait_event(), lambda: expected.wait_event()),
            (
                lambda: actual.wait_event(None, event=None),
                lambda: expected.wait_event(None, event=None),
            ),
            (
                lambda: actual.wait_stream(unexpected=None),
                lambda: expected.wait_stream(unexpected=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def reload_outcome(self, module):
        cpu = module.cpu
        old_type = cpu.Stream
        old_stream = old_type()
        type_payload = pickle.dumps(old_type)
        stream_payload = pickle.dumps(old_stream)

        reloaded = importlib.reload(cpu)
        new_type = reloaded.Stream
        restored_stream = pickle.loads(stream_payload)
        outcome = (
            reloaded is cpu,
            module.cpu is cpu,
            sys.modules[cpu.__name__] is cpu,
            new_type is old_type,
            type(old_stream) is old_type,
            isinstance(old_stream, new_type),
            old_stream.record_event(),
            old_stream.wait_event(object()),
            old_stream.wait_stream(object()),
            pickle.loads(type_payload) is new_type,
            type(restored_stream) is new_type,
            restored_stream.record_event(),
            pickle.loads(pickle.dumps(new_type)) is new_type,
            type(pickle.loads(pickle.dumps(new_type()))) is new_type,
        )

        errors = []
        for value in (old_type, old_stream):
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
                None,
                None,
                None,
                True,
                True,
                None,
                True,
                True,
            ),
        )
        self.assertTrue(all(error[0] == "PicklingError" for error in actual[1]))

    def test_stream_selection_and_context_apis_remain_unsupported(self):
        for name in ("current_stream", "stream", "StreamContext"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.cpu, name))
                self.assertTrue(hasattr(reference_torch.cpu, name))


if __name__ == "__main__":
    unittest.main()
