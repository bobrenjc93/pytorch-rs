import copy
import importlib
import inspect
import pickle
import pickletools
import re
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


class UnusableDevice:
    def __getattribute__(self, name):
        raise AssertionError(f"device attribute was inspected: {name}")

    def __repr__(self):
        raise AssertionError("device representation was inspected")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuCurrentStreamReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cpu.current_stream differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def normalized(self, value):
        return str(value).replace("torch_rs", "torch")

    def normalized_error(self, value):
        return re.sub(r"0x[0-9a-fA-F]+", "<address>", self.normalized(value))

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

    def identity_outcome(self, module):
        function = module.cpu.current_stream
        stream = function()
        devices = (
            None,
            0,
            -1,
            True,
            "cpu",
            "cuda:0",
            module.device("cpu"),
            module.tensor([1.0]),
            [],
            {"device": 0},
            UnusableDevice(),
        )
        return (
            type(stream) is module.cpu.Stream,
            stream is module.cpu._default_cpu_stream,
            stream is module.cpu._current_stream,
            tuple(function(device) is stream for device in devices),
            tuple(function(device=device) is stream for device in devices),
            vars(stream),
            function.__code__.co_names,
            function.__code__.co_freevars,
            function.__code__.co_cellvars,
        )

    def threaded_outcome(self, module):
        function = module.cpu.current_stream
        stream = function()
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    function(UnusableDevice()) is stream,
                    function(device=object()) is stream,
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
        return results, errors

    def tensor_outcome(self, module):
        leaf = module.tensor([[1.0, 2.0]], requires_grad=True)
        result = (leaf * 3.0).transpose(0, 1)
        before = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        )
        stream_is_canonical = (
            module.cpu.current_stream(result) is module.cpu.current_stream()
        )
        after = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        )
        values = result.tolist()
        result.sum().backward()
        return stream_is_canonical, before == after, values, leaf.grad.tolist()

    def test_identity_ignored_devices_and_threads_match_pytorch_2_13(self):
        actual = self.identity_outcome(torch)
        expected = self.identity_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (
                True,
                True,
                True,
                (True,) * 11,
                (True,) * 11,
                {},
                ("_current_stream",),
                (),
                (),
            ),
        )

        actual_threads = self.threaded_outcome(torch)
        expected_threads = self.threaded_outcome(reference_torch)
        self.assertEqual(actual_threads, expected_threads)
        self.assertEqual(actual_threads, ([(True, True)] * 16, []))

    def test_cuda_visible_h100_devices_and_tensors_are_ignored(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        current_device = reference_torch.cuda.current_device()
        cuda_device = reference_torch.device("cuda", current_device)
        cuda_tensor = reference_torch.arange(4, device=cuda_device)
        actual_stream = torch.cpu.current_stream()
        expected_stream = reference_torch.cpu.current_stream()

        for case, device in enumerate(
            (cuda_device, cuda_tensor, "cuda:0", current_device, object())
        ):
            with self.subTest(case=case):
                self.assertIs(torch.cpu.current_stream(device), actual_stream)
                self.assertIs(
                    reference_torch.cpu.current_stream(device), expected_stream
                )
                self.assertEqual(reference_torch.cuda.current_device(), current_device)

        reference_torch.cuda.synchronize(current_device)
        self.assertEqual(cuda_tensor.cpu().tolist(), [0, 1, 2, 3])

    def test_tensor_execution_behavior_matches_pytorch_2_13(self):
        actual = self.tensor_outcome(torch)
        expected = self.tensor_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (True, True, [[3.0], [6.0]], [[3.0, 3.0]]),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual = actual_cpu.current_stream
        expected = expected_cpu.current_stream

        self.assertIs(torch.cpu, actual_cpu)
        self.assertIs(reference_torch.cpu, expected_cpu)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(
            self.normalized(inspect.signature(actual)),
            self.normalized(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__.keys(), expected.__annotations__.keys())
        for name in actual.__annotations__:
            with self.subTest(annotation=name):
                self.assertEqual(
                    self.normalized(actual.__annotations__[name]),
                    self.normalized(expected.__annotations__[name]),
                )
        actual_hints = typing.get_type_hints(actual)
        expected_hints = typing.get_type_hints(expected)
        self.assertEqual(actual_hints.keys(), expected_hints.keys())
        for name in actual_hints:
            with self.subTest(type_hint=name):
                self.assertEqual(
                    self.normalized(actual_hints[name]),
                    self.normalized(expected_hints[name]),
                )
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_cpu)
        self.assertIs(inspect.getmodule(expected), expected_cpu)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )

    def test_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_cpu = torch.cpu
        expected_cpu = reference_torch.cpu
        supported = {
            "current_device",
            "current_stream",
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
        exec("from torch_rs.cpu import current_stream", actual_direct)
        exec("from torch.cpu import current_stream", expected_direct)
        self.assertIs(actual_direct["current_stream"], actual_cpu.current_stream)
        self.assertIs(expected_direct["current_stream"], expected_cpu.current_stream)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.cpu import *", actual_namespace)
        exec("from torch.cpu import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            supported,
        )
        for name in supported:
            with self.subTest(export=name):
                self.assertIs(actual_namespace[name], getattr(actual_cpu, name))
                self.assertIs(expected_namespace[name], getattr(expected_cpu, name))

        self.assertEqual(
            torch.__all__.count("current_stream"),
            reference_torch.__all__.count("current_stream"),
        )
        self.assertFalse(hasattr(torch, "current_stream"))
        self.assertFalse(hasattr(reference_torch, "current_stream"))

        actual_function = actual_cpu.current_stream
        expected_function = expected_cpu.current_stream
        self.assertIs(copy.copy(actual_function), actual_function)
        self.assertIs(copy.copy(expected_function), expected_function)
        self.assertIs(copy.deepcopy(actual_function), actual_function)
        self.assertIs(copy.deepcopy(expected_function), expected_function)

        actual_stream = actual_function()
        expected_stream = expected_function()
        actual_stream.payload = [1, 2, 3]
        expected_stream.payload = [1, 2, 3]
        try:
            for copier in (copy.copy, copy.deepcopy):
                with self.subTest(copier=copier.__name__):
                    actual_copy = copier(actual_stream)
                    expected_copy = copier(expected_stream)
                    self.assertEqual(vars(actual_copy), vars(expected_copy))
                    self.assertEqual(
                        actual_copy.payload is actual_stream.payload,
                        expected_copy.payload is expected_stream.payload,
                    )
                    self.assertIsNot(actual_copy, actual_stream)
                    self.assertIsNot(expected_copy, expected_stream)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                for actual_value, expected_value in (
                    (actual_function, expected_function),
                    (actual_stream, expected_stream),
                ):
                    with self.subTest(protocol=protocol, value=type(actual_value)):
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
                        if isinstance(actual_value, types.FunctionType):
                            self.assertIs(actual_restored, actual_value)
                            self.assertIs(expected_restored, expected_value)
                        else:
                            self.assertIsNot(actual_restored, actual_value)
                            self.assertIsNot(expected_restored, expected_value)
                            self.assertEqual(
                                vars(actual_restored), vars(expected_restored)
                            )
                            self.assertIsNot(actual_restored, actual_function())
                            self.assertIsNot(expected_restored, expected_function())
        finally:
            del actual_stream.payload
            del expected_stream.payload

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.cpu.current_stream
        expected = reference_torch.cpu.current_stream
        cases = (
            (lambda: actual(None, None), lambda: expected(None, None)),
            (
                lambda: actual(None, None, None),
                lambda: expected(None, None, None),
            ),
            (
                lambda: actual(unexpected=True),
                lambda: expected(unexpected=True),
            ),
            (
                lambda: actual(None, device=None),
                lambda: expected(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def reload_outcome(self, module):
        cpu = module.cpu
        old_function = cpu.current_stream
        old_type = cpu.Stream
        old_stream = old_function()
        old_stream.payload = [1, 2, 3]
        function_payload = pickle.dumps(old_function)
        type_payload = pickle.dumps(old_type)
        stream_payload = pickle.dumps(old_stream)

        reloaded = importlib.reload(cpu)
        new_function = reloaded.current_stream
        new_type = reloaded.Stream
        new_stream = new_function()
        restored_stream = pickle.loads(stream_payload)
        outcome = (
            reloaded is cpu,
            module.cpu is cpu,
            sys.modules[cpu.__name__] is cpu,
            new_function is old_function,
            new_type is old_type,
            new_stream is old_stream,
            type(new_stream) is new_type,
            new_stream is cpu._default_cpu_stream,
            new_stream is cpu._current_stream,
            old_function() is new_stream,
            type(old_stream) is old_type,
            isinstance(old_stream, new_type),
            pickle.loads(function_payload) is new_function,
            pickle.loads(type_payload) is new_type,
            restored_stream is new_stream,
            type(restored_stream) is new_type,
            vars(restored_stream),
            pickle.loads(pickle.dumps(new_function)) is new_function,
            type(pickle.loads(pickle.dumps(new_stream))) is new_type,
        )

        errors = []
        for value in (old_function, old_type, old_stream):
            try:
                pickle.dumps(value)
            except BaseException as error:
                errors.append(
                    (
                        type(error).__name__,
                        self.normalized_error(error),
                        tuple(
                            self.normalized_error(item)
                            if isinstance(item, str)
                            else item
                            for item in error.args
                        ),
                    )
                )
            else:
                errors.append(None)
        return outcome, errors

    def test_module_reload_behavior_matches_pytorch_2_13(self):
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
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
                True,
                {"payload": [1, 2, 3]},
                True,
                True,
            ),
        )
        self.assertTrue(all(error[0] == "PicklingError" for error in actual[1]))

    def test_stream_context_apis_remain_unsupported(self):
        for name in ("stream", "StreamContext"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.cpu, name))
                self.assertTrue(hasattr(reference_torch.cpu, name))


if __name__ == "__main__":
    unittest.main()
