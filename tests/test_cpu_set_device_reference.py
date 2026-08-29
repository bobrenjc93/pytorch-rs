import contextlib
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


class UnusableComparison:
    def __bool__(self):
        raise AssertionError("device truthiness was inspected")

    def __eq__(self, other):
        raise AssertionError("device equality was inspected")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuSetDeviceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cpu.set_device differentials require pinned PyTorch 2.13.0"
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

    def pickle_shape(self, function, protocol):
        shape = []
        for opcode, argument, _ in pickletools.genops(
            pickle.dumps(function, protocol=protocol)
        ):
            if opcode.name == "FRAME":
                argument = "<frame length>"
            elif isinstance(argument, str):
                argument = argument.replace("torch_rs", "torch")
            shape.append((opcode.name, argument))
        return shape

    def noop_outcome(self, module):
        cpu = module.cpu
        function = cpu.set_device
        baseline_stream = cpu.current_stream()
        baseline_device = cpu.current_device()
        devices = (
            None,
            0,
            -1,
            sys.maxsize,
            True,
            "cpu",
            "cpu:127",
            "cuda:0",
            "mps:0",
            "",
            module.device("cpu"),
            module.device("cpu", 0),
            module.tensor([1.0]),
            [],
            {"device": "cuda:0"},
            UnusableDevice(),
            UnusableComparison(),
        )
        observations = []
        for device in devices:
            observations.append(
                (
                    function(device),
                    cpu.current_device(),
                    cpu.current_stream() is baseline_stream,
                )
            )
            observations.append(
                (
                    function(device=device),
                    cpu.current_device(),
                    cpu.current_stream() is baseline_stream,
                )
            )
        return baseline_device, observations

    def threaded_outcome(self, module):
        cpu = module.cpu
        function = cpu.set_device
        baseline_stream = cpu.current_stream()
        baseline_device = cpu.current_device()
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        worker_states = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    worker_states[index] = (
                        module.is_grad_enabled(),
                        function(UnusableDevice()),
                        cpu.current_device(),
                        cpu.current_stream() is baseline_stream,
                        module.is_grad_enabled(),
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
        self.assertEqual(errors, [])
        return baseline_device, worker_states

    def tensor_outcome(self, module):
        leaf = module.tensor([[1.0, 2.0]], requires_grad=True)
        result = (leaf * 3.0).transpose(0, 1)
        metadata = (
            result.shape,
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        )

        set_result = module.cpu.set_device(result)
        unchanged = (
            result.shape,
            result.stride(),
            result.storage_offset(),
            result.data_ptr(),
            result.requires_grad,
            result.is_leaf,
        ) == metadata
        values = result.tolist()
        grad_before = leaf.grad
        result.sum().backward()
        return (
            set_result,
            module.cpu.current_device(),
            module.cpu.current_stream() is module.cpu._current_stream,
            unchanged,
            values,
            grad_before,
            leaf.grad.tolist(),
        )

    def reload_outcome(self, cpu):
        old_function = cpu.set_device
        namespace = cpu.__dict__

        reloaded = importlib.reload(cpu)
        new_function = reloaded.set_device
        pickle_error = None
        try:
            pickle.dumps(old_function)
        except pickle.PicklingError as error:
            pickle_error = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error))
        return (
            reloaded is cpu,
            reloaded.__dict__ is namespace,
            new_function is old_function,
            new_function(UnusableDevice()),
            reloaded.current_device(),
            reloaded.current_stream() is reloaded._default_cpu_stream,
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            pickle_error,
        )

    def test_noop_behavior_matches_pytorch_2_13(self):
        actual = self.noop_outcome(torch)
        expected = self.noop_outcome(reference_torch)

        self.assertEqual(actual, expected)
        baseline_device, observations = actual
        self.assertEqual(baseline_device, "cpu")
        self.assertEqual(observations, [(None, "cpu", True)] * len(observations))

    def test_threaded_and_grad_state_behavior_matches_pytorch_2_13(self):
        actual = self.threaded_outcome(torch)
        expected = self.threaded_outcome(reference_torch)

        self.assertEqual(actual, expected)
        baseline_device, worker_states = actual
        self.assertEqual(baseline_device, "cpu")
        for index, state in enumerate(worker_states):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                state,
                (expected_grad_state, None, "cpu", True, expected_grad_state),
            )

    def test_tensor_execution_behavior_matches_pytorch_2_13(self):
        actual = self.tensor_outcome(torch)
        expected = self.tensor_outcome(reference_torch)
        self.assertEqual(actual[:3], expected[:3])
        self.assertEqual(actual[3:], expected[3:])
        self.assertEqual(
            actual,
            (None, "cpu", True, True, [[3.0], [6.0]], None, [[3.0, 3.0]]),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual = actual_cpu.set_device
        expected = expected_cpu.set_device

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
        self.assertEqual(actual.__code__.co_names, expected.__code__.co_names)
        self.assertEqual(actual.__code__.co_freevars, expected.__code__.co_freevars)
        self.assertEqual(actual.__code__.co_cellvars, expected.__code__.co_cellvars)

    def test_imports_exports_copy_and_pickle_match_pytorch_2_13(self):
        actual_cpu = torch.cpu
        expected_cpu = reference_torch.cpu
        actual = actual_cpu.set_device
        expected = expected_cpu.set_device
        supported = {
            "current_device",
            "current_stream",
            "stream",
            "set_device",
            "device_count",
            "Event",
            "is_available",
            "is_initialized",
            "Stream",
            "StreamContext",
            "synchronize",
        }
        self.assertEqual(
            actual_cpu.__all__,
            [name for name in expected_cpu.__all__ if name in supported],
        )

        actual_direct = {}
        expected_direct = {}
        exec("from torch_rs.cpu import set_device", actual_direct)
        exec("from torch.cpu import set_device", expected_direct)
        self.assertIs(actual_direct["set_device"], actual)
        self.assertIs(expected_direct["set_device"], expected)

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

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cpu", namespace)
            self.assertNotIn("set_device", namespace)
            self.assertFalse(hasattr(module, "set_device"))

        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.copy(expected), expected)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(copy.deepcopy(expected), expected)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol)), actual)
                self.assertIs(pickle.loads(pickle.dumps(expected, protocol)), expected)
                self.assertEqual(
                    self.pickle_shape(actual, protocol),
                    self.pickle_shape(expected, protocol),
                )

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.cpu.set_device
        expected = reference_torch.cpu.set_device
        cases = (
            (lambda: actual(), lambda: expected()),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(unexpected=True), lambda: expected(unexpected=True)),
            (
                lambda: actual(None, device=None),
                lambda: expected(None, device=None),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_outcome(torch.cpu)
        expected = self.reload_outcome(reference_torch.cpu)
        self.assertEqual(
            actual[:-1],
            (True, True, False, None, "cpu", True, True, True, True),
        )
        self.assertEqual(actual[:-1], expected[:-1])
        self.assertEqual(
            actual[-1].replace("torch_rs", "torch"),
            expected[-1],
        )


if __name__ == "__main__":
    unittest.main()
