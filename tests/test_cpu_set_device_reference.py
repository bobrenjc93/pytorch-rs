import contextlib
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

    def ignored_device_outcome(self, module):
        function = module.cpu.set_device
        stream = module.cpu.current_stream()
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
            module.device("cpu"),
            module.device("cpu", 0),
            module.tensor([1.0]),
            [],
            {"device": "cuda:0"},
            UnusableDevice(),
        )

        positional_results = tuple(function(device) for device in devices)
        keyword_results = tuple(function(device=device) for device in devices)
        return (
            positional_results,
            keyword_results,
            module.cpu.current_device(),
            module.cpu.current_stream() is stream,
            function.__code__.co_names,
            function.__code__.co_freevars,
            function.__code__.co_cellvars,
        )

    def threaded_outcome(self, module):
        function = module.cpu.set_device
        devices = (
            None,
            0,
            -1,
            "cpu",
            "cuda:0",
            module.device("cpu"),
            module.tensor(1.0),
            UnusableDevice(),
        )
        worker_count = len(devices)
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
                        function(devices[index]),
                        module.is_grad_enabled(),
                        function(device=devices[-index - 1]),
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
        return worker_states

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
        set_result = module.cpu.set_device(result)
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
        return set_result, before == after, values, leaf.grad.tolist()

    def test_ignored_devices_threading_and_grad_states_match_pytorch_2_13(self):
        actual = self.ignored_device_outcome(torch)
        expected = self.ignored_device_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (
                (None,) * 16,
                (None,) * 16,
                "cpu",
                True,
                (),
                (),
                (),
            ),
        )

        actual_threads = self.threaded_outcome(torch)
        expected_threads = self.threaded_outcome(reference_torch)
        self.assertEqual(actual_threads, expected_threads)
        for index, state in enumerate(actual_threads):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                state,
                (
                    expected_grad_state,
                    None,
                    expected_grad_state,
                    None,
                    expected_grad_state,
                ),
            )
            self.assertIs(state[1], None)
            self.assertIs(state[3], None)

    def test_cuda_visible_h100_devices_and_tensors_are_ignored(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        current_device = reference_torch.cuda.current_device()
        cuda_device = reference_torch.device("cuda", current_device)
        cuda_tensor = reference_torch.arange(4, device=cuda_device)

        for case, device in enumerate(
            (cuda_device, cuda_tensor, "cuda:0", current_device, object())
        ):
            with self.subTest(case=case):
                self.assertIs(reference_torch.cpu.set_device(device), None)
                self.assertIs(torch.cpu.set_device(device), None)
                self.assertEqual(reference_torch.cuda.current_device(), current_device)
                self.assertEqual(reference_torch.cpu.current_device(), "cpu")
                self.assertEqual(torch.cpu.current_device(), "cpu")

        reference_torch.cuda.synchronize(current_device)
        self.assertEqual(cuda_tensor.cpu().tolist(), [0, 1, 2, 3])

    def test_tensor_execution_behavior_matches_pytorch_2_13(self):
        actual = self.tensor_outcome(torch)
        expected = self.tensor_outcome(reference_torch)
        self.assertEqual(actual, expected)
        self.assertEqual(
            actual,
            (None, True, [[3.0], [6.0]], [[3.0, 3.0]]),
        )

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual = actual_cpu.set_device
        expected = expected_cpu.set_device

        self.assertIs(torch.cpu, actual_cpu)
        self.assertIs(reference_torch.cpu, expected_cpu)
        self.assertIs(sys.modules["torch_rs.cpu"], actual_cpu)
        self.assertIs(sys.modules["torch.cpu"], expected_cpu)
        self.assertEqual(actual_cpu.__doc__, expected_cpu.__doc__)
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

    def test_imports_exports_copy_pickle_and_reload_match_pytorch_2_13(self):
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
            "Stream",
            "StreamContext",
            "Event",
            "is_available",
            "is_initialized",
            "synchronize",
        }

        self.assertEqual(
            actual_cpu.__all__,
            [name for name in expected_cpu.__all__ if name in supported],
        )
        for name in ("cpu", *sorted(supported - {"Event", "Stream"})):
            with self.subTest(top_level_export=name):
                self.assertEqual(
                    torch.__all__.count(name), reference_torch.__all__.count(name)
                )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import cpu", actual_package_import)
        exec("from torch import cpu", expected_package_import)
        self.assertIs(actual_package_import["cpu"], actual_cpu)
        self.assertIs(expected_package_import["cpu"], expected_cpu)

        actual_direct_import = {}
        expected_direct_import = {}
        exec("from torch_rs.cpu import set_device", actual_direct_import)
        exec("from torch.cpu import set_device", expected_direct_import)
        self.assertIs(actual_direct_import["set_device"], actual)
        self.assertIs(expected_direct_import["set_device"], expected)

        actual_cpu_namespace = {}
        expected_cpu_namespace = {}
        exec("from torch_rs.cpu import *", actual_cpu_namespace)
        exec("from torch.cpu import *", expected_cpu_namespace)
        self.assertEqual(
            {name for name in actual_cpu_namespace if not name.startswith("__")},
            supported,
        )
        for name in supported:
            with self.subTest(cpu_export=name):
                self.assertIs(actual_cpu_namespace[name], getattr(actual_cpu, name))
                self.assertIs(expected_cpu_namespace[name], getattr(expected_cpu, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("cpu", namespace)
            self.assertNotIn("set_device", namespace)

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

        actual_payload = pickle.dumps(actual)
        expected_payload = pickle.dumps(expected)
        self.assertIs(importlib.reload(actual_cpu), actual_cpu)
        self.assertIs(importlib.reload(expected_cpu), expected_cpu)
        new_actual = actual_cpu.set_device
        new_expected = expected_cpu.set_device
        self.assertIsNot(new_actual, actual)
        self.assertIsNot(new_expected, expected)
        self.assertIs(pickle.loads(actual_payload), new_actual)
        self.assertIs(pickle.loads(expected_payload), new_expected)
        self.assertIs(new_actual(object()), None)
        self.assertIs(new_expected(object()), None)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(actual)
        with self.assertRaises(pickle.PicklingError):
            pickle.dumps(expected)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.cpu.set_device
        expected = reference_torch.cpu.set_device
        cases = (
            (lambda: actual(), lambda: expected()),
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

        self.assertIs(actual(device=None), None)
        self.assertIs(expected(device=None), None)


if __name__ == "__main__":
    unittest.main()
