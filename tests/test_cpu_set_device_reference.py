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


class ExplodingDevice:
    def __repr__(self):
        raise AssertionError("device repr was inspected")

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

    def normalized_annotation(self, annotation):
        return str(annotation).replace("torch_rs", "torch")

    def ignored_devices(self, module):
        return (
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
            ExplodingDevice(),
            object(),
        )

    def threaded_outcome(self, module):
        function = module.cpu.set_device
        devices = self.ignored_devices(module)
        baseline = tuple(function(device) for device in devices)
        keyword_baseline = tuple(function(device=device) for device in devices)
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
                        function(devices[index]),
                        module.is_grad_enabled(),
                        function(device=devices[-index - 1]),
                        module.is_grad_enabled(),
                        module.cpu.current_device(),
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
        return baseline, keyword_baseline, worker_states

    def state_outcome(self, module):
        original_stream = module.cpu.current_stream()
        selected_stream = module.cpu.Stream()
        with module.cpu.stream(selected_stream):
            before = module.cpu.current_stream() is selected_stream
            result = module.cpu.set_device("cuda:0")
            after_device = module.cpu.current_device()
            after_stream = module.cpu.current_stream() is selected_stream
        restored = module.cpu.current_stream() is original_stream
        return before, result, after_device, after_stream, restored

    def test_ignored_devices_threading_grad_and_state_match_pytorch_2_13(self):
        actual_baseline, actual_keywords, actual_workers = self.threaded_outcome(torch)
        expected_baseline, expected_keywords, expected_workers = self.threaded_outcome(
            reference_torch
        )

        self.assertEqual(actual_baseline, expected_baseline)
        self.assertEqual(actual_keywords, expected_keywords)
        self.assertEqual(actual_workers, expected_workers)
        self.assertEqual(actual_baseline, (None,) * len(actual_baseline))
        self.assertEqual(actual_keywords, (None,) * len(actual_keywords))
        for index, state in enumerate(actual_workers):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                state,
                (
                    expected_grad_state,
                    None,
                    expected_grad_state,
                    None,
                    expected_grad_state,
                    "cpu",
                ),
            )
            self.assertIs(state[1], None)
            self.assertIs(state[3], None)
        self.assertEqual(
            self.state_outcome(torch),
            self.state_outcome(reference_torch),
        )
        self.assertEqual(self.state_outcome(torch), (True, None, "cpu", True, True))

    def test_cuda_visible_h100_device_selection_is_ignored(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        current_device = reference_torch.cuda.current_device()
        cuda_device = reference_torch.device("cuda", current_device)
        cuda_tensor = reference_torch.arange(4, device=cuda_device)

        for case, device in enumerate((cuda_device, cuda_tensor, "cuda:0")):
            with self.subTest(case=case):
                self.assertIs(reference_torch.cpu.set_device(device), None)
                self.assertIs(torch.cpu.set_device(device), None)
                self.assertEqual(reference_torch.cuda.current_device(), current_device)
                self.assertEqual(torch.cpu.current_device(), "cpu")

        reference_torch.cuda.synchronize(current_device)
        self.assertEqual(cuda_tensor.cpu().tolist(), [0, 1, 2, 3])

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
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(actual.__annotations__.keys(), expected.__annotations__.keys())
        for name in actual.__annotations__:
            with self.subTest(annotation=name):
                self.assertEqual(
                    self.normalized_annotation(actual.__annotations__[name]),
                    self.normalized_annotation(expected.__annotations__[name]),
                )
        actual_hints = typing.get_type_hints(actual)
        expected_hints = typing.get_type_hints(expected)
        self.assertEqual(actual_hints.keys(), expected_hints.keys())
        for name in actual_hints:
            with self.subTest(type_hint=name):
                self.assertEqual(
                    self.normalized_annotation(actual_hints[name]),
                    self.normalized_annotation(expected_hints[name]),
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

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
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

    def reload_contract(self, root):
        module = root.cpu
        old_function = module.set_device
        namespace = module.__dict__
        reloaded = importlib.reload(module)
        new_function = module.set_device

        try:
            pickle.dumps(old_function)
        except Exception as error:
            stale_pickle_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-fA-F]+", "0x...", str(error)).replace(
                    "torch_rs", "torch"
                ),
            )
        else:
            self.fail("a stale cpu.set_device function remained pickleable")

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.cpu is module,
            sys.modules[module.__name__] is module,
            old_function is not new_function,
            new_function("cpu") is None,
            module.current_device() == "cpu",
            copy.copy(new_function) is new_function,
            copy.deepcopy(new_function) is new_function,
            pickle.loads(pickle.dumps(new_function)) is new_function,
            stale_pickle_error,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        actual = torch.cpu.set_device
        expected = reference_torch.cpu.set_device
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
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


if __name__ == "__main__":
    unittest.main()
