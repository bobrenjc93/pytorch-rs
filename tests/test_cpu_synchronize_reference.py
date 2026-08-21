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


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CpuSynchronizeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "cpu.synchronize differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def threaded_outcome(self, module):
        function = module.cpu.synchronize
        devices = (
            None,
            0,
            -1,
            "cpu",
            "cuda:0",
            module.device("cpu"),
            module.tensor(1.0),
            object(),
        )
        baseline = tuple(function(device) for device in devices)
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
        return baseline, worker_states

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

    def test_ignored_devices_threading_and_grad_states_match_pytorch_2_13(self):
        actual_baseline, actual_workers = self.threaded_outcome(torch)
        expected_baseline, expected_workers = self.threaded_outcome(reference_torch)

        self.assertEqual(actual_baseline, expected_baseline)
        self.assertEqual(actual_workers, expected_workers)
        for baseline, worker_states in (
            (actual_baseline, actual_workers),
            (expected_baseline, expected_workers),
        ):
            self.assertEqual(baseline, (None,) * 8)
            for index, state in enumerate(worker_states):
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

        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertIn("H100", device_name)
        current_device = reference_torch.cuda.current_device()
        cuda_device = reference_torch.device("cuda", current_device)
        cuda_tensor = reference_torch.arange(4, device=cuda_device)

        for case, device in enumerate(
            (cuda_device, cuda_tensor, "cuda:0", current_device, object())
        ):
            with self.subTest(case=case):
                self.assertIs(reference_torch.cpu.synchronize(device), None)
                self.assertIs(torch.cpu.synchronize(device), None)
                self.assertEqual(reference_torch.cuda.current_device(), current_device)

        reference_torch.cuda.synchronize(current_device)
        self.assertEqual(cuda_tensor.cpu().tolist(), [0, 1, 2, 3])

    def test_signature_annotations_documentation_and_identity_match(self):
        actual_cpu = importlib.import_module("torch_rs.cpu")
        expected_cpu = importlib.import_module("torch.cpu")
        actual = actual_cpu.synchronize
        expected = expected_cpu.synchronize

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

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_cpu = torch.cpu
        expected_cpu = reference_torch.cpu
        actual = actual_cpu.synchronize
        expected = expected_cpu.synchronize
        supported = {
            "current_device",
            "device_count",
            "is_available",
            "is_initialized",
            "set_device",
            "synchronize",
        }

        self.assertEqual(
            actual_cpu.__all__,
            [name for name in expected_cpu.__all__ if name in supported],
        )
        for name in (
            "cpu",
            "device_count",
            "is_available",
            "is_initialized",
            "synchronize",
        ):
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
        exec("from torch_rs.cpu import synchronize", actual_direct_import)
        exec("from torch.cpu import synchronize", expected_direct_import)
        self.assertIs(actual_direct_import["synchronize"], actual)
        self.assertIs(expected_direct_import["synchronize"], expected)

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
            self.assertNotIn("synchronize", namespace)

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
        actual = torch.cpu.synchronize
        expected = reference_torch.cpu.synchronize
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

    def test_streams_events_and_device_mutation_remain_unsupported(self):
        actual_cpu = torch.cpu
        expected_cpu = reference_torch.cpu
        actual_public = {
            name for name in vars(actual_cpu) if not name.startswith("_")
        }
        expected_public = {
            name for name in vars(expected_cpu) if not name.startswith("_")
        }

        self.assertEqual(
            actual_public,
            {
                "current_device",
                "device_count",
                "is_available",
                "is_initialized",
                "set_device",
                "synchronize",
            },
        )
        unsupported = expected_public - actual_public
        self.assertTrue(
            {
                "amp",
                "current_stream",
                "Event",
                "get_capabilities",
                "Stream",
                "StreamContext",
                "stream",
            }.issubset(unsupported)
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual_cpu, name))


if __name__ == "__main__":
    unittest.main()
