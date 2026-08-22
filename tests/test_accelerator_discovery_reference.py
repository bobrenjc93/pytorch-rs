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


class ExplosiveBool:
    def __bool__(self):
        raise AssertionError("check_available truthiness was evaluated")


@contextlib.contextmanager
def reference_without_accelerator():
    original = reference_torch._C._accelerator_getAccelerator
    reference_torch._C._accelerator_getAccelerator = lambda: None
    try:
        yield
    finally:
        reference_torch._C._accelerator_getAccelerator = original


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AcceleratorDiscoveryReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "accelerator discovery differentials require pinned PyTorch 2.13.0"
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

    def normalized_annotations(self, function, *, resolve=False):
        annotations = (
            typing.get_type_hints(function) if resolve else function.__annotations__
        )
        return {
            name: repr(annotation).replace("torch_rs", "torch")
            for name, annotation in annotations.items()
        }

    def threaded_outcome(self, module):
        accelerator = module.accelerator
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = module.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        module.is_grad_enabled(),
                        accelerator.current_accelerator(),
                        accelerator.current_accelerator(True),
                        accelerator.is_available(),
                        accelerator.device_count(),
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
        return results

    def reload_outcome(self, module):
        original_module = module.accelerator
        original_functions = {
            name: getattr(original_module, name)
            for name in ("current_accelerator", "device_count", "is_available")
        }
        module_name = original_module.__name__

        reloaded = importlib.reload(original_module)
        reloaded_functions = {
            name: getattr(reloaded, name) for name in original_functions
        }
        after_reload = (
            reloaded is original_module,
            module.accelerator is original_module,
            tuple(
                reloaded_functions[name] is not original_functions[name]
                for name in original_functions
            ),
            reloaded.current_accelerator(),
            reloaded.current_accelerator(True),
            reloaded.is_available(),
            reloaded.device_count(),
        )

        try:
            removed = sys.modules.pop(module_name)
            replacement = importlib.import_module(module_name)
            after_reimport = (
                removed is original_module,
                replacement is not original_module,
                sys.modules[module_name] is replacement,
                module.accelerator is replacement,
                tuple(
                    getattr(replacement, name) is not reloaded_functions[name]
                    for name in original_functions
                ),
                replacement.current_accelerator(),
                replacement.current_accelerator(True),
                replacement.is_available(),
                replacement.device_count(),
            )
        finally:
            sys.modules[module_name] = original_module
            module.accelerator = original_module

        return after_reload, after_reimport

    def test_cpu_only_values_and_thread_behavior_match_pytorch_2_13(self):
        actual = (
            torch.accelerator.current_accelerator(),
            torch.accelerator.current_accelerator(False),
            torch.accelerator.current_accelerator(True),
            torch.accelerator.current_accelerator(None),
            torch.accelerator.current_accelerator(ExplosiveBool()),
            torch.accelerator.is_available(),
            torch.accelerator.device_count(),
        )
        with reference_without_accelerator():
            expected = (
                reference_torch.accelerator.current_accelerator(),
                reference_torch.accelerator.current_accelerator(False),
                reference_torch.accelerator.current_accelerator(True),
                reference_torch.accelerator.current_accelerator(None),
                reference_torch.accelerator.current_accelerator(ExplosiveBool()),
                reference_torch.accelerator.is_available(),
                reference_torch.accelerator.device_count(),
            )
            expected_threads = self.threaded_outcome(reference_torch)

        actual_threads = self.threaded_outcome(torch)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_threads, expected_threads)
        self.assertEqual(actual, (None, None, None, None, None, False, 0))
        self.assertIs(actual[5], False)
        self.assertIs(type(actual[6]), int)

    def test_signatures_annotations_documentation_and_identity_match(self):
        actual_module = importlib.import_module("torch_rs.accelerator")
        expected_module = importlib.import_module("torch.accelerator")

        self.assertIs(torch.accelerator, actual_module)
        self.assertIs(reference_torch.accelerator, expected_module)
        self.assertIs(sys.modules["torch_rs.accelerator"], actual_module)
        self.assertIs(sys.modules["torch.accelerator"], expected_module)
        self.assertEqual(actual_module.__doc__, expected_module.__doc__)

        for name in ("current_accelerator", "device_count", "is_available"):
            with self.subTest(name=name):
                actual = getattr(actual_module, name)
                expected = getattr(expected_module, name)
                self.assertIs(type(actual), types.FunctionType)
                self.assertIs(type(expected), types.FunctionType)
                self.assertEqual(
                    str(inspect.signature(actual)).replace("torch_rs", "torch"),
                    str(inspect.signature(expected)),
                )
                self.assertEqual(
                    self.normalized_annotations(actual),
                    self.normalized_annotations(expected),
                )
                self.assertEqual(
                    self.normalized_annotations(actual, resolve=True),
                    self.normalized_annotations(expected, resolve=True),
                )
                self.assertEqual(actual.__name__, expected.__name__)
                self.assertEqual(actual.__qualname__, expected.__qualname__)
                self.assertEqual(
                    actual.__module__.replace("torch_rs", "torch"),
                    expected.__module__,
                )
                self.assertIs(inspect.getmodule(actual), actual_module)
                self.assertIs(inspect.getmodule(expected), expected_module)
                self.assertEqual(actual.__doc__, expected.__doc__)
                self.assertEqual(actual.__defaults__, expected.__defaults__)
                self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
                self.assertEqual(actual.__dict__, expected.__dict__)
                self.assertEqual(
                    hasattr(actual, "__text_signature__"),
                    hasattr(expected, "__text_signature__"),
                )

    def test_imports_exports_copy_and_pickle_match_the_supported_scope(self):
        actual_module = torch.accelerator
        expected_module = reference_torch.accelerator
        supported = {"current_accelerator", "device_count", "is_available"}

        self.assertEqual(
            actual_module.__all__,
            [name for name in expected_module.__all__ if name in supported],
        )
        self.assertEqual(
            torch.__all__.count("accelerator"),
            reference_torch.__all__.count("accelerator"),
        )

        actual_package_import = {}
        expected_package_import = {}
        exec("from torch_rs import accelerator", actual_package_import)
        exec("from torch import accelerator", expected_package_import)
        self.assertIs(actual_package_import["accelerator"], actual_module)
        self.assertIs(expected_package_import["accelerator"], expected_module)

        actual_namespace = {}
        expected_namespace = {}
        exec("from torch_rs.accelerator import *", actual_namespace)
        exec("from torch.accelerator import *", expected_namespace)
        self.assertEqual(
            {name for name in actual_namespace if not name.startswith("__")},
            supported,
        )
        for name in supported:
            self.assertIs(actual_namespace[name], getattr(actual_module, name))
            self.assertIs(expected_namespace[name], getattr(expected_module, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("accelerator", namespace)

        for name in supported:
            actual = getattr(actual_module, name)
            expected = getattr(expected_module, name)
            with self.subTest(name=name):
                self.assertIs(copy.copy(actual), actual)
                self.assertIs(copy.copy(expected), expected)
                self.assertIs(copy.deepcopy(actual), actual)
                self.assertIs(copy.deepcopy(expected), expected)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual, protocol)), actual
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected, protocol)), expected
                    )
                    self.assertEqual(
                        self.pickle_shape(actual, protocol),
                        self.pickle_shape(expected, protocol),
                    )

    def test_argument_forms_and_errors_match_pytorch_2_13(self):
        actual = torch.accelerator
        expected = reference_torch.accelerator

        current_cases = (
            (
                lambda: actual.current_accelerator(None, None),
                lambda: expected.current_accelerator(None, None),
            ),
            (
                lambda: actual.current_accelerator(device=True),
                lambda: expected.current_accelerator(device=True),
            ),
            (
                lambda: actual.current_accelerator(None, device=True),
                lambda: expected.current_accelerator(None, device=True),
            ),
            (
                lambda: actual.current_accelerator(
                    False, check_available=True
                ),
                lambda: expected.current_accelerator(
                    False, check_available=True
                ),
            ),
        )
        no_argument_cases = []
        for name in ("is_available", "device_count"):
            actual_function = getattr(actual, name)
            expected_function = getattr(expected, name)
            no_argument_cases.extend(
                (
                    (
                        lambda function=actual_function: function(None),
                        lambda function=expected_function: function(None),
                    ),
                    (
                        lambda function=actual_function: function(None, None),
                        lambda function=expected_function: function(None, None),
                    ),
                    (
                        lambda function=actual_function: function(device=True),
                        lambda function=expected_function: function(device=True),
                    ),
                )
            )

        for case, (actual_call, expected_call) in enumerate(
            (*current_cases, *no_argument_cases)
        ):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        with reference_without_accelerator():
            self.assertIs(
                actual.current_accelerator(check_available=True),
                expected.current_accelerator(check_available=True),
            )
            self.assertIs(actual.is_available(), expected.is_available())
            self.assertEqual(actual.device_count(), expected.device_count())

    def test_reload_and_reimport_match_pytorch_2_13(self):
        actual = self.reload_outcome(torch)
        with reference_without_accelerator():
            expected = self.reload_outcome(reference_torch)
        self.assertEqual(actual, expected)

    def test_visible_h100_cuda_hardware_does_not_leak_into_torch_rs(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        current = reference_torch.accelerator.current_accelerator()
        self.assertIsNotNone(current)
        self.assertEqual(current.type, "cuda")
        self.assertIsNone(current.index)
        self.assertIs(reference_torch.accelerator.is_available(), True)
        self.assertGreaterEqual(reference_torch.accelerator.device_count(), 1)
        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)

        self.assertIs(torch.accelerator.current_accelerator(), None)
        self.assertIs(torch.accelerator.current_accelerator(True), None)
        self.assertIs(torch.accelerator.is_available(), False)
        self.assertEqual(torch.accelerator.device_count(), 0)
        self.assertFalse(hasattr(torch, "cuda"))

        probe = reference_torch.ones(
            1, device=reference_torch.device("cuda", 0)
        )
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(0)

        self.assertIs(torch.accelerator.current_accelerator(), None)
        self.assertIs(torch.accelerator.is_available(), False)
        self.assertEqual(torch.accelerator.device_count(), 0)

    def test_selection_stream_memory_graph_and_execution_apis_stay_unsupported(self):
        actual_public = {
            name for name in vars(torch.accelerator) if not name.startswith("_")
        }
        expected_public = {
            name
            for name in vars(reference_torch.accelerator)
            if not name.startswith("_")
        }
        self.assertEqual(
            actual_public,
            {"current_accelerator", "device_count", "is_available"},
        )
        unsupported = expected_public - actual_public
        self.assertTrue(
            {
                "Graph",
                "current_device_index",
                "current_stream",
                "device_index",
                "empty_cache",
                "get_memory_info",
                "memory_stats",
                "set_device_index",
                "set_stream",
                "synchronize",
            }.issubset(unsupported)
        )
        for name in unsupported:
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.accelerator, name))


if __name__ == "__main__":
    unittest.main()
