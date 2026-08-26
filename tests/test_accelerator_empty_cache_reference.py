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
class AcceleratorEmptyCacheReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "accelerator.empty_cache differentials require pinned PyTorch 2.13.0"
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

    def test_signature_documentation_identity_and_exports_match(self):
        actual_accelerator = importlib.import_module("torch_rs.accelerator")
        expected_accelerator = importlib.import_module("torch.accelerator")
        actual_memory = importlib.import_module("torch_rs.accelerator.memory")
        expected_memory = importlib.import_module("torch.accelerator.memory")
        actual = actual_memory.empty_cache
        expected = expected_memory.empty_cache

        self.assertIs(actual_accelerator.memory, actual_memory)
        self.assertIs(expected_accelerator.memory, expected_memory)
        self.assertIs(sys.modules["torch_rs.accelerator.memory"], actual_memory)
        self.assertIs(sys.modules["torch.accelerator.memory"], expected_memory)
        self.assertIs(actual_accelerator.empty_cache, actual)
        self.assertIs(expected_accelerator.empty_cache, expected)
        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(str(inspect.signature(actual)), str(inspect.signature(expected)))
        self.assertEqual(inspect.get_annotations(actual), inspect.get_annotations(expected))
        self.assertEqual(typing.get_type_hints(actual), typing.get_type_hints(expected))
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertIs(inspect.getmodule(actual), actual_memory)
        self.assertIs(inspect.getmodule(expected), expected_memory)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__dict__, expected.__dict__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(actual_memory.__doc__, expected_memory.__doc__)
        self.assertEqual(
            actual_memory.__all__,
            [name for name in expected_memory.__all__ if name == "empty_cache"],
        )
        self.assertEqual(
            actual_accelerator.__all__.count("empty_cache"),
            expected_accelerator.__all__.count("empty_cache"),
        )

        actual_memory_namespace = {}
        expected_memory_namespace = {}
        actual_accelerator_namespace = {}
        expected_accelerator_namespace = {}
        exec("from torch_rs.accelerator.memory import *", actual_memory_namespace)
        exec("from torch.accelerator.memory import *", expected_memory_namespace)
        exec("from torch_rs.accelerator import *", actual_accelerator_namespace)
        exec("from torch.accelerator import *", expected_accelerator_namespace)
        self.assertEqual(
            {name for name in actual_memory_namespace if not name.startswith("__")},
            {"empty_cache"},
        )
        self.assertIs(actual_memory_namespace["empty_cache"], actual)
        self.assertIs(expected_memory_namespace["empty_cache"], expected)
        self.assertIs(actual_accelerator_namespace["empty_cache"], actual)
        self.assertIs(expected_accelerator_namespace["empty_cache"], expected)

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

    def threaded_outcome(self, module):
        function = module.accelerator.empty_cache
        memory_function = module.accelerator.memory.empty_cache
        baseline = tuple(
            call()
            for _ in range(8)
            for call in (function, memory_function)
        )
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
                        tuple(
                            call()
                            for _ in range(8)
                            for call in (function, memory_function)
                        ),
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
        return baseline, results

    def test_repeatability_threading_and_grad_mode_match(self):
        actual = self.threaded_outcome(torch)
        expected = self.threaded_outcome(reference_torch)
        self.assertEqual(actual, expected)
        baseline, results = actual
        self.assertEqual(baseline, (None,) * 16)
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (expected_grad_state, (None,) * 16, expected_grad_state),
            )

    def reload_outcome(self, module):
        accelerator = module.accelerator
        memory = accelerator.memory
        original = memory.empty_cache
        old_accelerator_all = accelerator.__all__
        old_memory_all = memory.__all__

        reloaded_accelerator = importlib.reload(accelerator)
        after_accelerator_reload = (
            reloaded_accelerator is accelerator,
            module.accelerator is accelerator,
            accelerator.memory is memory,
            accelerator.empty_cache is original,
            accelerator.__all__ is not old_accelerator_all,
            accelerator.empty_cache(),
        )

        reloaded_memory = importlib.reload(memory)
        replacement = memory.empty_cache
        try:
            pickle.dumps(original)
        except pickle.PicklingError:
            original_pickleable = False
        else:
            original_pickleable = True
        after_memory_reload = (
            reloaded_memory is memory,
            accelerator.memory is memory,
            memory.__all__ is not old_memory_all,
            replacement is not original,
            accelerator.empty_cache is original,
            original(),
            replacement(),
            original_pickleable,
            pickle.loads(pickle.dumps(replacement)) is replacement,
        )

        importlib.reload(accelerator)
        restored = (
            accelerator.empty_cache is replacement,
            accelerator.memory.empty_cache is replacement,
            accelerator.empty_cache(),
        )
        return after_accelerator_reload, after_memory_reload, restored

    def test_package_and_memory_reload_behavior_matches(self):
        self.assertEqual(
            self.reload_outcome(torch),
            self.reload_outcome(reference_torch),
        )

    def test_real_cuda_allocation_keeps_torch_rs_cpu_build_behavior(self):
        torch_rs_state = (
            torch.accelerator.current_accelerator(),
            torch.accelerator.is_available(),
            torch.accelerator.device_count(),
            torch.accelerator.empty_cache(),
            torch.accelerator.memory.empty_cache(),
            torch._C._has_cuda,
            torch.version.cuda,
        )
        self.assertEqual(torch_rs_state, (None, False, 0, None, None, False, None))

        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch build")

        device_index = reference_torch.cuda.current_device()
        device = reference_torch.device("cuda", device_index)
        self.assertIs(reference_torch.accelerator.empty_cache(), None)

        allocation = reference_torch.full((4096,), 2.0, device=device)
        reference_torch.cuda.synchronize(device_index)
        self.assertGreater(allocation.data_ptr(), 0)
        self.assertEqual(allocation.sum().item(), 8192.0)

        self.assertIs(reference_torch.accelerator.empty_cache(), None)
        self.assertEqual(allocation[0].item(), 2.0)
        self.assertIs(torch.accelerator.empty_cache(), None)
        self.assertEqual(
            (
                torch.accelerator.current_accelerator(),
                torch.accelerator.is_available(),
                torch.accelerator.device_count(),
                torch.accelerator.empty_cache(),
                torch.accelerator.memory.empty_cache(),
                torch._C._has_cuda,
                torch.version.cuda,
            ),
            torch_rs_state,
        )

        del allocation
        self.assertIs(reference_torch.accelerator.empty_cache(), None)
        self.assertIs(reference_torch.accelerator.empty_cache(), None)
        self.assertIs(torch.accelerator.empty_cache(), None)

    def test_argument_errors_match_pytorch_2_13(self):
        actual = torch.accelerator.empty_cache
        expected = reference_torch.accelerator.empty_cache
        cases = (
            (lambda: actual(None), lambda: expected(None)),
            (lambda: actual(None, None), lambda: expected(None, None)),
            (lambda: actual(device=True), lambda: expected(device=True)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
