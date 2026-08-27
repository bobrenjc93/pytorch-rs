import importlib
import sys
import threading
import unittest

import numpy as np

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cudnn.benchmark must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnBenchmarkReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.benchmark differentials require pinned "
                "PyTorch 2.13.0"
            )

    def fresh_cudnn_module(self, root):
        module_name = f"{root.__name__}.backends.cudnn"
        sys.modules.pop(module_name, None)
        if hasattr(root.backends, "cudnn"):
            del root.backends.cudnn
        module = importlib.import_module(module_name)
        root.backends.cudnn = module
        return module

    def setUp(self):
        self.actual = self.fresh_cudnn_module(torch)
        self.expected = self.fresh_cudnn_module(reference_torch)
        self.actual_original_benchmark = self.actual.benchmark
        self.expected_original_benchmark = self.expected.benchmark
        self.actual_original_enabled = self.actual.enabled
        self.expected_original_enabled = self.expected.enabled
        self.actual.benchmark = False
        self.expected.benchmark = False
        self.actual.enabled = True
        self.expected.enabled = True

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        actual.benchmark = self.actual_original_benchmark
        expected.benchmark = self.expected_original_benchmark
        actual.enabled = self.actual_original_enabled
        expected.enabled = self.expected_original_enabled

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_exact_bool_updates_errors_and_independence_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        actual.enabled = False
        expected.enabled = False
        for benchmark in (True, False, False, True, True, False):
            with self.subTest(benchmark=benchmark):
                actual.benchmark = benchmark
                expected.benchmark = benchmark
                self.assertIs(actual.benchmark, expected.benchmark)
                self.assertIs(type(actual.benchmark), type(expected.benchmark))
                self.assertIs(actual.enabled, expected.enabled)

        invalid_values = (
            None,
            0,
            1,
            0.0,
            np.bool_(True),
            "",
            [],
            object(),
            _RejectTruthiness(),
        )
        for state in (False, True):
            actual.benchmark = state
            expected.benchmark = state
            actual.enabled = not state
            expected.enabled = not state
            for value in invalid_values:
                with self.subTest(state=state, value_type=type(value).__name__):
                    self.assert_error_matches(
                        lambda value=value: setattr(actual, "benchmark", value),
                        lambda value=value: setattr(expected, "benchmark", value),
                    )
                    self.assertIs(actual.benchmark, state)
                    self.assertIs(expected.benchmark, state)
                    self.assertIs(actual.enabled, not state)
                    self.assertIs(expected.enabled, not state)

        for actual_value, expected_value in (
            (torch.tensor(True), reference_torch.tensor(True)),
            (torch.float32, reference_torch.float32),
            (torch.device("cpu"), reference_torch.device("cpu")),
            (torch.strided, reference_torch.strided),
            (torch.Size([1]), reference_torch.Size([1])),
            (torch.finfo(torch.float32), reference_torch.finfo(reference_torch.float32)),
        ):
            with self.subTest(value_type=type(actual_value).__name__):
                self.assert_error_matches(
                    lambda: setattr(actual, "benchmark", actual_value),
                    lambda: setattr(expected, "benchmark", expected_value),
                )
                self.assertIs(actual.benchmark, True)
                self.assertIs(expected.benchmark, True)
                self.assertIs(actual.enabled, False)
                self.assertIs(expected.enabled, False)

    def thread_contract(self, module):
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        module.benchmark = False

        def worker():
            try:
                observations.append(module.benchmark)
                module.benchmark = True
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(module.benchmark)
                module.benchmark = True
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        main_saw_worker = module.benchmark is True
        module.benchmark = False
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            not thread.is_alive(),
            errors,
            observations,
            module.benchmark,
        )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def reload_contract(self, root, module):
        parent = root.backends
        namespace = module.__dict__
        module.benchmark = True

        reloaded = importlib.reload(module)
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            module.benchmark,
            reloaded.benchmark,
        )
        reloaded.benchmark = False
        old_saw_new = module.benchmark
        module.benchmark = True
        new_saw_old = reloaded.benchmark
        fresh = self.fresh_cudnn_module(root)
        fresh_state = fresh.benchmark
        return initial, old_saw_new, new_saw_old, fresh_state

    def test_reload_and_fresh_import_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_proxy_import_deletion_and_accessor_visibility_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_descriptor = vars(type(actual))["benchmark"]
        expected_descriptor = vars(type(expected))["benchmark"]

        self.assertNotIn("benchmark", vars(actual))
        self.assertNotIn("benchmark", vars(expected))
        self.assertNotIn("benchmark", vars(actual.m))
        self.assertNotIn("benchmark", vars(expected.m))
        self.assertEqual(
            hasattr(actual_descriptor, "getter"),
            hasattr(expected_descriptor, "getter"),
        )
        self.assertEqual(
            hasattr(actual_descriptor, "setter"),
            hasattr(expected_descriptor, "setter"),
        )
        self.assertIs(
            actual_descriptor.__get__(actual, type(actual)),
            expected_descriptor.__get__(expected, type(expected)),
        )

        actual_import = {}
        expected_import = {}
        exec("from torch_rs.backends.cudnn import benchmark", actual_import)
        exec("from torch.backends.cudnn import benchmark", expected_import)
        self.assertIs(actual_import["benchmark"], expected_import["benchmark"])
        actual.benchmark = True
        expected.benchmark = True
        exec("from torch_rs.backends.cudnn import benchmark", actual_import)
        exec("from torch.backends.cudnn import benchmark", expected_import)
        self.assertIs(actual_import["benchmark"], expected_import["benchmark"])

        self.assert_error_matches(
            lambda: delattr(actual, "benchmark"),
            lambda: delattr(expected, "benchmark"),
        )
        self.assertIs(actual.benchmark, True)
        self.assertIs(expected.benchmark, True)

        for root in (torch, reference_torch):
            with self.subTest(package=root.__name__):
                self.assertTrue(hasattr(root._C, "_get_cudnn_benchmark"))
                self.assertTrue(hasattr(root._C, "_set_cudnn_benchmark"))
                self.assertFalse(hasattr(root, "_get_cudnn_benchmark"))
                self.assertFalse(hasattr(root, "_set_cudnn_benchmark"))


if __name__ == "__main__":
    unittest.main()
