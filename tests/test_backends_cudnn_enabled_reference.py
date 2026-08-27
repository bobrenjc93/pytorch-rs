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
        raise AssertionError("cudnn.enabled must not request truthiness")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CudnnEnabledReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "backends.cudnn.enabled differentials require pinned "
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
        self.actual_original = self.actual.enabled
        self.expected_original = self.expected.enabled
        self.actual.enabled = True
        self.expected.enabled = True

    def tearDown(self):
        actual = self.fresh_cudnn_module(torch)
        expected = self.fresh_cudnn_module(reference_torch)
        actual.enabled = self.actual_original
        expected.enabled = self.expected_original

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    def test_exact_bool_updates_and_errors_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected

        for enabled in (False, True, True, False, False, True):
            with self.subTest(enabled=enabled):
                actual.enabled = enabled
                expected.enabled = enabled
                self.assertIs(actual.enabled, expected.enabled)
                self.assertIs(type(actual.enabled), type(expected.enabled))

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
            actual.enabled = state
            expected.enabled = state
            for value in invalid_values:
                with self.subTest(state=state, value_type=type(value).__name__):
                    self.assert_error_matches(
                        lambda value=value: setattr(actual, "enabled", value),
                        lambda value=value: setattr(expected, "enabled", value),
                    )
                    self.assertIs(actual.enabled, state)
                    self.assertIs(expected.enabled, state)

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
                    lambda: setattr(actual, "enabled", actual_value),
                    lambda: setattr(expected, "enabled", expected_value),
                )
                self.assertIs(actual.enabled, True)
                self.assertIs(expected.enabled, True)

    def thread_contract(self, module):
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        module.enabled = True

        def worker():
            try:
                observations.append(module.enabled)
                module.enabled = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(module.enabled)
                module.enabled = False
            except BaseException as error:
                errors.append((type(error).__name__, str(error)))
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        worker_ready = worker_changed.wait(timeout=10)
        main_saw_worker = module.enabled is False
        module.enabled = True
        main_changed.set()
        thread.join(timeout=10)
        return (
            worker_ready,
            main_saw_worker,
            not thread.is_alive(),
            errors,
            observations,
            module.enabled,
        )

    def test_process_global_thread_visibility_matches_pytorch_2_13(self):
        self.assertEqual(
            self.thread_contract(self.actual),
            self.thread_contract(self.expected),
        )

    def reload_contract(self, root, module):
        parent = root.backends
        namespace = module.__dict__
        module.enabled = False

        reloaded = importlib.reload(module)
        initial = (
            reloaded is module,
            module.__dict__ is namespace,
            parent.cudnn is module,
            sys.modules[module.__name__] is reloaded,
            reloaded.m is module,
            module.enabled,
            reloaded.enabled,
        )
        reloaded.enabled = True
        old_saw_new = module.enabled
        module.enabled = False
        new_saw_old = reloaded.enabled
        fresh = self.fresh_cudnn_module(root)
        fresh_state = fresh.enabled
        return initial, old_saw_new, new_saw_old, fresh_state

    def test_reload_and_fresh_import_state_match_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch, self.actual),
            self.reload_contract(reference_torch, self.expected),
        )

    def test_proxy_access_and_import_match_pytorch_2_13(self):
        actual = self.actual
        expected = self.expected
        actual_descriptor = vars(type(actual))["enabled"]
        expected_descriptor = vars(type(expected))["enabled"]

        self.assertNotIn("enabled", vars(actual))
        self.assertNotIn("enabled", vars(expected))
        self.assertNotIn("enabled", vars(actual.m))
        self.assertNotIn("enabled", vars(expected.m))
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
        exec("from torch_rs.backends.cudnn import enabled", actual_import)
        exec("from torch.backends.cudnn import enabled", expected_import)
        self.assertIs(actual_import["enabled"], expected_import["enabled"])
        actual.enabled = False
        expected.enabled = False
        exec("from torch_rs.backends.cudnn import enabled", actual_import)
        exec("from torch.backends.cudnn import enabled", expected_import)
        self.assertIs(actual_import["enabled"], expected_import["enabled"])

        self.assert_error_matches(
            lambda: delattr(actual, "enabled"),
            lambda: delattr(expected, "enabled"),
        )
        self.assertIs(actual.enabled, False)
        self.assertIs(expected.enabled, False)


if __name__ == "__main__":
    unittest.main()
