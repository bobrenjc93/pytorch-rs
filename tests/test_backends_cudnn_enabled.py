import importlib
import json
import subprocess
import sys
import threading
import unittest

import numpy as np

import torch_rs as torch


class _RejectTruthiness:
    def __bool__(self):
        raise AssertionError("cudnn.enabled must not request truthiness")


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnEnabledTests(unittest.TestCase):
    def setUp(self):
        self.cudnn = fresh_cudnn_module()
        self.original = self.cudnn.enabled
        self.cudnn.enabled = True

    def tearDown(self):
        cudnn = fresh_cudnn_module()
        cudnn.enabled = self.original

    def test_fresh_process_defaults_to_exact_true_without_cudnn_support(self):
        script = r'''
import json

import torch_rs as torch

cudnn = torch.backends.cudnn
initial = cudnn.enabled
cudnn.enabled = False
disabled = cudnn.enabled
cudnn.enabled = True
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "disabled": disabled,
    "restored": cudnn.enabled,
    "available": cudnn.is_available(),
    "version": cudnn.version(),
    "cuda": hasattr(torch, "cuda"),
    "execution": hasattr(torch, "cudnn_convolution"),
}))
'''
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {
                "initial": True,
                "initial_type": "bool",
                "disabled": False,
                "restored": True,
                "available": False,
                "version": None,
                "cuda": True,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_assignments_are_global_preferences_only(self):
        cudnn = self.cudnn

        self.assertIs(cudnn.enabled, True)
        self.assertIs(type(cudnn.enabled), bool)
        for enabled in (False, True, True, False, False, True):
            with self.subTest(enabled=enabled):
                cudnn.enabled = enabled
                self.assertIs(cudnn.enabled, enabled)
                self.assertIs(torch._C._get_cudnn_enabled(), enabled)
                self.assertIs(cudnn.is_available(), False)
                self.assertIs(cudnn.version(), None)

    def test_non_bool_values_are_rejected_without_coercion_or_state_change(self):
        invalid_values = (
            (None, "NoneType"),
            (0, "int"),
            (1, "int"),
            (0.0, "float"),
            (np.bool_(True), "numpy.bool"),
            ("", "str"),
            ([], "list"),
            (object(), "object"),
            (_RejectTruthiness(), "_RejectTruthiness"),
            (torch.tensor(True), "Tensor"),
            (torch.float32, "torch.dtype"),
            (torch.device("cpu"), "torch.device"),
            (torch.strided, "torch.layout"),
            (torch.Size([1]), "torch.Size"),
            (torch.finfo(torch.float32), "torch.finfo"),
        )
        for state in (False, True):
            self.cudnn.enabled = state
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_enabled_cudnn expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        self.cudnn.enabled = value
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(self.cudnn.enabled, state)
                    self.assertIs(torch._C._get_cudnn_enabled(), state)
                    self.assertIs(self.cudnn.is_available(), False)
                    self.assertIs(self.cudnn.version(), None)

    def test_state_is_process_global_across_threads_and_module_aliases(self):
        cudnn = self.cudnn
        imported = importlib.import_module("torch_rs.backends.cudnn")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []

        self.assertIs(imported, cudnn)

        def worker():
            try:
                observations.append(cudnn.enabled)
                imported.enabled = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(imported.enabled)
                cudnn.enabled = False
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cudnn.enabled, False)
        cudnn.enabled = True
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assertIs(cudnn.enabled, False)
        self.assertIs(torch._C._get_cudnn_enabled(), False)

    def test_reload_and_fresh_import_preserve_shared_state(self):
        backends = torch.backends
        cudnn = self.cudnn
        namespace = cudnn.__dict__
        cudnn.enabled = False

        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertIs(cudnn.enabled, False)
        self.assertIs(reloaded.enabled, False)

        reloaded.enabled = True
        self.assertIs(cudnn.enabled, True)
        cudnn.enabled = False
        self.assertIs(reloaded.enabled, False)

        fresh = fresh_cudnn_module()
        self.assertIs(torch.backends.cudnn, fresh)
        self.assertIs(fresh.enabled, False)
        fresh.enabled = True
        self.assertIs(cudnn.enabled, True)
        self.assertIs(reloaded.enabled, True)

    def test_proxy_property_imports_and_private_accessors(self):
        cudnn = self.cudnn
        descriptor = vars(type(cudnn))["enabled"]

        self.assertNotIn("enabled", vars(cudnn))
        self.assertNotIn("enabled", vars(cudnn.m))
        self.assertIs(descriptor.getter, torch._C._get_cudnn_enabled)
        self.assertIs(descriptor.setter, torch._C._set_cudnn_enabled)
        self.assertIs(descriptor.__get__(cudnn, type(cudnn)), True)

        imported = {}
        exec("from torch_rs.backends.cudnn import enabled", imported)
        self.assertIs(imported["enabled"], True)
        cudnn.enabled = False
        self.assertIs(imported["enabled"], True)
        exec("from torch_rs.backends.cudnn import enabled", imported)
        self.assertIs(imported["enabled"], False)
        self.assertNotIn("enabled", vars(cudnn))

        with self.assertRaises(AttributeError) as raised:
            del cudnn.enabled
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertIs(cudnn.enabled, False)

        self.assertTrue(hasattr(torch._C, "_get_cudnn_enabled"))
        self.assertTrue(hasattr(torch._C, "_set_cudnn_enabled"))
        self.assertFalse(hasattr(torch, "_get_cudnn_enabled"))
        self.assertFalse(hasattr(torch, "_set_cudnn_enabled"))
        self.assertNotIn("_get_cudnn_enabled", torch._C.__all__)
        self.assertNotIn("_set_cudnn_enabled", torch._C.__all__)


if __name__ == "__main__":
    unittest.main()
