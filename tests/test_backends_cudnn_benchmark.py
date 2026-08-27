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
        raise AssertionError("cudnn.benchmark must not request truthiness")


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.cudnn = fresh_cudnn_module()
        self.original_benchmark = self.cudnn.benchmark
        self.original_enabled = self.cudnn.enabled
        self.cudnn.benchmark = False
        self.cudnn.enabled = True

    def tearDown(self):
        cudnn = fresh_cudnn_module()
        cudnn.benchmark = self.original_benchmark
        cudnn.enabled = self.original_enabled

    def test_fresh_process_defaults_to_exact_false_without_cudnn_support(self):
        script = r'''
import json

import torch_rs as torch

cudnn = torch.backends.cudnn
initial = cudnn.benchmark
cudnn.benchmark = True
enabled_after_benchmark = cudnn.enabled
cudnn.enabled = False
benchmark_after_enabled = cudnn.benchmark
cudnn.benchmark = False
print(json.dumps({
    "initial": initial,
    "initial_type": type(initial).__name__,
    "enabled_after_benchmark": enabled_after_benchmark,
    "benchmark_after_enabled": benchmark_after_enabled,
    "restored": cudnn.benchmark,
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
                "initial": False,
                "initial_type": "bool",
                "enabled_after_benchmark": True,
                "benchmark_after_enabled": True,
                "restored": False,
                "available": False,
                "version": None,
                "cuda": False,
                "execution": False,
            },
        )

    def test_repeated_exact_bool_assignments_are_independent_preferences(self):
        cudnn = self.cudnn

        self.assertIs(cudnn.benchmark, False)
        self.assertIs(type(cudnn.benchmark), bool)
        for enabled in (False, True):
            cudnn.enabled = enabled
            for benchmark in (True, False, False, True, True, False):
                with self.subTest(enabled=enabled, benchmark=benchmark):
                    cudnn.benchmark = benchmark
                    self.assertIs(cudnn.benchmark, benchmark)
                    self.assertIs(torch._C._get_cudnn_benchmark(), benchmark)
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
            enabled = not state
            self.cudnn.benchmark = state
            self.cudnn.enabled = enabled
            for value, type_name in invalid_values:
                with self.subTest(state=state, value_type=type_name):
                    message = (
                        "set_benchmark_cudnn expects a bool, but got "
                        f"{type_name}"
                    )
                    with self.assertRaises(RuntimeError) as raised:
                        self.cudnn.benchmark = value
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))
                    self.assertIs(self.cudnn.benchmark, state)
                    self.assertIs(torch._C._get_cudnn_benchmark(), state)
                    self.assertIs(self.cudnn.enabled, enabled)
                    self.assertIs(self.cudnn.is_available(), False)
                    self.assertIs(self.cudnn.version(), None)

    def test_state_is_process_global_across_threads_and_module_aliases(self):
        cudnn = self.cudnn
        imported = importlib.import_module("torch_rs.backends.cudnn")
        worker_changed = threading.Event()
        main_changed = threading.Event()
        observations = []
        errors = []
        cudnn.benchmark = True
        cudnn.enabled = False

        self.assertIs(imported, cudnn)

        def worker():
            try:
                observations.append(cudnn.benchmark)
                imported.benchmark = False
                worker_changed.set()
                if not main_changed.wait(timeout=10):
                    raise RuntimeError("timed out waiting for main-thread update")
                observations.append(imported.benchmark)
                cudnn.benchmark = False
            except BaseException as error:
                errors.append(error)
                worker_changed.set()

        thread = threading.Thread(target=worker)
        thread.start()
        self.assertTrue(worker_changed.wait(timeout=10))
        self.assertEqual(errors, [])
        self.assertIs(cudnn.benchmark, False)
        cudnn.benchmark = True
        main_changed.set()
        thread.join(timeout=10)

        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(observations, [True, True])
        self.assertIs(cudnn.benchmark, False)
        self.assertIs(torch._C._get_cudnn_benchmark(), False)
        self.assertIs(cudnn.enabled, False)

    def test_reload_and_fresh_import_preserve_shared_state(self):
        backends = torch.backends
        cudnn = self.cudnn
        namespace = cudnn.__dict__
        cudnn.benchmark = True
        cudnn.enabled = False

        reloaded = importlib.reload(cudnn)

        self.assertIsNot(reloaded, cudnn)
        self.assertIs(cudnn.__dict__, namespace)
        self.assertIs(backends.cudnn, cudnn)
        self.assertIs(sys.modules[cudnn.__name__], reloaded)
        self.assertIs(reloaded.m, cudnn)
        self.assertIs(cudnn.benchmark, True)
        self.assertIs(reloaded.benchmark, True)
        self.assertIs(cudnn.enabled, False)
        self.assertIs(reloaded.enabled, False)

        reloaded.benchmark = False
        self.assertIs(cudnn.benchmark, False)
        cudnn.benchmark = True
        self.assertIs(reloaded.benchmark, True)

        fresh = fresh_cudnn_module()
        self.assertIs(torch.backends.cudnn, fresh)
        self.assertIs(fresh.benchmark, True)
        fresh.benchmark = False
        self.assertIs(cudnn.benchmark, False)
        self.assertIs(reloaded.benchmark, False)
        self.assertIs(fresh.enabled, False)

    def test_proxy_imports_deletion_and_private_accessors(self):
        cudnn = self.cudnn
        descriptor = vars(type(cudnn))["benchmark"]

        self.assertNotIn("benchmark", vars(cudnn))
        self.assertNotIn("benchmark", vars(cudnn.m))
        self.assertIs(descriptor.getter, torch._C._get_cudnn_benchmark)
        self.assertIs(descriptor.setter, torch._C._set_cudnn_benchmark)
        self.assertIs(descriptor.__get__(cudnn, type(cudnn)), False)

        imported = {}
        exec("from torch_rs.backends.cudnn import benchmark", imported)
        self.assertIs(imported["benchmark"], False)
        cudnn.benchmark = True
        self.assertIs(imported["benchmark"], False)
        exec("from torch_rs.backends.cudnn import benchmark", imported)
        self.assertIs(imported["benchmark"], True)
        self.assertNotIn("benchmark", vars(cudnn))

        wildcard = {}
        exec("from torch_rs.backends.cudnn import *", wildcard)
        self.assertEqual(
            {name for name in wildcard if not name.startswith("__")},
            {"m"},
        )

        with self.assertRaises(AttributeError) as raised:
            del cudnn.benchmark
        self.assertEqual(str(raised.exception), "__delete__")
        self.assertEqual(raised.exception.args, ("__delete__",))
        self.assertIs(cudnn.benchmark, True)

        self.assertTrue(hasattr(torch._C, "_get_cudnn_benchmark"))
        self.assertTrue(hasattr(torch._C, "_set_cudnn_benchmark"))
        self.assertFalse(hasattr(torch, "_get_cudnn_benchmark"))
        self.assertFalse(hasattr(torch, "_set_cudnn_benchmark"))
        self.assertNotIn("_get_cudnn_benchmark", torch._C.__all__)
        self.assertNotIn("_set_cudnn_benchmark", torch._C.__all__)


if __name__ == "__main__":
    unittest.main()
