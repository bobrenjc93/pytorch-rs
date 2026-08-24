import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = "Return a bool indicating if CUDNN is currently available."


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnIsAvailableTests(unittest.TestCase):
    def test_returns_exact_false_native_build_metadata_without_runtime_probes(self):
        function = torch.backends.cudnn.is_available
        self.assertEqual(function.__code__.co_names, ("torch", "_C", "_has_cudnn"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_CUDNN": "1"},
            {"CUDA_VISIBLE_DEVICES": ""},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
                "USE_CUDNN": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_cudnn)

        self.assertFalse(hasattr(torch, "_has_cudnn"))
        self.assertNotIn("_has_cudnn", torch.__all__)
        self.assertNotIn("_has_cudnn", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        cudnn = importlib.import_module("torch_rs.backends.cudnn")
        function = cudnn.is_available

        self.assertIs(torch.backends.cudnn, cudnn)
        self.assertIs(sys.modules["torch_rs.backends.cudnn"], cudnn)
        self.assertIsInstance(cudnn, types.ModuleType)
        self.assertEqual(type(cudnn).__name__, "CudnnModule")
        self.assertEqual(type(cudnn).__module__, "torch_rs.backends.cudnn")
        self.assertIsNone(cudnn.__doc__)
        self.assertFalse(hasattr(cudnn, "__all__"))
        self.assertEqual(
            {name for name in vars(cudnn) if not name.startswith("_")},
            {"m"},
        )
        self.assertIs(type(cudnn.m), types.ModuleType)
        self.assertIsNot(cudnn.m, cudnn)
        self.assertEqual(cudnn.m.__name__, cudnn.__name__)
        self.assertIs(function, cudnn.m.is_available)
        self.assertIs(cudnn.torch, torch)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.backends.cudnn")
        self.assertIs(inspect.getmodule(function), cudnn)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        cudnn = importlib.import_module("torch_rs.backends.cudnn")
        function = cudnn.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.cudnn, cudnn)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {"cuda", "cudnn", "mkl", "nnpack", "openmp"},
        )

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import cudnn", backend_import)
        exec("from torch_rs.backends.cudnn import is_available", function_import)
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.cudnn import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["cudnn"], cudnn)
        self.assertIs(function_import["is_available"], function)
        self.assertIs(parent_wildcard["cudnn"], cudnn)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"m"},
        )
        self.assertIs(child_wildcard["m"], cudnn.m)

        self.assertNotIn("backends", torch.__all__)
        self.assertFalse(hasattr(torch, "cudnn"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("cudnn", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for copier in (copy.copy, copy.deepcopy):
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'CudnnModule' object$",
            ):
                copier(cudnn)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cudnn", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        backends = torch.backends
        cudnn = backends.cudnn
        old_function = cudnn.is_available
        namespace = cudnn.__dict__

        try:
            reloaded = importlib.reload(cudnn)

            self.assertIsNot(reloaded, cudnn)
            self.assertIs(cudnn.__dict__, namespace)
            self.assertIs(backends.cudnn, cudnn)
            self.assertIs(sys.modules[cudnn.__name__], reloaded)
            self.assertIs(reloaded.m, cudnn)
            self.assertIsNot(cudnn.is_available, old_function)
            self.assertIs(reloaded.is_available, cudnn.is_available)
            self.assertIs(cudnn.is_available(), False)
            self.assertIs(copy.copy(cudnn.is_available), cudnn.is_available)
            self.assertIs(copy.deepcopy(cudnn.is_available), cudnn.is_available)
            self.assertIs(
                pickle.loads(pickle.dumps(cudnn.is_available)),
                cudnn.is_available,
            )
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(old_function)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                "Can't pickle <function is_available at 0x...>: "
                "it's not the same object as torch_rs.backends.cudnn.is_available",
            )
        finally:
            fresh_cudnn_module()

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.cudnn.is_available
        cases = (
            (
                lambda: function(None),
                "is_available() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_available() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_version_configuration_execution_and_cuda_tensors_remain_unsupported(self):
        cudnn = torch.backends.cudnn
        for name in (
            "CUDNN_TENSOR_DTYPES",
            "allow_tf32",
            "benchmark",
            "benchmark_limit",
            "conv",
            "depthwise_kernel",
            "deterministic",
            "enabled",
            "flags",
            "fp32_precision",
            "is_acceptable",
            "rnn",
            "set_flags",
            "version",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cudnn, name))

        for name in (
            "_cudnn",
            "_get_cudnn_enabled",
            "_set_cudnn_enabled",
            "_get_cudnn_benchmark",
            "_set_cudnn_benchmark",
        ):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch._C, name))

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cudnn", "cupy", "nvidia", "numpy", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
    USE_CUDNN="1",
)
import torch_rs as torch
from torch_rs.backends import cudnn
from torch_rs.backends.cudnn import is_available

assert torch.backends.cudnn is cudnn
assert cudnn.is_available is is_available
assert is_available.__code__.co_names == ("torch", "_C", "_has_cudnn")
assert is_available() is torch._C._has_cudnn is False
assert not hasattr(torch, "_has_cudnn")
assert not hasattr(torch, "cuda")
assert not hasattr(cudnn, "version")
assert not hasattr(cudnn, "flags")
assert not any(
    name.split(".", 1)[0] in RejectExternalRuntimeImport.blocked
    for name in sys.modules
)
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


if __name__ == "__main__":
    unittest.main()
