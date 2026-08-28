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


IS_AVAILABLE_DOC = "Return a bool indicating if CUDNN is currently available."
VERSION_DOC = "Return the version of cuDNN."


def fresh_cudnn_module():
    module_name = "torch_rs.backends.cudnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "cudnn"):
        del torch.backends.cudnn
    module = importlib.import_module(module_name)
    torch.backends.cudnn = module
    return module


class CudnnIsAvailableTests(unittest.TestCase):
    def test_returns_cpu_build_metadata_without_runtime_probes(self):
        cudnn = torch.backends.cudnn
        is_available = cudnn.is_available
        version = cudnn.version
        self.assertEqual(
            is_available.__code__.co_names,
            ("torch", "_C", "_has_cudnn"),
        )
        self.assertEqual(is_available.__code__.co_freevars, ())
        self.assertEqual(is_available.__code__.co_cellvars, ())
        self.assertEqual(version.__code__.co_names, ("_init", "__cudnn_version"))
        self.assertEqual(version.__code__.co_freevars, ())
        self.assertEqual(version.__code__.co_cellvars, ())
        self.assertEqual(cudnn._init.__code__.co_names, ("torch", "_C", "_has_cudnn"))

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
                    availability = is_available()
                    self.assertIs(type(availability), bool)
                    self.assertIs(availability, False)
                    self.assertIs(availability, torch._C._has_cudnn)
                    self.assertIs(cudnn._init(), torch._C._has_cudnn)
                    self.assertIs(version(), None)

        self.assertFalse(hasattr(torch, "_has_cudnn"))
        self.assertNotIn("_has_cudnn", torch.__all__)
        self.assertNotIn("_has_cudnn", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        cudnn = importlib.import_module("torch_rs.backends.cudnn")
        is_available = cudnn.is_available
        version = cudnn.version

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
        self.assertIs(is_available, cudnn.m.is_available)
        self.assertIs(version, cudnn.m.version)
        self.assertIs(cudnn.torch, torch)

        for function, name, doc in (
            (is_available, "is_available", IS_AVAILABLE_DOC),
            (version, "version", VERSION_DOC),
        ):
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), "()")
                self.assertEqual(inspect.get_annotations(function), {})
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.backends.cudnn")
                self.assertIs(inspect.getmodule(function), cudnn)
                self.assertEqual(function.__doc__, doc)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        cudnn = importlib.import_module("torch_rs.backends.cudnn")
        functions = {
            "is_available": cudnn.is_available,
            "version": cudnn.version,
        }

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.cudnn, cudnn)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {
                "cpu",
                "cuda",
                "cudnn",
                "cusparselt",
                "kleidiai",
                "mha",
                "mkl",
                "nnpack",
                "openmp",
            },
        )

        package_import = {}
        backend_import = {}
        function_imports = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import cudnn", backend_import)
        for name in functions:
            namespace = {}
            exec(f"from torch_rs.backends.cudnn import {name}", namespace)
            function_imports[name] = namespace[name]
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.cudnn import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["cudnn"], cudnn)
        for name, function in functions.items():
            self.assertIs(function_imports[name], function)
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

        for name, function in functions.items():
            with self.subTest(function=name):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.backends.cudnn", payload)
                    self.assertIs(pickle.loads(payload), function)
        for copier in (copy.copy, copy.deepcopy):
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'CudnnModule' object$",
            ):
                copier(cudnn)

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        backends = torch.backends
        cudnn = backends.cudnn
        old_functions = {
            "is_available": cudnn.is_available,
            "version": cudnn.version,
        }
        namespace = cudnn.__dict__

        try:
            reloaded = importlib.reload(cudnn)

            self.assertIsNot(reloaded, cudnn)
            self.assertIs(cudnn.__dict__, namespace)
            self.assertIs(backends.cudnn, cudnn)
            self.assertIs(sys.modules[cudnn.__name__], reloaded)
            self.assertIs(reloaded.m, cudnn)
            for name, old_function in old_functions.items():
                new_function = getattr(cudnn, name)
                self.assertIsNot(new_function, old_function)
                self.assertIs(getattr(reloaded, name), new_function)
                self.assertIs(copy.copy(new_function), new_function)
                self.assertIs(copy.deepcopy(new_function), new_function)
                self.assertIs(
                    pickle.loads(pickle.dumps(new_function)),
                    new_function,
                )
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(old_function)
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
                self.assertEqual(
                    message,
                    f"Can't pickle <function {name} at 0x...>: "
                    "it's not the same object as "
                    f"torch_rs.backends.cudnn.{name}",
                )
            self.assertIs(cudnn.is_available(), False)
            self.assertIs(cudnn.version(), None)
        finally:
            fresh_cudnn_module()

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        for name in ("is_available", "version"):
            function = getattr(torch.backends.cudnn, name)
            cases = (
                ((None,), {}, f"{name}() takes 0 positional arguments but 1 was given"),
                (
                    (None, None),
                    {},
                    f"{name}() takes 0 positional arguments but 2 were given",
                ),
                (
                    (),
                    {"enabled": True},
                    f"{name}() got an unexpected keyword argument 'enabled'",
                ),
                (
                    (None,),
                    {"enabled": True},
                    f"{name}() got an unexpected keyword argument 'enabled'",
                ),
            )
            for args, kwargs, message in cases:
                with self.subTest(function=name, args=args, kwargs=kwargs):
                    with self.assertRaises(TypeError) as raised:
                        function(*args, **kwargs)
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))

    def test_configuration_execution_and_cuda_tensors_remain_unsupported(self):
        cudnn = torch.backends.cudnn
        self.assertIs(cudnn.version(), None)
        for name in (
            "CUDNN_TENSOR_DTYPES",
            "conv",
            "depthwise_kernel",
            "fp32_precision",
            "is_acceptable",
            "rnn",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cudnn, name))

        self.assertIs(type(cudnn.enabled), bool)
        self.assertIs(type(cudnn.benchmark), bool)
        self.assertIs(type(cudnn.benchmark_limit), int)
        self.assertIs(type(cudnn.deterministic), bool)
        self.assertIs(type(cudnn.allow_tf32), bool)
        self.assertTrue(hasattr(torch._C, "_get_cudnn_enabled"))
        self.assertTrue(hasattr(torch._C, "_set_cudnn_enabled"))
        self.assertTrue(hasattr(torch._C, "_get_cudnn_benchmark"))
        self.assertTrue(hasattr(torch._C, "_set_cudnn_benchmark"))
        self.assertTrue(hasattr(torch._C, "_cuda_get_cudnn_benchmark_limit"))
        self.assertTrue(hasattr(torch._C, "_cuda_set_cudnn_benchmark_limit"))
        self.assertTrue(hasattr(torch._C, "_get_cudnn_deterministic"))
        self.assertTrue(hasattr(torch._C, "_set_cudnn_deterministic"))
        self.assertTrue(hasattr(torch._C, "_get_cudnn_allow_tf32"))
        self.assertTrue(hasattr(torch._C, "_set_cudnn_allow_tf32"))
        for name in ("_cudnn",):
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
from torch_rs.backends.cudnn import flags, is_available, set_flags, version

assert torch.backends.cudnn is cudnn
assert cudnn.flags is flags
assert cudnn.is_available is is_available
assert cudnn.set_flags is set_flags
assert cudnn.version is version
assert is_available.__code__.co_names == ("torch", "_C", "_has_cudnn")
assert version.__code__.co_names == ("_init", "__cudnn_version")
assert cudnn._init.__code__.co_names == ("torch", "_C", "_has_cudnn")
assert is_available() is torch._C._has_cudnn is False
assert version() is None
assert cudnn.enabled is True
assert cudnn.benchmark is False
assert cudnn.benchmark_limit == 10
assert cudnn.deterministic is False
assert cudnn.allow_tf32 is True
cudnn.enabled = False
assert cudnn.enabled is False
assert cudnn.benchmark is False
assert cudnn.benchmark_limit == 10
assert cudnn.deterministic is False
assert cudnn.allow_tf32 is True
cudnn.benchmark = True
assert cudnn.enabled is False
assert cudnn.benchmark is True
assert cudnn.benchmark_limit == 10
assert cudnn.deterministic is False
assert cudnn.allow_tf32 is True
cudnn.benchmark_limit = 2**31
assert cudnn.enabled is False
assert cudnn.benchmark is True
assert cudnn.benchmark_limit == -(2**31)
assert cudnn.deterministic is False
assert cudnn.allow_tf32 is True
cudnn.deterministic = True
assert cudnn.enabled is False
assert cudnn.benchmark is True
assert cudnn.benchmark_limit == -(2**31)
assert cudnn.deterministic is True
assert cudnn.allow_tf32 is True
cudnn.allow_tf32 = False
assert cudnn.enabled is False
assert cudnn.benchmark is True
assert cudnn.benchmark_limit == -(2**31)
assert cudnn.deterministic is True
assert cudnn.allow_tf32 is False
assert is_available() is False
assert version() is None
cudnn.enabled = True
cudnn.benchmark = False
cudnn.benchmark_limit = 10
cudnn.deterministic = False
cudnn.allow_tf32 = True
assert not hasattr(torch, "_has_cudnn")
assert not hasattr(torch, "cuda")
assert set_flags() == (True, False, 10, False, True, "none", "auto")
with flags(False, True, 11, True, False):
    assert cudnn.enabled is False
    assert cudnn.benchmark is True
    assert cudnn.benchmark_limit == 11
    assert cudnn.deterministic is True
    assert cudnn.allow_tf32 is False
assert cudnn.enabled is True
assert cudnn.benchmark is False
assert cudnn.benchmark_limit == 10
assert cudnn.deterministic is False
assert cudnn.allow_tf32 is True
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
