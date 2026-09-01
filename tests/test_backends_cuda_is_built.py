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


FUNCTION_DOC = """
    Return whether PyTorch is built with CUDA support.

    Note that this doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run on a machine with working CUDA drivers and devices, we would be able to use it.
    """


class CudaIsBuiltTests(unittest.TestCase):
    def test_returns_exact_false_native_build_metadata_without_runtime_probes(self):
        function = torch.backends.cuda.is_built
        self.assertEqual(function.__code__.co_names, ("torch", "_C", "_has_cuda"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_cuda)

        self.assertFalse(hasattr(torch, "_has_cuda"))
        self.assertNotIn("_has_cuda", torch.__all__)
        self.assertNotIn("_has_cuda", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        cuda = importlib.import_module("torch_rs.backends.cuda")
        function = cuda.is_built

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIsNone(cuda.__doc__)
        self.assertEqual(
            cuda.__all__,
            [
                "is_built",
                "cuBLASModule",
                "is_ck_sdpa_available",
                "matmul",
                "enable_cudnn_sdp",
                "cudnn_sdp_enabled",
                "enable_flash_sdp",
                "flash_sdp_enabled",
                "enable_mem_efficient_sdp",
                "mem_efficient_sdp_enabled",
                "math_sdp_enabled",
                "enable_math_sdp",
                "allow_fp16_bf16_reduction_math_sdp",
                "fp16_bf16_reduction_math_sdp_allowed",
                "is_flash_attention_available",
            ],
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_built")
        self.assertEqual(function.__qualname__, "is_built")
        self.assertEqual(function.__module__, "torch_rs.backends.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(
            inspect.cleandoc(function.__doc__), inspect.cleandoc(FUNCTION_DOC)
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        cuda = importlib.import_module("torch_rs.backends.cuda")
        function = cuda.is_built

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.cuda, cuda)
        self.assertIs(cuda.torch, torch)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {
                "cpu",
                "cuda",
                "cusparselt",
                "cudnn",
                "kleidiai",
                "m",
                "mha",
                "mkl",
                "nnpack",
                "openmp",
            },
        )

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import cuda", backend_import)
        exec("from torch_rs.backends.cuda import is_built", function_import)
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.cuda import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["cuda"], cuda)
        self.assertIs(function_import["is_built"], function)
        self.assertIs(parent_wildcard["cuda"], cuda)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {
                "allow_fp16_bf16_reduction_math_sdp",
                "cuBLASModule",
                "cudnn_sdp_enabled",
                "enable_cudnn_sdp",
                "enable_flash_sdp",
                "enable_math_sdp",
                "enable_mem_efficient_sdp",
                "flash_sdp_enabled",
                "fp16_bf16_reduction_math_sdp_allowed",
                "is_built",
                "is_ck_sdpa_available",
                "is_flash_attention_available",
                "math_sdp_enabled",
                "matmul",
                "mem_efficient_sdp_enabled",
            },
        )
        self.assertIs(child_wildcard["is_built"], function)

        self.assertNotIn("backends", torch.__all__)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("cuda", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cuda", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_function_and_preserves_canonical_module(self):
        backends = torch.backends
        cuda = backends.cuda
        old_function = cuda.is_built
        namespace = cuda.__dict__

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.is_built, old_function)
        self.assertIs(cuda.is_built(), False)
        self.assertIs(copy.copy(cuda.is_built), cuda.is_built)
        self.assertIs(copy.deepcopy(cuda.is_built), cuda.is_built)
        self.assertIs(pickle.loads(pickle.dumps(cuda.is_built)), cuda.is_built)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_built at 0x...>: "
            "it's not the same object as torch_rs.backends.cuda.is_built",
        )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.cuda.is_built
        cases = (
            (
                lambda: function(None),
                "is_built() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_built() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_built() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_built() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_cuda_execution_surface_remains_unsupported(self):
        cuda_backend = torch.backends.cuda

        for name in (
            "Event",
            "Stream",
            "current_stream",
            "device_count",
            "is_available",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda_backend, name))

        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(sys.modules["torch_rs.cuda"], torch.cuda)
        self.assertIs(importlib.import_module("torch_rs.cuda"), torch.cuda)
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertTrue(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            NotImplementedError, r"device conversions are not supported"
        ):
            torch.tensor([1.0]).to("cuda:0")
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
    blocked = {"cupy", "nvidia", "numpy", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)
import torch_rs as torch
from torch_rs.backends import cuda
from torch_rs.backends.cuda import is_built

assert torch.backends.cuda is cuda
assert cuda.is_built is is_built
assert is_built.__code__.co_names == ("torch", "_C", "_has_cuda")
assert is_built() is torch._C._has_cuda is False
assert not hasattr(torch, "_has_cuda")
assert torch.cuda.is_available() is False
assert torch.cuda.device_count() == 0
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
