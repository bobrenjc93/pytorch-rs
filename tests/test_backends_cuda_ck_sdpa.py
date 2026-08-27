import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import threading
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """
    .. warning:: This flag is beta and subject to change.

    Returns whether composable_kernel may be used as the backend for
    scaled-dot-product-attention.
    """


class CudaCkSdpaAvailabilityTests(unittest.TestCase):
    def test_returns_exact_false_native_build_metadata_without_runtime_probes(self):
        function = torch.backends.cuda.is_ck_sdpa_available
        native_function = torch._C._is_ck_sdpa_available
        self.assertEqual(
            function.__code__.co_names,
            ("torch", "_C", "_is_ck_sdpa_available"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"HIP_VISIBLE_DEVICES": ""},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "HIP_VISIBLE_DEVICES": "0",
                "ROCR_VISIBLE_DEVICES": "0",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, native_function())

        with (
            mock.patch.object(torch._C, "_has_cuda", True),
            mock.patch.object(
                torch._C,
                "_is_flash_attention_available",
                return_value=True,
            ),
        ):
            self.assertIs(torch.backends.cuda.is_built(), True)
            self.assertIs(torch.backends.cuda.is_flash_attention_available(), True)
            self.assertIs(function(), False)

        self.assertFalse(hasattr(torch, "_is_ck_sdpa_available"))
        self.assertNotIn("_is_ck_sdpa_available", torch.__all__)
        self.assertNotIn("_is_ck_sdpa_available", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        cuda = importlib.import_module("torch_rs.backends.cuda")
        function = cuda.is_ck_sdpa_available

        self.assertIs(torch.backends.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.backends.cuda"], cuda)
        self.assertIsNone(cuda.__doc__)
        self.assertEqual(
            cuda.__all__,
            [
                "is_built",
                "is_ck_sdpa_available",
                "is_flash_attention_available",
            ],
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(inspect.get_annotations(function), {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_ck_sdpa_available")
        self.assertEqual(function.__qualname__, "is_ck_sdpa_available")
        self.assertEqual(function.__module__, "torch_rs.backends.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        cuda = importlib.import_module("torch_rs.backends.cuda")
        function = cuda.is_ck_sdpa_available

        function_import = {}
        child_wildcard = {}
        exec(
            "from torch_rs.backends.cuda import is_ck_sdpa_available",
            function_import,
        )
        exec("from torch_rs.backends.cuda import *", child_wildcard)
        self.assertIs(function_import["is_ck_sdpa_available"], function)
        self.assertIs(child_wildcard["is_ck_sdpa_available"], function)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {
                "is_built",
                "is_ck_sdpa_available",
                "is_flash_attention_available",
            },
        )

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cuda", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_value_and_identity_are_stable_across_threads(self):
        function = torch.backends.cuda.is_ck_sdpa_available
        native_function = torch._C._is_ck_sdpa_available
        barrier = threading.Barrier(16)
        results = [None] * 16
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    value,
                    type(value) is bool,
                    value is native_function(),
                    function is torch.backends.cuda.is_ck_sdpa_available,
                    native_function is torch._C._is_ck_sdpa_available,
                )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [(False, True, True, True, True)] * 16)

    def test_reload_replaces_public_function_and_preserves_native_function(self):
        backends = torch.backends
        cuda = backends.cuda
        native = torch._C
        old_function = cuda.is_ck_sdpa_available
        native_function = native._is_ck_sdpa_available
        namespace = cuda.__dict__

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(backends.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.is_ck_sdpa_available, old_function)
        self.assertIs(cuda.is_ck_sdpa_available(), False)
        self.assertIs(copy.copy(cuda.is_ck_sdpa_available), cuda.is_ck_sdpa_available)
        self.assertIs(
            copy.deepcopy(cuda.is_ck_sdpa_available),
            cuda.is_ck_sdpa_available,
        )
        self.assertIs(
            pickle.loads(pickle.dumps(cuda.is_ck_sdpa_available)),
            cuda.is_ck_sdpa_available,
        )
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_ck_sdpa_available at 0x...>: "
            "it's not the same object as "
            "torch_rs.backends.cuda.is_ck_sdpa_available",
        )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(torch._C, native)
        self.assertIs(native._is_ck_sdpa_available, native_function)
        self.assertIs(native_function(), False)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.cuda.is_ck_sdpa_available
        cases = (
            (
                lambda: function(None),
                "is_ck_sdpa_available() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_ck_sdpa_available() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_ck_sdpa_available() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_ck_sdpa_available() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_cuda_and_sdpa_execution_surfaces_remain_unsupported(self):
        cuda_backend = torch.backends.cuda

        for name in (
            "SDPAParams",
            "enable_cudnn_sdp",
            "enable_flash_sdp",
            "enable_math_sdp",
            "enable_mem_efficient_sdp",
            "preferred_rocm_fa_library",
            "sdp_kernel",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda_backend, name))

        self.assertFalse(
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )
        self.assertFalse(hasattr(torch.nn, "attention"))
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
    blocked = {
        "composable_kernel",
        "cupy",
        "nvidia",
        "numpy",
        "pynvml",
        "rocm",
        "torch",
        "triton",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    HIP_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
    ROCR_VISIBLE_DEVICES="0",
)
import torch_rs as torch
from torch_rs.backends import cuda
from torch_rs.backends.cuda import is_ck_sdpa_available

assert torch.backends.cuda is cuda
assert cuda.is_ck_sdpa_available is is_ck_sdpa_available
assert is_ck_sdpa_available.__code__.co_names == (
    "torch",
    "_C",
    "_is_ck_sdpa_available",
)
assert is_ck_sdpa_available() is False
assert torch._C._is_ck_sdpa_available() is False
assert not hasattr(torch, "_is_ck_sdpa_available")
assert not hasattr(torch, "cuda")
assert not hasattr(torch.nn.functional, "scaled_dot_product_attention")
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
