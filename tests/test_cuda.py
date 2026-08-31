import contextlib
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


MODULE_DOC = """
CPU-build CUDA compatibility probes.

This module intentionally exposes only availability metadata. CUDA tensors,
streams, events, synchronization, memory APIs, device selection, runtime
initialization, and `torch.compile` CUDA execution remain unsupported.
"""

IS_AVAILABLE_DOC = """
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """
DEVICE_COUNT_DOC = """
    Return the number of GPUs available.

    .. note:: This API will NOT poison fork if NVML discovery succeeds.
        See :ref:`multiprocessing-poison-fork-note` for more details.
    """


class CudaProbeTests(unittest.TestCase):
    def test_probe_values_are_literals_without_runtime_or_environment_probes(self):
        probes = (
            (torch.cuda.is_available, False, bool),
            (torch.cuda.device_count, 0, int),
        )

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
        for function, expected, expected_type in probes:
            with self.subTest(function=function.__name__):
                self.assertEqual(function.__code__.co_names, ())
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())
                for environment in environments:
                    with self.subTest(environment=environment):
                        with mock.patch.dict(os.environ, environment, clear=True):
                            with mock.patch(
                                "os.cpu_count",
                                side_effect=AssertionError("hardware was probed"),
                            ):
                                result = function()
                        self.assertIs(type(result), expected_type)
                        self.assertEqual(result, expected)

    def test_values_are_stable_across_threads_and_grad_modes(self):
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    barrier.wait(timeout=10)
                    results[index] = (
                        torch.is_grad_enabled(),
                        torch.cuda.is_available(),
                        torch.cuda.device_count(),
                        torch.is_grad_enabled(),
                    )
            except BaseException as error:
                errors.append(error)

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
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    False,
                    0,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(type(result[2]), int)

    def test_signature_documentation_and_module_identity(self):
        cuda = importlib.import_module("torch_rs.cuda")

        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertEqual(inspect.cleandoc(cuda.__doc__), inspect.cleandoc(MODULE_DOC))
        self.assertEqual(cuda.__all__, ["device_count", "is_available"])
        self.assertIsNot(cuda, torch.backends.cuda)

        metadata = (
            (
                cuda.is_available,
                "() -> bool",
                {"return": bool},
                "is_available",
                IS_AVAILABLE_DOC,
            ),
            (
                cuda.device_count,
                "() -> int",
                {"return": int},
                "device_count",
                DEVICE_COUNT_DOC,
            ),
        )
        for function, signature, annotations, name, doc in metadata:
            with self.subTest(function=name):
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, "torch_rs.cuda")
                self.assertIs(inspect.getmodule(function), cuda)
                self.assertEqual(
                    inspect.cleandoc(function.__doc__), inspect.cleandoc(doc)
                )
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        cuda = importlib.import_module("torch_rs.cuda")

        package_import = {}
        function_import = {}
        top_level_wildcard = {}
        cuda_wildcard = {}
        exec("from torch_rs import cuda", package_import)
        exec("from torch_rs.cuda import is_available, device_count", function_import)
        exec("from torch_rs import *", top_level_wildcard)
        exec("from torch_rs.cuda import *", cuda_wildcard)

        self.assertIs(package_import["cuda"], cuda)
        self.assertIs(function_import["is_available"], cuda.is_available)
        self.assertIs(function_import["device_count"], cuda.device_count)
        self.assertNotIn("cuda", torch.__all__)
        self.assertNotIn("cuda", top_level_wildcard)
        self.assertEqual(
            {name for name in cuda_wildcard if not name.startswith("__")},
            {"is_available", "device_count"},
        )
        self.assertIs(cuda_wildcard["is_available"], cuda.is_available)
        self.assertIs(cuda_wildcard["device_count"], cuda.device_count)

        for function in (cuda.is_available, cuda.device_count):
            with self.subTest(function=function.__name__):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(b"torch_rs.cuda", payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_functions_and_preserves_canonical_module(self):
        cuda = torch.cuda
        old_is_available = cuda.is_available
        old_device_count = cuda.device_count
        namespace = cuda.__dict__

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.is_available, old_is_available)
        self.assertIsNot(cuda.device_count, old_device_count)
        self.assertIs(cuda.is_available(), False)
        self.assertEqual(cuda.device_count(), 0)
        for function in (cuda.is_available, cuda.device_count):
            with self.subTest(function=function.__name__):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)

        stale_functions = (
            ("is_available", old_is_available),
            ("device_count", old_device_count),
        )
        for name, function in stale_functions:
            with self.subTest(stale=name):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(function)
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
                self.assertEqual(
                    message,
                    f"Can't pickle <function {name} at 0x...>: "
                    f"it's not the same object as torch_rs.cuda.{name}",
                )

    def test_rejects_arguments_with_pytorch_2_13_style_errors(self):
        cases = (
            (
                lambda: torch.cuda.is_available(None),
                "is_available() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: torch.cuda.is_available(None, None),
                "is_available() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.is_available(enabled=True),
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: torch.cuda.is_available(None, enabled=True),
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: torch.cuda.device_count(None),
                "device_count() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: torch.cuda.device_count(None, None),
                "device_count() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.device_count(device=True),
                "device_count() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.cuda.device_count(None, device=True),
                "device_count() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_cuda_execution_surface_remains_unsupported(self):
        cuda = torch.cuda

        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {"is_available", "device_count"},
        )
        for name in (
            "Event",
            "Stream",
            "current_device",
            "current_stream",
            "empty_cache",
            "get_device_name",
            "init",
            "is_initialized",
            "memory_allocated",
            "set_device",
            "stream",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))

        self.assertFalse(hasattr(torch, "compile"))
        self.assertFalse(hasattr(torch.nn.functional, "scaled_dot_product_attention"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(RuntimeError, "only 'cpu' is implemented"):
            torch.device("cuda")
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_backends_cuda_preferences_are_preserved(self):
        backend_cuda = torch.backends.cuda
        self.assertIsNot(torch.cuda, backend_cuda)
        self.assertEqual(
            backend_cuda.__all__,
            [
                "is_built",
                "cuBLASModule",
                "is_ck_sdpa_available",
                "matmul",
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
        self.assertIs(backend_cuda.is_built(), False)
        self.assertIs(backend_cuda.is_ck_sdpa_available(), False)
        self.assertIs(backend_cuda.is_flash_attention_available(), False)

        original_tf32 = backend_cuda.matmul.allow_tf32
        original_flash = backend_cuda.flash_sdp_enabled()
        original_math = backend_cuda.math_sdp_enabled()
        try:
            backend_cuda.matmul.allow_tf32 = True
            self.assertIs(backend_cuda.matmul.allow_tf32, True)
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
            self.assertIs(torch.cuda.is_available(), False)
            self.assertEqual(torch.cuda.device_count(), 0)

            backend_cuda.enable_flash_sdp(False)
            backend_cuda.enable_math_sdp(False)
            self.assertIs(backend_cuda.flash_sdp_enabled(), False)
            self.assertIs(backend_cuda.math_sdp_enabled(), False)
            self.assertIs(torch.cuda.is_available(), False)
            self.assertEqual(torch.cuda.device_count(), 0)
        finally:
            backend_cuda.matmul.allow_tf32 = original_tf32
            backend_cuda.enable_flash_sdp(original_flash)
            backend_cuda.enable_math_sdp(original_math)

    def test_subprocess_import_does_not_import_pytorch_or_external_cuda_runtimes(self):
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
from torch_rs import cuda
from torch_rs.cuda import device_count, is_available

assert torch.cuda is cuda
assert cuda.is_available is is_available
assert cuda.device_count is device_count
assert is_available.__code__.co_names == ()
assert device_count.__code__.co_names == ()
assert is_available() is False
assert type(device_count()) is int
assert device_count() == 0
assert torch.backends.cuda.is_built() is False
assert torch.backends.cuda.matmul.allow_tf32 is False
assert not hasattr(torch.cuda, "is_initialized")
assert not hasattr(torch.cuda, "synchronize")
assert not hasattr(torch, "compile")
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
