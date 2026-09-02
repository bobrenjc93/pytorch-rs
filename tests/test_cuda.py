import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


MODULE_DOC = """
CPU-build CUDA compatibility probes.
"""

FUNCTION_DOCS = {
    "is_available": "Returns a bool indicating if CUDA is currently available.",
    "device_count": "Returns the number of GPUs available.",
    "empty_cache": """Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~torch.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for PyTorch. However, it may help reduce fragmentation
        of GPU memory in certain cases. See :ref:`cuda-memory-management` for
        more details about GPU memory management.
    """,
    "is_initialized": "Return whether PyTorch's CUDA state has been initialized.",
}


def _backend_preferences():
    backend = torch.backends.cuda
    return (
        backend.matmul.allow_tf32,
        backend.matmul.allow_fp16_reduced_precision_reduction,
        backend.matmul.allow_bf16_reduced_precision_reduction,
        backend.flash_sdp_enabled(),
        backend.math_sdp_enabled(),
        backend.mem_efficient_sdp_enabled(),
        backend.fp16_bf16_reduction_math_sdp_allowed(),
        backend.cudnn_sdp_enabled(),
    )


def _restore_backend_preferences(snapshot):
    backend = torch.backends.cuda
    (
        allow_tf32,
        allow_fp16,
        allow_bf16,
        flash_sdp,
        math_sdp,
        mem_efficient_sdp,
        fp16_bf16_reduction,
        cudnn_sdp,
    ) = snapshot
    backend.matmul.allow_tf32 = allow_tf32
    backend.matmul.allow_fp16_reduced_precision_reduction = allow_fp16
    backend.matmul.allow_bf16_reduced_precision_reduction = allow_bf16
    backend.enable_flash_sdp(flash_sdp)
    backend.enable_math_sdp(math_sdp)
    backend.enable_mem_efficient_sdp(mem_efficient_sdp)
    backend.allow_fp16_bf16_reduction_math_sdp(fp16_bf16_reduction)
    backend.enable_cudnn_sdp(cudnn_sdp)


class CudaProbeTests(unittest.TestCase):
    def test_returns_cpu_build_probe_values_without_runtime_probes(self):
        cases = (
            (torch.cuda.is_available, False, bool),
            (torch.cuda.device_count, 0, int),
            (torch.cuda.empty_cache, None, type(None)),
            (torch.cuda.is_initialized, False, bool),
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

        for function, expected_value, expected_type in cases:
            with self.subTest(function=function.__name__):
                self.assertEqual(function.__code__.co_names, ())
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

            for environment in environments:
                with self.subTest(function=function.__name__, environment=environment):
                    with mock.patch.dict(os.environ, environment, clear=True):
                        result = function()
                    self.assertIs(type(result), expected_type)
                    self.assertEqual(result, expected_value)

        self.assertFalse(hasattr(torch.cuda, "_initialized"))
        self.assertFalse(hasattr(torch.cuda, "_cached_device_count"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.empty_cache(), None)
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.is_initialized(), False)

    def test_signature_documentation_and_module_identity(self):
        cuda = importlib.import_module("torch_rs.cuda")
        memory = importlib.import_module("torch_rs.cuda.memory")

        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertIs(cuda.memory, memory)
        self.assertIs(sys.modules["torch_rs.cuda.memory"], memory)
        self.assertIs(cuda.empty_cache, memory.empty_cache)
        self.assertEqual(inspect.cleandoc(cuda.__doc__), inspect.cleandoc(MODULE_DOC))
        cases = (
            ("device_count", "() -> int", {"return": int}, {"return": int}, cuda),
            (
                "empty_cache",
                "() -> None",
                {"return": None},
                {"return": type(None)},
                memory,
            ),
            ("is_available", "() -> bool", {"return": bool}, {"return": bool}, cuda),
            ("is_initialized", "()", {}, {}, cuda),
        )
        for name, signature, annotations, type_hints, owner in cases:
            with self.subTest(name=name):
                function = getattr(cuda, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                self.assertEqual(typing.get_type_hints(function), type_hints)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, owner.__name__)
                self.assertIs(inspect.getmodule(function), owner)
                self.assertEqual(function.__doc__, FUNCTION_DOCS[name])
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        cuda = torch.cuda

        self.assertEqual(
            cuda.__all__,
            ["device_count", "empty_cache", "is_available", "is_initialized"],
        )
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {"device_count", "empty_cache", "is_available", "is_initialized", "memory"},
        )
        self.assertEqual(cuda.memory.__all__, ["empty_cache"])
        self.assertEqual(
            {name for name in vars(cuda.memory) if not name.startswith("_")},
            {"empty_cache"},
        )

        package_import = {}
        direct_import = {}
        direct_memory_import = {}
        module_wildcard = {}
        memory_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import cuda", package_import)
        exec(
            "from torch_rs.cuda import "
            "device_count, empty_cache, is_available, is_initialized",
            direct_import,
        )
        exec("from torch_rs.cuda.memory import empty_cache", direct_memory_import)
        exec("from torch_rs.cuda import *", module_wildcard)
        exec("from torch_rs.cuda.memory import *", memory_wildcard)
        exec("from torch_rs import *", top_level_wildcard)
        self.assertIs(package_import["cuda"], cuda)
        self.assertIs(direct_import["device_count"], cuda.device_count)
        self.assertIs(direct_import["empty_cache"], cuda.empty_cache)
        self.assertIs(direct_memory_import["empty_cache"], cuda.empty_cache)
        self.assertIs(direct_import["is_available"], cuda.is_available)
        self.assertIs(direct_import["is_initialized"], cuda.is_initialized)
        self.assertEqual(
            {name for name in module_wildcard if not name.startswith("__")},
            {"device_count", "empty_cache", "is_available", "is_initialized"},
        )
        self.assertEqual(
            {name for name in memory_wildcard if not name.startswith("__")},
            {"empty_cache"},
        )
        self.assertIs(module_wildcard["device_count"], cuda.device_count)
        self.assertIs(module_wildcard["empty_cache"], cuda.empty_cache)
        self.assertIs(memory_wildcard["empty_cache"], cuda.empty_cache)
        self.assertIs(module_wildcard["is_available"], cuda.is_available)
        self.assertIs(module_wildcard["is_initialized"], cuda.is_initialized)
        self.assertNotIn("cuda", torch.__all__)
        self.assertNotIn("empty_cache", torch.__all__)
        self.assertNotIn("is_initialized", torch.__all__)
        self.assertNotIn("cuda", top_level_wildcard)
        self.assertNotIn("empty_cache", top_level_wildcard)
        self.assertNotIn("is_initialized", top_level_wildcard)

        for function in (
            cuda.device_count,
            cuda.empty_cache,
            cuda.is_available,
            cuda.is_initialized,
        ):
            with self.subTest(function=function.__name__):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    with self.subTest(protocol=protocol):
                        payload = pickle.dumps(function, protocol=protocol)
                        self.assertIn(b"torch_rs.cuda", payload)
                        self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_functions_and_preserves_canonical_module(self):
        cuda = torch.cuda
        old_device_count = cuda.device_count
        old_empty_cache = cuda.empty_cache
        old_is_available = cuda.is_available
        old_is_initialized = cuda.is_initialized
        old_memory = cuda.memory
        namespace = cuda.__dict__

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(torch.cuda, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIs(cuda.memory, old_memory)
        self.assertIs(cuda.empty_cache, old_empty_cache)
        self.assertIs(cuda.empty_cache, cuda.memory.empty_cache)
        self.assertIsNot(cuda.device_count, old_device_count)
        self.assertIsNot(cuda.is_available, old_is_available)
        self.assertIsNot(cuda.is_initialized, old_is_initialized)
        self.assertEqual(cuda.device_count(), 0)
        self.assertIs(cuda.empty_cache(), None)
        self.assertIs(cuda.is_available(), False)
        self.assertIs(cuda.is_initialized(), False)

        for function, old_function in (
            (cuda.device_count, old_device_count),
            (cuda.is_available, old_is_available),
            (cuda.is_initialized, old_is_initialized),
        ):
            with self.subTest(function=function.__name__):
                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                self.assertIs(pickle.loads(pickle.dumps(function)), function)
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(old_function)
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
                self.assertEqual(
                    message,
                    f"Can't pickle <function {function.__name__} at 0x...>: "
                    f"it's not the same object as torch_rs.cuda.{function.__name__}",
                )

        self.assertIs(copy.copy(cuda.empty_cache), cuda.empty_cache)
        self.assertIs(copy.deepcopy(cuda.empty_cache), cuda.empty_cache)
        self.assertIs(
            pickle.loads(pickle.dumps(cuda.empty_cache)),
            cuda.empty_cache,
        )
        self.assertIs(
            pickle.loads(pickle.dumps(old_empty_cache)),
            cuda.empty_cache,
        )

    def test_rejects_arguments_with_cpu_build_pytorch_style_errors(self):
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
            (
                lambda: torch.cuda.empty_cache(None),
                "empty_cache() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: torch.cuda.empty_cache(None, None),
                "empty_cache() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.empty_cache(device=True),
                "empty_cache() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.cuda.empty_cache(None, device=True),
                "empty_cache() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: torch.cuda.is_initialized(None),
                "is_initialized() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: torch.cuda.is_initialized(None, None),
                "is_initialized() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.is_initialized(enabled=True),
                "is_initialized() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: torch.cuda.is_initialized(None, enabled=True),
                "is_initialized() got an unexpected keyword argument 'enabled'",
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

        for name in (
            "Event",
            "Stream",
            "current_device",
            "current_stream",
            "init",
            "memory_allocated",
            "memory_reserved",
            "memory_snapshot",
            "memory_stats",
            "memory_summary",
            "reset_accumulated_memory_stats",
            "reset_peak_memory_stats",
            "set_device",
            "stream",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))

        self.assertTrue(hasattr(cuda, "empty_cache"))
        self.assertTrue(hasattr(cuda, "memory"))
        self.assertIs(cuda.empty_cache, cuda.memory.empty_cache)
        for name in (
            "change_current_allocator",
            "get_allocator_backend",
            "memory_allocated",
            "memory_reserved",
            "memory_snapshot",
            "memory_stats",
            "memory_summary",
            "reset_accumulated_memory_stats",
            "reset_peak_memory_stats",
        ):
            with self.subTest(memory_name=name):
                self.assertFalse(hasattr(cuda.memory, name))

        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_probes_preserve_backends_cuda_preference_apis(self):
        original = _backend_preferences()
        self.addCleanup(_restore_backend_preferences, original)

        backend = torch.backends.cuda
        self.assertIsNot(torch.cuda, backend)
        self.assertTrue(hasattr(backend, "matmul"))
        self.assertFalse(hasattr(torch.cuda, "matmul"))

        backend.matmul.allow_tf32 = True
        backend.matmul.allow_fp16_reduced_precision_reduction = False
        backend.matmul.allow_bf16_reduced_precision_reduction = False
        backend.enable_flash_sdp(False)
        backend.enable_math_sdp(False)
        backend.enable_mem_efficient_sdp(False)
        backend.allow_fp16_bf16_reduction_math_sdp(True)
        backend.enable_cudnn_sdp(False)
        expected = (True, False, False, False, False, False, True, False)
        self.assertEqual(_backend_preferences(), expected)

        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)
        self.assertIs(torch.cuda.empty_cache(), None)
        self.assertIs(torch.cuda.is_initialized(), False)
        importlib.reload(torch.cuda)

        self.assertEqual(_backend_preferences(), expected)
        self.assertIs(torch.backends.cuda, backend)
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertIs(torch.backends.cuda.is_ck_sdpa_available(), False)
        self.assertIs(torch.backends.cuda.is_flash_attention_available(), False)

    def test_subprocess_import_and_calls_do_not_import_external_runtimes(self):
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
from torch_rs.cuda import device_count, empty_cache, is_available, is_initialized
from torch_rs.cuda import memory
from torch_rs.cuda.memory import empty_cache as memory_empty_cache

assert torch.cuda is cuda
assert cuda.device_count is device_count
assert cuda.empty_cache is empty_cache
assert cuda.empty_cache is memory.empty_cache
assert memory.empty_cache is memory_empty_cache
assert cuda.is_available is is_available
assert cuda.is_initialized is is_initialized
assert cuda.__all__ == ["device_count", "empty_cache", "is_available", "is_initialized"]
assert memory.__all__ == ["empty_cache"]
assert device_count.__code__.co_names == ()
assert empty_cache.__code__.co_names == ()
assert is_available.__code__.co_names == ()
assert is_initialized.__code__.co_names == ()
assert type(device_count()) is int and device_count() == 0
assert empty_cache() is None
assert is_available() is False
assert is_initialized() is False
assert not hasattr(cuda, "_initialized")
assert not hasattr(cuda, "memory_stats")
assert not hasattr(cuda, "synchronize")
assert not hasattr(cuda, "Stream")
assert not hasattr(memory, "memory_stats")
assert not hasattr(memory, "memory_snapshot")
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
