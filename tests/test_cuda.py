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
    "is_initialized": "Return whether PyTorch's CUDA state has been initialized.",
    "max_memory_reserved": """Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    By default, this returns the peak cached memory since the beginning of this
    program. :func:`~torch.cuda.reset_peak_memory_stats` can be used to reset
    the starting point in tracking this metric. For example, these two functions
    can measure the peak cached memory amount of each iteration in a training
    loop.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        See :ref:`cuda-memory-management` for more details about GPU memory
        management.
    """,
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
            (torch.cuda.is_initialized, False, bool),
            (torch.cuda.max_memory_reserved, 0, int),
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

    def test_signature_documentation_and_module_identity(self):
        cuda = importlib.import_module("torch_rs.cuda")
        memory = importlib.import_module("torch_rs.cuda.memory")

        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertIs(sys.modules["torch_rs.cuda.memory"], memory)
        self.assertIs(cuda.memory, memory)
        self.assertIs(cuda.max_memory_reserved, memory.max_memory_reserved)
        self.assertEqual(inspect.cleandoc(cuda.__doc__), inspect.cleandoc(MODULE_DOC))
        cases = (
            ("device_count", "() -> int", {"return": int}, cuda),
            ("is_available", "() -> bool", {"return": bool}, cuda),
            ("is_initialized", "()", {}, cuda),
            (
                "max_memory_reserved",
                "(device: 'Device' = None) -> int",
                {"device": "Device", "return": int},
                memory,
            ),
        )
        for name, signature, annotations, owner in cases:
            with self.subTest(name=name):
                function = getattr(cuda, name)
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                if name == "max_memory_reserved":
                    with self.assertRaises(NameError) as raised:
                        typing.get_type_hints(function)
                    self.assertEqual(
                        str(raised.exception),
                        "name 'Device' is not defined",
                    )
                else:
                    self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, owner.__name__)
                self.assertIs(inspect.getmodule(function), owner)
                self.assertEqual(function.__doc__, FUNCTION_DOCS[name])
                self.assertIsNone(function.__kwdefaults__)
                if name == "max_memory_reserved":
                    self.assertEqual(function.__defaults__, (None,))
                else:
                    self.assertIsNone(function.__defaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        cuda = torch.cuda

        self.assertEqual(
            cuda.__all__,
            [
                "device_count",
                "is_available",
                "is_initialized",
                "max_memory_reserved",
            ],
        )
        self.assertEqual(cuda.memory.__all__, ["max_memory_reserved"])
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {
                "device_count",
                "is_available",
                "is_initialized",
                "max_memory_reserved",
                "memory",
            },
        )

        package_import = {}
        direct_import = {}
        memory_import = {}
        module_wildcard = {}
        memory_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import cuda", package_import)
        exec(
            "from torch_rs.cuda import device_count, is_available, "
            "is_initialized, max_memory_reserved",
            direct_import,
        )
        exec("from torch_rs.cuda.memory import max_memory_reserved", memory_import)
        exec("from torch_rs.cuda import *", module_wildcard)
        exec("from torch_rs.cuda.memory import *", memory_wildcard)
        exec("from torch_rs import *", top_level_wildcard)
        self.assertIs(package_import["cuda"], cuda)
        self.assertIs(direct_import["device_count"], cuda.device_count)
        self.assertIs(direct_import["is_available"], cuda.is_available)
        self.assertIs(direct_import["is_initialized"], cuda.is_initialized)
        self.assertIs(
            direct_import["max_memory_reserved"],
            cuda.max_memory_reserved,
        )
        self.assertIs(
            memory_import["max_memory_reserved"],
            cuda.max_memory_reserved,
        )
        self.assertEqual(
            {name for name in module_wildcard if not name.startswith("__")},
            {
                "device_count",
                "is_available",
                "is_initialized",
                "max_memory_reserved",
            },
        )
        self.assertIs(module_wildcard["device_count"], cuda.device_count)
        self.assertIs(module_wildcard["is_available"], cuda.is_available)
        self.assertIs(module_wildcard["is_initialized"], cuda.is_initialized)
        self.assertIs(module_wildcard["max_memory_reserved"], cuda.max_memory_reserved)
        self.assertEqual(
            {name for name in memory_wildcard if not name.startswith("__")},
            {"max_memory_reserved"},
        )
        self.assertIs(memory_wildcard["max_memory_reserved"], cuda.max_memory_reserved)
        self.assertNotIn("cuda", torch.__all__)
        self.assertNotIn("is_initialized", torch.__all__)
        self.assertNotIn("max_memory_reserved", torch.__all__)
        self.assertNotIn("cuda", top_level_wildcard)
        self.assertNotIn("is_initialized", top_level_wildcard)
        self.assertNotIn("max_memory_reserved", top_level_wildcard)

        for function in (
            cuda.device_count,
            cuda.is_available,
            cuda.is_initialized,
            cuda.max_memory_reserved,
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
        old_is_available = cuda.is_available
        old_is_initialized = cuda.is_initialized
        old_max_memory_reserved = cuda.max_memory_reserved
        old_memory = cuda.memory
        namespace = cuda.__dict__

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(torch.cuda, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIs(cuda.memory, old_memory)
        self.assertIsNot(cuda.device_count, old_device_count)
        self.assertIsNot(cuda.is_available, old_is_available)
        self.assertIsNot(cuda.is_initialized, old_is_initialized)
        self.assertIs(cuda.max_memory_reserved, old_max_memory_reserved)
        self.assertIs(cuda.max_memory_reserved, cuda.memory.max_memory_reserved)
        self.assertEqual(cuda.device_count(), 0)
        self.assertIs(cuda.is_available(), False)
        self.assertIs(cuda.is_initialized(), False)
        self.assertEqual(cuda.max_memory_reserved(), 0)

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

        reloaded_memory = importlib.reload(cuda.memory)
        self.assertIs(reloaded_memory, cuda.memory)
        self.assertIsNot(cuda.memory.max_memory_reserved, old_max_memory_reserved)
        self.assertIs(cuda.max_memory_reserved, old_max_memory_reserved)
        self.assertIs(importlib.reload(cuda), cuda)
        self.assertIs(cuda.max_memory_reserved, cuda.memory.max_memory_reserved)
        self.assertIsNot(cuda.max_memory_reserved, old_max_memory_reserved)
        self.assertEqual(cuda.max_memory_reserved(), 0)
        self.assertIs(copy.copy(cuda.max_memory_reserved), cuda.max_memory_reserved)
        self.assertIs(copy.deepcopy(cuda.max_memory_reserved), cuda.max_memory_reserved)
        self.assertIs(
            pickle.loads(pickle.dumps(cuda.max_memory_reserved)),
            cuda.max_memory_reserved,
        )
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_max_memory_reserved)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function max_memory_reserved at 0x...>: "
            "it's not the same object as torch_rs.cuda.memory.max_memory_reserved",
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
            (
                lambda: torch.cuda.max_memory_reserved(None, None),
                "max_memory_reserved() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.max_memory_reserved(None, device=0),
                "max_memory_reserved() got multiple values for argument 'device'",
            ),
            (
                lambda: torch.cuda.max_memory_reserved(device_index=0),
                "max_memory_reserved() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.max_memory_reserved(None, device_index=0),
                "max_memory_reserved() got an unexpected keyword argument 'device_index'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_max_memory_reserved_accepts_cpu_build_device_forms_without_state_changes(self):
        class ExplodingDeviceToken:
            def __bool__(self):
                raise AssertionError("device token truth value was inspected")

            def __index__(self):
                raise AssertionError("device token index was inspected")

            def __int__(self):
                raise AssertionError("device token integer value was inspected")

            def __str__(self):
                raise AssertionError("device token string value was inspected")

        cuda = torch.cuda
        tokens = (
            None,
            0,
            -1,
            True,
            False,
            "cuda",
            "cuda:0",
            "cuda:-1",
            "cuda:01",
            "cpu",
            "cpu:0",
            "",
            "banana",
            1.5,
            object(),
            [],
            {},
            torch.device("cpu"),
            torch.device("cpu", 0),
            ExplodingDeviceToken(),
        )
        before = (cuda.is_available(), cuda.device_count(), cuda.is_initialized())
        self.assertEqual(before, (False, 0, False))

        with mock.patch.object(
            cuda,
            "is_available",
            side_effect=AssertionError("CUDA availability was checked"),
        ), mock.patch.object(
            cuda,
            "device_count",
            side_effect=AssertionError("CUDA device count was checked"),
        ), mock.patch.object(
            cuda,
            "is_initialized",
            side_effect=AssertionError("CUDA initialization state was checked"),
        ):
            for token in tokens:
                with self.subTest(token=type(token).__name__):
                    result = cuda.max_memory_reserved(token)
                    self.assertIs(type(result), int)
                    self.assertEqual(result, 0)
            self.assertEqual(cuda.max_memory_reserved(device=None), 0)
            self.assertEqual(cuda.max_memory_reserved(device=0), 0)

        after = (cuda.is_available(), cuda.device_count(), cuda.is_initialized())
        self.assertEqual(after, before)

    def test_cuda_execution_surface_remains_unsupported(self):
        cuda = torch.cuda

        for name in (
            "Event",
            "Stream",
            "current_device",
            "current_stream",
            "empty_cache",
            "init",
            "max_memory_allocated",
            "memory_allocated",
            "memory_reserved",
            "memory_snapshot",
            "memory_summary",
            "memory_stats",
            "reset_accumulated_memory_stats",
            "reset_max_memory_allocated",
            "reset_max_memory_cached",
            "reset_peak_memory_stats",
            "set_device",
            "stream",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))

        self.assertTrue(hasattr(cuda, "memory"))
        for name in (
            "memory_allocated",
            "memory_reserved",
            "memory_snapshot",
            "memory_summary",
            "memory_stats",
            "reset_accumulated_memory_stats",
            "reset_max_memory_allocated",
            "reset_max_memory_cached",
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
        self.assertIs(torch.cuda.is_initialized(), False)
        self.assertEqual(torch.cuda.max_memory_reserved(), 0)
        importlib.reload(torch.cuda)

        self.assertEqual(_backend_preferences(), expected)
        self.assertEqual(torch.cuda.max_memory_reserved(), 0)
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
from torch_rs.cuda import (
    device_count,
    is_available,
    is_initialized,
    max_memory_reserved,
)

assert torch.cuda is cuda
assert cuda.device_count is device_count
assert cuda.is_available is is_available
assert cuda.is_initialized is is_initialized
assert cuda.max_memory_reserved is max_memory_reserved
assert cuda.max_memory_reserved is cuda.memory.max_memory_reserved
assert cuda.memory.__all__ == ["max_memory_reserved"]
assert cuda.__all__ == [
    "device_count",
    "is_available",
    "is_initialized",
    "max_memory_reserved",
]
assert device_count.__code__.co_names == ()
assert is_available.__code__.co_names == ()
assert is_initialized.__code__.co_names == ()
assert max_memory_reserved.__code__.co_names == ()
assert type(device_count()) is int and device_count() == 0
assert is_available() is False
assert is_initialized() is False
assert type(max_memory_reserved()) is int and max_memory_reserved() == 0
assert max_memory_reserved(device=None) == 0
assert max_memory_reserved("cuda:0") == 0
assert not hasattr(cuda, "_initialized")
assert not hasattr(cuda, "synchronize")
assert not hasattr(cuda, "Stream")
assert not hasattr(cuda, "memory_stats")
assert not hasattr(cuda.memory, "memory_stats")
assert not hasattr(cuda.memory, "reset_peak_memory_stats")
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
