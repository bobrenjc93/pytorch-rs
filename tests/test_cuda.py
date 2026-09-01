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
from collections import OrderedDict
from unittest import mock

import torch_rs as torch


MODULE_DOC = """
CPU-build CUDA compatibility probes and memory query no-ops.
"""

SUPPORTED = {
    "device_count",
    "empty_cache",
    "is_available",
    "is_initialized",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
}
MEMORY_LOCAL = {
    "empty_cache",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
}

FUNCTION_DOCS = {
    "is_available": "Returns a bool indicating if CUDA is currently available.",
    "device_count": "Returns the number of GPUs available.",
    "is_initialized": "Return whether PyTorch's CUDA state has been initialized.",
}
MEMORY_DOC_PREFIXES = {
    "empty_cache": "Release all unoccupied cached memory currently held by the caching\nallocator",
    "memory_allocated": "Return the current GPU memory occupied by tensors in bytes",
    "max_memory_allocated": "Return the maximum GPU memory occupied by tensors in bytes",
    "memory_reserved": "Return the current GPU memory managed by the caching allocator",
    "max_memory_reserved": "Return the maximum GPU memory managed by the caching allocator",
    "memory_stats": "Return a dictionary of CUDA memory allocator statistics",
    "reset_accumulated_memory_stats": 'Reset the "accumulated" (historical) stats tracked',
    "reset_peak_memory_stats": 'Reset the "peak" stats tracked by the CUDA memory allocator.',
}


class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")


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

    def test_cuda_memory_noops_are_probe_free_and_ignore_device_tokens(self):
        cuda = torch.cuda
        memory = cuda.memory
        tokens = (
            None,
            0,
            -1,
            True,
            1.5,
            "cpu",
            "cuda:0",
            torch.device("cpu"),
            object(),
            [],
            {},
            ExplodingDeviceToken(),
        )
        expected_state = (
            cuda.is_available(),
            cuda.device_count(),
            cuda.is_initialized(),
        )

        with mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
            clear=True,
        ):
            with mock.patch(
                "os.cpu_count",
                side_effect=AssertionError("hardware was probed"),
            ):
                results = []
                for token in tokens:
                    results.append(cuda.memory_stats(token))
                    results.append(cuda.memory_stats(device=token))
                    self.assertEqual(
                        (
                            cuda.memory_allocated(token),
                            cuda.max_memory_allocated(token),
                            cuda.memory_reserved(token),
                            cuda.max_memory_reserved(token),
                            cuda.memory_allocated(device=token),
                            cuda.max_memory_allocated(device=token),
                            cuda.memory_reserved(device=token),
                            cuda.max_memory_reserved(device=token),
                        ),
                        (0, 0, 0, 0, 0, 0, 0, 0),
                    )
                    self.assertIs(cuda.reset_accumulated_memory_stats(token), None)
                    self.assertIs(
                        cuda.reset_accumulated_memory_stats(device=token),
                        None,
                    )
                    self.assertIs(cuda.reset_peak_memory_stats(token), None)
                    self.assertIs(cuda.reset_peak_memory_stats(device=token), None)
                results.extend(cuda.memory_stats() for _ in range(8))
                self.assertEqual(
                    tuple(cuda.empty_cache() for _ in range(8)),
                    (None,) * 8,
                )

        self.assertEqual(
            (cuda.is_available(), cuda.device_count(), cuda.is_initialized()),
            expected_state,
        )
        self.assertFalse(hasattr(cuda, "_initialized"))
        self.assertFalse(hasattr(cuda, "_cached_device_count"))
        self.assertTrue(all(type(result) is OrderedDict for result in results))
        self.assertTrue(all(result == OrderedDict() for result in results))
        self.assertEqual(len({id(result) for result in results}), len(results))
        results[0]["mutated"] = 1
        self.assertEqual(results[0], OrderedDict((("mutated", 1),)))
        self.assertTrue(all(not result for result in results[1:]))

        sentinel = object()
        with mock.patch.object(
            memory,
            "memory_stats",
            return_value=OrderedDict(
                (
                    ("allocated_bytes.all.current", 37),
                    ("allocated_bytes.all.peak", 43),
                    ("reserved_bytes.all.current", 53),
                    ("reserved_bytes.all.peak", 59),
                )
            ),
        ) as memory_stats:
            self.assertEqual(cuda.memory_allocated(sentinel), 37)
            self.assertEqual(cuda.max_memory_allocated(sentinel), 43)
            self.assertEqual(cuda.memory_reserved(sentinel), 53)
            self.assertEqual(cuda.max_memory_reserved(sentinel), 59)
        self.assertEqual(
            memory_stats.call_args_list,
            [mock.call(device=sentinel)] * 4,
        )

    def test_signature_documentation_and_module_identity(self):
        cuda = importlib.import_module("torch_rs.cuda")
        memory = importlib.import_module("torch_rs.cuda.memory")

        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertIs(cuda.memory, memory)
        self.assertIs(sys.modules["torch_rs.cuda.memory"], memory)
        self.assertEqual(
            memory.__doc__,
            "This package adds support for device memory management implemented in CUDA.",
        )
        self.assertEqual(inspect.cleandoc(cuda.__doc__), inspect.cleandoc(MODULE_DOC))
        cases = (
            ("device_count", "() -> int", {"return": int}),
            ("is_available", "() -> bool", {"return": bool}),
            ("is_initialized", "()", {}),
            ("empty_cache", "() -> None", {"return": None}),
            (
                "max_memory_allocated",
                "(device: 'Device' = None) -> int",
                {"device": "Device", "return": int},
            ),
            (
                "max_memory_reserved",
                "(device: 'Device' = None) -> int",
                {"device": "Device", "return": int},
            ),
            (
                "memory_allocated",
                "(device: 'Device' = None) -> int",
                {"device": "Device", "return": int},
            ),
            (
                "memory_reserved",
                "(device: 'Device' = None) -> int",
                {"device": "Device", "return": int},
            ),
            (
                "memory_stats",
                "(device: 'Device' = None) -> dict[str, typing.Any]",
                {"device": "Device", "return": dict[str, typing.Any]},
            ),
            (
                "reset_accumulated_memory_stats",
                "(device: 'Device' = None) -> None",
                {"device": "Device", "return": None},
            ),
            (
                "reset_peak_memory_stats",
                "(device: 'Device' = None) -> None",
                {"device": "Device", "return": None},
            ),
        )
        for name, signature, annotations in cases:
            with self.subTest(name=name):
                function = getattr(cuda, name)
                defining_module = memory if name in MEMORY_LOCAL else cuda
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), signature)
                self.assertEqual(function.__annotations__, annotations)
                if "device" in annotations:
                    with self.assertRaisesRegex(
                        NameError,
                        "name 'Device' is not defined",
                    ):
                        typing.get_type_hints(function)
                elif name == "empty_cache":
                    self.assertEqual(
                        typing.get_type_hints(function),
                        {"return": type(None)},
                    )
                else:
                    self.assertEqual(typing.get_type_hints(function), annotations)
                self.assertEqual(function.__name__, name)
                self.assertEqual(function.__qualname__, name)
                self.assertEqual(function.__module__, defining_module.__name__)
                self.assertIs(inspect.getmodule(function), defining_module)
                if name in FUNCTION_DOCS:
                    self.assertEqual(function.__doc__, FUNCTION_DOCS[name])
                else:
                    self.assertIn(
                        MEMORY_DOC_PREFIXES[name],
                        inspect.cleandoc(function.__doc__),
                    )
                expected_defaults = (
                    (None,) if name in MEMORY_LOCAL - {"empty_cache"} else None
                )
                self.assertEqual(function.__defaults__, expected_defaults)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        cuda = torch.cuda
        memory = importlib.import_module("torch_rs.cuda.memory")

        self.assertEqual(
            cuda.__all__,
            [
                "device_count",
                "empty_cache",
                "is_available",
                "is_initialized",
                "max_memory_allocated",
                "max_memory_reserved",
                "memory_allocated",
                "memory_reserved",
                "memory_stats",
                "reset_accumulated_memory_stats",
                "reset_peak_memory_stats",
            ],
        )
        self.assertEqual(
            memory.__all__,
            [
                "empty_cache",
                "memory_stats",
                "reset_accumulated_memory_stats",
                "reset_peak_memory_stats",
                "memory_allocated",
                "max_memory_allocated",
                "memory_reserved",
                "max_memory_reserved",
            ],
        )
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            SUPPORTED | {"memory"},
        )
        self.assertEqual(
            {name for name in vars(memory) if not name.startswith("_")},
            MEMORY_LOCAL,
        )
        for name in MEMORY_LOCAL:
            with self.subTest(memory_alias=name):
                self.assertIs(getattr(cuda, name), getattr(memory, name))

        package_import = {}
        direct_import = {}
        module_wildcard = {}
        memory_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import cuda", package_import)
        exec(
            "from torch_rs.cuda import device_count, empty_cache, is_available, is_initialized, max_memory_allocated, max_memory_reserved, memory_allocated, memory_reserved, memory_stats, reset_accumulated_memory_stats, reset_peak_memory_stats",
            direct_import,
        )
        exec("from torch_rs.cuda import *", module_wildcard)
        exec("from torch_rs.cuda.memory import *", memory_wildcard)
        exec("from torch_rs import *", top_level_wildcard)
        self.assertIs(package_import["cuda"], cuda)
        for name in SUPPORTED:
            self.assertIs(direct_import[name], getattr(cuda, name))
        self.assertEqual(
            {name for name in module_wildcard if not name.startswith("__")},
            SUPPORTED,
        )
        self.assertEqual(
            {name for name in memory_wildcard if not name.startswith("__")},
            MEMORY_LOCAL,
        )
        for name in SUPPORTED:
            self.assertIs(module_wildcard[name], getattr(cuda, name))
            if name in MEMORY_LOCAL:
                self.assertIs(memory_wildcard[name], getattr(cuda, name))
        self.assertNotIn("cuda", torch.__all__)
        self.assertNotIn("is_initialized", torch.__all__)
        self.assertNotIn("cuda", top_level_wildcard)
        self.assertNotIn("is_initialized", top_level_wildcard)

        for name in SUPPORTED:
            function = getattr(cuda, name)
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
        old_memory_functions = {name: getattr(cuda, name) for name in MEMORY_LOCAL}
        namespace = cuda.__dict__
        memory = cuda.memory

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(torch.cuda, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIs(cuda.memory, memory)
        self.assertIsNot(cuda.device_count, old_device_count)
        self.assertIsNot(cuda.is_available, old_is_available)
        self.assertIsNot(cuda.is_initialized, old_is_initialized)
        self.assertEqual(cuda.device_count(), 0)
        self.assertIs(cuda.is_available(), False)
        self.assertIs(cuda.is_initialized(), False)
        for name, old_function in old_memory_functions.items():
            with self.subTest(memory_name=name):
                self.assertIs(getattr(cuda, name), old_function)
                self.assertIs(getattr(cuda, name), getattr(memory, name))

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

    def test_memory_reload_keeps_old_and_new_cpu_build_functions_usable(self):
        cuda = torch.cuda
        memory = cuda.memory
        old_all = memory.__all__
        old_functions = {name: getattr(memory, name) for name in MEMORY_LOCAL}

        reloaded = importlib.reload(memory)
        new_functions = {name: getattr(memory, name) for name in MEMORY_LOCAL}

        self.assertIs(reloaded, memory)
        self.assertIs(cuda.memory, memory)
        self.assertIs(sys.modules["torch_rs.cuda.memory"], memory)
        self.assertIsNot(memory.__all__, old_all)
        for name in MEMORY_LOCAL:
            with self.subTest(name=name):
                old_function = old_functions[name]
                new_function = new_functions[name]
                self.assertIsNot(new_function, old_function)
                self.assertIs(getattr(cuda, name), old_function)
                self.assertIsNot(getattr(cuda, name), new_function)
                self.assertIs(copy.copy(new_function), new_function)
                self.assertIs(copy.deepcopy(new_function), new_function)
                self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
                with self.assertRaises(pickle.PicklingError):
                    pickle.dumps(old_function)

        self.assertIs(old_functions["empty_cache"](), None)
        self.assertIs(new_functions["empty_cache"](), None)
        self.assertEqual(
            (
                old_functions["memory_allocated"](object()),
                new_functions["memory_allocated"](object()),
                old_functions["max_memory_allocated"](object()),
                new_functions["max_memory_allocated"](object()),
                old_functions["memory_reserved"](object()),
                new_functions["memory_reserved"](object()),
                old_functions["max_memory_reserved"](object()),
                new_functions["max_memory_reserved"](object()),
            ),
            (0, 0, 0, 0, 0, 0, 0, 0),
        )
        old_stats = old_functions["memory_stats"](object())
        new_stats = new_functions["memory_stats"](object())
        self.assertIs(type(old_stats), OrderedDict)
        self.assertIs(type(new_stats), OrderedDict)
        self.assertEqual((old_stats, new_stats), (OrderedDict(), OrderedDict()))
        self.assertIsNot(old_stats, new_stats)
        self.assertIs(old_functions["reset_accumulated_memory_stats"](object()), None)
        self.assertIs(new_functions["reset_accumulated_memory_stats"](object()), None)
        self.assertIs(old_functions["reset_peak_memory_stats"](object()), None)
        self.assertIs(new_functions["reset_peak_memory_stats"](object()), None)

        self.assertIs(importlib.reload(cuda), cuda)
        for name in MEMORY_LOCAL:
            with self.subTest(reloaded_name=name):
                self.assertIs(getattr(cuda, name), new_functions[name])
                self.assertIs(getattr(cuda, name), getattr(memory, name))

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
                lambda: torch.cuda.memory_allocated(device_index=None),
                "memory_allocated() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.memory_allocated(None, None),
                "memory_allocated() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.memory_allocated(unexpected=True),
                "memory_allocated() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.cuda.max_memory_allocated(device_index=None),
                "max_memory_allocated() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.max_memory_allocated(None, None),
                "max_memory_allocated() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.max_memory_allocated(unexpected=True),
                "max_memory_allocated() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.cuda.memory_reserved(device_index=None),
                "memory_reserved() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.memory_reserved(None, None),
                "memory_reserved() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.memory_reserved(unexpected=True),
                "memory_reserved() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.cuda.max_memory_reserved(device_index=None),
                "max_memory_reserved() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.max_memory_reserved(None, None),
                "max_memory_reserved() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.max_memory_reserved(unexpected=True),
                "max_memory_reserved() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.cuda.memory_stats(device_index=None),
                "memory_stats() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.memory_stats(None, None),
                "memory_stats() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.memory_stats(unexpected=True),
                "memory_stats() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.cuda.reset_accumulated_memory_stats(device_index=None),
                "reset_accumulated_memory_stats() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.reset_accumulated_memory_stats(None, None),
                "reset_accumulated_memory_stats() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.reset_accumulated_memory_stats(unexpected=True),
                "reset_accumulated_memory_stats() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: torch.cuda.reset_peak_memory_stats(device_index=None),
                "reset_peak_memory_stats() got an unexpected keyword argument 'device_index'",
            ),
            (
                lambda: torch.cuda.reset_peak_memory_stats(None, None),
                "reset_peak_memory_stats() takes from 0 to 1 positional arguments but 2 were given",
            ),
            (
                lambda: torch.cuda.reset_peak_memory_stats(unexpected=True),
                "reset_peak_memory_stats() got an unexpected keyword argument 'unexpected'",
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
            "list_gpu_processes",
            "memory_snapshot",
            "memory_summary",
            "mem_get_info",
            "set_device",
            "stream",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))
                self.assertNotIn(name, cuda.__all__)

        self.assertIs(cuda.memory, importlib.import_module("torch_rs.cuda.memory"))
        for name in ("memory_snapshot", "memory_summary", "mem_get_info"):
            with self.subTest(memory_name=name):
                self.assertFalse(hasattr(cuda.memory, name))
                self.assertNotIn(name, cuda.memory.__all__)

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
from collections import OrderedDict

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
    empty_cache,
    is_available,
    is_initialized,
    max_memory_allocated,
    max_memory_reserved,
    memory_allocated,
    memory_reserved,
    memory_stats,
    reset_accumulated_memory_stats,
    reset_peak_memory_stats,
)
from torch_rs.cuda.memory import memory_stats as memory_stats_from_module

class ExplodingDeviceToken:
    def __bool__(self):
        raise AssertionError("device token truth value was inspected")

    def __index__(self):
        raise AssertionError("device token index was inspected")

    def __int__(self):
        raise AssertionError("device token integer value was inspected")

    def __str__(self):
        raise AssertionError("device token string value was inspected")

assert torch.cuda is cuda
assert cuda.device_count is device_count
assert cuda.is_available is is_available
assert cuda.is_initialized is is_initialized
assert cuda.empty_cache is empty_cache
assert cuda.max_memory_allocated is max_memory_allocated
assert cuda.max_memory_reserved is max_memory_reserved
assert cuda.memory_allocated is memory_allocated
assert cuda.memory_reserved is memory_reserved
assert cuda.memory_stats is memory_stats
assert cuda.reset_accumulated_memory_stats is reset_accumulated_memory_stats
assert cuda.reset_peak_memory_stats is reset_peak_memory_stats
assert cuda.memory.memory_stats is memory_stats_from_module
assert cuda.__all__ == [
    "device_count",
    "empty_cache",
    "is_available",
    "is_initialized",
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated",
    "memory_reserved",
    "memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
]
assert device_count.__code__.co_names == ()
assert is_available.__code__.co_names == ()
assert is_initialized.__code__.co_names == ()
assert empty_cache.__code__.co_names == ()
assert memory_stats.__code__.co_names == ("_OrderedDict",)
assert memory_allocated.__code__.co_names == ("memory_stats", "get")
assert max_memory_allocated.__code__.co_names == ("memory_stats", "get")
assert memory_reserved.__code__.co_names == ("memory_stats", "get")
assert max_memory_reserved.__code__.co_names == ("memory_stats", "get")
assert reset_accumulated_memory_stats.__code__.co_names == ()
assert reset_peak_memory_stats.__code__.co_names == ()
assert type(device_count()) is int and device_count() == 0
assert is_available() is False
assert is_initialized() is False
state = (is_available(), device_count(), is_initialized())
stats = [
    memory_stats(),
    memory_stats("cuda:0"),
    memory_stats(device="cpu"),
    memory_stats(ExplodingDeviceToken()),
]
assert all(type(value) is OrderedDict and not value for value in stats)
assert len({id(value) for value in stats}) == len(stats)
assert memory_allocated() == 0
assert memory_allocated(device="cuda:0") == 0
assert max_memory_allocated(ExplodingDeviceToken()) == 0
assert memory_reserved() == 0
assert memory_reserved(device="cuda:0") == 0
assert max_memory_reserved(ExplodingDeviceToken()) == 0
assert reset_accumulated_memory_stats() is None
assert reset_accumulated_memory_stats(device=ExplodingDeviceToken()) is None
assert reset_peak_memory_stats() is None
assert reset_peak_memory_stats(device=ExplodingDeviceToken()) is None
assert empty_cache() is None
assert (is_available(), device_count(), is_initialized()) == state
assert not hasattr(cuda, "_initialized")
assert not hasattr(cuda, "synchronize")
assert not hasattr(cuda, "Stream")
assert not hasattr(cuda, "memory_snapshot")
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
