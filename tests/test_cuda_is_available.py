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


MODULE_DOC = r"""
This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.

:ref:`cuda-semantics` has more details about working with CUDA.
"""

FUNCTION_DOC = r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set. For more details, see
        :ref:`multiprocessing-poison-fork-note`.
    """


class CudaIsAvailableTests(unittest.TestCase):
    def test_returns_exact_native_build_flag_without_runtime_probes(self):
        cuda = torch.cuda
        function = cuda.is_available

        self.assertIs(cuda._C, torch._C)
        self.assertEqual(function.__code__.co_names, ("_C", "_has_cuda"))
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

    def test_false_is_stable_across_threads_and_grad_modes(self):
        function = torch.cuda.is_available
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
                        function(),
                        torch.is_grad_enabled(),
                        function(),
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
                    expected_grad_state,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[3], False)

    def test_signature_annotations_documentation_and_module_identity(self):
        cuda = importlib.import_module("torch_rs.cuda")
        function = cuda.is_available

        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules["torch_rs.cuda"], cuda)
        self.assertIs(type(cuda), types.ModuleType)
        self.assertEqual(cuda.__name__, "torch_rs.cuda")
        self.assertEqual(cuda.__package__, "torch_rs.cuda")
        self.assertEqual(cuda.__spec__.name, "torch_rs.cuda")
        self.assertEqual(cuda.__doc__, MODULE_DOC)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.cuda")
        self.assertIs(inspect.getmodule(function), cuda)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_exports_copy_and_pickle_use_the_canonical_module(self):
        cuda = torch.cuda
        function = cuda.is_available

        self.assertEqual(cuda.__all__, ["is_available"])
        self.assertEqual(
            {name for name in vars(cuda) if not name.startswith("_")},
            {"is_available"},
        )
        self.assertFalse(hasattr(cuda, "__getattr__"))

        package_import = {}
        exec("from torch_rs import cuda", package_import)
        self.assertIs(package_import["cuda"], cuda)

        direct_import = {}
        exec("from torch_rs.cuda import is_available", direct_import)
        self.assertIs(direct_import["is_available"], function)

        cuda_namespace = {}
        exec("from torch_rs.cuda import *", cuda_namespace)
        self.assertEqual(
            {name for name in cuda_namespace if not name.startswith("__")},
            {"is_available"},
        )
        self.assertIs(cuda_namespace["is_available"], function)

        self.assertNotIn("cuda", torch.__all__)
        self.assertNotIn("is_available", torch.__all__)
        top_level_namespace = {}
        exec("from torch_rs import *", top_level_namespace)
        self.assertNotIn("cuda", top_level_namespace)
        self.assertNotIn("is_available", top_level_namespace)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.cuda", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_function_and_preserves_canonical_module(self):
        cuda = torch.cuda
        old_all = cuda.__all__
        old_function = cuda.is_available
        namespace = cuda.__dict__

        reloaded = importlib.reload(cuda)

        self.assertIs(reloaded, cuda)
        self.assertIs(cuda.__dict__, namespace)
        self.assertIs(torch.cuda, cuda)
        self.assertIs(sys.modules[cuda.__name__], cuda)
        self.assertIsNot(cuda.__all__, old_all)
        self.assertEqual(cuda.__all__, ["is_available"])
        self.assertIsNot(cuda.is_available, old_function)
        self.assertIs(cuda.is_available(), torch._C._has_cuda)
        self.assertIs(copy.copy(cuda.is_available), cuda.is_available)
        self.assertIs(copy.deepcopy(cuda.is_available), cuda.is_available)
        self.assertIs(pickle.loads(pickle.dumps(cuda.is_available)), cuda.is_available)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_available at 0x...>: "
            "it's not the same object as torch_rs.cuda.is_available",
        )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.cuda.is_available
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
                lambda: function(device=True),
                "is_available() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: function(None, device=True),
                "is_available() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_every_other_cuda_api_remains_unsupported(self):
        cuda = torch.cuda

        for name in (
            "amp",
            "current_device",
            "current_stream",
            "device",
            "device_count",
            "empty_cache",
            "Event",
            "get_device_name",
            "init",
            "is_initialized",
            "manual_seed",
            "memory",
            "set_device",
            "Stream",
            "synchronize",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cuda, name))
                self.assertNotIn(name, cuda.__all__)

        for module_name in (
            "torch_rs.cuda.amp",
            "torch_rs.cuda.graphs",
            "torch_rs.cuda.memory",
            "torch_rs.cuda.random",
            "torch_rs.cuda.streams",
        ):
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda:0' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda:0")

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import builtins
import ctypes
import ctypes.util
import os
import pathlib
import sys

blocked_roots = {"amdsmi", "cupy", "nvidia", "pynvml", "torch"}
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.partition(".")[0] in blocked_roots:
        raise RuntimeError(f"external runtime import was attempted: {name}")
    return original_import(name, globals, locals, fromlist, level)

def accelerator_mappings():
    try:
        mappings = pathlib.Path("/proc/self/maps").read_text().splitlines()
    except OSError:
        return ()
    markers = ("libcuda", "libnvidia-ml", "libamd_smi")
    return tuple(
        line for line in mappings
        if any(marker in line.lower() for marker in markers)
    )

builtins.__import__ = guarded_import
environment_before_import = os.environ.copy()
mappings_before_import = accelerator_mappings()

import torch_rs as torch
from torch_rs import cuda
from torch_rs.cuda import is_available

assert os.environ == environment_before_import
assert accelerator_mappings() == mappings_before_import
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)

def forbidden_probe(*args, **kwargs):
    raise AssertionError("unexpected accelerator library probe")

ctypes.CDLL = forbidden_probe
ctypes.PyDLL = forbidden_probe
ctypes.cdll.LoadLibrary = forbidden_probe
ctypes.util.find_library = forbidden_probe

modules_before_calls = set(sys.modules)
environment_before_calls = os.environ.copy()
assert torch.cuda is cuda
assert cuda.is_available is is_available
assert is_available.__code__.co_names == ("_C", "_has_cuda")
assert is_available() is torch._C._has_cuda is False
assert is_available() is False
assert set(sys.modules) == modules_before_calls
assert os.environ == environment_before_calls
assert accelerator_mappings() == mappings_before_import
assert not any(name.partition(".")[0] in blocked_roots for name in sys.modules)
'''
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES="0",
            NVIDIA_VISIBLE_DEVICES="all",
            PYTHONDONTWRITEBYTECODE="1",
            PYTORCH_NVML_BASED_CUDA_CHECK="1",
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
