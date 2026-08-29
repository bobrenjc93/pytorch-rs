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


FUNCTION_DOC = '''Return cpu capability as a string value.

    Possible values:
    - "DEFAULT"
    - "VSX"
    - "Z VECTOR"
    - "NO AVX"
    - "AVX2"
    - "AVX512"
    - "SVE256"
    '''


class CpuCapabilityTests(unittest.TestCase):
    def test_returns_exact_default_native_dispatch_without_runtime_probes(self):
        function = torch.backends.cpu.get_cpu_capability
        native_function = torch._C._get_cpu_capability
        self.assertEqual(
            function.__code__.co_names,
            ("torch", "_C", "_get_cpu_capability"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"ATEN_CPU_CAPABILITY": "default"},
            {"ATEN_CPU_CAPABILITY": "avx2"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "MKL_DEBUG_CPU_TYPE": "5",
                "OMP_NUM_THREADS": "64",
                "OPENBLAS_CORETYPE": "SAPPHIRERAPIDS",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), str)
                    self.assertEqual(result, "DEFAULT")
                    self.assertEqual(result, native_function())

        self.assertFalse(hasattr(torch, "_get_cpu_capability"))
        self.assertNotIn("_get_cpu_capability", torch.__all__)
        self.assertNotIn("_get_cpu_capability", torch._C.__all__)

    def test_value_and_identity_are_stable_across_threads(self):
        function = torch.backends.cpu.get_cpu_capability
        native_function = torch._C._get_cpu_capability
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                results[index] = (
                    function(),
                    native_function(),
                    function is torch.backends.cpu.get_cpu_capability,
                    native_function is torch._C._get_cpu_capability,
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
        self.assertEqual(
            results,
            [("DEFAULT", "DEFAULT", True, True)] * worker_count,
        )

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        cpu = importlib.import_module("torch_rs.backends.cpu")
        function = cpu.get_cpu_capability

        self.assertIs(torch.backends.cpu, cpu)
        self.assertIs(sys.modules["torch_rs.backends.cpu"], cpu)
        self.assertIs(type(cpu), types.ModuleType)
        self.assertIsNone(cpu.__doc__)
        self.assertEqual(cpu.__all__, ["get_cpu_capability"])
        self.assertEqual(cpu.__annotations__, {})
        self.assertEqual(
            {name for name in vars(cpu) if not name.startswith("_")},
            {"get_cpu_capability", "torch"},
        )
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> str")
        self.assertEqual(inspect.get_annotations(function), {"return": str})
        self.assertEqual(typing.get_type_hints(function), {"return": str})
        self.assertEqual(function.__name__, "get_cpu_capability")
        self.assertEqual(function.__qualname__, "get_cpu_capability")
        self.assertEqual(function.__module__, "torch_rs.backends.cpu")
        self.assertIs(inspect.getmodule(function), cpu)
        self.assertEqual(
            inspect.cleandoc(function.__doc__),
            inspect.cleandoc(FUNCTION_DOC),
        )
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        cpu = importlib.import_module("torch_rs.backends.cpu")
        function = cpu.get_cpu_capability

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.cpu, cpu)
        self.assertIs(cpu.torch, torch)
        self.assertIsNot(torch.cpu, cpu)
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
        exec("from torch_rs.backends import cpu", backend_import)
        exec(
            "from torch_rs.backends.cpu import get_cpu_capability",
            function_import,
        )
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.cpu import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["cpu"], cpu)
        self.assertIs(function_import["get_cpu_capability"], function)
        self.assertIs(parent_wildcard["cpu"], cpu)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"get_cpu_capability"},
        )
        self.assertIs(child_wildcard["get_cpu_capability"], function)

        self.assertNotIn("backends", torch.__all__)
        self.assertNotIn("cpu", torch.__all__)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("cpu", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cpu", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_function_and_preserves_canonical_module(self):
        backends = torch.backends
        cpu = backends.cpu
        old_function = cpu.get_cpu_capability
        namespace = cpu.__dict__

        reloaded = importlib.reload(cpu)

        self.assertIs(reloaded, cpu)
        self.assertIs(cpu.__dict__, namespace)
        self.assertIs(backends.cpu, cpu)
        self.assertIs(sys.modules[cpu.__name__], cpu)
        self.assertIsNot(cpu.get_cpu_capability, old_function)
        self.assertEqual(cpu.get_cpu_capability(), "DEFAULT")
        self.assertIs(copy.copy(cpu.get_cpu_capability), cpu.get_cpu_capability)
        self.assertIs(copy.deepcopy(cpu.get_cpu_capability), cpu.get_cpu_capability)
        self.assertIs(
            pickle.loads(pickle.dumps(cpu.get_cpu_capability)),
            cpu.get_cpu_capability,
        )
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function get_cpu_capability at 0x...>: "
            "it's not the same object as "
            "torch_rs.backends.cpu.get_cpu_capability",
        )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.cpu.get_cpu_capability
        cases = (
            (
                lambda: function(None),
                "get_cpu_capability() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "get_cpu_capability() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(device="cpu"),
                "get_cpu_capability() got an unexpected keyword argument 'device'",
            ),
            (
                lambda: function(None, device="cpu"),
                "get_cpu_capability() got an unexpected keyword argument 'device'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_importing_and_calling_does_not_probe_host_features(self):
        script = r'''
import builtins
import os
import sys

class RejectFeatureProbeImport:
    blocked = {"cpuinfo", "numpy", "psutil", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"feature-probe import was attempted: {fullname}")
        return None

original_open = builtins.open

def guarded_open(file, *args, **kwargs):
    if os.fspath(file) == "/proc/cpuinfo":
        raise RuntimeError("/proc/cpuinfo probe was attempted")
    return original_open(file, *args, **kwargs)

sys.meta_path.insert(0, RejectFeatureProbeImport())
builtins.open = guarded_open
os.environ.update(
    ATEN_CPU_CAPABILITY="avx512",
    MKL_DEBUG_CPU_TYPE="5",
    OMP_NUM_THREADS="64",
    OPENBLAS_CORETYPE="SAPPHIRERAPIDS",
)
import torch_rs as torch
from torch_rs.backends import cpu
from torch_rs.backends.cpu import get_cpu_capability

assert torch.backends.cpu is cpu
assert cpu.get_cpu_capability is get_cpu_capability
assert get_cpu_capability.__code__.co_names == (
    "torch",
    "_C",
    "_get_cpu_capability",
)
assert get_cpu_capability() == "DEFAULT"
assert torch._C._get_cpu_capability() == "DEFAULT"
assert not hasattr(torch, "_get_cpu_capability")
assert not any(
    name.split(".", 1)[0] in RejectFeatureProbeImport.blocked
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
