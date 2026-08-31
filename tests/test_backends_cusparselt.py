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


FUNCTION_DOC = "Return a bool indicating if cuSPARSELt is currently available."


class CuSparseLtAvailabilityTests(unittest.TestCase):
    def test_returns_exact_false_private_native_build_flag_without_probes(self):
        function = torch.backends.cusparselt.is_available
        self.assertEqual(
            function.__code__.co_names,
            ("torch", "_C", "_has_cusparselt"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {"USE_CUSPARSELT": "1"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "CUSPARSELT_PATH": "/not/a/cusparselt/install",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
                "USE_CUSPARSELT": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_cusparselt)

        self.assertFalse(hasattr(torch, "_has_cusparselt"))
        self.assertIn("_has_cusparselt", vars(torch._C))
        self.assertNotIn("_has_cusparselt", torch.__all__)
        self.assertNotIn("_has_cusparselt", torch._C.__all__)
        self.assertIs(torch.backends.cuda.is_built(), False)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        cusparselt = importlib.import_module("torch_rs.backends.cusparselt")
        function = cusparselt.is_available

        self.assertIs(torch.backends.cusparselt, cusparselt)
        self.assertIs(sys.modules["torch_rs.backends.cusparselt"], cusparselt)
        self.assertIs(type(cusparselt), types.ModuleType)
        self.assertIsNone(cusparselt.__doc__)
        self.assertEqual(cusparselt.__all__, ["is_available"])
        self.assertEqual(
            {name for name in vars(cusparselt) if not name.startswith("_")},
            {"is_available", "torch"},
        )
        self.assertIs(cusparselt.torch, torch)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(inspect.get_annotations(function), {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.backends.cusparselt")
        self.assertIs(inspect.getmodule(function), cusparselt)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        cusparselt = importlib.import_module("torch_rs.backends.cusparselt")
        function = cusparselt.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.cusparselt, cusparselt)
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
        private_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        native_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import cusparselt", backend_import)
        exec(
            "from torch_rs.backends.cusparselt import is_available",
            function_import,
        )
        exec("from torch_rs._C import _has_cusparselt", private_import)
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.cusparselt import *", child_wildcard)
        exec("from torch_rs._C import *", native_wildcard)

        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["cusparselt"], cusparselt)
        self.assertIs(function_import["is_available"], function)
        self.assertIs(private_import["_has_cusparselt"], False)
        self.assertIs(parent_wildcard["cusparselt"], cusparselt)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"is_available"},
        )
        self.assertIs(child_wildcard["is_available"], function)
        self.assertNotIn("_has_cusparselt", native_wildcard)

        self.assertNotIn("backends", torch.__all__)
        self.assertFalse(hasattr(torch, "cusparselt"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("cusparselt", top_level_wildcard)
        self.assertNotIn("_has_cusparselt", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(copy.copy(torch._C._has_cusparselt), False)
        self.assertIs(copy.deepcopy(torch._C._has_cusparselt), False)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.cusparselt", payload)
                self.assertIs(pickle.loads(payload), function)
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(
                            torch._C._has_cusparselt,
                            protocol=protocol,
                        )
                    ),
                    False,
                )

    def test_value_and_identity_are_stable_across_threads(self):
        function = torch.backends.cusparselt.is_available
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    value,
                    type(value) is bool,
                    value is torch._C._has_cusparselt,
                    function is torch.backends.cusparselt.is_available,
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
        self.assertEqual(results, [(False, True, True, True)] * worker_count)

    def test_reload_replaces_function_and_preserves_native_flag(self):
        package = torch
        native = torch._C
        backends = torch.backends
        cusparselt = backends.cusparselt
        namespace = cusparselt.__dict__
        old_function = cusparselt.is_available

        reloaded = importlib.reload(cusparselt)
        function = cusparselt.is_available

        self.assertIs(reloaded, cusparselt)
        self.assertIs(cusparselt.__dict__, namespace)
        self.assertIs(backends.cusparselt, cusparselt)
        self.assertIs(sys.modules[cusparselt.__name__], cusparselt)
        self.assertIsNot(function, old_function)
        self.assertIs(function(), native._has_cusparselt)
        self.assertIs(function(), False)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_available at 0x...>: "
            "it's not the same object as "
            "torch_rs.backends.cusparselt.is_available",
        )

        self.assertIs(importlib.reload(package), package)
        self.assertIs(package._C, native)
        self.assertIs(package.backends, backends)
        self.assertIs(backends.cusparselt, cusparselt)
        self.assertIs(cusparselt.is_available, function)
        self.assertIs(cusparselt.is_available(), False)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(native._has_cusparselt, False)
        self.assertIs(cusparselt.is_available(), native._has_cusparselt)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.cusparselt.is_available
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

    def test_version_execution_and_algorithm_surface_remain_unsupported(self):
        cusparselt = torch.backends.cusparselt
        for name in (
            "get_max_alg_id",
            "version",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(cusparselt, name))
                with self.assertRaises(ImportError):
                    exec(f"from torch_rs.backends.cusparselt import {name}", {})

        self.assertFalse(hasattr(torch._C, "_cusparselt"))
        self.assertFalse(hasattr(torch, "cusparselt"))
        self.assertIs(torch.cuda.is_available(), False)
        self.assertEqual(torch.cuda.device_count(), 0)

    def test_import_and_call_are_probe_free_with_cuda_visibility(self):
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
    CUSPARSELT_PATH="/not/a/cusparselt/install",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
    USE_CUSPARSELT="1",
)

import torch_rs as torch
from torch_rs.backends import cusparselt
from torch_rs.backends.cusparselt import is_available
from torch_rs._C import _has_cusparselt

assert torch.backends.cusparselt is cusparselt
assert cusparselt.is_available is is_available
assert is_available.__code__.co_names == ("torch", "_C", "_has_cusparselt")
assert is_available() is _has_cusparselt is False
assert torch.backends.cuda.is_built() is False
assert not hasattr(cusparselt, "version")
assert not hasattr(cusparselt, "get_max_alg_id")
assert not hasattr(torch._C, "_cusparselt")
assert not hasattr(torch, "_has_cusparselt")
assert not hasattr(torch, "cusparselt")
assert torch.cuda.is_available() is False
assert type(torch.cuda.device_count()) is int
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
