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


BACKENDS = {
    "openmp": (
        "has_openmp",
        "Return whether PyTorch is built with OpenMP support.",
    ),
    "mkl": (
        "has_mkl",
        "Return whether PyTorch is built with MKL support.",
    ),
}


class NativeCpuBackendAvailabilityTests(unittest.TestCase):
    def test_queries_return_the_exact_native_build_flags(self):
        environments = (
            {},
            {"USE_OPENMP": "1", "USE_MKL": "1"},
            {"OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "64"},
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for backend, (flag, _) in BACKENDS.items():
                        with self.subTest(backend=backend):
                            function = getattr(torch.backends, backend).is_available
                            result = function()
                            self.assertIs(type(result), bool)
                            self.assertIs(result, False)
                            self.assertIs(result, getattr(torch._C, flag))
                            self.assertIs(result, getattr(torch, flag))

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        for backend, (flag, documentation) in BACKENDS.items():
            with self.subTest(backend=backend):
                module = importlib.import_module(f"torch_rs.backends.{backend}")
                function = module.is_available

                self.assertIsNone(module.__doc__)
                self.assertFalse(hasattr(module, "__all__"))
                self.assertIs(type(function), types.FunctionType)
                self.assertEqual(str(inspect.signature(function)), "()")
                self.assertEqual(inspect.get_annotations(function), {})
                self.assertEqual(function.__name__, "is_available")
                self.assertEqual(function.__qualname__, "is_available")
                self.assertEqual(function.__module__, module.__name__)
                self.assertIs(inspect.getmodule(function), module)
                self.assertEqual(function.__doc__, documentation)
                self.assertIsNone(function.__defaults__)
                self.assertIsNone(function.__kwdefaults__)
                self.assertEqual(function.__dict__, {})
                self.assertFalse(hasattr(function, "__text_signature__"))
                self.assertEqual(function.__code__.co_names, ("torch", "_C", flag))
                self.assertEqual(function.__code__.co_freevars, ())
                self.assertEqual(function.__code__.co_cellvars, ())

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        self.assertIs(torch.backends, backends)
        self.assertIs(sys.modules["torch_rs.backends"], backends)
        self.assertIsNone(backends.__doc__)
        self.assertFalse(hasattr(backends, "__all__"))
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            set(BACKENDS),
        )

        package_import = {}
        exec("from torch_rs import backends", package_import)
        self.assertIs(package_import["backends"], backends)

        parent_wildcard = {}
        exec("from torch_rs.backends import *", parent_wildcard)
        self.assertEqual(
            {name for name in parent_wildcard if not name.startswith("__")},
            set(BACKENDS),
        )
        self.assertNotIn("backends", torch.__all__)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)

        for backend in BACKENDS:
            with self.subTest(backend=backend):
                module_name = f"torch_rs.backends.{backend}"
                module = importlib.import_module(module_name)
                function = module.is_available
                self.assertIs(getattr(backends, backend), module)
                self.assertIs(sys.modules[module_name], module)
                self.assertIs(module.torch, torch)

                backend_import = {}
                exec(f"from torch_rs.backends import {backend}", backend_import)
                self.assertIs(backend_import[backend], module)

                function_import = {}
                exec(
                    f"from {module_name} import is_available",
                    function_import,
                )
                self.assertIs(function_import["is_available"], function)

                child_wildcard = {}
                exec(f"from {module_name} import *", child_wildcard)
                self.assertEqual(
                    {
                        name
                        for name in child_wildcard
                        if not name.startswith("__")
                    },
                    {"is_available", "torch"},
                )
                self.assertIs(child_wildcard["is_available"], function)
                self.assertIs(child_wildcard["torch"], torch)

                self.assertIs(copy.copy(function), function)
                self.assertIs(copy.deepcopy(function), function)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                    payload = pickle.dumps(function, protocol=protocol)
                    self.assertIn(module_name.encode(), payload)
                    self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_functions_and_preserves_canonical_modules(self):
        backends = torch.backends
        for backend, (flag, _) in BACKENDS.items():
            with self.subTest(backend=backend):
                module = getattr(backends, backend)
                old_function = module.is_available
                namespace = module.__dict__

                reloaded = importlib.reload(module)

                self.assertIs(reloaded, module)
                self.assertIs(module.__dict__, namespace)
                self.assertIs(getattr(backends, backend), module)
                self.assertIs(sys.modules[module.__name__], module)
                self.assertIsNot(module.is_available, old_function)
                self.assertIs(module.is_available(), getattr(torch._C, flag))
                self.assertIs(copy.copy(module.is_available), module.is_available)
                self.assertIs(
                    copy.deepcopy(module.is_available), module.is_available
                )
                self.assertIs(
                    pickle.loads(pickle.dumps(module.is_available)),
                    module.is_available,
                )
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(old_function)
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
                self.assertEqual(
                    message,
                    "Can't pickle <function is_available at 0x...>: "
                    f"it's not the same object as {module.__name__}.is_available",
                )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        for backend in BACKENDS:
            function = getattr(torch.backends, backend).is_available
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
                with self.subTest(backend=backend, message=message):
                    with self.assertRaises(TypeError) as raised:
                        call()
                    self.assertEqual(str(raised.exception), message)
                    self.assertEqual(raised.exception.args, (message,))

    def test_configuration_verbosity_execution_and_other_backends_are_unsupported(self):
        backends = torch.backends
        for name in (
            "disable_global_flags",
            "flags",
            "flags_frozen",
            "set_flags",
        ):
            with self.subTest(parent_name=name):
                self.assertFalse(hasattr(backends, name))

        for name in ("VERBOSE_OFF", "VERBOSE_ON", "verbose"):
            with self.subTest(mkl_name=name):
                self.assertFalse(hasattr(backends.mkl, name))

        for name in (
            "cpu",
            "cuda",
            "cudnn",
            "mkldnn",
            "mps",
            "nnpack",
            "quantized",
        ):
            with self.subTest(backend=name):
                self.assertFalse(hasattr(backends, name))
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(f"torch_rs.backends.{name}")

        self.assertFalse(hasattr(torch._C, "_verbose"))

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"torch", "numpy", "scipy", "mkl", "mkl_service"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    USE_OPENMP="1",
    USE_MKL="1",
    OMP_NUM_THREADS="64",
    MKL_NUM_THREADS="64",
)
import torch_rs as torch
from torch_rs.backends import mkl, openmp
from torch_rs.backends.mkl import is_available as is_mkl_available
from torch_rs.backends.openmp import is_available as is_openmp_available

assert torch.backends.mkl is mkl
assert torch.backends.openmp is openmp
assert mkl.is_available is is_mkl_available
assert openmp.is_available is is_openmp_available
assert mkl.is_available() is torch._C.has_mkl is False
assert openmp.is_available() is torch._C.has_openmp is False
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
