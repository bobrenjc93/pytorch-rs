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
        None,
    ),
    "mkl": (
        "has_mkl",
        "Return whether PyTorch is built with MKL support.",
        None,
    ),
    "mkldnn": (
        "_has_mkldnn",
        None,
        None,
    ),
    "nnpack": (
        None,
        "Return whether PyTorch is built with NNPACK support.",
        ["is_available", "flags", "set_flags"],
    ),
}


class NativeCpuBackendAvailabilityTests(unittest.TestCase):
    def test_queries_return_the_exact_native_build_flags(self):
        environments = (
            {},
            {"USE_OPENMP": "1", "USE_MKL": "1", "USE_NNPACK": "1"},
            {
                "OMP_NUM_THREADS": "64",
                "MKL_NUM_THREADS": "64",
                "NNPACK_NUM_THREADS": "64",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for backend, (flag, _, _) in BACKENDS.items():
                        with self.subTest(backend=backend):
                            function = getattr(torch.backends, backend).is_available
                            result = function()
                            self.assertIs(type(result), bool)
                            self.assertIs(result, False)
                            if flag is None:
                                self.assertIs(result, torch._nnpack_available())
                            else:
                                self.assertIs(result, getattr(torch._C, flag))
                                if not flag.startswith("_"):
                                    self.assertIs(result, getattr(torch, flag))

        self.assertNotIn("_nnpack_available", torch.__all__)
        self.assertFalse(hasattr(torch._C, "_nnpack_available"))
        self.assertIs(torch._nnpack_available(None, enabled=True), False)
        self.assertIs(torch._C._has_mkldnn, False)
        self.assertFalse(hasattr(torch._C, "has_mkldnn"))
        self.assertNotIn("_has_mkldnn", torch._C.__all__)
        self.assertNotIn("has_mkldnn", torch.__all__)
        self.assertIs(torch.has_mkl, False)
        self.assertIs(torch._C.has_mkl, torch.has_mkl)
        self.assertIs(torch.has_lapack, False)
        self.assertIs(torch._C.has_lapack, torch.has_lapack)

    def test_private_nnpack_build_probe_matches_pytorch_2_13_ownership(self):
        function = torch._nnpack_available
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "_nnpack_available")
        self.assertEqual(
            function.__qualname__,
            "_VariableFunctionsClass._nnpack_available",
        )
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__text_signature__)
        self.assertIsNone(function.__self__)
        self.assertFalse(hasattr(function, "__annotations__"))
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner._nnpack_available, function)
        self.assertFalse(hasattr(torch._C, "_nnpack_available"))
        self.assertNotIn("_nnpack_available", torch.__all__)
        self.assertNotIn("_nnpack_available", torch._C.__all__)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        for backend, (flag, documentation, module_all) in BACKENDS.items():
            with self.subTest(backend=backend):
                module = importlib.import_module(f"torch_rs.backends.{backend}")
                function = module.is_available

                self.assertIsNone(module.__doc__)
                if module_all is None:
                    self.assertFalse(hasattr(module, "__all__"))
                else:
                    self.assertEqual(module.__all__, module_all)

                if backend == "mkldnn":
                    self.assertIsInstance(module, types.ModuleType)
                    self.assertEqual(type(module).__name__, "MkldnnModule")
                    self.assertEqual(
                        {name for name in vars(module) if not name.startswith("_")},
                        {"m"},
                    )
                    self.assertIs(module.m.torch, torch)
                    self.assertIs(type(function), types.MethodType)
                    self.assertIs(function.__self__, module)
                    self.assertEqual(str(inspect.signature(function)), "()")
                    self.assertEqual(inspect.get_annotations(function), {})
                    self.assertEqual(function.__name__, "is_available")
                    self.assertEqual(
                        function.__qualname__,
                        "MkldnnModule.is_available",
                    )
                    self.assertEqual(function.__module__, module.__name__)
                    self.assertIs(inspect.getmodule(function), module)
                    self.assertIsNone(function.__doc__)
                    self.assertIsNone(function.__defaults__)
                    self.assertIsNone(function.__kwdefaults__)
                    self.assertEqual(function.__dict__, {})
                    self.assertFalse(hasattr(function, "__text_signature__"))
                    self.assertEqual(
                        function.__func__.__code__.co_names,
                        ("is_available",),
                    )
                    self.assertEqual(
                        module.m.is_available.__code__.co_names,
                        ("torch", "_C", "_has_mkldnn"),
                    )
                    self.assertEqual(function.__func__.__code__.co_freevars, ())
                    self.assertEqual(function.__func__.__code__.co_cellvars, ())
                    continue

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
                expected_names = (
                    ("torch", "_nnpack_available")
                    if flag is None
                    else ("torch", "_C", flag)
                )
                self.assertEqual(function.__code__.co_names, expected_names)
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
            {
                *BACKENDS,
                "cpu",
                "cuda",
                "cusparselt",
                "cudnn",
                "kleidiai",
                "m",
                "mha",
            },
        )

        package_import = {}
        exec("from torch_rs import backends", package_import)
        self.assertIs(package_import["backends"], backends)

        parent_wildcard = {}
        exec("from torch_rs.backends import *", parent_wildcard)
        self.assertEqual(
            {name for name in parent_wildcard if not name.startswith("__")},
            {
                *BACKENDS,
                "cpu",
                "cuda",
                "cusparselt",
                "cudnn",
                "kleidiai",
                "m",
                "mha",
            },
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
                if backend == "mkldnn":
                    self.assertEqual(function_import["is_available"], function)
                    self.assertIs(function_import["is_available"].__self__, module)
                else:
                    self.assertIs(function_import["is_available"], function)

                child_wildcard = {}
                exec(f"from {module_name} import *", child_wildcard)
                expected_child_names = (
                    {"flags", "is_available", "set_flags"}
                    if backend == "nnpack"
                    else {"m"}
                    if backend == "mkldnn"
                    else {"is_available", "torch"}
                )
                self.assertEqual(
                    {
                        name
                        for name in child_wildcard
                        if not name.startswith("__")
                    },
                    expected_child_names,
                )
                if backend == "mkldnn":
                    self.assertIs(child_wildcard["m"], module.m)
                else:
                    self.assertIs(child_wildcard["is_available"], function)
                if backend not in {"mkldnn", "nnpack"}:
                    self.assertIs(child_wildcard["torch"], torch)

                if backend == "mkldnn":
                    copied = copy.copy(function)
                    self.assertEqual(copied, function)
                    self.assertIsNot(copied, function)
                    self.assertIs(copied.__self__, module)
                    with self.assertRaisesRegex(TypeError, "MkldnnModule"):
                        copy.deepcopy(function)
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                        with self.subTest(backend=backend, protocol=protocol):
                            with self.assertRaisesRegex(TypeError, "MkldnnModule"):
                                pickle.dumps(function, protocol=protocol)
                else:
                    self.assertIs(copy.copy(function), function)
                    self.assertIs(copy.deepcopy(function), function)
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                        payload = pickle.dumps(function, protocol=protocol)
                        self.assertIn(module_name.encode(), payload)
                        self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_functions_and_preserves_canonical_modules(self):
        backends = torch.backends
        for backend, (flag, _, _) in BACKENDS.items():
            with self.subTest(backend=backend):
                module = getattr(backends, backend)
                old_function = module.is_available

                if backend == "mkldnn":
                    reloaded = importlib.reload(module)
                    self.assertIsNot(reloaded, module)
                    self.assertIs(getattr(backends, backend), reloaded)
                    self.assertIs(sys.modules[reloaded.__name__], reloaded)
                    self.assertIsNot(reloaded.is_available, old_function)
                    self.assertEqual(
                        reloaded.is_available,
                        getattr(backends, backend).is_available,
                    )
                    self.assertIs(reloaded.is_available(), torch._C._has_mkldnn)
                    continue

                namespace = module.__dict__

                reloaded = importlib.reload(module)

                self.assertIs(reloaded, module)
                self.assertIs(module.__dict__, namespace)
                self.assertIs(getattr(backends, backend), module)
                self.assertIs(sys.modules[module.__name__], module)
                self.assertIsNot(module.is_available, old_function)
                expected = (
                    torch._nnpack_available()
                    if flag is None
                    else getattr(torch._C, flag)
                )
                self.assertIs(module.is_available(), expected)
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
            if backend == "mkldnn":
                cases = (
                    (
                        lambda: function(None),
                        "MkldnnModule.is_available() takes 1 positional "
                        "argument but 2 were given",
                    ),
                    (
                        lambda: function(None, None),
                        "MkldnnModule.is_available() takes 1 positional "
                        "argument but 3 were given",
                    ),
                    (
                        lambda: function(enabled=True),
                        "MkldnnModule.is_available() got an unexpected "
                        "keyword argument 'enabled'",
                    ),
                    (
                        lambda: function(None, enabled=True),
                        "MkldnnModule.is_available() got an unexpected "
                        "keyword argument 'enabled'",
                    ),
                )
            else:
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
            "flags",
            "set_flags",
        ):
            with self.subTest(parent_name=name):
                self.assertFalse(hasattr(backends, name))

        for name in ("VERBOSE_OFF", "VERBOSE_ON", "verbose"):
            with self.subTest(mkl_name=name):
                self.assertFalse(hasattr(backends.mkl, name))

        for name in (
            "VERBOSE_OFF",
            "VERBOSE_ON",
            "VERBOSE_ON_CREATION",
            "enabled",
            "flags",
            "set_flags",
            "verbose",
        ):
            with self.subTest(mkldnn_name=name):
                self.assertFalse(hasattr(backends.mkldnn, name))

        for name in (
            "mps",
            "quantized",
        ):
            with self.subTest(backend=name):
                self.assertFalse(hasattr(backends, name))
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(f"torch_rs.backends.{name}")

        self.assertFalse(hasattr(torch._C, "_verbose"))
        self.assertTrue(hasattr(torch._C, "_get_nnpack_enabled"))
        self.assertTrue(hasattr(torch._C, "_set_nnpack_enabled"))
        self.assertFalse(hasattr(torch, "_get_nnpack_enabled"))
        self.assertFalse(hasattr(torch, "_set_nnpack_enabled"))
        self.assertNotIn("_get_nnpack_enabled", torch._C.__all__)
        self.assertNotIn("_set_nnpack_enabled", torch._C.__all__)

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"torch", "numpy", "scipy", "dnnl", "mkl", "mkldnn", "mkl_service"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    USE_OPENMP="1",
    USE_MKL="1",
    USE_MKLDNN="1",
    USE_NNPACK="1",
    OMP_NUM_THREADS="64",
    MKL_NUM_THREADS="64",
    NNPACK_NUM_THREADS="64",
)
import torch_rs as torch
from torch_rs.backends import mkl, mkldnn, nnpack, openmp
from torch_rs.backends.mkl import is_available as is_mkl_available
from torch_rs.backends.mkldnn import is_available as is_mkldnn_available
from torch_rs.backends.nnpack import is_available as is_nnpack_available
from torch_rs.backends.nnpack import flags as nnpack_flags
from torch_rs.backends.nnpack import set_flags as set_nnpack_flags
from torch_rs.backends.openmp import is_available as is_openmp_available

assert torch.backends.mkl is mkl
assert torch.backends.mkldnn is mkldnn
assert torch.backends.nnpack is nnpack
assert torch.backends.openmp is openmp
assert mkl.is_available is is_mkl_available
assert mkldnn.is_available == is_mkldnn_available
assert nnpack.is_available is is_nnpack_available
assert nnpack.flags is nnpack_flags
assert nnpack.set_flags is set_nnpack_flags
assert openmp.is_available is is_openmp_available
assert mkl.is_available() is torch._C.has_mkl is False
assert mkldnn.is_available() is torch._C._has_mkldnn is False
assert not hasattr(torch._C, "has_mkldnn")
assert "has_mkldnn" not in torch.__all__
assert torch.has_mkl is False
assert torch.has_lapack is False
assert nnpack.is_available() is torch._nnpack_available() is False
assert nnpack.set_flags(False) == (True,)
assert nnpack.set_flags(True) == (False,)
with nnpack.flags(False) as entered:
    assert entered is None
    assert torch._C._get_nnpack_enabled() is False
assert torch._C._get_nnpack_enabled() is True
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
