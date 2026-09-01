import copy
import importlib
import inspect
import os
import pickle
import subprocess
import sys
import threading
import types
import unittest
import warnings
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = "Return whether PyTorch is built with MKL-DNN support."
DEPRECATED_HAS_MKLDNN_WARNING = (
    "'has_mkldnn' is deprecated, please use "
    "'torch_rs.backends.mkldnn.is_available()'"
)
BACKEND_MODULES = {
    "cpu",
    "cuda",
    "cusparselt",
    "cudnn",
    "kleidiai",
    "m",
    "mha",
    "mkl",
    "mkldnn",
    "nnpack",
    "openmp",
}


class MkldnnAvailabilityTests(unittest.TestCase):
    def test_returns_exact_false_native_build_flag_without_probes(self):
        function = torch.backends.mkldnn.is_available
        self.assertEqual(
            function.__func__.__code__.co_names,
            ("is_available",),
        )
        self.assertEqual(
            torch.backends.mkldnn.m.is_available.__code__.co_names,
            ("torch", "_C", "_has_mkldnn"),
        )
        self.assertEqual(function.__func__.__code__.co_freevars, ())
        self.assertEqual(function.__func__.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_MKLDNN": "1"},
            {"ONEDNN_VERBOSE": "1"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "DNNL_VERBOSE": "1",
                "MKL_DEBUG_CPU_TYPE": "5",
                "ONEDNN_VERBOSE": "1",
                "OMP_NUM_THREADS": "64",
                "USE_MKL": "1",
                "USE_MKLDNN": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_mkldnn)

        self.assertIs(torch.has_mkl, False)
        self.assertIs(torch._C.has_mkl, torch.has_mkl)
        self.assertIs(torch.has_lapack, False)
        self.assertIs(torch._C.has_lapack, torch.has_lapack)
        self.assertFalse(hasattr(torch._C, "has_mkldnn"))
        self.assertEqual(torch._C.__all__.count("_has_mkldnn"), 0)
        self.assertNotIn("has_mkldnn", torch.__all__)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertIs(torch.has_mkldnn, False)
        self.assertEqual([str(warning.message) for warning in caught], [
            DEPRECATED_HAS_MKLDNN_WARNING
        ])

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        function = mkldnn.is_available

        self.assertIs(torch.backends.mkldnn, mkldnn)
        self.assertIs(sys.modules["torch_rs.backends.mkldnn"], mkldnn)
        self.assertIsInstance(mkldnn, types.ModuleType)
        self.assertEqual(type(mkldnn).__name__, "MkldnnModule")
        self.assertIsNone(mkldnn.__doc__)
        self.assertFalse(hasattr(mkldnn, "__all__"))
        self.assertEqual(
            {name for name in vars(mkldnn) if not name.startswith("_")},
            {"m"},
        )
        self.assertIs(mkldnn.m.torch, torch)
        self.assertIs(type(mkldnn.m.is_available), types.FunctionType)
        self.assertEqual(mkldnn.m.is_available.__doc__, FUNCTION_DOC)
        self.assertIs(type(function), types.MethodType)
        self.assertIs(function.__self__, mkldnn)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "MkldnnModule.is_available")
        self.assertEqual(function.__module__, "torch_rs.backends.mkldnn")
        self.assertIs(inspect.getmodule(function), mkldnn)
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__func__.__defaults__)
        self.assertIsNone(function.__func__.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        function = mkldnn.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mkldnn, mkldnn)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            BACKEND_MODULES,
        )

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import mkldnn", backend_import)
        exec(
            "from torch_rs.backends.mkldnn import is_available",
            function_import,
        )
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.mkldnn import *", child_wildcard)
        exec("from torch_rs import *", top_level_wildcard)

        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mkldnn"], mkldnn)
        self.assertEqual(function_import["is_available"], function)
        self.assertIs(function_import["is_available"].__self__, mkldnn)
        self.assertIs(parent_wildcard["mkldnn"], mkldnn)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"m"},
        )
        self.assertIs(child_wildcard["m"], mkldnn.m)
        self.assertNotIn("backends", torch.__all__)
        self.assertNotIn("mkldnn", torch.__all__)
        self.assertNotIn("has_mkldnn", torch.__all__)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("mkldnn", top_level_wildcard)
        self.assertNotIn("has_mkldnn", top_level_wildcard)

        copied = copy.copy(function)
        self.assertEqual(copied, function)
        self.assertIsNot(copied, function)
        self.assertIs(copied.__self__, mkldnn)
        with self.assertRaisesRegex(TypeError, "MkldnnModule"):
            copy.deepcopy(function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(TypeError, "MkldnnModule"):
                    pickle.dumps(function, protocol=protocol)

    def test_value_and_identity_are_stable_across_threads(self):
        function = torch.backends.mkldnn.is_available
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
                    value is torch._C._has_mkldnn,
                    function == torch.backends.mkldnn.is_available,
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
        mkldnn = backends.mkldnn
        old_function = mkldnn.is_available

        reloaded = importlib.reload(mkldnn)
        function = reloaded.is_available

        self.assertIsNot(reloaded, mkldnn)
        self.assertIs(backends.mkldnn, reloaded)
        self.assertIs(sys.modules[reloaded.__name__], reloaded)
        self.assertIsNot(function, old_function)
        self.assertEqual(function, reloaded.is_available)
        self.assertIs(function(), native._has_mkldnn)
        self.assertIs(function(), False)
        copied = copy.copy(function)
        self.assertEqual(copied, function)
        self.assertIsNot(copied, function)
        self.assertIs(copied.__self__, reloaded)
        with self.assertRaisesRegex(TypeError, "MkldnnModule"):
            copy.deepcopy(function)
        with self.assertRaisesRegex(TypeError, "MkldnnModule"):
            pickle.dumps(function)
        with self.assertRaisesRegex(TypeError, "MkldnnModule"):
            pickle.dumps(old_function)

        self.assertIs(importlib.reload(package), package)
        self.assertIs(package._C, native)
        self.assertIs(package.backends, backends)
        self.assertIs(backends.mkldnn.is_available(), False)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(native._has_mkldnn, False)
        self.assertIs(backends.mkldnn.is_available(), native._has_mkldnn)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.mkldnn.is_available
        cases = (
            (
                lambda: function(None),
                "MkldnnModule.is_available() takes 1 positional argument but "
                "2 were given",
            ),
            (
                lambda: function(None, None),
                "MkldnnModule.is_available() takes 1 positional argument but "
                "3 were given",
            ),
            (
                lambda: function(enabled=True),
                "MkldnnModule.is_available() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "MkldnnModule.is_available() got an unexpected keyword "
                "argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_tensors_operators_flags_and_execution_remain_unsupported(self):
        mkldnn = torch.backends.mkldnn
        for name in (
            "VERBOSE_OFF",
            "VERBOSE_ON",
            "VERBOSE_ON_CREATION",
            "enabled",
            "flags",
            "set_flags",
            "verbose",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(mkldnn, name))
                with self.assertRaises(ImportError):
                    exec(f"from torch_rs.backends.mkldnn import {name}", {})

        self.assertFalse(hasattr(torch, "mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        tensor = torch.tensor([1.0])
        self.assertIs(tensor.is_mkldnn, False)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'mkldnn' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="mkldnn")

    def test_import_and_call_are_probe_free_without_execution_claims(self):
        script = r'''
import builtins
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {
        "dnnl",
        "mkl",
        "mkldnn",
        "mkl_service",
        "numpy",
        "onednn",
        "scipy",
        "torch",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

original_open = builtins.open

def guarded_open(file, *args, **kwargs):
    if os.fspath(file) == "/proc/cpuinfo":
        raise RuntimeError("/proc/cpuinfo probe was attempted")
    return original_open(file, *args, **kwargs)

sys.meta_path.insert(0, RejectExternalRuntimeImport())
builtins.open = guarded_open
os.environ.update(
    ATEN_CPU_CAPABILITY="avx512",
    DNNL_VERBOSE="1",
    MKL_DEBUG_CPU_TYPE="5",
    MKL_NUM_THREADS="64",
    ONEDNN_VERBOSE="1",
    OMP_NUM_THREADS="64",
    USE_MKL="1",
    USE_MKLDNN="1",
)

import torch_rs as torch
from torch_rs.backends import mkldnn
from torch_rs.backends.mkldnn import is_available

assert torch.backends.mkldnn is mkldnn
assert mkldnn.is_available == is_available
assert is_available.__func__.__code__.co_names == ("is_available",)
assert mkldnn.m.is_available.__code__.co_names == ("torch", "_C", "_has_mkldnn")
assert is_available() is torch._C._has_mkldnn is False
assert not hasattr(torch._C, "has_mkldnn")
assert "has_mkldnn" not in torch.__all__
assert torch.has_mkl is torch._C.has_mkl is False
assert torch.has_lapack is torch._C.has_lapack is False
assert not hasattr(mkldnn, "enabled")
assert not hasattr(mkldnn, "flags")
assert not hasattr(mkldnn, "set_flags")
assert not hasattr(mkldnn, "verbose")
assert not hasattr(torch, "mkldnn")
assert not hasattr(torch.Tensor, "to_mkldnn")
assert torch.tensor([1.0]).is_mkldnn is False
try:
    torch.tensor([1.0], device="mkldnn")
except RuntimeError as error:
    assert str(error) == (
        "tensor(): device 'mkldnn' is not supported; only 'cpu' is implemented"
    )
else:
    raise AssertionError("mkldnn tensor construction unexpectedly succeeded")
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
