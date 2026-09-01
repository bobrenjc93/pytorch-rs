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
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = "Return whether PyTorch is built with MKL-DNN support."
UNSUPPORTED_MKLDNN_NAMES = (
    "VERBOSE_OFF",
    "VERBOSE_ON",
    "VERBOSE_ON_CREATION",
    "allow_tf32",
    "conv",
    "deterministic",
    "enabled",
    "flags",
    "fp32_precision",
    "is_acl_available",
    "matmul",
    "rnn",
    "set_flags",
    "verbose",
)


def fresh_mkldnn_module():
    module_name = "torch_rs.backends.mkldnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "mkldnn"):
        del torch.backends.mkldnn
    module = importlib.import_module(module_name)
    torch.backends.mkldnn = module
    return module


class MkldnnAvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.mkldnn = fresh_mkldnn_module()

    def tearDown(self):
        fresh_mkldnn_module()

    def test_returns_exact_false_private_native_status_without_probes(self):
        mkldnn = self.mkldnn
        function = mkldnn.is_available
        implementation = mkldnn.m.is_available
        self.assertEqual(function.__func__.__code__.co_names, ("is_available",))
        self.assertEqual(function.__func__.__code__.co_freevars, ())
        self.assertEqual(function.__func__.__code__.co_cellvars, ())
        self.assertEqual(
            implementation.__code__.co_names,
            ("torch", "_C", "_has_mkldnn"),
        )
        self.assertEqual(implementation.__code__.co_freevars, ())
        self.assertEqual(implementation.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_MKLDNN": "1"},
            {"DNNL_VERBOSE": "1"},
            {"MKLDNN_VERBOSE": "1"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "DNNL_VERBOSE": "1",
                "MKL_NUM_THREADS": "64",
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
                    self.assertIs(implementation(), torch._C._has_mkldnn)

        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertFalse(hasattr(torch, "has_mkldnn"))
        self.assertIn("_has_mkldnn", vars(torch._C))
        self.assertNotIn("_has_mkldnn", torch.__all__)
        self.assertNotIn("_has_mkldnn", torch._C.__all__)
        self.assertIs(torch.has_mkl, torch._C.has_mkl)
        self.assertIs(torch.has_mkl, False)
        self.assertIs(torch.has_lapack, torch._C.has_lapack)
        self.assertIs(torch.has_lapack, False)
        self.assertEqual(torch.backends.cpu.get_cpu_capability(), "DEFAULT")

    def test_signature_documentation_and_module_identity_match_supported_subset(self):
        mkldnn = self.mkldnn
        function = mkldnn.is_available
        implementation = mkldnn.m.is_available

        self.assertIs(torch.backends.mkldnn, mkldnn)
        self.assertIs(sys.modules["torch_rs.backends.mkldnn"], mkldnn)
        self.assertIsInstance(mkldnn, types.ModuleType)
        self.assertEqual(type(mkldnn).__name__, "MkldnnModule")
        self.assertEqual(type(mkldnn).__module__, "torch_rs.backends.mkldnn")
        self.assertIsNone(mkldnn.__doc__)
        self.assertFalse(hasattr(mkldnn, "__all__"))
        self.assertEqual(
            {name for name in vars(mkldnn) if not name.startswith("_")},
            {"m"},
        )
        self.assertIs(type(mkldnn.m), types.ModuleType)
        self.assertIsNot(mkldnn.m, mkldnn)
        self.assertEqual(mkldnn.m.__name__, mkldnn.__name__)
        self.assertIs(mkldnn.torch, torch)

        self.assertIs(type(function), types.MethodType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "MkldnnModule.is_available")
        self.assertEqual(function.__module__, "torch_rs.backends.mkldnn")
        self.assertIs(inspect.getmodule(function), mkldnn)
        self.assertIsNone(function.__doc__)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertIs(function.__self__, mkldnn)
        self.assertIs(function.__func__, type(mkldnn).is_available)

        self.assertIs(type(implementation), types.FunctionType)
        self.assertEqual(str(inspect.signature(implementation)), "()")
        self.assertEqual(implementation.__name__, "is_available")
        self.assertEqual(implementation.__qualname__, "is_available")
        self.assertEqual(implementation.__module__, "torch_rs.backends.mkldnn")
        self.assertIs(inspect.getmodule(implementation), mkldnn)
        self.assertEqual(implementation.__doc__, FUNCTION_DOC)
        self.assertIsNone(implementation.__defaults__)
        self.assertIsNone(implementation.__kwdefaults__)
        self.assertEqual(implementation.__dict__, {})
        self.assertFalse(hasattr(implementation, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mkldnn = self.mkldnn
        function = mkldnn.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mkldnn, mkldnn)
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
                "mkldnn",
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
        exec("from torch_rs.backends import mkldnn", backend_import)
        exec(
            "from torch_rs.backends.mkldnn import is_available",
            function_import,
        )
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.mkldnn import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mkldnn"], mkldnn)
        self.assertEqual(function_import["is_available"], function)
        self.assertIs(function_import["is_available"].__self__, mkldnn)
        self.assertIs(function_import["is_available"].__func__, function.__func__)
        self.assertIs(parent_wildcard["mkldnn"], mkldnn)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"m"},
        )
        self.assertIs(child_wildcard["m"], mkldnn.m)

        self.assertNotIn("backends", torch.__all__)
        self.assertFalse(hasattr(torch, "mkldnn"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("mkldnn", top_level_wildcard)
        self.assertNotIn("_has_mkldnn", top_level_wildcard)

        copied = copy.copy(function)
        self.assertIsNot(copied, function)
        self.assertEqual(copied, function)
        self.assertIs(copied.__self__, mkldnn)
        self.assertIs(copied.__func__, function.__func__)
        self.assertIs(copy.copy(mkldnn.m.is_available), mkldnn.m.is_available)
        self.assertIs(copy.deepcopy(mkldnn.m.is_available), mkldnn.m.is_available)

        for copier in (copy.copy, copy.deepcopy):
            with self.subTest(copier=copier.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "^cannot pickle 'MkldnnModule' object$",
                ):
                    copier(mkldnn)

        with self.assertRaisesRegex(
            TypeError,
            "^cannot pickle 'MkldnnModule' object$",
        ):
            copy.deepcopy(function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                with self.assertRaises(TypeError) as raised:
                    pickle.dumps(function, protocol=protocol)
                self.assertIn(
                    str(raised.exception),
                    {
                        "module() argument 'name' must be str, not MkldnnModule",
                        "cannot pickle 'MkldnnModule' object",
                    },
                )
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(mkldnn.m.is_available, protocol=protocol)
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
                self.assertEqual(
                    message,
                    "Can't pickle <function is_available at 0x...>: "
                    "it's not the same object as "
                    "torch_rs.backends.mkldnn.is_available",
                )

    def test_value_and_identity_are_stable_across_threads(self):
        mkldnn = self.mkldnn
        function = mkldnn.is_available
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                current_function = torch.backends.mkldnn.is_available
                results[index] = (
                    value,
                    type(value) is bool,
                    value is torch._C._has_mkldnn,
                    current_function == function,
                    current_function.__self__ is mkldnn,
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
        self.assertEqual(results, [(False, True, True, True, True)] * worker_count)

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        mkldnn = self.mkldnn
        namespace = mkldnn.__dict__
        old_method = mkldnn.is_available

        try:
            reloaded = importlib.reload(mkldnn)

            self.assertIsNot(reloaded, mkldnn)
            self.assertIs(mkldnn.__dict__, namespace)
            self.assertIs(torch.backends.mkldnn, mkldnn)
            self.assertIs(sys.modules[mkldnn.__name__], reloaded)
            self.assertIs(reloaded.m, mkldnn)
            self.assertIs(type(reloaded.is_available), types.MethodType)
            self.assertIs(reloaded.is_available.__self__, reloaded)
            self.assertIs(reloaded.is_available(), False)
            self.assertIs(reloaded.is_available(), torch._C._has_mkldnn)

            self.assertIs(type(mkldnn.is_available), types.FunctionType)
            self.assertIsNot(mkldnn.is_available, old_method)
            self.assertIs(mkldnn.is_available(), False)
            self.assertIs(mkldnn.is_available(), torch._C._has_mkldnn)
            self.assertIs(copy.copy(mkldnn.is_available), mkldnn.is_available)
            self.assertIs(copy.deepcopy(mkldnn.is_available), mkldnn.is_available)
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(mkldnn.is_available)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                "Can't pickle <function is_available at 0x...>: "
                "it's not the same object as "
                "torch_rs.backends.mkldnn.is_available",
            )

            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                pickle.dumps(old_method)
        finally:
            fresh_mkldnn_module()

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = self.mkldnn.is_available
        cases = (
            (
                lambda: function(None),
                "MkldnnModule.is_available() takes 1 positional argument "
                "but 2 were given",
            ),
            (
                lambda: function(None, None),
                "MkldnnModule.is_available() takes 1 positional argument "
                "but 3 were given",
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

    def test_flags_tensors_operators_and_execution_remain_unsupported(self):
        mkldnn = self.mkldnn
        for name in UNSUPPORTED_MKLDNN_NAMES:
            with self.subTest(name=name):
                self.assertFalse(hasattr(mkldnn, name))
                with self.assertRaises(ImportError):
                    exec(f"from torch_rs.backends.mkldnn import {name}", {})

        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertFalse(hasattr(torch._C, "_get_mkldnn_enabled"))
        self.assertFalse(hasattr(torch._C, "_set_mkldnn_enabled"))
        self.assertFalse(hasattr(torch._C, "_get_mkldnn_deterministic"))
        self.assertFalse(hasattr(torch._C, "_set_mkldnn_deterministic"))
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)
        self.assertIs(torch.tensor([1.0]).layout, torch.strided)

    def test_import_and_call_are_probe_free_without_execution_claims(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"dnnl", "mkl", "mkl_service", "mkldnn", "numpy", "scipy", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    ATEN_CPU_CAPABILITY="avx512",
    DNNL_VERBOSE="1",
    MKL_NUM_THREADS="64",
    USE_MKLDNN="1",
)

import torch_rs as torch
from torch_rs.backends import mkldnn
from torch_rs.backends.mkldnn import is_available
from torch_rs._C import _has_mkldnn

assert torch.backends.mkldnn is mkldnn
assert is_available == mkldnn.is_available
assert is_available.__self__ is mkldnn
assert is_available() is _has_mkldnn is False
assert torch.has_mkl is torch._C.has_mkl is False
assert torch.has_lapack is torch._C.has_lapack is False
assert not hasattr(torch, "has_mkldnn")
assert not hasattr(torch, "_has_mkldnn")
assert not hasattr(mkldnn, "flags")
assert not hasattr(mkldnn, "enabled")
assert torch.tensor([1.0]).is_mkldnn is False
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
