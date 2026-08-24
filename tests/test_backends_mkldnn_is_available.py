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


FUNCTION_DOC = "Return whether PyTorch is built with MKL-DNN support."


def fresh_mkldnn_module():
    module_name = "torch_rs.backends.mkldnn"
    sys.modules.pop(module_name, None)
    if hasattr(torch.backends, "mkldnn"):
        del torch.backends.mkldnn
    module = importlib.import_module(module_name)
    torch.backends.mkldnn = module
    return module


class MkldnnIsAvailableTests(unittest.TestCase):
    def test_returns_exact_false_private_build_flag_without_runtime_probes(self):
        mkldnn = torch.backends.mkldnn
        function = mkldnn.is_available
        native_function = mkldnn.m.is_available

        self.assertEqual(function.__code__.co_names, ("is_available",))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())
        self.assertEqual(
            native_function.__code__.co_names,
            ("torch", "_C", "_has_mkldnn"),
        )
        self.assertEqual(native_function.__code__.co_freevars, ())
        self.assertEqual(native_function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_MKLDNN": "1"},
            {"DNNL_VERBOSE": "1", "MKLDNN_VERBOSE": "1"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "DNNL_VERBOSE": "2",
                "MKLDNN_VERBOSE": "2",
                "ONEDNN_VERBOSE": "2",
                "USE_MKLDNN": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, native_function())
                    self.assertIs(result, torch._C._has_mkldnn)

        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertNotIn("_has_mkldnn", torch.__all__)
        self.assertNotIn("_has_mkldnn", torch._C.__all__)

    def test_signature_documentation_and_module_proxy_match_pytorch_2_13(self):
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        function = mkldnn.is_available
        native_function = mkldnn.m.is_available

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
        self.assertIs(function.__self__, mkldnn)
        self.assertIs(
            function.__func__,
            type(mkldnn).__dict__["is_available"],
        )
        self.assertIsNot(function, mkldnn.is_available)
        self.assertEqual(function, mkldnn.is_available)
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

        self.assertIs(type(native_function), types.FunctionType)
        self.assertEqual(str(inspect.signature(native_function)), "()")
        self.assertEqual(inspect.get_annotations(native_function), {})
        self.assertEqual(native_function.__name__, "is_available")
        self.assertEqual(native_function.__qualname__, "is_available")
        self.assertEqual(native_function.__module__, "torch_rs.backends.mkldnn")
        self.assertIs(inspect.getmodule(native_function), mkldnn)
        self.assertEqual(native_function.__doc__, FUNCTION_DOC)
        self.assertIsNone(native_function.__defaults__)
        self.assertIsNone(native_function.__kwdefaults__)
        self.assertEqual(native_function.__dict__, {})
        self.assertFalse(hasattr(native_function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        function = mkldnn.is_available
        native_function = mkldnn.m.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mkldnn, mkldnn)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {"cuda", "cudnn", "mkl", "mkldnn", "nnpack", "openmp"},
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
        imported_function = function_import["is_available"]
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mkldnn"], mkldnn)
        self.assertIs(parent_wildcard["mkldnn"], mkldnn)
        self.assertIsNot(imported_function, function)
        self.assertEqual(imported_function, function)
        self.assertIs(imported_function.__self__, mkldnn)
        self.assertIs(imported_function.__func__, function.__func__)
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

        copied = copy.copy(function)
        self.assertIsNot(copied, function)
        self.assertEqual(copied, function)
        self.assertIs(copied.__self__, mkldnn)
        self.assertIs(copied.__func__, function.__func__)
        with self.assertRaisesRegex(
            TypeError,
            "^cannot pickle 'MkldnnModule' object$",
        ):
            copy.deepcopy(function)
        for copier in (copy.copy, copy.deepcopy):
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                copier(mkldnn)

        self.assertIs(copy.copy(native_function), native_function)
        self.assertIs(copy.deepcopy(native_function), native_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol, callable="bound method"):
                expected_message = (
                    "module() argument 'name' must be str, not MkldnnModule"
                    if protocol < 2
                    else "cannot pickle 'MkldnnModule' object"
                )
                with self.assertRaises(TypeError) as raised:
                    pickle.dumps(function, protocol=protocol)
                self.assertEqual(str(raised.exception), expected_message)
                self.assertEqual(raised.exception.args, (expected_message,))

            with self.subTest(protocol=protocol, callable="native function"):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(native_function, protocol=protocol)
                message = re.sub(
                    r"0x[0-9a-fA-F]+",
                    "0x...",
                    str(raised.exception),
                )
                self.assertEqual(
                    message,
                    "Can't pickle <function is_available at 0x...>: "
                    "it's not the same object as "
                    "torch_rs.backends.mkldnn.is_available",
                )

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        backends = torch.backends
        mkldnn = backends.mkldnn
        inner = mkldnn.m
        old_method = mkldnn.is_available
        old_native_function = inner.is_available
        namespace = mkldnn.__dict__

        try:
            reloaded = importlib.reload(mkldnn)
            new_method = reloaded.is_available
            reloaded_native_function = mkldnn.is_available

            self.assertIsNot(reloaded, mkldnn)
            self.assertIs(mkldnn.__dict__, namespace)
            self.assertIs(backends.mkldnn, mkldnn)
            self.assertIs(sys.modules[mkldnn.__name__], reloaded)
            self.assertIs(reloaded.m, mkldnn)
            self.assertIs(mkldnn.m, inner)
            self.assertIs(type(new_method), types.MethodType)
            self.assertIs(new_method.__self__, reloaded)
            self.assertIs(type(reloaded_native_function), types.FunctionType)
            self.assertIs(reloaded.m.is_available, reloaded_native_function)
            self.assertIs(inner.is_available, old_native_function)
            self.assertIs(old_method.__self__, mkldnn)

            self.assertIs(old_method(), False)
            self.assertIs(reloaded_native_function(), False)
            self.assertIs(new_method(), False)
            copied = copy.copy(new_method)
            self.assertIsNot(copied, new_method)
            self.assertEqual(copied, new_method)
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                copy.deepcopy(new_method)
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                pickle.dumps(new_method)

            for stale_function in (
                old_native_function,
                reloaded_native_function,
            ):
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(stale_function)
                message = re.sub(
                    r"0x[0-9a-fA-F]+",
                    "0x...",
                    str(raised.exception),
                )
                self.assertEqual(
                    message,
                    "Can't pickle <function is_available at 0x...>: "
                    "it's not the same object as "
                    "torch_rs.backends.mkldnn.is_available",
                )
        finally:
            fresh_mkldnn_module()

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.mkldnn.is_available
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

    def test_configuration_tensors_and_execution_remain_unsupported(self):
        mkldnn = torch.backends.mkldnn
        for name in (
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
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(mkldnn, name))

        for name in (
            "_get_mkldnn_enabled",
            "_set_mkldnn_enabled",
            "_get_mkldnn_deterministic",
            "_set_mkldnn_deterministic",
            "_get_onednn_allow_tf32",
            "_set_onednn_allow_tf32",
            "_has_mkldnn_acl",
            "_verbose",
        ):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch._C, name))

        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertIs(tensor.is_mkldnn, False)
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"dnnl", "mkldnn", "numpy", "onednn", "scipy", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    ATEN_CPU_CAPABILITY="avx512",
    DNNL_VERBOSE="2",
    MKLDNN_VERBOSE="2",
    ONEDNN_VERBOSE="2",
    USE_MKLDNN="1",
)
import torch_rs as torch
from torch_rs.backends import mkldnn
from torch_rs.backends.mkldnn import is_available

assert torch.backends.mkldnn is mkldnn
assert is_available == mkldnn.is_available
assert is_available.__self__ is mkldnn
assert is_available.__func__ is type(mkldnn).__dict__["is_available"]
assert mkldnn.m.is_available.__code__.co_names == (
    "torch",
    "_C",
    "_has_mkldnn",
)
assert is_available() is mkldnn.m.is_available() is torch._C._has_mkldnn is False
assert not hasattr(torch, "_has_mkldnn")
assert not hasattr(torch.Tensor, "to_mkldnn")
assert not hasattr(torch, "_mkldnn")
assert not hasattr(mkldnn, "flags")
assert not hasattr(mkldnn, "enabled")
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
