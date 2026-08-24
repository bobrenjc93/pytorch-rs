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
    def test_returns_exact_false_native_build_metadata_without_runtime_probes(self):
        module = torch.backends.mkldnn
        method = module.is_available
        function = module.m.is_available
        self.assertEqual(method.__code__.co_names, ("is_available",))
        self.assertEqual(function.__code__.co_names, ("torch", "_C", "_has_mkldnn"))
        self.assertEqual(method.__code__.co_freevars, ())
        self.assertEqual(method.__code__.co_cellvars, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_MKLDNN": "1"},
            {"DNNL_VERBOSE": "1", "MKLDNN_VERBOSE": "1"},
            {"ONEDNN_VERBOSE": "all", "OMP_NUM_THREADS": "64"},
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = method()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, function())
                    self.assertIs(result, torch._C._has_mkldnn)

        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertNotIn("_has_mkldnn", torch.__all__)
        self.assertNotIn("_has_mkldnn", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        method = mkldnn.is_available
        function = mkldnn.m.is_available

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

        self.assertIs(type(method), types.MethodType)
        self.assertEqual(str(inspect.signature(method)), "()")
        self.assertEqual(inspect.get_annotations(method), {})
        self.assertEqual(method.__name__, "is_available")
        self.assertEqual(method.__qualname__, "MkldnnModule.is_available")
        self.assertEqual(method.__module__, "torch_rs.backends.mkldnn")
        self.assertIs(inspect.getmodule(method), mkldnn)
        self.assertIsNone(method.__doc__)
        self.assertIsNone(method.__defaults__)
        self.assertIsNone(method.__kwdefaults__)
        self.assertEqual(method.__dict__, {})
        self.assertFalse(hasattr(method, "__text_signature__"))
        self.assertIs(method.__self__, mkldnn)
        self.assertIs(method.__func__, type(mkldnn).is_available)
        self.assertIsNot(method, mkldnn.is_available)
        self.assertEqual(method, mkldnn.is_available)

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.backends.mkldnn")
        self.assertIs(inspect.getmodule(function), mkldnn)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_copying_and_pickling_match_the_proxy_contract(self):
        backends = importlib.import_module("torch_rs.backends")
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        method = mkldnn.is_available
        function = mkldnn.m.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mkldnn, mkldnn)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {"cuda", "cudnn", "mkl", "mkldnn", "nnpack", "openmp"},
        )

        package_import = {}
        backend_import = {}
        method_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import mkldnn", backend_import)
        exec("from torch_rs.backends.mkldnn import is_available", method_import)
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.mkldnn import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mkldnn"], mkldnn)
        self.assertEqual(method_import["is_available"], method)
        self.assertIs(method_import["is_available"].__self__, mkldnn)
        self.assertIs(method_import["is_available"].__func__, method.__func__)
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

        copied = copy.copy(method)
        self.assertIsNot(copied, method)
        self.assertEqual(copied, method)
        self.assertIs(copied.__self__, mkldnn)
        self.assertIs(copied.__func__, method.__func__)
        with self.assertRaisesRegex(
            TypeError,
            "^cannot pickle 'MkldnnModule' object$",
        ):
            copy.deepcopy(method)
        for copier in (copy.copy, copy.deepcopy):
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                copier(mkldnn)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                with self.assertRaises(TypeError):
                    pickle.dumps(method, protocol=protocol)
                with self.assertRaises(pickle.PicklingError) as raised:
                    pickle.dumps(function, protocol=protocol)
                message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
                self.assertEqual(
                    message,
                    "Can't pickle <function is_available at 0x...>: "
                    "it's not the same object as "
                    "torch_rs.backends.mkldnn.is_available",
                )
                class_function = type(mkldnn).is_available
                self.assertIs(
                    pickle.loads(pickle.dumps(class_function, protocol=protocol)),
                    class_function,
                )

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        backends = torch.backends
        mkldnn = backends.mkldnn
        inner = mkldnn.m
        old_method = mkldnn.is_available
        old_class_function = type(mkldnn).is_available
        namespace = mkldnn.__dict__

        try:
            reloaded = importlib.reload(mkldnn)

            self.assertIsNot(reloaded, mkldnn)
            self.assertIs(mkldnn.__dict__, namespace)
            self.assertIs(backends.mkldnn, mkldnn)
            self.assertIs(sys.modules[mkldnn.__name__], reloaded)
            self.assertIs(reloaded.m, mkldnn)
            self.assertIs(mkldnn.m, inner)
            self.assertIs(type(mkldnn.is_available), types.FunctionType)
            self.assertIs(mkldnn.is_available, namespace["is_available"])
            self.assertIs(type(reloaded.is_available), types.MethodType)
            self.assertIs(reloaded.is_available.__self__, reloaded)
            self.assertIs(reloaded.is_available.__func__, type(reloaded).is_available)
            self.assertIs(reloaded.is_available(), False)

            copied_old_method = copy.copy(old_method)
            self.assertIs(copied_old_method, mkldnn.is_available)
            copied_new_method = copy.copy(reloaded.is_available)
            self.assertEqual(copied_new_method, reloaded.is_available)
            self.assertIsNot(copied_new_method, reloaded.is_available)
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                copy.deepcopy(reloaded.is_available)
            with self.assertRaises(TypeError):
                pickle.dumps(reloaded.is_available)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(mkldnn.is_available)
            with self.assertRaises(pickle.PicklingError) as raised:
                pickle.dumps(old_class_function)
            message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
            self.assertEqual(
                message,
                "Can't pickle <function MkldnnModule.is_available at 0x...>: "
                "it's not the same object as "
                "torch_rs.backends.mkldnn.MkldnnModule.is_available",
            )
        finally:
            fresh_mkldnn_module()

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        method = torch.backends.mkldnn.is_available
        cases = (
            (
                lambda: method(None),
                "MkldnnModule.is_available() takes 1 positional argument "
                "but 2 were given",
            ),
            (
                lambda: method(None, None),
                "MkldnnModule.is_available() takes 1 positional argument "
                "but 3 were given",
            ),
            (
                lambda: method(enabled=True),
                "MkldnnModule.is_available() got an unexpected keyword "
                "argument 'enabled'",
            ),
            (
                lambda: method(None, enabled=True),
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

    def test_configuration_verbosity_and_execution_remain_unsupported(self):
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
            "_get_mkldnn_deterministic",
            "_get_mkldnn_enabled",
            "_get_onednn_allow_tf32",
            "_has_mkldnn_acl",
            "_set_mkldnn_deterministic",
            "_set_mkldnn_enabled",
            "_set_onednn_allow_tf32",
            "_verbose",
        ):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch._C, name))

        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertIs(torch.tensor([1.0]).is_mkldnn, False)

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"dnnl", "mkl", "mkldnn", "numpy", "onednn", "scipy", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    DNNL_VERBOSE="1",
    MKLDNN_VERBOSE="1",
    ONEDNN_VERBOSE="all",
    OMP_NUM_THREADS="64",
    USE_MKLDNN="1",
)
import torch_rs as torch
from torch_rs.backends import mkldnn
from torch_rs.backends.mkldnn import is_available

assert torch.backends.mkldnn is mkldnn
assert is_available == mkldnn.is_available
assert is_available.__self__ is mkldnn
assert is_available.__func__ is type(mkldnn).is_available
assert is_available.__code__.co_names == ("is_available",)
assert mkldnn.m.is_available.__code__.co_names == ("torch", "_C", "_has_mkldnn")
assert is_available() is torch._C._has_mkldnn is False
assert not hasattr(torch, "_has_mkldnn")
assert not hasattr(torch.Tensor, "to_mkldnn")
assert not hasattr(mkldnn, "flags")
assert not hasattr(mkldnn, "verbose")
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
