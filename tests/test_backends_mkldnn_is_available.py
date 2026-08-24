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
        mkldnn = torch.backends.mkldnn
        method = mkldnn.is_available
        function = mkldnn.m.is_available
        self.assertEqual(method.__code__.co_names, ("is_available",))
        self.assertEqual(function.__code__.co_names, ("torch", "_C", "_has_mkldnn"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_MKLDNN": "1"},
            {"MKLDNN_VERBOSE": "1"},
            {
                "DNNL_VERBOSE": "all",
                "MKLDNN_VERBOSE": "1",
                "OMP_NUM_THREADS": "64",
                "USE_MKLDNN": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = method()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_mkldnn)
                    self.assertIs(function(), torch._C._has_mkldnn)

        self.assertFalse(hasattr(torch, "_has_mkldnn"))
        self.assertNotIn("_has_mkldnn", torch.__all__)
        self.assertNotIn("_has_mkldnn", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        method = mkldnn.is_available
        second_method = mkldnn.is_available
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
        self.assertIsNot(method, second_method)
        self.assertEqual(method, second_method)
        self.assertIs(method.__self__, mkldnn)
        self.assertIs(method.__func__, type(mkldnn).is_available)
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
        self.assertEqual(method.__code__.co_varnames, ("self",))

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

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mkldnn = importlib.import_module("torch_rs.backends.mkldnn")
        method = mkldnn.is_available

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
        for value in (method, mkldnn):
            with self.subTest(value=type(value).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "^cannot pickle 'MkldnnModule' object$",
                ):
                    copy.deepcopy(value)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                expected = (
                    "module() argument 'name' must be str, not MkldnnModule"
                    if protocol < 2
                    else "cannot pickle 'MkldnnModule' object"
                )
                with self.assertRaises(TypeError) as raised:
                    pickle.dumps(method, protocol=protocol)
                self.assertEqual(str(raised.exception), expected)
                self.assertEqual(raised.exception.args, (expected,))

                class_method = type(mkldnn).is_available
                self.assertIs(
                    pickle.loads(pickle.dumps(class_method, protocol=protocol)),
                    class_method,
                )

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(mkldnn.m.is_available)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_available at 0x...>: "
            "it's not the same object as torch_rs.backends.mkldnn.is_available",
        )

    def test_reload_matches_pytorch_module_replacement_behavior(self):
        backends = torch.backends
        mkldnn = backends.mkldnn
        old_method = mkldnn.is_available
        old_class_method = old_method.__func__
        old_underlying = mkldnn.m
        namespace = mkldnn.__dict__

        try:
            reloaded = importlib.reload(mkldnn)

            self.assertIsNot(reloaded, mkldnn)
            self.assertIs(mkldnn.__dict__, namespace)
            self.assertIs(backends.mkldnn, mkldnn)
            self.assertIs(sys.modules[mkldnn.__name__], reloaded)
            self.assertIs(reloaded.m, mkldnn)
            self.assertIs(mkldnn.m, old_underlying)

            parent_function = mkldnn.is_available
            reloaded_method = reloaded.is_available
            self.assertIs(type(parent_function), types.FunctionType)
            self.assertIs(type(reloaded_method), types.MethodType)
            self.assertIs(reloaded_method.__self__, reloaded)
            self.assertIs(reloaded_method.__func__, type(reloaded).is_available)
            self.assertIsNot(reloaded_method.__func__, old_class_method)
            self.assertIs(parent_function, mkldnn.__dict__["is_available"])
            self.assertIs(old_method(), False)
            self.assertIs(parent_function(), False)
            self.assertIs(reloaded_method(), False)

            copied = copy.copy(reloaded_method)
            self.assertIsNot(copied, reloaded_method)
            self.assertEqual(copied, reloaded_method)
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                copy.deepcopy(reloaded_method)
            with self.assertRaisesRegex(
                TypeError,
                "^cannot pickle 'MkldnnModule' object$",
            ):
                pickle.dumps(reloaded_method)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(parent_function)
            with self.assertRaises(pickle.PicklingError):
                pickle.dumps(old_class_method)
        finally:
            fresh_mkldnn_module()

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        method = torch.backends.mkldnn.is_available
        cases = (
            (
                lambda: method(None),
                "MkldnnModule.is_available() takes 1 positional argument but "
                "2 were given",
            ),
            (
                lambda: method(None, None),
                "MkldnnModule.is_available() takes 1 positional argument but "
                "3 were given",
            ),
            (
                lambda: method(enabled=True),
                "MkldnnModule.is_available() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: method(None, enabled=True),
                "MkldnnModule.is_available() got an unexpected keyword argument "
                "'enabled'",
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
            "_mkldnn",
            "_verbose",
        ):
            with self.subTest(native_name=name):
                self.assertFalse(hasattr(torch._C, name))

        tensor = torch.tensor([1.0])
        self.assertIs(tensor.is_mkldnn, False)
        self.assertFalse(hasattr(torch.Tensor, "to_mkldnn"))
        self.assertFalse(hasattr(torch, "_mkldnn"))
        self.assertFalse(hasattr(torch, "mkldnn_convolution"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'mkldnn' is not supported; "
            r"only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="mkldnn")

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
    DNNL_VERBOSE="all",
    MKLDNN_VERBOSE="1",
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
assert mkldnn.m.is_available.__code__.co_names == ("torch", "_C", "_has_mkldnn")
assert is_available() is torch._C._has_mkldnn is False
assert mkldnn.m.is_available() is torch._C._has_mkldnn
assert not hasattr(torch, "_has_mkldnn")
assert not hasattr(torch.Tensor, "to_mkldnn")
assert not hasattr(mkldnn, "flags")
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
