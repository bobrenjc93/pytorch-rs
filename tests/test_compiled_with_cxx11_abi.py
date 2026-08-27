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


FUNCTION_DOC = "Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"


class CompiledWithCxx11AbiTests(unittest.TestCase):
    def test_reports_exact_false_native_build_capability_without_probes(self):
        function = torch.compiled_with_cxx11_abi
        environments = (
            {},
            {"CXX": "/not/a/compiler"},
            {"CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1"},
            {
                "GLIBCXX_USE_CXX11_ABI": "1",
                "TORCH_CXX11_ABI": "1",
                "USE_CXX11_ABI": "1",
            },
        )

        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._GLIBCXX_USE_CXX11_ABI)

        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_signature_documentation_and_identity_match_pytorch_shape(self):
        function = torch.compiled_with_cxx11_abi

        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(inspect.get_annotations(function), {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "compiled_with_cxx11_abi")
        self.assertEqual(function.__qualname__, "compiled_with_cxx11_abi")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), torch)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_direct_imports_and_private_native_placement(self):
        native = importlib.import_module("torch_rs._C")
        package_import = {}
        native_import = {}
        package_wildcard = {}
        native_wildcard = {}

        exec(
            "from torch_rs import compiled_with_cxx11_abi",
            package_import,
        )
        exec(
            "from torch_rs._C import _GLIBCXX_USE_CXX11_ABI",
            native_import,
        )
        exec("from torch_rs import *", package_wildcard)
        exec("from torch_rs._C import *", native_wildcard)

        self.assertIs(native, torch._C)
        self.assertIs(
            package_import["compiled_with_cxx11_abi"],
            torch.compiled_with_cxx11_abi,
        )
        self.assertIs(native_import["_GLIBCXX_USE_CXX11_ABI"], False)
        self.assertIn("compiled_with_cxx11_abi", vars(torch))
        self.assertNotIn("compiled_with_cxx11_abi", vars(native))
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", vars(torch))
        self.assertIn("_GLIBCXX_USE_CXX11_ABI", vars(native))
        self.assertNotIn("compiled_with_cxx11_abi", torch.__all__)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", torch.__all__)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", native.__all__)
        self.assertNotIn("compiled_with_cxx11_abi", package_wildcard)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", package_wildcard)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", native_wildcard)

    def test_public_query_ignores_native_flag_mutation_and_deletion(self):
        function = torch.compiled_with_cxx11_abi
        native = torch._C
        original = native._GLIBCXX_USE_CXX11_ABI

        try:
            for replacement in (None, True, 1, "cxx11", object()):
                with self.subTest(replacement=replacement):
                    native._GLIBCXX_USE_CXX11_ABI = replacement
                    self.assertIs(function(), False)

            del native._GLIBCXX_USE_CXX11_ABI
            self.assertFalse(hasattr(native, "_GLIBCXX_USE_CXX11_ABI"))
            self.assertIs(function(), False)
        finally:
            native._GLIBCXX_USE_CXX11_ABI = original

        self.assertIs(native._GLIBCXX_USE_CXX11_ABI, False)
        self.assertIs(function(), False)

    def test_copying_and_pickling_are_canonical(self):
        function = torch.compiled_with_cxx11_abi

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(copy.copy(torch._C._GLIBCXX_USE_CXX11_ABI), False)
        self.assertIs(copy.deepcopy(torch._C._GLIBCXX_USE_CXX11_ABI), False)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIn(b"compiled_with_cxx11_abi", payload)
                self.assertIs(pickle.loads(payload), function)
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(
                            torch._C._GLIBCXX_USE_CXX11_ABI,
                            protocol=protocol,
                        )
                    ),
                    False,
                )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiled_with_cxx11_abi
        cases = (
            (
                lambda: function(None),
                "compiled_with_cxx11_abi() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "compiled_with_cxx11_abi() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "compiled_with_cxx11_abi() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "compiled_with_cxx11_abi() got an unexpected keyword argument 'enabled'",
            ),
        )

        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_value_and_identity_are_stable_across_threads(self):
        function = torch.compiled_with_cxx11_abi
        barrier = threading.Barrier(16)
        results = [None] * 16
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                value = function()
                results[index] = (
                    value,
                    type(value) is bool,
                    value is torch._C._GLIBCXX_USE_CXX11_ABI,
                    function is torch.compiled_with_cxx11_abi,
                )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(16)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [(False, True, True, True)] * 16)

    def test_package_and_native_reloads_preserve_the_contract(self):
        package = torch
        native = torch._C
        namespace = package.__dict__
        old_function = package.compiled_with_cxx11_abi

        reloaded = importlib.reload(package)
        function = package.compiled_with_cxx11_abi

        self.assertIs(reloaded, package)
        self.assertIs(package.__dict__, namespace)
        self.assertIs(package._C, native)
        self.assertIsNot(function, old_function)
        self.assertIs(function(), native._GLIBCXX_USE_CXX11_ABI)
        self.assertIs(function(), False)
        self.assertNotIn("compiled_with_cxx11_abi", package.__all__)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertIn("compiled_with_cxx11_abi", message)
        self.assertIn("not the same object", message)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(package.compiled_with_cxx11_abi, function)
        self.assertIs(native._GLIBCXX_USE_CXX11_ABI, False)
        self.assertIs(function(), native._GLIBCXX_USE_CXX11_ABI)

    def test_import_is_probe_free_and_does_not_import_pytorch(self):
        script = r'''
import importlib
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cppimport", "mkl", "numpy", "pybind11", "scipy", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CXX="/not/a/compiler",
    CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1",
    GLIBCXX_USE_CXX11_ABI="1",
    TORCH_CXX11_ABI="1",
    USE_CXX11_ABI="1",
)

import torch_rs as torch
from torch_rs import compiled_with_cxx11_abi
from torch_rs._C import _GLIBCXX_USE_CXX11_ABI

assert compiled_with_cxx11_abi is torch.compiled_with_cxx11_abi
assert compiled_with_cxx11_abi() is _GLIBCXX_USE_CXX11_ABI is False
assert not hasattr(torch, "_GLIBCXX_USE_CXX11_ABI")
torch._C._GLIBCXX_USE_CXX11_ABI = None
assert compiled_with_cxx11_abi() is False
del torch._C._GLIBCXX_USE_CXX11_ABI
assert compiled_with_cxx11_abi() is False
torch._C._GLIBCXX_USE_CXX11_ABI = False
package_wildcard = {}
native_wildcard = {}
exec("from torch_rs import *", package_wildcard)
exec("from torch_rs._C import *", native_wildcard)
assert "compiled_with_cxx11_abi" not in package_wildcard
assert "_GLIBCXX_USE_CXX11_ABI" not in package_wildcard
assert "_GLIBCXX_USE_CXX11_ABI" not in native_wildcard
old_function = compiled_with_cxx11_abi
assert importlib.reload(torch) is torch
assert torch.compiled_with_cxx11_abi is not old_function
assert torch.compiled_with_cxx11_abi() is False
assert importlib.reload(torch._C) is torch._C
assert torch.compiled_with_cxx11_abi() is torch._C._GLIBCXX_USE_CXX11_ABI is False
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
