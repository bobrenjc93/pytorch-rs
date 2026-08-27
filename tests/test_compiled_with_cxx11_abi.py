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


FLAG_NAME = "_GLIBCXX_USE_CXX11_ABI"
FUNCTION_DOC = "Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"


class CompiledWithCxx11AbiTests(unittest.TestCase):
    def test_rust_build_reports_exact_false_without_runtime_probes(self):
        function = torch.compiled_with_cxx11_abi
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"_GLIBCXX_USE_CXX11_ABI": "1"},
            {"CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1"},
            {
                "CC": "g++",
                "CXX": "g++",
                "CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1",
                "LD_LIBRARY_PATH": "/pretend/libstdc++",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    native = getattr(torch._C, FLAG_NAME)
                self.assertIs(type(result), bool)
                self.assertIs(result, False)
                self.assertIs(type(native), bool)
                self.assertIs(native, False)
                self.assertIs(result, native)

    def test_metadata_matches_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.compiled_with_cxx11_abi

        self.assertIs(package, torch)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(inspect.get_annotations(function), {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "compiled_with_cxx11_abi")
        self.assertEqual(function.__qualname__, "compiled_with_cxx11_abi")
        self.assertEqual(function.__module__, "torch_rs")
        self.assertIs(inspect.getmodule(function), package)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_direct_imports_and_private_export_placement(self):
        package = importlib.import_module("torch_rs")
        native = importlib.import_module("torch_rs._C")
        function = torch.compiled_with_cxx11_abi

        self.assertIs(package, torch)
        self.assertIs(native, torch._C)
        self.assertFalse(hasattr(torch, FLAG_NAME))
        self.assertFalse(hasattr(torch._C, "compiled_with_cxx11_abi"))
        self.assertNotIn(FLAG_NAME, torch.__all__)
        self.assertNotIn(FLAG_NAME, torch._C.__all__)
        self.assertNotIn("compiled_with_cxx11_abi", torch.__all__)
        self.assertNotIn("compiled_with_cxx11_abi", torch._C.__all__)

        package_direct = {}
        native_direct = {}
        package_wildcard = {}
        native_wildcard = {}
        exec(
            "from torch_rs import compiled_with_cxx11_abi",
            package_direct,
        )
        exec(
            "from torch_rs._C import _GLIBCXX_USE_CXX11_ABI",
            native_direct,
        )
        exec("from torch_rs import *", package_wildcard)
        exec("from torch_rs._C import *", native_wildcard)

        self.assertIs(package_direct["compiled_with_cxx11_abi"], function)
        self.assertIs(native_direct[FLAG_NAME], False)
        self.assertNotIn("compiled_with_cxx11_abi", package_wildcard)
        self.assertNotIn(FLAG_NAME, package_wildcard)
        self.assertNotIn(FLAG_NAME, native_wildcard)

    def test_copying_and_pickling_use_the_canonical_package_function(self):
        function = torch.compiled_with_cxx11_abi
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs", payload)
                self.assertIs(pickle.loads(payload), function)

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

    def test_build_metadata_is_stable_across_threads(self):
        function = torch.compiled_with_cxx11_abi
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=10)
                results[index] = (
                    function(),
                    getattr(torch._C, FLAG_NAME),
                    function(),
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
        for result in results:
            self.assertEqual(result, (False, False, False))
            self.assertTrue(all(type(value) is bool for value in result))

    def test_native_and_package_reloads_preserve_the_surface(self):
        package = torch
        native = torch._C
        package_namespace = package.__dict__
        old_function = package.compiled_with_cxx11_abi

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(package.compiled_with_cxx11_abi, old_function)
        self.assertIs(getattr(native, FLAG_NAME), False)
        self.assertIs(old_function(), False)

        self.assertIs(importlib.reload(package), package)
        new_function = package.compiled_with_cxx11_abi
        self.assertIs(package.__dict__, package_namespace)
        self.assertIs(package._C, native)
        self.assertIsNot(new_function, old_function)
        self.assertIs(getattr(native, FLAG_NAME), False)
        self.assertIs(new_function(), False)
        self.assertNotIn("compiled_with_cxx11_abi", package.__all__)
        self.assertIs(copy.copy(new_function), new_function)
        self.assertIs(copy.deepcopy(new_function), new_function)
        self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function compiled_with_cxx11_abi at 0x...>: "
            "it's not the same object as torch_rs.compiled_with_cxx11_abi",
        )

    def test_import_and_query_do_not_probe_or_import_external_toolchains(self):
        script = r'''
import os
import sys

class RejectExternalToolchainImport:
    blocked = {"cppyy", "ninja", "numpy", "setuptools", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external toolchain import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalToolchainImport())
os.environ.update(
    CC="g++",
    CXX="g++",
    CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1",
    LD_LIBRARY_PATH="/pretend/libstdc++",
    _GLIBCXX_USE_CXX11_ABI="1",
)
import torch_rs as torch
from torch_rs import compiled_with_cxx11_abi
from torch_rs._C import _GLIBCXX_USE_CXX11_ABI

assert torch.compiled_with_cxx11_abi is compiled_with_cxx11_abi
assert compiled_with_cxx11_abi() is False
assert torch._C._GLIBCXX_USE_CXX11_ABI is _GLIBCXX_USE_CXX11_ABI is False
assert not hasattr(torch, "_GLIBCXX_USE_CXX11_ABI")
assert "compiled_with_cxx11_abi" not in torch.__all__
assert "_GLIBCXX_USE_CXX11_ABI" not in torch._C.__all__
assert not any(
    name.split(".", 1)[0] in RejectExternalToolchainImport.blocked
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
