import contextlib
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
    def test_rust_build_reports_exact_false_without_runtime_probes(self):
        function = torch.compiled_with_cxx11_abi
        self.assertEqual(function.__code__.co_names, ())
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"_GLIBCXX_USE_CXX11_ABI": "1"},
            {"CXX": "g++", "CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1"},
            {
                "CC": "gcc",
                "CXX": "clang++",
                "CFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1",
                "CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._GLIBCXX_USE_CXX11_ABI)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        package = importlib.import_module("torch_rs")
        function = package.compiled_with_cxx11_abi

        self.assertIs(torch, package)
        self.assertIs(sys.modules["torch_rs"], package)
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

    def test_direct_imports_and_private_native_placement(self):
        package = importlib.import_module("torch_rs")
        native = importlib.import_module("torch_rs._C")
        function = package.compiled_with_cxx11_abi
        flag_name = "_GLIBCXX_USE_CXX11_ABI"

        self.assertIs(package, torch)
        self.assertIs(native, torch._C)
        self.assertIn("compiled_with_cxx11_abi", vars(package))
        self.assertNotIn("compiled_with_cxx11_abi", vars(native))
        self.assertNotIn(flag_name, vars(package))
        self.assertIn(flag_name, vars(native))
        self.assertIs(getattr(native, flag_name), False)
        self.assertNotIn("compiled_with_cxx11_abi", package.__all__)
        self.assertNotIn(flag_name, package.__all__)
        self.assertNotIn(flag_name, native.__all__)

        direct_function = {}
        direct_flag = {}
        package_wildcard = {}
        native_wildcard = {}
        exec(
            "from torch_rs import compiled_with_cxx11_abi",
            direct_function,
        )
        exec(
            "from torch_rs._C import _GLIBCXX_USE_CXX11_ABI",
            direct_flag,
        )
        exec("from torch_rs import *", package_wildcard)
        exec("from torch_rs._C import *", native_wildcard)

        self.assertIs(direct_function["compiled_with_cxx11_abi"], function)
        self.assertIs(direct_flag[flag_name], False)
        self.assertNotIn("compiled_with_cxx11_abi", package_wildcard)
        self.assertNotIn(flag_name, package_wildcard)
        self.assertNotIn(flag_name, native_wildcard)

    def test_copying_and_pickling_use_the_canonical_package_function(self):
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

    def test_false_build_metadata_is_stable_across_threads_and_grad_modes(self):
        function = torch.compiled_with_cxx11_abi
        worker_count = 8
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                context = torch.no_grad() if index % 2 else contextlib.nullcontext()
                with context:
                    expected_grad_state = torch.is_grad_enabled()
                    barrier.wait(timeout=10)
                    first = function()
                    native = torch._C._GLIBCXX_USE_CXX11_ABI
                    second = function()
                    results[index] = (
                        expected_grad_state,
                        first,
                        native,
                        second,
                        torch.is_grad_enabled(),
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
        for index, result in enumerate(results):
            expected_grad_state = index % 2 == 0
            self.assertEqual(
                result,
                (
                    expected_grad_state,
                    False,
                    False,
                    False,
                    expected_grad_state,
                ),
            )
            self.assertIs(result[1], False)
            self.assertIs(result[2], False)
            self.assertIs(result[3], False)

    def test_package_reload_replaces_the_wrapper_and_preserves_native_metadata(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        old_function = package.compiled_with_cxx11_abi
        package_namespace = package.__dict__
        native_namespace = native.__dict__

        reloaded = importlib.reload(package)
        new_function = package.compiled_with_cxx11_abi

        self.assertIs(reloaded, package)
        self.assertIs(package.__dict__, package_namespace)
        self.assertIs(torch, package)
        self.assertIs(package._C, native)
        self.assertIs(native.__dict__, native_namespace)
        self.assertIsNot(new_function, old_function)
        self.assertIs(old_function(), False)
        self.assertIs(new_function(), False)
        self.assertIs(native._GLIBCXX_USE_CXX11_ABI, False)
        self.assertNotIn("compiled_with_cxx11_abi", package.__all__)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", native.__all__)
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

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(package.compiled_with_cxx11_abi, new_function)
        self.assertIs(native._GLIBCXX_USE_CXX11_ABI, False)

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

    def test_import_and_reload_do_not_probe_toolchains_or_external_runtimes(self):
        script = r'''
import ctypes.util
import importlib
import os
import subprocess
import sys

class RejectExternalRuntimeImport:
    blocked = {"numpy", "setuptools", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

def reject_probe(*args, **kwargs):
    raise AssertionError(f"host toolchain probe was attempted: {args!r} {kwargs!r}")

sys.meta_path.insert(0, RejectExternalRuntimeImport())
ctypes.util.find_library = reject_probe
subprocess.call = reject_probe
subprocess.check_call = reject_probe
subprocess.check_output = reject_probe
subprocess.Popen = reject_probe
subprocess.run = reject_probe
os.environ.update(
    CC="gcc",
    CXX="g++",
    CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1",
    CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1",
    _GLIBCXX_USE_CXX11_ABI="1",
)

import torch_rs as torch
from torch_rs import compiled_with_cxx11_abi
from torch_rs._C import _GLIBCXX_USE_CXX11_ABI

assert torch.compiled_with_cxx11_abi is compiled_with_cxx11_abi
assert compiled_with_cxx11_abi() is False
assert _GLIBCXX_USE_CXX11_ABI is False
assert torch._C._GLIBCXX_USE_CXX11_ABI is False
assert not hasattr(torch, "_GLIBCXX_USE_CXX11_ABI")
assert "compiled_with_cxx11_abi" not in torch.__all__
assert "_GLIBCXX_USE_CXX11_ABI" not in torch._C.__all__

package_wildcard = {}
native_wildcard = {}
exec("from torch_rs import *", package_wildcard)
exec("from torch_rs._C import *", native_wildcard)
assert "compiled_with_cxx11_abi" not in package_wildcard
assert "_GLIBCXX_USE_CXX11_ABI" not in package_wildcard
assert "_GLIBCXX_USE_CXX11_ABI" not in native_wildcard

assert importlib.reload(torch) is torch
assert torch.compiled_with_cxx11_abi() is False
assert importlib.reload(torch._C) is torch._C
assert torch._C._GLIBCXX_USE_CXX11_ABI is False
assert not any(
    name == "torch" or name.startswith("torch.") or name == "numpy"
    or name.startswith("numpy.") or name == "setuptools"
    or name.startswith("setuptools.")
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
