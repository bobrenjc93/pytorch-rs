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
    def test_reports_the_exact_false_native_build_fact(self):
        function = torch.compiled_with_cxx11_abi
        flag_name = "_GLIBCXX_USE_CXX11_ABI"
        environments = (
            {},
            {"CXX": "g++", "CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1"},
            {
                "CXX": "clang++",
                "CXXFLAGS": "-D_GLIBCXX_USE_CXX11_ABI=0",
                "TORCH_CXX_FLAGS": "-D_GLIBCXX_USE_CXX11_ABI=1",
            },
        )

        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    native = getattr(torch._C, flag_name)
                    self.assertIs(type(result), bool)
                    self.assertIs(type(native), bool)
                    self.assertIs(result, False)
                    self.assertIs(native, False)
                    self.assertIs(result, native)

        with mock.patch.object(torch._C, flag_name, True):
            self.assertIs(function(), True)
        self.assertIs(function(), False)

    def test_function_metadata_matches_pytorch_2_13(self):
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
        self.assertEqual(function.__annotations__, {"return": bool})
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            function.__code__.co_names,
            ("_C", "_GLIBCXX_USE_CXX11_ABI"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

    def test_direct_imports_and_private_wildcard_placement(self):
        package = importlib.import_module("torch_rs")
        native = importlib.import_module("torch_rs._C")
        function = torch.compiled_with_cxx11_abi
        flag = torch._C._GLIBCXX_USE_CXX11_ABI

        self.assertIs(package, torch)
        self.assertIs(native, torch._C)
        self.assertIn("compiled_with_cxx11_abi", vars(torch))
        self.assertNotIn("compiled_with_cxx11_abi", vars(native))
        self.assertIn("_GLIBCXX_USE_CXX11_ABI", vars(native))
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", vars(torch))
        self.assertEqual(torch.__all__.count("compiled_with_cxx11_abi"), 0)
        self.assertEqual(torch.__all__.count("_GLIBCXX_USE_CXX11_ABI"), 0)
        self.assertEqual(native.__all__.count("_GLIBCXX_USE_CXX11_ABI"), 0)

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
        self.assertIs(native_direct["_GLIBCXX_USE_CXX11_ABI"], flag)
        self.assertNotIn("compiled_with_cxx11_abi", package_wildcard)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", package_wildcard)
        self.assertNotIn("_GLIBCXX_USE_CXX11_ABI", native_wildcard)

    def test_copying_and_pickling_preserve_canonical_objects(self):
        function = torch.compiled_with_cxx11_abi
        flag = torch._C._GLIBCXX_USE_CXX11_ABI

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(copy.copy(flag), flag)
        self.assertIs(copy.deepcopy(flag), flag)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                function_payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"compiled_with_cxx11_abi", function_payload)
                self.assertIs(pickle.loads(function_payload), function)
                self.assertIs(pickle.loads(pickle.dumps(flag, protocol)), flag)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.compiled_with_cxx11_abi
        cases = (
            (
                lambda: function(None),
                "compiled_with_cxx11_abi() takes 0 positional arguments but 1 "
                "was given",
            ),
            (
                lambda: function(None, None),
                "compiled_with_cxx11_abi() takes 0 positional arguments but 2 "
                "were given",
            ),
            (
                lambda: function(enabled=True),
                "compiled_with_cxx11_abi() got an unexpected keyword argument "
                "'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "compiled_with_cxx11_abi() got an unexpected keyword argument "
                "'enabled'",
            ),
        )

        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

        self.assertIs(function(*(), **{}), False)

    def test_threads_observe_one_process_wide_build_fact(self):
        function = torch.compiled_with_cxx11_abi
        worker_count = 16
        ready = threading.Barrier(worker_count + 1)
        release = threading.Barrier(worker_count + 1)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                ready.wait(timeout=10)
                release.wait(timeout=10)
                results[index] = (
                    function(),
                    function is torch.compiled_with_cxx11_abi,
                    torch._C._GLIBCXX_USE_CXX11_ABI,
                )
            except BaseException as error:
                errors.append(error)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()

        ready.wait(timeout=10)
        self.assertIs(function(), False)
        release.wait(timeout=10)
        for thread in threads:
            thread.join(timeout=10)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(results, [(False, True, False)] * worker_count)

    def test_package_and_native_reloads_preserve_the_contract(self):
        package = torch
        native = torch._C
        old_function = package.compiled_with_cxx11_abi
        flag = native._GLIBCXX_USE_CXX11_ABI
        package_namespace = package.__dict__

        reloaded_package = importlib.reload(package)
        new_function = package.compiled_with_cxx11_abi

        self.assertIs(reloaded_package, package)
        self.assertIs(package.__dict__, package_namespace)
        self.assertIs(package._C, native)
        self.assertIs(native._GLIBCXX_USE_CXX11_ABI, flag)
        self.assertIsNot(new_function, old_function)
        self.assertIs(new_function(), flag)
        self.assertIs(copy.copy(new_function), new_function)
        self.assertIs(copy.deepcopy(new_function), new_function)
        self.assertIs(pickle.loads(pickle.dumps(new_function)), new_function)
        self.assertNotIn("compiled_with_cxx11_abi", package.__all__)

        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function compiled_with_cxx11_abi at 0x...>: "
            "it's not the same object as "
            "torch_rs.compiled_with_cxx11_abi",
        )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(native._GLIBCXX_USE_CXX11_ABI, flag)
        self.assertIs(package.compiled_with_cxx11_abi, new_function)
        self.assertIs(new_function(), False)

    def test_import_and_calls_do_not_probe_compilers_or_external_runtimes(self):
        script = r'''
import importlib
import os
import pickle
import subprocess
import sys

class RejectExternalRuntimeImport:
    blocked = {
        "cupy",
        "nvidia",
        "numpy",
        "pybind11",
        "setuptools",
        "torch",
    }

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

def reject_probe(*args, **kwargs):
    raise AssertionError((args, kwargs))

sys.meta_path.insert(0, RejectExternalRuntimeImport())
subprocess.Popen = reject_probe
subprocess.call = reject_probe
subprocess.check_call = reject_probe
subprocess.check_output = reject_probe
subprocess.run = reject_probe
os.system = reject_probe
os.environ.update(
    CXX="missing-cxx-compiler",
    CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1",
    TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1",
)

import torch_rs as torch
from torch_rs import compiled_with_cxx11_abi
from torch_rs._C import _GLIBCXX_USE_CXX11_ABI

assert compiled_with_cxx11_abi is torch.compiled_with_cxx11_abi
assert compiled_with_cxx11_abi.__code__.co_names == (
    "_C",
    "_GLIBCXX_USE_CXX11_ABI",
)
assert compiled_with_cxx11_abi() is _GLIBCXX_USE_CXX11_ABI is False
assert not hasattr(torch, "_GLIBCXX_USE_CXX11_ABI")
assert not hasattr(torch._C, "compiled_with_cxx11_abi")

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
assert pickle.loads(pickle.dumps(torch.compiled_with_cxx11_abi)) is (
    torch.compiled_with_cxx11_abi
)
assert importlib.reload(torch._C) is torch._C
assert torch.compiled_with_cxx11_abi() is False
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
