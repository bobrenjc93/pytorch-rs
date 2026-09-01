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


FUNCTION_DOC = "Return whether PyTorch is built with KleidiAI support."


class KleidiAIAvailabilityTests(unittest.TestCase):
    def test_returns_exact_false_private_native_build_flag_without_probes(self):
        function = torch.backends.kleidiai.is_available
        self.assertEqual(
            function.__code__.co_names,
            ("torch", "_C", "_has_kleidiai"),
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"USE_KLEIDIAI": "1"},
            {"KLEIDIAI_PATH": "/not/a/kleidiai/install"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "KLEIDIAI_NUM_THREADS": "64",
                "USE_KLEIDIAI": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_kleidiai)

        self.assertFalse(hasattr(torch, "_has_kleidiai"))
        self.assertIn("_has_kleidiai", vars(torch._C))
        self.assertNotIn("_has_kleidiai", torch.__all__)
        self.assertNotIn("_has_kleidiai", torch._C.__all__)
        self.assertEqual(torch.backends.cpu.get_cpu_capability(), "DEFAULT")

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        kleidiai = importlib.import_module("torch_rs.backends.kleidiai")
        function = kleidiai.is_available

        self.assertIs(torch.backends.kleidiai, kleidiai)
        self.assertIs(sys.modules["torch_rs.backends.kleidiai"], kleidiai)
        self.assertIs(type(kleidiai), types.ModuleType)
        self.assertIsNone(kleidiai.__doc__)
        self.assertFalse(hasattr(kleidiai, "__all__"))
        self.assertEqual(
            {name for name in vars(kleidiai) if not name.startswith("_")},
            {"is_available", "torch"},
        )
        self.assertIs(kleidiai.torch, torch)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "()")
        self.assertEqual(inspect.get_annotations(function), {})
        self.assertEqual(function.__name__, "is_available")
        self.assertEqual(function.__qualname__, "is_available")
        self.assertEqual(function.__module__, "torch_rs.backends.kleidiai")
        self.assertIs(inspect.getmodule(function), kleidiai)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        kleidiai = importlib.import_module("torch_rs.backends.kleidiai")
        function = kleidiai.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.kleidiai, kleidiai)
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
        private_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        native_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import kleidiai", backend_import)
        exec(
            "from torch_rs.backends.kleidiai import is_available",
            function_import,
        )
        exec("from torch_rs._C import _has_kleidiai", private_import)
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.kleidiai import *", child_wildcard)
        exec("from torch_rs._C import *", native_wildcard)

        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["kleidiai"], kleidiai)
        self.assertIs(function_import["is_available"], function)
        self.assertIs(private_import["_has_kleidiai"], False)
        self.assertIs(parent_wildcard["kleidiai"], kleidiai)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"is_available", "torch"},
        )
        self.assertIs(child_wildcard["is_available"], function)
        self.assertIs(child_wildcard["torch"], torch)
        self.assertNotIn("_has_kleidiai", native_wildcard)

        self.assertNotIn("backends", torch.__all__)
        self.assertFalse(hasattr(torch, "kleidiai"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("kleidiai", top_level_wildcard)
        self.assertNotIn("_has_kleidiai", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(copy.copy(torch._C._has_kleidiai), False)
        self.assertIs(copy.deepcopy(torch._C._has_kleidiai), False)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.kleidiai", payload)
                self.assertIs(pickle.loads(payload), function)
                self.assertIs(
                    pickle.loads(
                        pickle.dumps(
                            torch._C._has_kleidiai,
                            protocol=protocol,
                        )
                    ),
                    False,
                )

    def test_value_and_identity_are_stable_across_threads(self):
        function = torch.backends.kleidiai.is_available
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
                    value is torch._C._has_kleidiai,
                    function is torch.backends.kleidiai.is_available,
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
        kleidiai = backends.kleidiai
        namespace = kleidiai.__dict__
        old_function = kleidiai.is_available

        reloaded = importlib.reload(kleidiai)
        function = kleidiai.is_available

        self.assertIs(reloaded, kleidiai)
        self.assertIs(kleidiai.__dict__, namespace)
        self.assertIs(backends.kleidiai, kleidiai)
        self.assertIs(sys.modules[kleidiai.__name__], kleidiai)
        self.assertIsNot(function, old_function)
        self.assertIs(function(), native._has_kleidiai)
        self.assertIs(function(), False)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        self.assertIs(pickle.loads(pickle.dumps(function)), function)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_available at 0x...>: "
            "it's not the same object as "
            "torch_rs.backends.kleidiai.is_available",
        )

        self.assertIs(importlib.reload(package), package)
        self.assertIs(package._C, native)
        self.assertIs(package.backends, backends)
        self.assertIs(backends.kleidiai, kleidiai)
        self.assertIs(kleidiai.is_available, function)
        self.assertIs(kleidiai.is_available(), False)

        self.assertIs(importlib.reload(native), native)
        self.assertIs(package._C, native)
        self.assertIs(native._has_kleidiai, False)
        self.assertIs(kleidiai.is_available(), native._has_kleidiai)

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.kleidiai.is_available
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
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_import_and_call_are_probe_free_without_execution_claims(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cpuinfo", "kai", "kleidiai", "numpy", "scipy", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    ATEN_CPU_CAPABILITY="avx512",
    KLEIDIAI_NUM_THREADS="64",
    KLEIDIAI_PATH="/not/a/kleidiai/install",
    USE_KLEIDIAI="1",
)

import torch_rs as torch
from torch_rs.backends import kleidiai
from torch_rs.backends.kleidiai import is_available
from torch_rs._C import _has_kleidiai

assert torch.backends.kleidiai is kleidiai
assert kleidiai.is_available is is_available
assert is_available.__code__.co_names == ("torch", "_C", "_has_kleidiai")
assert is_available() is _has_kleidiai is False
assert torch.backends.cpu.get_cpu_capability() == "DEFAULT"
assert not hasattr(torch, "_has_kleidiai")
assert not hasattr(torch, "kleidiai")
assert {name for name in vars(kleidiai) if not name.startswith("_")} == {
    "is_available",
    "torch",
}
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
