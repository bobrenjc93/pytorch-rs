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
SUPPORTED_BACKENDS = {
    "cpu",
    "cuda",
    "cudnn",
    "kleidiai",
    "mha",
    "mkl",
    "nnpack",
    "openmp",
}


class KleidiAIAvailabilityTests(unittest.TestCase):
    def test_returns_exact_false_private_build_flag_without_runtime_probes(self):
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
            {"ATEN_CPU_CAPABILITY": "default"},
            {
                "ATEN_CPU_CAPABILITY": "avx512",
                "KLEIDIAI_ENABLED": "1",
                "OMP_NUM_THREADS": "64",
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
                    self.assertEqual(
                        torch.backends.cpu.get_cpu_capability(),
                        "DEFAULT",
                    )

        self.assertIn("_has_kleidiai", vars(torch._C))
        self.assertFalse(hasattr(torch, "_has_kleidiai"))
        self.assertNotIn("_has_kleidiai", torch.__all__)
        self.assertNotIn("_has_kleidiai", torch._C.__all__)

        explicit_native_import = {}
        native_wildcard = {}
        exec(
            "from torch_rs._C import _has_kleidiai",
            explicit_native_import,
        )
        exec("from torch_rs._C import *", native_wildcard)
        self.assertIs(explicit_native_import["_has_kleidiai"], False)
        self.assertNotIn("_has_kleidiai", native_wildcard)

    def test_query_reads_the_live_private_native_attribute(self):
        function = torch.backends.kleidiai.is_available
        original = torch._C._has_kleidiai
        marker = object()
        try:
            for value in (True, 0, None, marker):
                with self.subTest(value=value):
                    torch._C._has_kleidiai = value
                    self.assertIs(function(), value)

            del torch._C._has_kleidiai
            with self.assertRaises(AttributeError):
                function()
        finally:
            torch._C._has_kleidiai = original

        self.assertIs(function(), False)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        backends = importlib.import_module("torch_rs.backends")
        kleidiai = importlib.import_module("torch_rs.backends.kleidiai")
        function = kleidiai.is_available

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.kleidiai, kleidiai)
        self.assertIs(sys.modules["torch_rs.backends.kleidiai"], kleidiai)
        self.assertIs(type(kleidiai), types.ModuleType)
        self.assertIsNone(kleidiai.__doc__)
        self.assertFalse(hasattr(kleidiai, "__all__"))
        self.assertEqual(kleidiai.__annotations__, {})
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

        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            SUPPORTED_BACKENDS,
        )

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import kleidiai", backend_import)
        exec(
            "from torch_rs.backends.kleidiai import is_available",
            function_import,
        )
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.kleidiai import *", child_wildcard)
        exec("from torch_rs import *", top_level_wildcard)

        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["kleidiai"], kleidiai)
        self.assertIs(function_import["is_available"], function)
        self.assertEqual(
            {name for name in parent_wildcard if not name.startswith("__")},
            SUPPORTED_BACKENDS,
        )
        self.assertIs(parent_wildcard["kleidiai"], kleidiai)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"is_available", "torch"},
        )
        self.assertIs(child_wildcard["is_available"], function)
        self.assertIs(child_wildcard["torch"], torch)
        self.assertNotIn("backends", torch.__all__)
        self.assertNotIn("kleidiai", torch.__all__)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("kleidiai", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.kleidiai", payload)
                self.assertIs(pickle.loads(payload), function)

        for copier in (copy.copy, copy.deepcopy):
            with self.subTest(copier=copier.__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    "^cannot pickle 'module' object$",
                ):
                    copier(kleidiai)

    def test_value_and_identity_are_stable_across_threads(self):
        kleidiai = torch.backends.kleidiai
        function = kleidiai.is_available
        worker_count = 16
        barrier = threading.Barrier(worker_count)
        results = [None] * worker_count
        errors = []

        def worker(index):
            try:
                barrier.wait(timeout=5)
                results[index] = (
                    function() is False,
                    type(function()) is bool,
                    function() is torch._C._has_kleidiai,
                    function is kleidiai.is_available,
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
        self.assertEqual(results, [(True, True, True, True)] * worker_count)

    def test_reload_replaces_function_and_preserves_canonical_module(self):
        backends = torch.backends
        kleidiai = backends.kleidiai
        old_function = kleidiai.is_available
        namespace = kleidiai.__dict__

        reloaded = importlib.reload(kleidiai)

        self.assertIs(reloaded, kleidiai)
        self.assertIs(kleidiai.__dict__, namespace)
        self.assertIs(backends.kleidiai, kleidiai)
        self.assertIs(sys.modules[kleidiai.__name__], kleidiai)
        self.assertIsNot(kleidiai.is_available, old_function)
        self.assertIs(kleidiai.is_available(), False)
        self.assertIs(copy.copy(kleidiai.is_available), kleidiai.is_available)
        self.assertIs(copy.deepcopy(kleidiai.is_available), kleidiai.is_available)
        self.assertIs(
            pickle.loads(pickle.dumps(kleidiai.is_available)),
            kleidiai.is_available,
        )
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_available at 0x...>: "
            "it's not the same object as "
            "torch_rs.backends.kleidiai.is_available",
        )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.kleidiai.is_available
        cases = (
            (
                (None,),
                {},
                "is_available() takes 0 positional arguments but 1 was given",
            ),
            (
                (None, None),
                {},
                "is_available() takes 0 positional arguments but 2 were given",
            ),
            (
                (),
                {"enabled": True},
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
            (
                (None,),
                {"enabled": True},
                "is_available() got an unexpected keyword argument 'enabled'",
            ),
        )
        for args, kwargs, message in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(TypeError) as raised:
                    function(*args, **kwargs)
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_no_kleidiai_execution_or_optimized_cpu_dispatch_is_claimed(self):
        kleidiai = torch.backends.kleidiai
        self.assertEqual(
            {name for name in vars(kleidiai) if not name.startswith("_")},
            {"is_available", "torch"},
        )
        for name in ("linear", "matmul", "mm", "quantized", "version"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(kleidiai, name))
        self.assertEqual(torch.backends.cpu.get_cpu_capability(), "DEFAULT")

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import builtins
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cpuinfo", "kleidiai", "numpy", "psutil", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"feature-probe import was attempted: {fullname}")
        return None

original_open = builtins.open

def guarded_open(file, *args, **kwargs):
    path = os.fspath(file)
    if path == "/proc/cpuinfo" or path.startswith("/sys/devices/system/cpu/"):
        raise RuntimeError(f"CPU feature probe was attempted: {path}")
    return original_open(file, *args, **kwargs)

sys.meta_path.insert(0, RejectExternalRuntimeImport())
builtins.open = guarded_open
os.environ.update(
    ATEN_CPU_CAPABILITY="avx512",
    KLEIDIAI_ENABLED="1",
    OMP_NUM_THREADS="64",
    USE_KLEIDIAI="1",
)
import torch_rs as torch
from torch_rs.backends import kleidiai
from torch_rs.backends.kleidiai import is_available

assert torch.backends.kleidiai is kleidiai
assert kleidiai.is_available is is_available
assert is_available.__code__.co_names == ("torch", "_C", "_has_kleidiai")
assert is_available() is torch._C._has_kleidiai is False
assert torch.backends.cpu.get_cpu_capability() == "DEFAULT"
assert not hasattr(torch, "_has_kleidiai")
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
