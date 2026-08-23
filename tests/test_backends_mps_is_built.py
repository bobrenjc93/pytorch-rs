import copy
import importlib
import inspect
import os
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


FUNCTION_DOC = """Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run a machine with working MPS drivers
    and devices, we would be able to use it.
    """


class MpsIsBuiltTests(unittest.TestCase):
    def test_returns_exact_false_native_build_metadata_without_runtime_probes(self):
        function = torch.backends.mps.is_built
        self.assertEqual(function.__code__.co_names, ("torch", "_C", "_has_mps"))
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(function.__code__.co_cellvars, ())

        environments = (
            {},
            {"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
            {
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.0",
                "PYTORCH_MPS_PREFER_METAL": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    result = function()
                    self.assertIs(type(result), bool)
                    self.assertIs(result, False)
                    self.assertIs(result, torch._C._has_mps)

        self.assertFalse(hasattr(torch, "_has_mps"))
        self.assertNotIn("_has_mps", torch.__all__)
        self.assertNotIn("_has_mps", torch._C.__all__)

    def test_signature_documentation_and_module_identity_match_pytorch_2_13(self):
        mps = importlib.import_module("torch_rs.backends.mps")
        function = mps.is_built

        self.assertIs(torch.backends.mps, mps)
        self.assertIs(sys.modules["torch_rs.backends.mps"], mps)
        self.assertIsNone(mps.__doc__)
        self.assertEqual(mps.__all__, ["is_built"])
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(str(inspect.signature(function)), "() -> bool")
        self.assertEqual(inspect.get_annotations(function), {"return": bool})
        self.assertEqual(typing.get_type_hints(function), {"return": bool})
        self.assertEqual(function.__name__, "is_built")
        self.assertEqual(function.__qualname__, "is_built")
        self.assertEqual(function.__module__, "torch_rs.backends.mps")
        self.assertIs(inspect.getmodule(function), mps)
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertIsNone(function.__defaults__)
        self.assertIsNone(function.__kwdefaults__)
        self.assertEqual(function.__dict__, {})
        self.assertFalse(hasattr(function, "__text_signature__"))

    def test_imports_wildcards_copying_and_pickling_are_canonical(self):
        backends = importlib.import_module("torch_rs.backends")
        mps = importlib.import_module("torch_rs.backends.mps")
        function = mps.is_built

        self.assertIs(torch.backends, backends)
        self.assertIs(backends.mps, mps)
        self.assertIs(mps.torch, torch)
        self.assertEqual(
            {name for name in vars(backends) if not name.startswith("_")},
            {"cuda", "mkl", "mps", "nnpack", "openmp"},
        )

        package_import = {}
        backend_import = {}
        function_import = {}
        parent_wildcard = {}
        child_wildcard = {}
        exec("from torch_rs import backends", package_import)
        exec("from torch_rs.backends import mps", backend_import)
        exec("from torch_rs.backends.mps import is_built", function_import)
        exec("from torch_rs.backends import *", parent_wildcard)
        exec("from torch_rs.backends.mps import *", child_wildcard)
        self.assertIs(package_import["backends"], backends)
        self.assertIs(backend_import["mps"], mps)
        self.assertIs(function_import["is_built"], function)
        self.assertIs(parent_wildcard["mps"], mps)
        self.assertEqual(
            {name for name in child_wildcard if not name.startswith("__")},
            {"is_built"},
        )
        self.assertIs(child_wildcard["is_built"], function)

        self.assertNotIn("backends", torch.__all__)
        self.assertFalse(hasattr(torch, "mps"))
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("backends", top_level_wildcard)
        self.assertNotIn("mps", top_level_wildcard)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                payload = pickle.dumps(function, protocol=protocol)
                self.assertIn(b"torch_rs.backends.mps", payload)
                self.assertIs(pickle.loads(payload), function)

    def test_reload_replaces_function_and_preserves_canonical_module(self):
        backends = torch.backends
        mps = backends.mps
        old_function = mps.is_built
        namespace = mps.__dict__

        reloaded = importlib.reload(mps)

        self.assertIs(reloaded, mps)
        self.assertIs(mps.__dict__, namespace)
        self.assertIs(backends.mps, mps)
        self.assertIs(sys.modules[mps.__name__], mps)
        self.assertIsNot(mps.is_built, old_function)
        self.assertIs(mps.is_built(), False)
        self.assertIs(copy.copy(mps.is_built), mps.is_built)
        self.assertIs(copy.deepcopy(mps.is_built), mps.is_built)
        self.assertIs(pickle.loads(pickle.dumps(mps.is_built)), mps.is_built)
        with self.assertRaises(pickle.PicklingError) as raised:
            pickle.dumps(old_function)
        message = re.sub(r"0x[0-9a-fA-F]+", "0x...", str(raised.exception))
        self.assertEqual(
            message,
            "Can't pickle <function is_built at 0x...>: "
            "it's not the same object as torch_rs.backends.mps.is_built",
        )

    def test_rejects_arguments_with_pytorch_2_13_errors(self):
        function = torch.backends.mps.is_built
        cases = (
            (
                lambda: function(None),
                "is_built() takes 0 positional arguments but 1 was given",
            ),
            (
                lambda: function(None, None),
                "is_built() takes 0 positional arguments but 2 were given",
            ),
            (
                lambda: function(enabled=True),
                "is_built() got an unexpected keyword argument 'enabled'",
            ),
            (
                lambda: function(None, enabled=True),
                "is_built() got an unexpected keyword argument 'enabled'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)
                self.assertEqual(raised.exception.args, (message,))

    def test_mps_runtime_and_execution_surface_remains_unsupported(self):
        mps_backend = torch.backends.mps

        for name in (
            "get_core_count",
            "get_name",
            "is_available",
            "is_macos13_or_newer",
            "is_macos_or_newer",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(mps_backend, name))

        self.assertFalse(hasattr(torch, "mps"))
        self.assertNotIn("torch_rs.mps", sys.modules)
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("torch_rs.mps")
        self.assertFalse(hasattr(torch.Tensor, "mps"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        for specification in ("mps", "mps:0"):
            with self.subTest(specification=specification, action="device"):
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.device(specification)
            with self.subTest(specification=specification, action="tensor"):
                with self.assertRaisesRegex(
                    RuntimeError, r"only 'cpu' is implemented"
                ):
                    torch.tensor([1.0], device=specification)

    def test_importing_and_calling_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"Metal", "numpy", "objc", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    PYTORCH_ENABLE_MPS_FALLBACK="1",
    PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.0",
    PYTORCH_MPS_PREFER_METAL="1",
)
import torch_rs as torch
from torch_rs.backends import mps
from torch_rs.backends.mps import is_built

assert torch.backends.mps is mps
assert mps.is_built is is_built
assert is_built.__code__.co_names == ("torch", "_C", "_has_mps")
assert is_built.__annotations__ == {"return": bool}
assert is_built() is torch._C._has_mps is False
assert not hasattr(torch, "_has_mps")
assert not hasattr(torch, "mps")
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
