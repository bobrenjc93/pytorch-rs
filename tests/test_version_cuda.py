import importlib
import importlib.metadata
import os
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


class VersionCudaTests(unittest.TestCase):
    def test_cpu_build_metadata_uses_package_and_native_truth(self):
        version = torch.version

        self.assertIs(torch._C._has_cuda, False)
        self.assertFalse(hasattr(torch._C, "_cuda_version"))
        self.assertIs(version.__version__, torch.__version__)
        self.assertEqual(version.__version__, importlib.metadata.version("torch-rs"))

        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(version.cuda, None)

    def test_module_identity_direct_imports_and_wildcards_are_canonical(self):
        version = importlib.import_module("torch_rs.version")

        self.assertIs(type(version), types.ModuleType)
        self.assertIs(torch.version, version)
        self.assertIs(sys.modules["torch_rs.version"], version)
        self.assertEqual(version.__name__, "torch_rs.version")
        self.assertEqual(version.__package__, "torch_rs")
        self.assertIsNone(version.__doc__)
        self.assertEqual(version.__all__, ["__version__", "cuda"])
        self.assertEqual(
            typing.get_type_hints(version),
            {"cuda": typing.Optional[str]},
        )
        self.assertEqual(
            {name for name in vars(version) if not name.startswith("_")},
            {"Optional", "cuda"},
        )

        module_import = {}
        parent_import = {}
        value_imports = {}
        module_wildcard = {}
        package_wildcard = {}
        exec("import torch_rs.version as version", module_import)
        exec("from torch_rs import version", parent_import)
        exec(
            "from torch_rs.version import __version__, cuda",
            value_imports,
        )
        exec("from torch_rs.version import *", module_wildcard)
        exec("from torch_rs import *", package_wildcard)

        self.assertIs(module_import["version"], version)
        self.assertIs(parent_import["version"], version)
        self.assertIs(value_imports["__version__"], version.__version__)
        self.assertIs(value_imports["cuda"], version.cuda)
        self.assertEqual(
            {
                name
                for name in module_wildcard
                if name != "__builtins__"
            },
            {"__version__", "cuda"},
        )
        self.assertIs(module_wildcard["__version__"], version.__version__)
        self.assertIs(module_wildcard["cuda"], version.cuda)
        self.assertNotIn("version", torch.__all__)
        self.assertNotIn("version", package_wildcard)

    def test_reload_preserves_the_canonical_module(self):
        version = torch.version
        namespace = version.__dict__
        old_exports = version.__all__

        reloaded = importlib.reload(version)

        self.assertIs(reloaded, version)
        self.assertIs(version.__dict__, namespace)
        self.assertIs(torch.version, version)
        self.assertIs(sys.modules[version.__name__], version)
        self.assertIsNot(version.__all__, old_exports)
        self.assertEqual(version.__all__, ["__version__", "cuda"])
        self.assertIs(version.__version__, torch.__version__)
        self.assertIs(version.cuda, None)

    def test_import_does_not_probe_or_import_external_runtimes(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cupy", "nvidia", "numpy", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
)

import torch_rs as torch
import torch_rs.version as direct_version
from torch_rs import version
from torch_rs.version import __version__, cuda

wildcard = {}
exec("from torch_rs.version import *", wildcard)

assert torch.version is version is direct_version
assert version.__version__ is __version__ is torch.__version__
assert version.cuda is cuda is None
assert torch._C._has_cuda is False
assert not hasattr(torch._C, "_cuda_version")
assert set(version.__all__) == {"__version__", "cuda"}
assert {
    name for name in wildcard if name != "__builtins__"
} == {"__version__", "cuda"}
assert not hasattr(torch, "cuda")
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
