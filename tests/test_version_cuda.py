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
    def test_cpu_build_metadata_uses_package_and_native_sources_of_truth(self):
        version = torch.version

        self.assertEqual(version.__version__, torch.__version__)
        self.assertEqual(
            version.__version__,
            importlib.metadata.version("torch-rs"),
        )
        self.assertIs(type(version.__version__), str)
        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(version.cuda, None)
        self.assertIs(version.hip, None)
        self.assertIs(version.rocm, None)
        self.assertIs(version.xpu, None)

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
                    self.assertIs(torch._C._has_cuda, False)
                    self.assertIs(version.cuda, None)
                    self.assertIs(version.hip, None)
                    self.assertIs(version.rocm, None)
                    self.assertIs(version.xpu, None)

    def test_module_identity_and_supported_metadata_match_pytorch_shape(self):
        version = importlib.import_module("torch_rs.version")

        self.assertIs(torch.version, version)
        self.assertIs(sys.modules["torch_rs.version"], version)
        self.assertIs(type(version), types.ModuleType)
        self.assertEqual(version.__name__, "torch_rs.version")
        self.assertEqual(version.__package__, "torch_rs")
        self.assertIsNone(version.__doc__)
        self.assertEqual(
            version.__all__,
            ["__version__", "cuda", "hip", "rocm", "xpu"],
        )
        self.assertEqual(
            version.__annotations__,
            {
                "cuda": typing.Optional[str],
                "hip": typing.Optional[str],
                "rocm": typing.Optional[str],
                "xpu": typing.Optional[str],
            },
        )
        self.assertIs(version.Optional, typing.Optional)
        self.assertEqual(
            {name for name in vars(version) if not name.startswith("_")},
            {"Optional", "cuda", "hip", "rocm", "xpu"},
        )
        for name in ("debug", "git_version"):
            with self.subTest(unsupported=name):
                self.assertFalse(hasattr(version, name))

    def test_direct_and_wildcard_imports_use_the_canonical_module(self):
        version = torch.version

        package_import = {}
        module_import = {}
        direct_import = {}
        child_wildcard = {}
        top_level_wildcard = {}
        exec("from torch_rs import version", package_import)
        exec("import torch_rs.version as version", module_import)
        exec(
            "from torch_rs.version import __version__, cuda, hip, rocm, xpu",
            direct_import,
        )
        exec("from torch_rs.version import *", child_wildcard)
        exec("from torch_rs import *", top_level_wildcard)

        self.assertIs(package_import["version"], version)
        self.assertIs(module_import["version"], version)
        self.assertIs(direct_import["__version__"], version.__version__)
        self.assertIs(direct_import["cuda"], version.cuda)
        self.assertIs(direct_import["hip"], version.hip)
        self.assertIs(direct_import["rocm"], version.rocm)
        self.assertIs(direct_import["xpu"], version.xpu)
        self.assertEqual(
            {name for name in child_wildcard if name != "__builtins__"},
            {"__version__", "cuda", "hip", "rocm", "xpu"},
        )
        self.assertIs(child_wildcard["__version__"], version.__version__)
        self.assertIs(child_wildcard["cuda"], version.cuda)
        self.assertIs(child_wildcard["hip"], version.hip)
        self.assertIs(child_wildcard["rocm"], version.rocm)
        self.assertIs(child_wildcard["xpu"], version.xpu)
        self.assertNotIn("version", torch.__all__)
        self.assertNotIn("__version__", torch.__all__)
        self.assertNotIn("cuda", torch.__all__)
        self.assertNotIn("hip", torch.__all__)
        self.assertNotIn("rocm", torch.__all__)
        self.assertNotIn("xpu", torch.__all__)
        self.assertNotIn("version", top_level_wildcard)
        self.assertNotIn("__version__", top_level_wildcard)
        self.assertNotIn("cuda", top_level_wildcard)
        self.assertNotIn("hip", top_level_wildcard)
        self.assertNotIn("rocm", top_level_wildcard)
        self.assertNotIn("xpu", top_level_wildcard)

    def test_reload_preserves_canonical_identity_and_restores_metadata(self):
        version = torch.version
        namespace = version.__dict__
        old_all = version.__all__
        expected_package_version = torch.__version__
        version.__version__ = "stale"
        version.cuda = "stale"
        version.hip = "stale"
        version.rocm = "stale"
        version.xpu = "stale"

        reloaded = importlib.reload(version)

        self.assertIs(reloaded, version)
        self.assertIs(version.__dict__, namespace)
        self.assertIs(torch.version, version)
        self.assertIs(sys.modules["torch_rs.version"], version)
        self.assertIsNot(version.__all__, old_all)
        self.assertEqual(
            version.__all__,
            ["__version__", "cuda", "hip", "rocm", "xpu"],
        )
        self.assertIs(version.__version__, expected_package_version)
        self.assertIs(version.cuda, None)
        self.assertIs(version.hip, None)
        self.assertIs(version.rocm, None)
        self.assertIs(version.xpu, None)

    def test_import_does_not_import_pytorch_or_probe_external_cuda_runtimes(self):
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
import torch_rs.version as version
from torch_rs import version as package_version
from torch_rs.version import __version__, cuda, hip, rocm, xpu

assert version is package_version is torch.version
assert version is sys.modules["torch_rs.version"]
assert __version__ == version.__version__ == torch.__version__
assert torch._C._has_cuda is False
assert cuda is version.cuda is None
assert hip is version.hip is None
assert rocm is version.rocm is None
assert xpu is version.xpu is None
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
