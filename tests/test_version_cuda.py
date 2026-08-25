import importlib
import importlib.metadata
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch_rs as torch


SUPPORTED_EXPORTS = ("__version__", "cuda")


class VersionCudaTests(unittest.TestCase):
    def test_reports_native_cpu_build_and_package_version(self):
        version = torch.version
        environments = (
            {},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": "0"},
            {
                "CUDA_VISIBLE_DEVICES": "0",
                "NVIDIA_VISIBLE_DEVICES": "all",
                "PYTORCH_NVML_BASED_CUDA_CHECK": "1",
                "CUDA_HOME": "/opt/cuda",
            },
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    self.assertIs(torch._C._has_cuda, False)
                    self.assertIs(version.cuda, None)
                    self.assertEqual(version.__version__, torch.__version__)
                    self.assertIs(version.__version__, torch._C.__version__)

        self.assertIs(type(version.__version__), str)
        self.assertEqual(
            version.__version__,
            importlib.metadata.version("torch-rs"),
        )
        self.assertFalse(hasattr(torch, "_has_cuda"))
        self.assertNotIn("_has_cuda", torch.__all__)
        self.assertNotIn("_has_cuda", torch._C.__all__)

    def test_module_identity_metadata_and_supported_exports(self):
        version = importlib.import_module("torch_rs.version")

        self.assertIs(torch.version, version)
        self.assertIs(sys.modules["torch_rs.version"], version)
        self.assertIs(type(version), types.ModuleType)
        self.assertEqual(version.__name__, "torch_rs.version")
        self.assertEqual(version.__package__, "torch_rs")
        self.assertIsNone(version.__doc__)
        self.assertEqual(version.__all__, list(SUPPORTED_EXPORTS))
        self.assertEqual(version.__annotations__, {"cuda": str | None})
        self.assertEqual(
            {name for name in vars(version) if not name.startswith("_")},
            {"cuda"},
        )

    def test_direct_imports_and_wildcards_are_canonical(self):
        version = importlib.import_module("torch_rs.version")
        parent_import = {}
        module_import = {}
        direct_import = {}
        child_wildcard = {}
        top_level_wildcard = {}

        exec("from torch_rs import version", parent_import)
        exec("import torch_rs.version as version", module_import)
        exec("from torch_rs.version import __version__, cuda", direct_import)
        exec("from torch_rs.version import *", child_wildcard)
        exec("from torch_rs import *", top_level_wildcard)

        self.assertIs(parent_import["version"], version)
        self.assertIs(module_import["version"], version)
        self.assertIs(direct_import["__version__"], version.__version__)
        self.assertIs(direct_import["cuda"], version.cuda)
        self.assertEqual(
            {name for name in child_wildcard if name != "__builtins__"},
            set(SUPPORTED_EXPORTS),
        )
        for name in SUPPORTED_EXPORTS:
            self.assertIs(child_wildcard[name], getattr(version, name))

        self.assertNotIn("version", torch.__all__)
        self.assertNotIn("version", top_level_wildcard)
        self.assertNotIn("cuda", top_level_wildcard)

    def test_reload_preserves_canonical_module_and_restores_metadata(self):
        version = torch.version
        namespace = version.__dict__
        old_exports = version.__all__
        sentinel = object()
        version.__version__ = sentinel
        version.cuda = sentinel
        version.__all__.append("unsupported")
        version._reload_marker = sentinel

        try:
            reloaded = importlib.reload(version)

            self.assertIs(reloaded, version)
            self.assertIs(version.__dict__, namespace)
            self.assertIs(torch.version, version)
            self.assertIs(sys.modules[version.__name__], version)
            self.assertIsNot(version.__all__, old_exports)
            self.assertEqual(version.__all__, list(SUPPORTED_EXPORTS))
            self.assertIs(version.__version__, torch.__version__)
            self.assertIs(version.cuda, None)
            self.assertIs(version._reload_marker, sentinel)
        finally:
            version.__dict__.pop("_reload_marker", None)
            importlib.reload(version)

    def test_unimplemented_build_metadata_remains_absent(self):
        for name in ("debug", "git_version", "hip", "rocm", "xpu"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.version, name))

    def test_import_does_not_probe_drivers_or_import_pytorch(self):
        script = r'''
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"cuda", "cupy", "nvidia", "numpy", "pynvml", "torch"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    CUDA_VISIBLE_DEVICES="0",
    NVIDIA_VISIBLE_DEVICES="all",
    PYTORCH_NVML_BASED_CUDA_CHECK="1",
    CUDA_HOME="/opt/cuda",
)
import torch_rs as torch
from torch_rs import version
from torch_rs.version import __version__, cuda

wildcard = {}
exec("from torch_rs.version import *", wildcard)
assert torch.version is version
assert version.__version__ is __version__ is torch.__version__
assert version.cuda is cuda is None
assert torch._C._has_cuda is False
assert version.__all__ == ["__version__", "cuda"]
assert {name for name in wildcard if name != "__builtins__"} == {
    "__version__",
    "cuda",
}
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
