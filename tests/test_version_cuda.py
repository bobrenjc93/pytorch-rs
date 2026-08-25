import importlib
import os
import subprocess
import sys
import types
import typing
import unittest
from unittest import mock

import torch_rs as torch


SUPPORTED_EXPORTS = ["__version__", "cuda"]


class VersionCudaTests(unittest.TestCase):
    def test_cpu_build_metadata_uses_package_and_native_sources_of_truth(self):
        module = torch.version
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
                    self.assertIs(module._C, torch._C)
                    self.assertIs(module._C._has_cuda, False)
                    self.assertIs(torch.backends.cuda.is_built(), False)
                    self.assertEqual(module.__version__, torch.__version__)
                    self.assertIs(type(module.__version__), str)
                    self.assertIs(module.cuda, None)

        self.assertFalse(hasattr(torch, "_has_cuda"))
        self.assertNotIn("_has_cuda", torch.__all__)
        self.assertNotIn("_has_cuda", torch._C.__all__)

    def test_module_identity_metadata_and_supported_scope(self):
        module = importlib.import_module("torch_rs.version")

        self.assertIs(torch.version, module)
        self.assertIs(sys.modules["torch_rs.version"], module)
        self.assertIs(type(module), types.ModuleType)
        self.assertEqual(module.__name__, "torch_rs.version")
        self.assertEqual(module.__package__, "torch_rs")
        self.assertEqual(module.__spec__.name, "torch_rs.version")
        self.assertIsNone(module.__doc__)
        self.assertEqual(module.__all__, SUPPORTED_EXPORTS)
        self.assertEqual(module.__annotations__, {"cuda": typing.Optional[str]})
        self.assertEqual(
            {name for name in vars(module) if not name.startswith("_")},
            {"Optional", "cuda"},
        )

        for name in ("debug", "git_version", "hip", "rocm", "xpu"):
            with self.subTest(unsupported_name=name):
                self.assertFalse(hasattr(module, name))

    def test_direct_imports_and_wildcards_are_canonical(self):
        module = torch.version

        parent_import = {}
        version_import = {}
        child_wildcard = {}
        parent_wildcard = {}
        exec("from torch_rs import version", parent_import)
        exec(
            "from torch_rs.version import __version__, cuda",
            version_import,
        )
        exec("from torch_rs.version import *", child_wildcard)
        exec("from torch_rs import *", parent_wildcard)

        self.assertIs(parent_import["version"], module)
        self.assertIs(version_import["__version__"], module.__version__)
        self.assertIs(version_import["cuda"], module.cuda)
        self.assertEqual(
            {
                name
                for name in child_wildcard
                if name != "__builtins__"
            },
            set(SUPPORTED_EXPORTS),
        )
        for name in SUPPORTED_EXPORTS:
            self.assertIs(child_wildcard[name], getattr(module, name))

        self.assertNotIn("version", torch.__all__)
        self.assertNotIn("version", parent_wildcard)

    def test_reload_preserves_canonical_module_and_refreshes_metadata(self):
        module = torch.version
        namespace = module.__dict__
        old_exports = module.__all__

        module.__version__ = "stale"
        module.cuda = "stale"
        reloaded = importlib.reload(module)

        self.assertIs(reloaded, module)
        self.assertIs(module.__dict__, namespace)
        self.assertIs(torch.version, module)
        self.assertIs(sys.modules[module.__name__], module)
        self.assertIsNot(module.__all__, old_exports)
        self.assertEqual(module.__all__, SUPPORTED_EXPORTS)
        self.assertEqual(module.__version__, torch.__version__)
        self.assertIs(module.cuda, None)

        package_reloaded = importlib.reload(torch)
        self.assertIs(package_reloaded, torch)
        self.assertIs(torch.version, module)
        self.assertIs(sys.modules[module.__name__], module)
        self.assertEqual(module.__version__, torch.__version__)
        self.assertIs(module.cuda, None)

    def test_import_does_not_probe_drivers_or_import_external_runtimes(self):
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
)
import torch_rs as torch
from torch_rs import version
from torch_rs.version import __version__, cuda

wildcard = {}
exec("from torch_rs.version import *", wildcard)
assert torch.version is version
assert version._C is torch._C
assert version.__version__ == __version__ == torch.__version__
assert version.cuda is cuda is None
assert version._C._has_cuda is False
assert wildcard["__version__"] is version.__version__
assert wildcard["cuda"] is None
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
