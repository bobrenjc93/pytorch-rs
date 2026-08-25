import importlib
import importlib.metadata
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

import torch_rs as torch


class VersionCudaTests(unittest.TestCase):
    def test_cpu_build_metadata_uses_package_version_and_cuda_none(self):
        module = torch.version

        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(module.cuda, None)
        self.assertIs(type(module.__version__), str)
        self.assertEqual(module.__version__, torch.__version__)
        self.assertEqual(module.__version__, importlib.metadata.version("torch-rs"))

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
                    self.assertIs(module.cuda, None)
                    self.assertEqual(module.__version__, torch.__version__)

    def test_module_identity_direct_imports_and_supported_wildcard(self):
        module = importlib.import_module("torch_rs.version")

        self.assertIs(torch.version, module)
        self.assertIs(sys.modules["torch_rs.version"], module)
        self.assertIs(type(module), types.ModuleType)
        self.assertEqual(module.__name__, "torch_rs.version")
        self.assertEqual(module.__package__, "torch_rs")
        self.assertIsNone(module.__doc__)
        self.assertEqual(module.__all__, ["__version__", "cuda"])

        package_import = {}
        attribute_import = {}
        wildcard_import = {}
        exec("from torch_rs import version", package_import)
        exec(
            "from torch_rs.version import __version__, cuda",
            attribute_import,
        )
        exec("from torch_rs.version import *", wildcard_import)

        self.assertIs(package_import["version"], module)
        self.assertIs(attribute_import["__version__"], module.__version__)
        self.assertIs(attribute_import["cuda"], module.cuda)
        self.assertEqual(
            set(wildcard_import) - {"__builtins__"},
            {"__version__", "cuda"},
        )
        self.assertIs(wildcard_import["__version__"], module.__version__)
        self.assertIs(wildcard_import["cuda"], module.cuda)

        self.assertNotIn("version", torch.__all__)
        top_level_wildcard = {}
        exec("from torch_rs import *", top_level_wildcard)
        self.assertNotIn("version", top_level_wildcard)

    def test_reload_preserves_identity_and_resets_generated_metadata(self):
        module = torch.version
        namespace = module.__dict__
        old_exports = module.__all__
        marker = object()

        module.__version__ = "mutated"
        module.cuda = "mutated"
        module.__all__.append("mutated")
        module._reload_marker = marker

        try:
            reloaded = importlib.reload(module)

            self.assertIs(reloaded, module)
            self.assertIs(module.__dict__, namespace)
            self.assertIs(torch.version, module)
            self.assertIs(sys.modules[module.__name__], module)
            self.assertIsNot(module.__all__, old_exports)
            self.assertEqual(module.__all__, ["__version__", "cuda"])
            self.assertEqual(module.__version__, torch.__version__)
            self.assertIs(module.cuda, None)
            self.assertIs(module._reload_marker, marker)
        finally:
            module.__dict__.pop("_reload_marker", None)
            importlib.reload(module)

    def test_import_does_not_import_pytorch_or_probe_external_runtimes(self):
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
from torch_rs import version
from torch_rs.version import __version__, cuda

assert torch.version is version
assert version.__version__ == __version__ == torch.__version__
assert torch._C._has_cuda is False
assert version.cuda is cuda is None
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
