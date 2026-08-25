import importlib
import importlib.metadata
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_EXPORTS = {"__version__", "cuda"}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class VersionCudaReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.version differentials require pinned PyTorch 2.13.0"
            )

    def test_module_identity_values_and_supported_exports_match_pytorch_2_13(self):
        actual = importlib.import_module("torch_rs.version")
        expected = importlib.import_module("torch.version")

        self.assertIs(torch.version, actual)
        self.assertIs(reference_torch.version, expected)
        self.assertIs(sys.modules["torch_rs.version"], actual)
        self.assertIs(sys.modules["torch.version"], expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in SUPPORTED_EXPORTS],
        )

        self.assertIs(type(actual.__version__), type(expected.__version__))
        self.assertEqual(actual.__version__, torch.__version__)
        self.assertEqual(actual.__version__, importlib.metadata.version("torch-rs"))
        self.assertEqual(expected.__version__, reference_torch.__version__)
        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(actual.cuda, None)

    def test_direct_import_and_wildcard_behavior_matches_supported_subset(self):
        for package_name, root in (
            ("torch_rs", torch),
            ("torch", reference_torch),
        ):
            module = root.version
            package_import = {}
            attribute_import = {}
            wildcard_import = {}

            exec(f"from {package_name} import version", package_import)
            exec(
                f"from {package_name}.version import __version__, cuda",
                attribute_import,
            )
            exec(f"from {package_name}.version import *", wildcard_import)

            self.assertIs(package_import["version"], module)
            self.assertIs(attribute_import["__version__"], module.__version__)
            self.assertIs(attribute_import["cuda"], module.cuda)
            wildcard_names = set(wildcard_import) - {"__builtins__"}
            if package_name == "torch_rs":
                self.assertEqual(wildcard_names, SUPPORTED_EXPORTS)
            else:
                self.assertEqual(
                    wildcard_names & SUPPORTED_EXPORTS,
                    SUPPORTED_EXPORTS,
                )

            self.assertNotIn("version", root.__all__)
            top_level_wildcard = {}
            exec(f"from {package_name} import *", top_level_wildcard)
            self.assertNotIn("version", top_level_wildcard)

    def reload_contract(self, root):
        module = root.version
        namespace = module.__dict__
        original_version = module.__version__
        original_cuda = module.cuda
        old_exports = module.__all__
        marker = object()

        module.__version__ = "mutated"
        module.cuda = "mutated"
        module.__all__.append("mutated")
        module._reload_marker = marker

        try:
            reloaded = importlib.reload(module)
            return (
                reloaded is module,
                module.__dict__ is namespace,
                root.version is module,
                sys.modules[module.__name__] is module,
                module.__all__ is not old_exports,
                [name for name in module.__all__ if name in SUPPORTED_EXPORTS]
                == ["__version__", "cuda"],
                module.__version__ == original_version,
                module.cuda == original_cuda,
                module._reload_marker is marker,
            )
        finally:
            module.__dict__.pop("_reload_marker", None)
            importlib.reload(module)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_h100_reports_reference_cuda_build_version_and_cpu_only_torch_rs(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertIs(reference_torch._C._has_cuda, True)
        self.assertIs(type(reference_torch.version.cuda), str)
        self.assertRegex(reference_torch.version.cuda, r"^\d+\.\d+$")
        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(torch.version.cuda, None)
        self.assertFalse(hasattr(torch, "cuda"))


if __name__ == "__main__":
    unittest.main()
