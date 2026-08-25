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


SUPPORTED_EXPORTS = ("__version__", "cuda")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class VersionCudaReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.version differentials require pinned PyTorch 2.13.0"
            )

    def test_module_identity_and_supported_metadata_match_pytorch_2_13(self):
        actual = importlib.import_module("torch_rs.version")
        expected = importlib.import_module("torch.version")

        self.assertIs(torch.version, actual)
        self.assertIs(reference_torch.version, expected)
        self.assertIs(sys.modules[actual.__name__], actual)
        self.assertIs(sys.modules[expected.__name__], expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(
            actual.__name__.replace("torch_rs", "torch"),
            expected.__name__,
        )
        self.assertEqual(
            actual.__package__.replace("torch_rs", "torch"),
            expected.__package__,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in SUPPORTED_EXPORTS],
        )
        self.assertEqual(actual.__annotations__, {"cuda": str | None})

    def test_values_follow_each_packages_build_metadata(self):
        actual = torch.version
        expected = reference_torch.version

        self.assertEqual(actual.__version__, torch.__version__)
        self.assertIs(actual.__version__, torch._C.__version__)
        self.assertEqual(
            actual.__version__,
            importlib.metadata.version("torch-rs"),
        )
        self.assertEqual(expected.__version__, reference_torch.__version__)
        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(actual.cuda, None)
        self.assertEqual(expected.cuda is not None, reference_torch._C._has_cuda)
        if expected.cuda is not None:
            self.assertIs(type(expected.cuda), str)
            self.assertTrue(expected.cuda)

    def test_direct_import_and_supported_wildcards_match_pytorch_2_13(self):
        modules = (
            ("torch_rs", torch.version),
            ("torch", reference_torch.version),
        )
        wildcard_names = []
        for package_name, module in modules:
            with self.subTest(package=package_name):
                parent_import = {}
                direct_import = {}
                child_wildcard = {}
                top_level_wildcard = {}
                exec(f"from {package_name} import version", parent_import)
                exec(
                    f"from {package_name}.version import __version__, cuda",
                    direct_import,
                )
                exec(f"from {package_name}.version import *", child_wildcard)
                exec(f"from {package_name} import *", top_level_wildcard)

                self.assertIs(parent_import["version"], module)
                for name in SUPPORTED_EXPORTS:
                    self.assertIs(direct_import[name], getattr(module, name))
                    self.assertIs(child_wildcard[name], getattr(module, name))
                self.assertNotIn("version", top_level_wildcard)
                wildcard_names.append(
                    {
                        name
                        for name in child_wildcard
                        if name != "__builtins__" and name in SUPPORTED_EXPORTS
                    }
                )

        self.assertEqual(wildcard_names[0], wildcard_names[1])
        self.assertEqual(wildcard_names[0], set(SUPPORTED_EXPORTS))

    def reload_contract(self, root):
        module = root.version
        namespace = module.__dict__
        old_exports = module.__all__
        sentinel = object()
        module.__version__ = sentinel
        module.cuda = sentinel
        module.__all__.append("unsupported")
        module._reload_marker = sentinel

        try:
            reloaded = importlib.reload(module)
            return (
                reloaded is module,
                module.__dict__ is namespace,
                root.version is module,
                sys.modules[module.__name__] is module,
                module.__all__ is not old_exports,
                tuple(name for name in module.__all__ if name in SUPPORTED_EXPORTS),
                module.__version__ is not sentinel,
                module.cuda is not sentinel,
                module._reload_marker is sentinel,
            )
        finally:
            module.__dict__.pop("_reload_marker", None)
            importlib.reload(module)

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )
        self.assertEqual(torch.version.__version__, torch.__version__)
        self.assertIs(torch.version.cuda, None)

    def test_h100_reports_reference_cuda_build_while_torch_rs_stays_cpu_only(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        reference_version = reference_torch.version.cuda
        self.assertIs(reference_torch._C._has_cuda, True)
        self.assertIs(type(reference_version), str)
        self.assertRegex(reference_version, r"^\d+\.\d+$")
        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(torch.version.cuda, None)

        device = reference_torch.device("cuda", 0)
        source = reference_torch.tensor([2.0, 3.0], device=device)
        result = source.square()
        reference_torch.cuda.synchronize(device)
        self.assertEqual(result.cpu().tolist(), [4.0, 9.0])

        self.assertFalse(hasattr(torch, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "cuda"))
        self.assertFalse(hasattr(torch.Tensor, "to"))
        with self.assertRaises(RuntimeError):
            torch.tensor([2.0, 3.0], device="cuda:0")

    def test_other_generated_build_metadata_remains_unsupported(self):
        for name in ("debug", "git_version", "hip", "rocm", "xpu"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch.version, name))
                self.assertTrue(hasattr(reference_torch.version, name))


if __name__ == "__main__":
    unittest.main()
