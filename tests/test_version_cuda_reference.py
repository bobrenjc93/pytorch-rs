import importlib
import re
import sys
import types
import typing
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
                "version.cuda differentials require pinned PyTorch 2.13.0"
            )

    def test_module_identity_metadata_and_supported_scope_match_pytorch_2_13(self):
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
        self.assertEqual(
            actual.__spec__.name.replace("torch_rs", "torch"),
            expected.__spec__.name,
        )
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in SUPPORTED_EXPORTS],
        )
        self.assertEqual(
            actual.__annotations__,
            {
                name: annotation
                for name, annotation in expected.__annotations__.items()
                if name in SUPPORTED_EXPORTS
            },
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if not name.startswith("_")
                and name in {"Optional", "cuda"}
            },
        )
        self.assertEqual(actual.__version__, torch.__version__)
        self.assertEqual(expected.__version__, reference_torch.__version__)
        self.assertIs(type(actual.__version__), type(expected.__version__))
        self.assertEqual(
            actual.__annotations__["cuda"],
            typing.Optional[str],
        )

    def test_direct_import_and_wildcard_behavior_matches_supported_scope(self):
        actual = torch.version
        expected = reference_torch.version

        for package_name, module in (
            ("torch_rs", actual),
            ("torch", expected),
        ):
            parent_import = {}
            child_import = {}
            exec(f"from {package_name} import version", parent_import)
            exec(
                f"from {package_name}.version import __version__, cuda",
                child_import,
            )
            self.assertIs(parent_import["version"], module)
            self.assertIs(child_import["__version__"], module.__version__)
            self.assertIs(child_import["cuda"], module.cuda)

        actual_child_wildcard = {}
        expected_child_wildcard = {}
        exec("from torch_rs.version import *", actual_child_wildcard)
        exec("from torch.version import *", expected_child_wildcard)
        self.assertEqual(
            {
                name
                for name in actual_child_wildcard
                if name != "__builtins__"
            },
            {
                name
                for name in expected_child_wildcard
                if name in SUPPORTED_EXPORTS
            },
        )

        actual_parent_wildcard = {}
        expected_parent_wildcard = {}
        exec("from torch_rs import *", actual_parent_wildcard)
        exec("from torch import *", expected_parent_wildcard)
        self.assertNotIn("version", actual_parent_wildcard)
        self.assertNotIn("version", expected_parent_wildcard)
        self.assertEqual(
            torch.__all__.count("version"),
            reference_torch.__all__.count("version"),
        )

    def reload_contract(self, root):
        module = root.version
        namespace = module.__dict__
        old_exports = module.__all__
        reloaded = importlib.reload(module)
        has_cuda = root._C._has_cuda
        cuda_matches_build = (
            isinstance(module.cuda, str) and bool(module.cuda)
            if has_cuda
            else module.cuda is None
        )
        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.version is module,
            sys.modules[module.__name__] is module,
            module.__all__ is not old_exports,
            tuple(name for name in module.__all__ if name in SUPPORTED_EXPORTS),
            module.__version__ == root.__version__,
            cuda_matches_build,
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.reload_contract(torch),
            self.reload_contract(reference_torch),
        )

    def test_cuda_visible_h100_reports_reference_build_version_only(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertIs(reference_torch._C._has_cuda, True)
        self.assertIs(reference_torch.backends.cuda.is_built(), True)
        reference_version = reference_torch.version.cuda
        self.assertIs(type(reference_version), str)
        self.assertRegex(reference_version, r"^\d+\.\d+$")
        cuda_tag = "cu" + re.sub(r"\D", "", reference_version)
        self.assertTrue(reference_torch.__version__.endswith(f"+{cuda_tag}"))
        self.assertEqual(reference_torch.cuda.get_device_capability(0), (9, 0))

        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertEqual(torch.version.__version__, torch.__version__)
        self.assertIs(torch.version.cuda, None)

    def test_other_version_metadata_remains_explicitly_unsupported(self):
        actual = torch.version
        expected = reference_torch.version
        for name in ("debug", "git_version", "hip", "rocm", "xpu"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(actual, name))
                self.assertTrue(hasattr(expected, name))


if __name__ == "__main__":
    unittest.main()
