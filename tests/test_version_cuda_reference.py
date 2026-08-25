import importlib
import sys
import types
import typing
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
                "version metadata differentials require pinned PyTorch 2.13.0"
            )

    def test_supported_values_and_annotations_match_pytorch_2_13(self):
        actual = torch.version
        expected = reference_torch.version

        for root, module in (
            (torch, actual),
            (reference_torch, expected),
        ):
            with self.subTest(module=module.__name__):
                self.assertIsInstance(module.__version__, str)
                self.assertEqual(module.__version__, root.__version__)

        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(actual.cuda, None)
        self.assertIsInstance(expected.cuda, (str, type(None)))
        self.assertEqual(
            actual.__annotations__["cuda"],
            expected.__annotations__["cuda"],
        )
        self.assertEqual(
            typing.get_type_hints(actual)["cuda"],
            typing.get_type_hints(expected)["cuda"],
        )

    def test_module_identity_and_direct_imports_match_pytorch_2_13(self):
        for root in (torch, reference_torch):
            with self.subTest(module=root.__name__):
                module = importlib.import_module(f"{root.__name__}.version")
                namespace = {}
                exec(
                    f"import {root.__name__}.version as direct_version",
                    namespace,
                )
                exec(
                    f"from {root.__name__} import version as parent_version",
                    namespace,
                )
                exec(
                    f"from {root.__name__}.version import __version__, cuda",
                    namespace,
                )

                self.assertIs(type(module), types.ModuleType)
                self.assertIs(getattr(root, "version"), module)
                self.assertIs(sys.modules[module.__name__], module)
                self.assertIs(namespace["direct_version"], module)
                self.assertIs(namespace["parent_version"], module)
                self.assertIs(namespace["__version__"], module.__version__)
                self.assertIs(namespace["cuda"], module.cuda)
                self.assertIsNone(module.__doc__)

        self.assertEqual(
            torch.version.__name__.replace("torch_rs", "torch"),
            reference_torch.version.__name__,
        )
        self.assertEqual(
            torch.version.__package__.replace("torch_rs", "torch"),
            reference_torch.version.__package__,
        )

    def test_supported_wildcard_exports_match_pytorch_2_13(self):
        actual = torch.version
        expected = reference_torch.version
        expected_supported = [
            name for name in expected.__all__ if name in SUPPORTED_EXPORTS
        ]

        self.assertEqual(actual.__all__, expected_supported)
        self.assertEqual(actual.__all__, ["__version__", "cuda"])

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.version import *", actual_wildcard)
        exec("from torch.version import *", expected_wildcard)
        actual_names = {
            name for name in actual_wildcard if name != "__builtins__"
        }
        expected_supported_names = {
            name for name in expected_wildcard if name in SUPPORTED_EXPORTS
        }
        self.assertEqual(actual_names, expected_supported_names)
        for name in SUPPORTED_EXPORTS:
            self.assertIs(actual_wildcard[name], getattr(actual, name))
            self.assertIs(expected_wildcard[name], getattr(expected, name))

        self.assertEqual(
            torch.__all__.count("version"),
            reference_torch.__all__.count("version"),
        )
        self.assertEqual(torch.__all__.count("version"), 0)

    def reload_contract(self, root):
        module = root.version
        namespace = module.__dict__
        old_exports = module.__all__
        expected_version = module.__version__
        module.__version__ = "stale"
        module.cuda = "stale"

        reloaded = importlib.reload(module)
        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.version is module,
            sys.modules[module.__name__] is module,
            module.__all__ is not old_exports,
            module.__version__ == expected_version,
            type(module.__version__).__name__,
            module.cuda,
            tuple(name for name in module.__all__ if name in SUPPORTED_EXPORTS),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual_contract = self.reload_contract(torch)
        expected_contract = self.reload_contract(reference_torch)

        self.assertEqual(actual_contract[:7], expected_contract[:7])
        self.assertIs(actual_contract[7], None)
        self.assertEqual(expected_contract[7], reference_torch.version.cuda)
        self.assertEqual(
            actual_contract[8],
            tuple(
                name
                for name in expected_contract[8]
                if name in SUPPORTED_EXPORTS
            ),
        )
        self.assertIs(torch.version.cuda, None)

    def test_h100_reports_reference_cuda_build_but_torch_rs_stays_cpu_only(self):
        if reference_torch.version.cuda is None:
            self.skipTest("requires a CUDA-built reference PyTorch")
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertIsInstance(reference_torch.version.cuda, str)
        self.assertTrue(reference_torch.version.cuda)
        self.assertIs(reference_torch._C._has_cuda, True)
        self.assertIs(reference_torch.backends.cuda.is_built(), True)
        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)

        probe = reference_torch.ones(1, device="cuda:0")
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(0)

        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertIs(torch.version.cuda, None)
        self.assertFalse(hasattr(torch, "cuda"))


if __name__ == "__main__":
    unittest.main()
