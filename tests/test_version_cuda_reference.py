import copy
import importlib
import pickle
import sys
import types
import typing
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SUPPORTED_VERSION_NAMES = (
    "__version__",
    "debug",
    "cuda",
    "git_version",
    "hip",
    "rocm",
    "xpu",
)
SUPPORTED_PUBLIC_NAMES = (set(SUPPORTED_VERSION_NAMES) - {"__version__"}) | {
    "Optional",
}


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class VersionCudaReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.version differentials require pinned PyTorch 2.13.0"
            )

    def test_module_metadata_and_identity_match_the_supported_scope(self):
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
            [name for name in expected.__all__ if name in SUPPORTED_VERSION_NAMES],
        )
        self.assertEqual(
            actual.__annotations__,
            {
                name: annotation
                for name, annotation in expected.__annotations__.items()
                if name in SUPPORTED_VERSION_NAMES
            },
        )
        self.assertEqual(
            list(actual.__annotations__),
            [
                name
                for name in expected.__annotations__
                if name in SUPPORTED_VERSION_NAMES
            ],
        )
        self.assertIs(actual.Optional, typing.Optional)
        self.assertIs(expected.Optional, typing.Optional)
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {
                name
                for name in vars(expected)
                if name in SUPPORTED_PUBLIC_NAMES
            },
        )
        self.assertIs(type(actual.__version__), type(expected.__version__))
        self.assertIs(actual.debug, expected.debug)
        self.assertIs(actual.debug, False)
        self.assertTrue(hasattr(actual, "git_version"))
        self.assertTrue(hasattr(expected, "git_version"))
        self.assertIs(actual.git_version, None)
        self.assertIs(type(expected.git_version), str)
        self.assertTrue(expected.git_version)
        for name in ("cuda", "hip", "rocm", "xpu"):
            with self.subTest(name=name):
                self.assertIs(getattr(actual, name), None)

    def test_direct_and_wildcard_imports_match_pytorch_2_13(self):
        actual = torch.version
        expected = reference_torch.version

        for package_name, module in (
            ("torch_rs", actual),
            ("torch", expected),
        ):
            package_import = {}
            module_import = {}
            direct_import = {}
            exec(f"from {package_name} import version", package_import)
            exec(f"import {package_name}.version as version", module_import)
            exec(
                f"from {package_name}.version import "
                "__version__, debug, cuda, git_version, hip, rocm, xpu",
                direct_import,
            )
            self.assertIs(package_import["version"], module)
            self.assertIs(module_import["version"], module)
            for name in SUPPORTED_VERSION_NAMES:
                self.assertIs(direct_import[name], getattr(module, name))

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.version import *", actual_wildcard)
        exec("from torch.version import *", expected_wildcard)
        self.assertEqual(
            [name for name in actual_wildcard if name != "__builtins__"],
            [
                name
                for name in expected_wildcard
                if name in SUPPORTED_VERSION_NAMES
            ],
        )
        self.assertIs(actual_wildcard["__version__"], actual.__version__)
        self.assertIs(actual_wildcard["debug"], actual.debug)
        for name in ("cuda", "git_version", "hip", "rocm", "xpu"):
            self.assertIs(actual_wildcard[name], getattr(actual, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("version", namespace)
            self.assertNotIn("__version__", namespace)
            self.assertNotIn("debug", namespace)
            self.assertNotIn("git_version", namespace)
            for name in ("cuda", "hip", "rocm", "xpu"):
                self.assertNotIn(name, namespace)

    def reload_contract(self, root):
        module = root.version
        namespace = module.__dict__
        old_all = module.__all__
        old_annotations = module.__annotations__
        expected_version = module.__version__
        expected_git_version = module.git_version
        module.__version__ = "stale"
        module.debug = True
        for name in ("cuda", "git_version", "hip", "rocm", "xpu"):
            setattr(module, name, "stale")

        reloaded = importlib.reload(module)

        return (
            reloaded is module,
            module.__dict__ is namespace,
            root.version is module,
            sys.modules[module.__name__] is module,
            module.__all__ is not old_all,
            module.__all__,
            module.__version__ == expected_version,
            type(module.__version__).__name__,
            module.git_version == expected_git_version,
            type(module.git_version).__name__,
            module.debug is False,
            module.__annotations__ is old_annotations,
            tuple(module.__annotations__),
            tuple(getattr(module, name) for name in ("cuda", "hip", "rocm", "xpu")),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_contract(torch)
        expected = self.reload_contract(reference_torch)

        self.assertEqual(actual[:5], expected[:5])
        self.assertEqual(
            actual[5],
            [name for name in expected[5] if name in SUPPORTED_VERSION_NAMES],
        )
        self.assertEqual(actual[6:9], expected[6:9])
        self.assertEqual(actual[9], "NoneType")
        self.assertEqual(expected[9], "str")
        self.assertEqual(actual[10:13], expected[10:13])
        self.assertEqual(actual[13], (None, None, None, None))
        self.assertEqual(
            expected[13],
            tuple(
                getattr(reference_torch.version, name)
                for name in ("cuda", "hip", "rocm", "xpu")
            ),
        )

    def test_constant_copy_deepcopy_and_pickle_behavior_matches_supported_scope(self):
        for root in (torch, reference_torch):
            version = root.version
            for name in ("debug", "cuda", "git_version", "hip", "rocm", "xpu"):
                value = getattr(version, name)
                with self.subTest(package=root.__name__, name=name):
                    self.assertIs(copy.copy(value), value)
                    self.assertIs(copy.deepcopy(value), value)
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                        with self.subTest(protocol=protocol):
                            restored = pickle.loads(
                                pickle.dumps(value, protocol=protocol)
                            )
                            self.assertIs(type(restored), type(value))
                            self.assertEqual(restored, value)

            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(
                    package=root.__name__,
                    name="__version__",
                    protocol=protocol,
                ):
                    restored = pickle.loads(
                        pickle.dumps(version.__version__, protocol=protocol)
                    )
                    self.assertIs(type(restored), str)
                    self.assertEqual(restored, version.__version__)

        self.assertIs(torch.version.git_version, None)
        self.assertIs(type(reference_torch.version.git_version), str)

    def test_cuda_visible_h100_reports_reference_build_version_only(self):
        if not reference_torch.cuda.is_available():
            self.skipTest("requires a CUDA-visible reference PyTorch runtime")

        device_name = reference_torch.cuda.get_device_name(0)
        if "H100" not in device_name:
            self.skipTest(f"requires an NVIDIA H100, found {device_name}")

        self.assertGreaterEqual(reference_torch.cuda.device_count(), 1)
        self.assertIs(reference_torch._C._has_cuda, True)
        self.assertEqual(
            tuple(
                getattr(reference_torch.version, name)
                for name in ("cuda", "hip", "rocm", "xpu")
            ),
            ("13.0", None, None, None),
        )
        self.assertIn("+cu130", reference_torch.version.__version__)

        self.assertIs(torch._C._has_cuda, False)
        self.assertIs(torch.backends.cuda.is_built(), False)
        self.assertIs(torch.version.git_version, None)
        self.assertIs(type(reference_torch.version.git_version), str)
        for name in ("cuda", "hip", "rocm", "xpu"):
            with self.subTest(name=name):
                self.assertIs(getattr(torch.version, name), None)
        self.assertNotIn("+cu", torch.version.__version__)

        probe = reference_torch.ones(1, device=reference_torch.device("cuda", 0))
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(0)


if __name__ == "__main__":
    unittest.main()
