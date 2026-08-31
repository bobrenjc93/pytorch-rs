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


SUPPORTED_ALL = ["__version__", "debug", "cuda", "git_version", "hip", "rocm", "xpu"]
SUPPORTED_PUBLIC = {"Optional", "debug", "cuda", "git_version", "hip", "rocm", "xpu"}
ANNOTATED_METADATA = ("cuda", "hip", "rocm", "xpu")
CPU_ONLY_METADATA = ("cuda", "git_version", "hip", "rocm", "xpu")


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
        supported = tuple(SUPPORTED_ALL)

        self.assertIs(torch.version, actual)
        self.assertIs(reference_torch.version, expected)
        self.assertIs(sys.modules["torch_rs.version"], actual)
        self.assertIs(sys.modules["torch.version"], expected)
        self.assertIs(type(actual), types.ModuleType)
        self.assertIs(type(expected), types.ModuleType)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            actual.__all__,
            [name for name in expected.__all__ if name in supported],
        )
        self.assertEqual(actual.__all__, SUPPORTED_ALL)
        self.assertEqual(
            actual.__annotations__,
            {
                name: annotation
                for name, annotation in expected.__annotations__.items()
                if name in supported
            },
        )
        self.assertEqual(list(actual.__annotations__), list(ANNOTATED_METADATA))
        self.assertEqual(
            list(actual.__annotations__),
            [name for name in expected.__annotations__ if name in supported],
        )
        self.assertIs(actual.Optional, typing.Optional)
        self.assertIs(expected.Optional, typing.Optional)
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            {name for name in vars(expected) if name in SUPPORTED_PUBLIC},
        )
        self.assertEqual(
            {name for name in vars(actual) if not name.startswith("_")},
            SUPPORTED_PUBLIC,
        )
        self.assertIs(type(actual.__version__), type(expected.__version__))
        self.assertIs(actual.debug, expected.debug)
        self.assertIs(actual.debug, False)
        for name in CPU_ONLY_METADATA:
            with self.subTest(name=name):
                self.assertIs(getattr(actual, name), None)
        self.assertTrue(hasattr(expected, "git_version"))
        self.assertIs(type(expected.git_version), str)
        self.assertGreater(len(expected.git_version), 0)

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
            for name in SUPPORTED_ALL:
                self.assertIs(direct_import[name], getattr(module, name))

        actual_wildcard = {}
        expected_wildcard = {}
        exec("from torch_rs.version import *", actual_wildcard)
        exec("from torch.version import *", expected_wildcard)
        self.assertEqual(
            [name for name in actual_wildcard if name != "__builtins__"],
            [name for name in expected_wildcard if name in SUPPORTED_ALL],
        )
        self.assertEqual(
            [name for name in actual_wildcard if name != "__builtins__"],
            SUPPORTED_ALL,
        )
        self.assertIs(actual_wildcard["__version__"], actual.__version__)
        self.assertIs(actual_wildcard["debug"], actual.debug)
        for name in CPU_ONLY_METADATA:
            self.assertIs(actual_wildcard[name], getattr(actual, name))

        for module in (torch, reference_torch):
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            self.assertNotIn("version", namespace)
            self.assertNotIn("__version__", namespace)
            self.assertNotIn("debug", namespace)
            for name in CPU_ONLY_METADATA:
                self.assertNotIn(name, namespace)

    def reload_contract(self, root):
        module = root.version
        namespace = module.__dict__
        old_all = module.__all__
        old_annotations = module.__annotations__
        expected_version = module.__version__
        module.__version__ = "stale"
        module.debug = True
        for name in CPU_ONLY_METADATA:
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
            module.debug is False,
            module.__annotations__ is old_annotations,
            tuple(module.__annotations__),
            tuple(getattr(module, name) for name in CPU_ONLY_METADATA),
        )

    def test_reload_behavior_matches_pytorch_2_13(self):
        actual = self.reload_contract(torch)
        expected = self.reload_contract(reference_torch)

        self.assertEqual(actual[:5], expected[:5])
        self.assertEqual(
            actual[5],
            [name for name in expected[5] if name in SUPPORTED_ALL],
        )
        self.assertEqual(actual[6:11], expected[6:11])
        self.assertEqual(actual[11], (None, None, None, None, None))
        self.assertEqual(
            expected[11],
            tuple(getattr(reference_torch.version, name) for name in CPU_ONLY_METADATA),
        )

    def test_metadata_constants_copy_deepcopy_and_pickle_match_supported_scope(self):
        actual = torch.version
        expected = reference_torch.version

        for module in (actual, expected):
            with self.subTest(module=module.__name__, name="__version__"):
                value = module.__version__
                self.assertIs(copy.copy(value), value)
                self.assertIs(copy.deepcopy(value), value)
                roundtrip = pickle.loads(pickle.dumps(value))
                self.assertEqual(roundtrip, value)
                self.assertIs(type(roundtrip), str)

            for name in ("debug", *CPU_ONLY_METADATA):
                with self.subTest(module=module.__name__, name=name):
                    value = getattr(module, name)
                    self.assertIs(copy.copy(value), value)
                    self.assertIs(copy.deepcopy(value), value)
                    roundtrip = pickle.loads(pickle.dumps(value))
                    if value is None or value is False or value is True:
                        self.assertIs(roundtrip, value)
                    else:
                        self.assertEqual(roundtrip, value)
                        self.assertIs(type(roundtrip), type(value))

        self.assertIs(actual.git_version, None)
        self.assertIs(type(expected.git_version), str)

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
        for name in CPU_ONLY_METADATA:
            with self.subTest(name=name):
                self.assertIs(getattr(torch.version, name), None)
        self.assertNotIn("+cu", torch.version.__version__)

        probe = reference_torch.ones(1, device=reference_torch.device("cuda", 0))
        self.assertEqual(probe.item(), 1.0)
        reference_torch.cuda.synchronize(0)


if __name__ == "__main__":
    unittest.main()
