import importlib
import os
import subprocess
import sys
import unittest
from unittest import mock

import torch_rs as torch


CAPABILITY_NAMES = (
    "has_openmp",
    "has_mkl",
    "has_lapack",
    "has_spectral",
)


class NativeBuildCapabilityFlagsTests(unittest.TestCase):
    def test_current_cargo_build_reports_exact_false_singletons(self):
        environments = (
            {},
            {
                "USE_OPENMP": "1",
                "USE_MKL": "1",
                "USE_MKLDNN": "1",
                "USE_LAPACK": "1",
                "USE_SPECTRAL": "1",
            },
            {"OMP_NUM_THREADS": "64", "MKL_NUM_THREADS": "64"},
        )
        for environment in environments:
            with self.subTest(environment=environment):
                with mock.patch.dict(os.environ, environment, clear=True):
                    for name in CAPABILITY_NAMES:
                        package_value = getattr(torch, name)
                        native_value = getattr(torch._C, name)
                        self.assertIs(type(package_value), bool)
                        self.assertIs(package_value, False)
                        self.assertIs(native_value, package_value)

    def test_top_level_native_and_wildcard_placement(self):
        package = importlib.import_module("torch_rs")
        native = importlib.import_module("torch_rs._C")
        package_wildcard = {}
        native_wildcard = {}
        exec("from torch_rs import *", package_wildcard)
        exec("from torch_rs._C import *", native_wildcard)

        self.assertIs(package, torch)
        self.assertIs(native, torch._C)
        for name in CAPABILITY_NAMES:
            with self.subTest(name=name):
                value = getattr(torch, name)
                self.assertIn(name, vars(torch))
                self.assertIn(name, vars(torch._C))
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertEqual(torch._C.__all__.count(name), 1)
                self.assertIs(package_wildcard[name], value)
                self.assertIs(native_wildcard[name], value)

                explicit_package = {}
                explicit_native = {}
                exec(f"from torch_rs import {name}", explicit_package)
                exec(f"from torch_rs._C import {name}", explicit_native)
                self.assertIs(explicit_package[name], value)
                self.assertIs(explicit_native[name], value)

    def test_spectral_operations_remain_unsupported(self):
        for name in ("fft", "stft", "istft", "spectral"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))

        for module_name in ("torch_rs.fft", "torch_rs.spectral"):
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_reload_preserves_flags_tensor_type_and_matmul(self):
        package = torch
        native = torch._C
        tensor_type = torch.Tensor
        tensor_factory = torch.tensor
        matmul = torch.matmul
        backends = torch.backends
        nnpack_probe = torch._nnpack_available
        backend_modules = {
            name: getattr(backends, name)
            for name in ("openmp", "mkl", "mkldnn", "nnpack")
        }
        backend_functions = {
            name: module.is_available
            for name, module in backend_modules.items()
        }
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        def assert_stable_surface():
            self.assertIs(torch.Tensor, tensor_type)
            self.assertIs(torch.tensor, tensor_factory)
            self.assertIs(torch.matmul, matmul)
            self.assertIs(torch._nnpack_available, nnpack_probe)
            self.assertIs(
                torch._C._VariableFunctionsClass._nnpack_available,
                nnpack_probe,
            )
            self.assertEqual(torch.__all__.count("matmul"), 2)
            self.assertEqual(
                torch.matmul(left, right).tolist(),
                [[19.0, 22.0], [43.0, 50.0]],
            )
            self.assertEqual(
                (left @ right).tolist(),
                [[19.0, 22.0], [43.0, 50.0]],
            )
            for name in CAPABILITY_NAMES:
                self.assertIs(getattr(torch, name), False)
                self.assertIs(getattr(torch._C, name), False)
                self.assertEqual(torch.__all__.count(name), 1)
                self.assertEqual(torch._C.__all__.count(name), 1)
            self.assertIs(torch._C._has_mkldnn, False)
            self.assertFalse(hasattr(torch._C, "has_mkldnn"))
            self.assertNotIn("_has_mkldnn", torch._C.__all__)
            self.assertNotIn("has_mkldnn", torch.__all__)
            self.assertIs(torch.backends, backends)
            for name, module in backend_modules.items():
                self.assertIs(getattr(torch.backends, name), module)
                if name == "mkldnn":
                    self.assertEqual(module.is_available, backend_functions[name])
                else:
                    self.assertIs(module.is_available, backend_functions[name])
                expected = (
                    torch._nnpack_available()
                    if name == "nnpack"
                    else torch._C._has_mkldnn
                    if name == "mkldnn"
                    else getattr(torch._C, f"has_{name}")
                )
                self.assertIs(module.is_available(), expected)

        assert_stable_surface()
        self.assertIs(importlib.reload(package), package)
        assert_stable_surface()
        self.assertIs(importlib.reload(native), native)
        assert_stable_surface()

    def test_backend_availability_namespaces_are_the_only_supported_scope(self):
        backends = importlib.import_module("torch_rs.backends")
        self.assertIs(torch.backends, backends)
        for backend, flag in (
            ("openmp", "has_openmp"),
            ("mkl", "has_mkl"),
            ("mkldnn", "_has_mkldnn"),
            ("nnpack", None),
        ):
            with self.subTest(backend=backend):
                module = importlib.import_module(f"torch_rs.backends.{backend}")
                self.assertIs(getattr(backends, backend), module)
                expected = (
                    torch._nnpack_available()
                    if flag is None
                    else getattr(torch._C, flag)
                )
                self.assertIs(module.is_available(), expected)

        for module_name in (
            "torch_rs.backends.lapack",
        ):
            with self.subTest(module=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)

    def test_import_does_not_probe_external_runtimes_or_import_pytorch(self):
        script = r'''
import importlib
import os
import sys

class RejectExternalRuntimeImport:
    blocked = {"torch", "numpy", "scipy", "dnnl", "mkl", "mkldnn", "mkl_service"}

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in self.blocked:
            raise RuntimeError(f"external runtime import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectExternalRuntimeImport())
os.environ.update(
    USE_OPENMP="1",
    USE_MKL="1",
    USE_MKLDNN="1",
    USE_LAPACK="1",
    USE_SPECTRAL="1",
    OMP_NUM_THREADS="64",
    MKL_NUM_THREADS="64",
)
import torch_rs as torch

package_wildcard = {}
native_wildcard = {}
exec("from torch_rs import *", package_wildcard)
exec("from torch_rs._C import *", native_wildcard)
for name in ("has_openmp", "has_mkl", "has_lapack", "has_spectral"):
    package_value = getattr(torch, name)
    native_value = getattr(torch._C, name)
    assert package_value is native_value is False
    assert package_wildcard[name] is package_value
    assert native_wildcard[name] is native_value
assert torch.backends.mkldnn.is_available() is torch._C._has_mkldnn is False
assert not hasattr(torch._C, "has_mkldnn")
assert "has_mkldnn" not in torch.__all__
package = torch
native = torch._C
assert importlib.reload(package) is package
assert torch.has_spectral is torch._C.has_spectral is False
assert importlib.reload(native) is native
assert torch.has_spectral is torch._C.has_spectral is False
assert not hasattr(torch, "fft")
assert not hasattr(torch, "stft")
assert not hasattr(torch, "istft")
assert not hasattr(torch, "spectral")
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
