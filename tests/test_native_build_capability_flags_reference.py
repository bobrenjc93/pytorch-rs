import importlib
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


CAPABILITY_NAMES = ("has_openmp", "has_mkl", "has_lapack")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class NativeBuildCapabilityFlagsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "native build capability differentials require pinned PyTorch 2.13.0"
            )

    def metadata_contract(self, module):
        native = module._C
        package_wildcard = {}
        native_wildcard = {}
        exec(f"from {module.__name__} import *", package_wildcard)
        exec(f"from {native.__name__} import *", native_wildcard)
        return tuple(
            (
                name,
                name in vars(module),
                name in vars(native),
                type(getattr(module, name)).__name__,
                type(getattr(native, name)).__name__,
                getattr(module, name) is getattr(native, name),
                module.__all__.count(name),
                package_wildcard[name] is getattr(module, name),
                native_wildcard[name] is getattr(native, name),
            )
            for name in CAPABILITY_NAMES
        )

    def test_metadata_and_import_placement_match_pytorch_2_13(self):
        self.assertEqual(
            self.metadata_contract(torch),
            self.metadata_contract(reference_torch),
        )

    def test_values_are_build_specific_exact_booleans(self):
        for name in CAPABILITY_NAMES:
            with self.subTest(name=name):
                actual = getattr(torch, name)
                expected = getattr(reference_torch, name)
                self.assertIs(actual, False)
                self.assertIs(type(actual), bool)
                self.assertIs(type(expected), bool)

    def native_reload_contract(self, module):
        native = module._C
        tensor_type = module.Tensor
        matmul = module.matmul
        values = tuple(getattr(module, name) for name in CAPABILITY_NAMES)
        native_values = tuple(getattr(native, name) for name in CAPABILITY_NAMES)
        backends = module.backends
        backend_modules = tuple(
            getattr(backends, name) for name in ("openmp", "mkl", "nnpack")
        )
        backend_functions = tuple(
            backend.is_available for backend in backend_modules
        )

        reloaded = importlib.reload(native)
        return (
            reloaded is native,
            module.Tensor is tensor_type,
            module.matmul is matmul,
            tuple(
                getattr(module, name) is value
                for name, value in zip(CAPABILITY_NAMES, values, strict=True)
            ),
            tuple(
                getattr(native, name) is value
                for name, value in zip(
                    CAPABILITY_NAMES, native_values, strict=True
                )
            ),
            tuple(
                getattr(module, name) is getattr(native, name)
                for name in CAPABILITY_NAMES
            ),
            module.backends is backends,
            tuple(
                getattr(module.backends, name) is backend
                for name, backend in zip(
                    ("openmp", "mkl", "nnpack"),
                    backend_modules,
                    strict=True,
                )
            ),
            tuple(
                backend.is_available is function
                for backend, function in zip(
                    backend_modules, backend_functions, strict=True
                )
            ),
        )

    def test_native_reload_behavior_matches_pytorch_2_13(self):
        self.assertEqual(
            self.native_reload_contract(torch),
            self.native_reload_contract(reference_torch),
        )

    def matmul_contract(self, module):
        left = module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32)
        right = module.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=module.float32)
        product = module.matmul(left, right)
        operator_product = left @ right
        return (
            tuple(product.shape),
            product.stride(),
            product.storage_offset(),
            str(product.dtype).replace("torch_rs", "torch"),
            str(product.device),
            product.tolist(),
            operator_product.tolist(),
        )

    def test_reading_flags_preserves_tensor_and_matmul_behavior(self):
        for module in (torch, reference_torch):
            for name in CAPABILITY_NAMES:
                self.assertIs(type(getattr(module, name)), bool)

        self.assertEqual(
            self.matmul_contract(torch),
            self.matmul_contract(reference_torch),
        )

    def test_backend_availability_namespaces_match_the_supported_scope(self):
        self.assertTrue(hasattr(reference_torch, "backends"))
        self.assertTrue(hasattr(torch, "backends"))
        for backend in ("openmp", "mkl", "nnpack"):
            with self.subTest(backend=backend):
                actual = getattr(torch.backends, backend)
                expected = getattr(reference_torch.backends, backend)
                self.assertIs(type(actual.is_available()), bool)
                self.assertIs(type(expected.is_available()), bool)
                actual_expected = (
                    torch._nnpack_available()
                    if backend == "nnpack"
                    else getattr(torch._C, f"has_{backend}")
                )
                self.assertIs(actual.is_available(), actual_expected)

        self.assertFalse(hasattr(torch.backends, "lapack"))


if __name__ == "__main__":
    unittest.main()
