import copy
import importlib
import pickle
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ZerosLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "zeros_like differentials require pinned PyTorch 2.13.0"
            )

    def tensor_contract(self, module, tensor):
        source = tensor.detach() if tensor.requires_grad else tensor
        return (
            np.asarray(source).reshape(-1).view(np.uint32).tolist(),
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.numel(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            str(tensor.layout),
            tensor.layout is module.strided,
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.grad is None,
            tensor.is_contiguous(),
        )

    def inputs(self, module):
        base = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
        )
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("empty vector", module.zeros((0,), dtype=module.float32)),
            (
                "empty multidimensional",
                module.zeros((2, 0, 3), dtype=module.float32),
            ),
            ("matrix", base),
            ("offset row", base[1]),
            (
                "requires grad input",
                module.ones((2, 3), dtype=module.float32, requires_grad=True) * 2.0,
            ),
        )

    def test_supported_values_metadata_and_aliasing_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.preserve_format},
            lambda module: {"memory_format": module.contiguous_format},
        )
        actual_inputs = self.inputs(torch)
        expected_inputs = self.inputs(reference_torch)

        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_inputs, expected_inputs, strict=True
        ):
            self.assertEqual(case, expected_case)
            for option_factory in option_factories:
                actual_options = option_factory(torch)
                expected_options = option_factory(reference_torch)
                with self.subTest(case=case, options=actual_options):
                    actual = torch.zeros_like(actual_input, **actual_options)
                    expected = reference_torch.zeros_like(
                        expected_input, **expected_options
                    )
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )
                    self.assertEqual(
                        actual.is_set_to(actual_input),
                        expected.is_set_to(expected_input),
                    )
                    self.assertFalse(actual.is_set_to(actual_input))
                    if actual_input.numel() != 0:
                        self.assertEqual(
                            actual.data_ptr() == actual_input.data_ptr(),
                            expected.data_ptr() == expected_input.data_ptr(),
                        )
                        self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())

    def test_no_grad_and_fresh_storage_match_pytorch_2_13(self):
        def observe(module):
            source = (
                module.ones((2, 3), dtype=module.float32, requires_grad=True) * 5.0
            )
            with module.no_grad():
                default = module.zeros_like(source)
                requested = module.zeros_like(source, requires_grad=True)
            first = module.zeros_like(source)
            second = module.zeros_like(source)
            return {
                "default": self.tensor_contract(module, default),
                "requested": self.tensor_contract(module, requested),
                "fresh_storage": first.data_ptr() != second.data_ptr(),
                "fresh_view": not first.is_set_to(second),
                "source_alias": first.is_set_to(source),
            }

        self.assertEqual(observe(torch), observe(reference_torch))

    def test_callable_import_wildcard_copy_and_pickle_match_pytorch_2_13(self):
        def contract(module):
            function = module.zeros_like
            namespace = {}
            exec(f"from {module.__name__} import *", namespace)
            return {
                "type": type(function),
                "name": function.__name__,
                "qualname": function.__qualname__,
                "module": function.__module__,
                "text_signature": function.__text_signature__,
                "all_count": module.__all__.count("zeros_like"),
                "wildcard_identity": namespace["zeros_like"] is function,
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "pickle_identities": tuple(
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            }

        self.assertEqual(contract(torch), contract(reference_torch))
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(
            hasattr(importlib.import_module("torch_rs"), "_VariableFunctionsClass")
        )

    def test_like_family_remains_unexported(self):
        for name in ("ones_like", "empty_like", "full_like"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
