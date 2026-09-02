import copy
import importlib
import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EmptyLikeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "empty_like differentials require pinned PyTorch 2.13.0"
            )

    def source_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        return (
            ("scalar", module.tensor(-3.0, dtype=module.float32)),
            ("empty", module.ones((0,), dtype=module.float32)),
            (
                "empty multidimensional",
                module.ones((2, 0, 3), dtype=module.float32),
            ),
            ("multidimensional", module.ones((2, 3, 4), dtype=module.float32)),
            ("offset contiguous", base[1]),
        )

    def tensor_observation(self, module, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            str(tensor.layout),
            tensor.layout is module.strided,
            tensor.is_pinned(),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.is_contiguous(),
        )

    def test_supported_metadata_matches_pytorch_2_13(self):
        actual_sources = self.source_cases(torch)
        expected_sources = self.source_cases(reference_torch)
        metadata_cases = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"memory_format": None},
            lambda module: {"memory_format": module.preserve_format},
            lambda module: {"memory_format": module.contiguous_format},
            lambda module: {"requires_grad": True},
        )

        for (case, actual_source), (expected_case, expected_source) in zip(
            actual_sources, expected_sources, strict=True
        ):
            self.assertEqual(case, expected_case)
            for metadata_factory in metadata_cases:
                actual_kwargs = metadata_factory(torch)
                expected_kwargs = metadata_factory(reference_torch)
                with self.subTest(case=case, kwargs=actual_kwargs):
                    actual = torch.empty_like(actual_source, **actual_kwargs)
                    expected = reference_torch.empty_like(
                        expected_source, **expected_kwargs
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )
                    self.assertEqual(
                        actual.is_set_to(actual_source),
                        expected.is_set_to(expected_source),
                    )
                    self.assertFalse(actual.is_set_to(actual_source))

    def test_fresh_storage_matches_pytorch_2_13(self):
        def contract(module):
            source = module.ones((2, 3), dtype=module.float32)
            first = module.empty_like(source)
            second = module.empty_like(source)
            return {
                "first": self.tensor_observation(module, first),
                "second": self.tensor_observation(module, second),
                "first_not_source": not first.is_set_to(source),
                "first_not_second": not first.is_set_to(second),
                "source_storage": first.data_ptr() != source.data_ptr(),
                "peer_storage": first.data_ptr() != second.data_ptr(),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_no_grad_requires_grad_behavior_matches_pytorch_2_13(self):
        actual_source = torch.ones((2, 3), dtype=torch.float32, requires_grad=True) * 2.0
        expected_source = (
            reference_torch.ones(
                (2, 3), dtype=reference_torch.float32, requires_grad=True
            )
            * 2.0
        )

        with torch.no_grad():
            actual_default = torch.empty_like(actual_source)
            actual_tracked = torch.empty_like(actual_source, requires_grad=True)
        with reference_torch.no_grad():
            expected_default = reference_torch.empty_like(expected_source)
            expected_tracked = reference_torch.empty_like(
                expected_source, requires_grad=True
            )

        self.assertEqual(
            self.tensor_observation(torch, actual_default),
            self.tensor_observation(reference_torch, expected_default),
        )
        self.assertEqual(
            self.tensor_observation(torch, actual_tracked),
            self.tensor_observation(reference_torch, expected_tracked),
        )

    def callable_contract(self, module, native_name):
        _ = importlib.import_module(native_name)
        function = module.empty_like
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)

        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None

        reducer, reduction_args = function.__reduce__()
        owner = reduction_args[0]
        reduced_name = reduction_args[1] if len(reduction_args) > 1 else None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc_has_signature": (
                "empty_like(input, *, dtype=None, layout=None, device=None, "
                "requires_grad=False, memory_format=torch.preserve_format) -> Tensor"
            )
            in function.__doc__,
            "text_signature": function.__text_signature__,
            "signature_error": signature_error,
            "reduce_type": type(reducer).__name__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "reduced_name": reduced_name,
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_identity": owner.empty_like is function,
            "all_count": module.__all__.count("empty_like"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["empty_like"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_exports_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch, "torch_rs._C"),
            self.callable_contract(reference_torch, "torch._C"),
        )
        actual_function = torch.empty_like
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.empty_like, actual_function)


if __name__ == "__main__":
    unittest.main()
