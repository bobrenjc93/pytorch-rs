import inspect
import math
import pickle
import re
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ArangeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("arange differentials require pinned PyTorch 2.13.0")

    def tensor_contract(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
        }

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_shapes_and_default_metadata_match_pytorch_2_13(self):
        endpoints = (
            0.0,
            -0.0,
            math.nextafter(0.0, 1.0),
            0.25,
            math.nextafter(1.0, 0.0),
            1.0,
            math.nextafter(1.0, 2.0),
            2.5,
            8.0,
        )
        for end in endpoints:
            for form in ("positional", "keyword"):
                with self.subTest(end=end, form=form):
                    if form == "positional":
                        actual = torch.arange(end)
                        expected = reference_torch.arange(end)
                    else:
                        actual = torch.arange(end=end)
                        expected = reference_torch.arange(end=end)
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )

    def test_default_equivalent_options_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            with self.subTest(options=actual_options):
                actual = torch.arange(2.5, **actual_options)
                expected = reference_torch.arange(2.5, **expected_options)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_fresh_storage_matches_pytorch_2_13(self):
        actual_first = torch.arange(8.5)
        actual_second = torch.arange(8.5)
        expected_first = reference_torch.arange(8.5)
        expected_second = reference_torch.arange(8.5)
        self.assertEqual(
            actual_first.data_ptr() != actual_second.data_ptr(),
            expected_first.data_ptr() != expected_second.data_ptr(),
        )
        self.assertEqual(
            actual_first.is_set_to(actual_second),
            expected_first.is_set_to(expected_second),
        )

        actual_empty_first = torch.arange(0.0)
        actual_empty_second = torch.arange(-0.0)
        expected_empty_first = reference_torch.arange(0.0)
        expected_empty_second = reference_torch.arange(-0.0)
        self.assertEqual(
            actual_empty_first.is_set_to(actual_empty_second),
            expected_empty_first.is_set_to(expected_empty_second),
        )

    def test_negative_and_nonfinite_errors_match_pytorch_2_13(self):
        endpoints = (
            -math.nextafter(0.0, 1.0),
            -0.25,
            -1.0,
            float("nan"),
            float("-nan"),
            float("inf"),
            float("-inf"),
        )
        for end in endpoints:
            for form in ("positional", "keyword"):
                with self.subTest(end=end, form=form):
                    if form == "positional":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end),
                            lambda end=end: reference_torch.arange(end),
                        )
                    else:
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end=end),
                            lambda end=end: reference_torch.arange(end=end),
                        )

    def test_oversized_endpoint_errors_match_pytorch_2_13(self):
        endpoints = (
            math.nextafter(float(2**63), 0.0),
            float(2**63),
            math.nextafter(float(2**63), math.inf),
            1.0e100,
        )
        for end in endpoints:
            with self.subTest(end=end):
                self.assert_error_matches(
                    lambda end=end: torch.arange(end),
                    lambda end=end: reference_torch.arange(end),
                )

    def callable_contract(self, module):
        function = module.arange
        owner = function.__reduce__()[1][0]
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
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.arange is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("arange"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["arange"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
