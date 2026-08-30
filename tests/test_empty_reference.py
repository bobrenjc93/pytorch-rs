import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EmptyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("empty differentials require pinned PyTorch 2.13.0")

    def tensor_metadata(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is module.strided,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "data_ptr": tensor.data_ptr(),
        }

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def assert_error_matches(self, actual_call, expected_call):
        actual_type, actual_message = self.capture_error(actual_call)
        expected_type, expected_message = self.capture_error(expected_call)
        self.assertIs(actual_type, expected_type)
        self.assertEqual(actual_message, expected_message)

    def test_zero_element_shapes_and_metadata_match_pytorch_2_13(self):
        cases = (
            lambda module: module.empty(0),
            lambda module: module.empty((0,)),
            lambda module: module.empty([2, 0, 3]),
            lambda module: module.empty(size=(0,)),
            lambda module: module.empty(size=[2, 0]),
            lambda module: module.empty((0, sys.maxsize)),
            lambda module: module.empty((sys.maxsize, 0, sys.maxsize)),
        )
        for create in cases:
            with self.subTest(create=create):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_metadata(torch, actual),
                    self.tensor_metadata(reference_torch, expected),
                )

    def test_integer_protocol_dimensions_match_pytorch_2_13(self):
        actual_dynamic = IndexDimension(0)
        expected_dynamic = IndexDimension(0)
        actual = torch.empty(actual_dynamic)
        expected = reference_torch.empty(expected_dynamic)
        self.assertEqual(
            self.tensor_metadata(torch, actual),
            self.tensor_metadata(reference_torch, expected),
        )
        self.assertEqual(actual_dynamic.calls, 1)
        self.assertGreaterEqual(expected_dynamic.calls, 1)

        for dimension in (IntSubclass(0), np.int64(0), np.uint32(0)):
            with self.subTest(dimension=dimension):
                actual = torch.empty(dimension)
                expected = reference_torch.empty(dimension)
                self.assertEqual(
                    self.tensor_metadata(torch, actual),
                    self.tensor_metadata(reference_torch, expected),
                )

        actual_tuple_dimension = IndexDimension(0)
        expected_tuple_dimension = IndexDimension(0)
        actual = torch.empty((actual_tuple_dimension,))
        expected = reference_torch.empty((expected_tuple_dimension,))
        self.assertEqual(
            self.tensor_metadata(torch, actual),
            self.tensor_metadata(reference_torch, expected),
        )
        self.assertEqual(actual_tuple_dimension.calls, 1)
        self.assertGreaterEqual(expected_tuple_dimension.calls, 1)

    def test_default_equivalent_options_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {
                "out": None,
                "dtype": module.float,
                "device": module.device("cpu"),
                "requires_grad": True,
            },
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            with self.subTest(options=actual_options):
                with torch.no_grad():
                    actual = torch.empty((2, 0, 3), **actual_options)
                with reference_torch.no_grad():
                    expected = reference_torch.empty(
                        (2, 0, 3), **expected_options
                    )
                self.assertEqual(
                    self.tensor_metadata(torch, actual),
                    self.tensor_metadata(reference_torch, expected),
                )

    def test_empty_returns_fresh_storage_match_pytorch_2_13(self):
        def contract(module):
            first = module.empty((2, 0, 3))
            second = module.empty((2, 0, 3))
            return {
                "first": self.tensor_metadata(module, first),
                "second": self.tensor_metadata(module, second),
                "fresh_pair_storage": not first.is_set_to(second),
                "data_ptr_equal": first.data_ptr() == second.data_ptr(),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_parser_errors_match_pytorch_2_13(self):
        exact_cases = (
            lambda module: module.empty(),
            lambda module: module.empty(None),
            lambda module: module.empty(size=None),
            lambda module: module.empty(size=0),
            lambda module: module.empty(shape=(0,)),
            lambda module: module.empty(True),
            lambda module: module.empty(False),
            lambda module: module.empty(np.bool_(True)),
            lambda module: module.empty(0.0),
            lambda module: module.empty([True]),
            lambda module: module.empty([0.0]),
            lambda module: module.empty(range(0, 1)),
            lambda module: module.empty(-1),
            lambda module: module.empty(IndexDimension(-1)),
            lambda module: module.empty((-1, 0)),
            lambda module: module.empty((0,), out=[]),
            lambda module: module.empty((0,), dtype=object()),
            lambda module: module.empty((0,), requires_grad=1),
            lambda module: module.empty((0,), unexpected=True),
            lambda module: module.empty((0, sys.maxsize, sys.maxsize)),
            lambda module: module.empty(sys.maxsize),
        )
        for call in exact_cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch), lambda: call(reference_torch)
                )

        overflow_cases = (
            lambda module: module.empty(2**63),
            lambda module: module.empty((2**63,)),
            lambda module: module.empty((IndexDimension(2**63),)),
        )
        for call in overflow_cases:
            with self.subTest(call=call):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

    def callable_contract(self, module):
        function = module.empty
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
            "owner_callable_identity": owner.empty is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("empty"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["empty"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_exports_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def test_reload_preserves_empty_callable_identity(self):
        old_function = torch.empty
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(torch.empty, old_function)
        self.assertIs(torch._C._VariableFunctionsClass.empty, old_function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(torch.empty, protocol)), torch.empty)


if __name__ == "__main__":
    unittest.main()
