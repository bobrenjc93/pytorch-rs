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


class BadIndexDimension:
    def __index__(self):
        return 1.5


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EmptyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("empty differentials require pinned PyTorch 2.13.0")

    def tensor_contract(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
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

    def test_supported_sizes_and_metadata_match_pytorch_2_13(self):
        size_factories = (
            (lambda module: 2),
            (lambda module: IntSubclass(2)),
            (lambda module: np.int64(2)),
            (lambda module: np.uint32(2)),
            (lambda module: IndexDimension(2)),
            (lambda module: (2, 3, 4)),
            (lambda module: [2, 3]),
            (lambda module: module.Size([2, 3])),
            (lambda module: ()),
            (lambda module: 0),
            (lambda module: (2, 0, 3)),
            (lambda module: (sys.maxsize, 0, 2)),
        )
        option_factories = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {
                "dtype": module.float32,
                "device": module.device("cpu"),
                "requires_grad": True,
            },
        )

        for size_factory in size_factories:
            for option_factory in option_factories:
                actual_size = size_factory(torch)
                expected_size = size_factory(reference_torch)
                actual_options = option_factory(torch)
                expected_options = option_factory(reference_torch)
                with self.subTest(size=actual_size, options=actual_options):
                    actual = torch.empty(actual_size, **actual_options)
                    expected = reference_torch.empty(expected_size, **expected_options)
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )

        actual = torch.empty(size=(2, 0, 3))
        expected = reference_torch.empty(size=(2, 0, 3))
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )

    def test_storage_freshness_and_leaf_autograd_match_pytorch_2_13(self):
        actual_first = torch.empty((8,))
        actual_second = torch.empty((8,))
        expected_first = reference_torch.empty((8,))
        expected_second = reference_torch.empty((8,))
        self.assertEqual(
            actual_first.data_ptr() != actual_second.data_ptr(),
            expected_first.data_ptr() != expected_second.data_ptr(),
        )

        actual_leaf = torch.empty((2, 3), requires_grad=True)
        expected_leaf = reference_torch.empty((2, 3), requires_grad=True)
        actual_leaf.sum().backward()
        expected_leaf.sum().backward()
        self.assertEqual(
            self.tensor_contract(torch, actual_leaf.grad),
            self.tensor_contract(reference_torch, expected_leaf.grad),
        )
        self.assertEqual(actual_leaf.grad.tolist(), expected_leaf.grad.tolist())

    def test_index_protocol_and_supported_form_errors_match_pytorch_2_13(self):
        actual_scalar = IndexDimension(2)
        expected_scalar = IndexDimension(2)
        actual_element = IndexDimension(3)
        expected_element = IndexDimension(3)
        self.assertEqual(
            self.tensor_contract(torch, torch.empty(actual_scalar)),
            self.tensor_contract(reference_torch, reference_torch.empty(expected_scalar)),
        )
        self.assertEqual(
            self.tensor_contract(torch, torch.empty((actual_element,))),
            self.tensor_contract(
                reference_torch, reference_torch.empty((expected_element,))
            ),
        )
        self.assertGreater(actual_scalar.calls, 0)
        self.assertGreater(expected_scalar.calls, 0)
        self.assertGreater(actual_element.calls, 0)
        self.assertGreater(expected_element.calls, 0)

        exact_cases = (
            lambda module: module.empty(),
            lambda module: module.empty(None),
            lambda module: module.empty(size=2),
            lambda module: module.empty(-1),
            lambda module: module.empty((2, -1)),
            lambda module: module.empty(True),
            lambda module: module.empty(np.bool_(True)),
            lambda module: module.empty((True,)),
            lambda module: module.empty((np.bool_(True),)),
            lambda module: module.empty(2.0),
            lambda module: module.empty((2.0,)),
            lambda module: module.empty(range(2)),
            lambda module: module.empty(BadIndexDimension()),
            lambda module: module.empty((BadIndexDimension(),)),
            lambda module: module.empty((2,), dtype=object()),
            lambda module: module.empty((2,), device=object()),
            lambda module: module.empty((2,), requires_grad=1),
            lambda module: module.empty((2,), size=(3,)),
            lambda module: module.empty(shape=(2,)),
            lambda module: module.empty((2,), shape=(3,)),
            lambda module: module.empty(-1, dtype=object()),
            lambda module: module.empty((2, -1), dtype=object()),
            lambda module: module.empty(-1, device=object()),
            lambda module: module.empty(-1, requires_grad=1),
            lambda module: module.empty(-1, unexpected=True),
            lambda module: module.empty(True, requires_grad=1),
        )
        for call in exact_cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda call=call: call(torch),
                    lambda call=call: call(reference_torch),
                )

        overflow_cases = (
            2**63,
            -(2**63) - 1,
            np.uint64(2**63),
            IndexDimension(2**63),
            (2**63,),
        )
        for size in overflow_cases:
            with self.subTest(size=size):
                with self.assertRaises(Exception) as actual_raised:
                    torch.empty(size)
                with self.assertRaises(Exception) as expected_raised:
                    reference_torch.empty(size)
                self.assertIs(
                    type(actual_raised.exception), type(expected_raised.exception)
                )
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, str(actual_raised.exception))
                self.assertIn(marker, str(expected_raised.exception))
                self.assertIn(
                    "Overflow when unpacking long long", str(actual_raised.exception)
                )
                self.assertIn(
                    "Overflow when unpacking long long", str(expected_raised.exception)
                )

        self.assert_error_matches(
            lambda: torch.empty(sys.maxsize),
            lambda: reference_torch.empty(sys.maxsize),
        )
        self.assert_error_matches(
            lambda: torch.empty((sys.maxsize, 2)),
            lambda: reference_torch.empty((sys.maxsize, 2)),
        )

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "requires a CUDA-capable PyTorch runtime",
    )
    def test_cuda_storage_remains_explicitly_unsupported(self):
        self.assert_error_matches(
            lambda: torch.empty(-1, device="cuda"),
            lambda: reference_torch.empty(-1, device="cuda"),
        )

        expected = reference_torch.empty((2,), device="cuda")
        self.assertTrue(expected.is_cuda)
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                "empty(): device 'cuda' is not supported; only 'cpu' is implemented"
            ),
        ):
            torch.empty((2,), device="cuda")

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
