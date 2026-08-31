import copy
import pickle
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
            "layout_identity": tensor.layout is module.strided,
            "is_pinned": tensor.is_pinned(),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "data_ptr_is_zero": tensor.data_ptr() == 0,
        }

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def assert_error_matches(self, actual_call, expected_call):
        actual_type, actual_message = self.capture_error(actual_call)
        expected_type, expected_message = self.capture_error(expected_call)
        self.assertIs(actual_type, expected_type)
        self.assertEqual(
            actual_message.replace("torch.device or str", "torch.device"),
            expected_message,
        )

    def test_shape_and_metadata_match_pytorch_2_13_without_value_assertions(self):
        cases = (
            ("one positional dimension", lambda module: module.empty(2)),
            ("scalar tuple", lambda module: module.empty(())),
            ("scalar list", lambda module: module.empty([])),
            ("empty vector", lambda module: module.empty((0,))),
            ("empty middle", lambda module: module.empty([2, 0, 3])),
            ("multidimensional", lambda module: module.empty((2, 3))),
            ("size keyword", lambda module: module.empty(size=[2])),
        )
        for case, create in cases:
            with self.subTest(case=case):
                actual = create(torch)
                expected = create(reference_torch)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_integer_protocol_size_dimensions_match_pytorch_2_13(self):
        actual_dynamic = IndexDimension(2)
        expected_dynamic = IndexDimension(2)
        actual = torch.empty([actual_dynamic, np.int64(0), IntSubclass(3)])
        expected = reference_torch.empty(
            [expected_dynamic, np.int64(0), IntSubclass(3)]
        )

        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )
        self.assertEqual(actual_dynamic.calls, 1)
        self.assertGreaterEqual(expected_dynamic.calls, 1)

    def test_out_none_results_and_storage_freshness_match_pytorch_2_13(self):
        cases = (
            ("scalar", lambda module: module.empty((), out=None)),
            ("empty", lambda module: module.empty((0,), out=None)),
            (
                "multidimensional",
                lambda module: module.empty((2, 3), out=None),
            ),
            (
                "requires grad",
                lambda module: module.empty((2,), out=None, requires_grad=True),
            ),
            (
                "layout and pin defaults",
                lambda module: module.empty(
                    (2,),
                    out=None,
                    layout=module.strided,
                    pin_memory=False,
                ),
            ),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                actual_peer = factory(torch)
                expected = factory(reference_torch)
                expected_peer = factory(reference_torch)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )
                self.assertEqual(
                    actual.is_set_to(actual_peer),
                    expected.is_set_to(expected_peer),
                )
                self.assertEqual(
                    actual.data_ptr() == actual_peer.data_ptr(),
                    expected.data_ptr() == expected_peer.data_ptr(),
                )

    def test_dtype_device_layout_pin_and_requires_grad_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "layout": module.strided,
                "device": module.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            with self.subTest(options=actual_options):
                with torch.no_grad():
                    actual = torch.empty((2, 3), **actual_options)
                with reference_torch.no_grad():
                    expected = reference_torch.empty((2, 3), **expected_options)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_negative_dimensions_match_pytorch_2_13(self):
        cases = (
            ("positional scalar", lambda module: module.empty(-1)),
            (
                "positional index scalar",
                lambda module: module.empty(IndexDimension(-1)),
            ),
            ("tuple dimension", lambda module: module.empty((-1,))),
            ("list dimension", lambda module: module.empty([1, -2])),
            ("size keyword dimension", lambda module: module.empty(size=(-1,))),
        )
        for case, create in cases:
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda create=create: create(torch),
                    lambda create=create: create(reference_torch),
                )

    def test_unsupported_boundaries_are_pinned(self):
        self.assertFalse(hasattr(torch, "empty_like"))
        self.assertTrue(hasattr(reference_torch, "empty_like"))

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=torch.empty((1,)))
        out = reference_torch.empty((1,), dtype=reference_torch.float32)
        self.assertIs(reference_torch.empty((1,), out=out), out)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.empty((1,), pin_memory=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")
        self.assertEqual(str(reference_torch.empty((1,), device="meta").device), "meta")

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.empty((1,), dtype=reference_torch.float64)
        self.assertIs(
            reference_torch.empty((1,), dtype=reference_torch.float64).dtype,
            reference_torch.float64,
        )

        for layout in (object(), reference_torch.strided, reference_torch.sparse_coo):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.empty((1,), layout=layout)
        self.assertIs(
            reference_torch.empty((1,), layout=reference_torch.strided).layout,
            reference_torch.strided,
        )

    def test_callable_import_and_wildcard_exports_match_pytorch_2_13(self):
        def contract(module):
            function = module.empty
            import_namespace = {}
            wildcard_namespace = {}
            exec(
                f"from {module.__name__} import empty as imported_empty",
                import_namespace,
            )
            exec(f"from {module.__name__} import *", wildcard_namespace)
            return {
                "callable": callable(function),
                "type": type(function).__name__,
                "name": function.__name__,
                "all_count": module.__all__.count("empty"),
                "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
                "import_identity": import_namespace["imported_empty"] is function,
                "wildcard_identity": wildcard_namespace["empty"] is function,
                "copy_identity": copy.copy(function) is function,
                "deepcopy_identity": copy.deepcopy(function) is function,
                "pickle_identities": tuple(
                    pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                    for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                ),
            }

        self.assertEqual(contract(torch), contract(reference_torch))


if __name__ == "__main__":
    unittest.main()
