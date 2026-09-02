import copy
import math
import pickle
import sys
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
class FullReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("full differentials require pinned PyTorch 2.13.0")

    def float32_value_contract(self, tensor):
        source = tensor.detach() if tensor.requires_grad else tensor
        values = np.asarray(source, dtype=np.float32).reshape(-1)
        bits = values.view(np.uint32)
        result = []
        for value, raw_bits in zip(values, bits, strict=True):
            if math.isnan(float(value)):
                result.append(("nan", bool(np.signbit(value))))
            else:
                result.append(("bits", int(raw_bits)))
        return tuple(result)

    def tensor_contract(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": self.float32_value_contract(tensor),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is module.strided,
            "is_pinned": tensor.is_pinned(),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
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

    def test_shape_values_and_keyword_forms_match_pytorch_2_13(self):
        cases = (
            ("scalar tuple", lambda module: module.full((), -2.5)),
            ("scalar list", lambda module: module.full([], -2.5)),
            ("empty middle", lambda module: module.full([2, 0, 3], 7.0)),
            (
                "multidimensional",
                lambda module: module.full((2, 3), 1.25),
            ),
            (
                "keyword",
                lambda module: module.full(size=[2], fill_value=3.0),
            ),
            (
                "explicit float alias",
                lambda module: module.full((2,), 3, dtype=module.float),
            ),
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
        actual = torch.full(
            [actual_dynamic, np.int64(3), IntSubclass(1)],
            np.float32(1.5),
        )
        expected = reference_torch.full(
            [expected_dynamic, np.int64(3), IntSubclass(1)],
            np.float32(1.5),
        )

        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )
        self.assertEqual(actual_dynamic.calls, 1)
        self.assertGreaterEqual(expected_dynamic.calls, 1)

    def test_out_none_results_and_storage_freshness_match_pytorch_2_13(self):
        cases = (
            ("scalar tuple", lambda module: module.full((), -2.5, out=None)),
            ("scalar list", lambda module: module.full([], -2.5, out=None)),
            (
                "empty middle",
                lambda module: module.full([2, 0, 3], 7.0, out=None),
            ),
            (
                "multidimensional",
                lambda module: module.full((2, 3), 1.25, out=None),
            ),
            (
                "size keyword",
                lambda module: module.full(size=[2], fill_value=3.0, out=None),
            ),
            (
                "requires grad",
                lambda module: module.full(
                    (2,),
                    4.0,
                    out=None,
                    dtype=module.float32,
                    device=module.device("cpu"),
                    requires_grad=True,
                ),
            ),
            (
                "integer protocol dimensions",
                lambda module: module.full(
                    [IndexDimension(2), np.int64(0), IntSubclass(3)],
                    np.float32(1.5),
                    out=None,
                ),
            ),
            (
                "layout and pin defaults",
                lambda module: module.full(
                    (2,),
                    -0.0,
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

    def test_nonfinite_and_signed_zero_values_match_pytorch_2_13(self):
        for fill_value in (math.nan, math.inf, -math.inf, 0.0, -0.0):
            with self.subTest(fill_value=repr(fill_value)):
                actual = torch.full((2,), fill_value)
                expected = reference_torch.full((2,), fill_value)
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_dtype_device_and_requires_grad_metadata_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"out": None},
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
                    actual = torch.full((2, 3), 1.25, **actual_options)
                with reference_torch.no_grad():
                    expected = reference_torch.full(
                        (2, 3), 1.25, **expected_options
                    )
                self.assertEqual(
                    self.tensor_contract(torch, actual),
                    self.tensor_contract(reference_torch, expected),
                )

    def test_full_returns_fresh_storage_match_pytorch_2_13(self):
        def contract(module):
            first = module.full((2, 3), -1.25)
            second = module.full((2, 3), -1.25)
            scalar = module.tensor(4.5)
            from_scalar = module.full((2,), scalar)
            return {
                "first": self.tensor_contract(module, first),
                "second": self.tensor_contract(module, second),
                "fresh_pair_storage": first.data_ptr() != second.data_ptr(),
                "fresh_pair_view": not first.is_set_to(second),
                "scalar_fill": self.tensor_contract(module, from_scalar),
                "scalar_fill_fresh": from_scalar.data_ptr() != scalar.data_ptr(),
                "scalar_fill_not_view": not from_scalar.is_set_to(scalar),
            }

        self.assertEqual(contract(torch), contract(reference_torch))

    def test_tensor_fill_value_boundary_matches_pytorch_2_13(self):
        actual = torch.full((2,), torch.tensor(-0.0))
        expected = reference_torch.full((2,), reference_torch.tensor(-0.0))
        self.assertEqual(
            self.tensor_contract(torch, actual),
            self.tensor_contract(reference_torch, expected),
        )

        error_cases = (
            (
                lambda: torch.full((2,), torch.tensor(2.0, requires_grad=True)),
                lambda: reference_torch.full(
                    (2,),
                    reference_torch.tensor(2.0, requires_grad=True),
                ),
            ),
            (
                lambda: torch.full((2,), torch.tensor([2.0])),
                lambda: reference_torch.full(
                    (2,), reference_torch.tensor([2.0])
                ),
            ),
        )
        for actual_call, expected_call in error_cases:
            with self.subTest(actual_call=actual_call):
                self.assert_error_matches(actual_call, expected_call)

    def test_negative_and_overflow_errors_match_pytorch_2_13(self):
        exact_cases = (
            lambda module: module.full((-1,), 3.0),
            lambda module: module.full((1, -2), 3.0),
            lambda module: module.full((2,), 1e40),
            lambda module: module.full((2,), -1e40),
        )
        for call in exact_cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch), lambda: call(reference_torch)
                )

        overflow_cases = (
            lambda module: module.full((2**63,), 3.0),
            lambda module: module.full((np.uint64(2**63),), 3.0),
            lambda module: module.full((IndexDimension(2**63),), 3.0),
        )
        for call in overflow_cases:
            with self.subTest(call=call):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertIn("size element at index 0 is invalid", actual_message)
                self.assertIn("failed to unpack the object at pos 1", expected_message)
                self.assertIn("Overflow", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        storage_cases = (
            lambda module: module.full((2**62, 4), 1.0),
            lambda module: module.full((sys.maxsize // 4 + 1,), 1.0),
        )
        for call in storage_cases:
            with self.subTest(call=call):
                self.assert_error_matches(
                    lambda: call(torch), lambda: call(reference_torch)
                )

    def test_out_none_error_order_matches_pytorch_2_13(self):
        cases = (
            ("missing size", lambda module: module.full(out=None)),
            ("missing fill", lambda module: module.full((1,), out=None)),
            ("negative size", lambda module: module.full((-1,), 3.0, out=None)),
            (
                "invalid dtype",
                lambda module: module.full((1,), 3.0, dtype=object(), out=None),
            ),
            (
                "invalid device",
                lambda module: module.full((1,), 3.0, device=object(), out=None),
            ),
            (
                "unknown keyword",
                lambda module: module.full((1,), 3.0, unexpected=True, out=None),
            ),
            (
                "duplicate size",
                lambda module: module.full((1,), 3.0, size=(1,), out=None),
            ),
            (
                "invalid layout before negative size",
                lambda module: module.full((-1,), 3.0, layout=object(), out=None),
            ),
            (
                "invalid pin before negative size",
                lambda module: module.full((-1,), 3.0, pin_memory=0, out=None),
            ),
            (
                "invalid pin before requires_grad",
                lambda module: module.full(
                    (1,),
                    3.0,
                    pin_memory=0,
                    requires_grad=0,
                    out=None,
                ),
            ),
            (
                "unknown after accepted layout",
                lambda module: module.full(
                    (1,),
                    3.0,
                    layout=module.strided,
                    unexpected=True,
                    out=None,
                ),
            ),
            (
                "duplicate size after accepted layout",
                lambda module: module.full(
                    (1,),
                    3.0,
                    size=(1,),
                    layout=module.strided,
                    out=None,
                ),
            ),
        )
        for case, call in cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(
                    actual_message.replace("torch.device or str", "torch.device"),
                    expected_message,
                )

    def test_out_type_error_order_matches_pytorch_2_13(self):
        cases = (
            ("missing size", lambda module: module.full(out=[])),
            ("missing fill", lambda module: module.full((1,), out=[])),
            ("valid shape", lambda module: module.full((1,), 1.0, out=[])),
            ("negative size", lambda module: module.full((-1,), 1.0, out=[])),
            (
                "invalid dtype",
                lambda module: module.full((1,), 1.0, dtype=object(), out=[]),
            ),
            (
                "unknown keyword",
                lambda module: module.full((1,), 1.0, unexpected=True, out=[]),
            ),
            (
                "duplicate size",
                lambda module: module.full((1,), 1.0, size=(1,), out=[]),
            ),
        )
        for case, call in cases:
            with self.subTest(case=case):
                self.assert_error_matches(
                    lambda: call(torch), lambda: call(reference_torch)
                )

    def test_unsupported_dtype_device_layout_and_out_errors_are_pinned(self):
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^full\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.full((1,), 1.0, dtype=reference_torch.float64)
        with self.assertRaisesRegex(
            TypeError,
            r"^full\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.full((1,), 1.0, out=None, dtype=reference_torch.float64)
        self.assertIs(
            reference_torch.full((1,), 1.0, dtype=reference_torch.float64).dtype,
            reference_torch.float64,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"^full\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.full((1,), 1.0, device="meta")
        with self.assertRaisesRegex(
            RuntimeError,
            r"^full\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.full((1,), 1.0, out=None, device="meta")
        meta = reference_torch.full((1,), 1.0, device="meta")
        self.assertEqual(str(meta.device), "meta")
        self.assertIs(meta.dtype, reference_torch.float32)
        self.assertIs(meta.layout, reference_torch.strided)

        for layout in (object(), reference_torch.strided, reference_torch.sparse_coo):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^full\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.full((1,), 1.0, layout=layout)

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^full\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.full((1,), 1.0, pin_memory=pin_memory)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^full\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.full((1,), 1.0, pin_memory=True)

        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^full\(\): the 'out' argument is not supported$",
        ):
            torch.full(
                (1,),
                1.0,
                out=out,
                layout=torch.strided,
                pin_memory=False,
            )
        self.assertEqual(out.tolist(), [0.0])

        for fill_value in ([1.0], object(), 1 + 2j):
            with self.subTest(fill_value=type(fill_value).__name__):
                with self.assertRaisesRegex(TypeError, "fill_value"):
                    torch.full((1,), fill_value, out=None)

        self.assertIs(
            reference_torch.full(
                (1,), 1.0, layout=reference_torch.strided
            ).layout,
            reference_torch.strided,
        )
        out = reference_torch.empty((1,), dtype=reference_torch.float32)
        self.assertIs(reference_torch.full((1,), 1.0, out=out), out)
        self.assertEqual(out.tolist(), [1.0])
        self.assertFalse(
            reference_torch.full((1,), 1.0, pin_memory=False).is_pinned()
        )

    def test_callable_import_and_wildcard_exports_match_pytorch_2_13(self):
        def contract(module):
            function = module.full
            import_namespace = {}
            wildcard_namespace = {}
            exec(
                f"from {module.__name__} import full as imported_full",
                import_namespace,
            )
            exec(f"from {module.__name__} import *", wildcard_namespace)
            return {
                "callable": callable(function),
                "type": type(function).__name__,
                "name": function.__name__,
                "all_count": module.__all__.count("full"),
                "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
                "import_identity": import_namespace["imported_full"] is function,
                "wildcard_identity": wildcard_namespace["full"] is function,
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
