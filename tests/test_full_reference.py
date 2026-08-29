import math
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
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
            lambda module: {"requires_grad": True},
            lambda module: {
                "dtype": module.float32,
                "device": module.device("cpu"),
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

    def test_unsupported_dtype_device_layout_and_out_errors_are_pinned(self):
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^full\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.full((1,), 1.0, dtype=reference_torch.float64)
        self.assertIs(
            reference_torch.full((1,), 1.0, dtype=reference_torch.float64).dtype,
            reference_torch.float64,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"^full\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.full((1,), 1.0, device="meta")
        meta = reference_torch.full((1,), 1.0, device="meta")
        self.assertEqual(str(meta.device), "meta")
        self.assertIs(meta.dtype, reference_torch.float32)
        self.assertIs(meta.layout, reference_torch.strided)

        for keyword, value in (
            ("layout", torch.strided),
            ("out", None),
            ("pin_memory", False),
        ):
            with self.subTest(keyword=keyword):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"^full\(\) got an unexpected keyword argument '{keyword}'$",
                ):
                    torch.full((1,), 1.0, **{keyword: value})

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


if __name__ == "__main__":
    unittest.main()
