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


class StatefulIndexDimension:
    def __init__(self, values):
        self.values = values
        self.calls = 0

    def __index__(self):
        value = self.values[min(self.calls, len(self.values) - 1)]
        self.calls += 1
        return value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class EmptyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("empty differentials require pinned PyTorch 2.13.0")

    def tensor_metadata(self, module, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.storage_offset(),
            tensor.numel(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            str(tensor.layout),
            tensor.layout is module.strided,
            tensor.is_pinned(),
            tensor.requires_grad,
            tensor.is_leaf,
            tensor.grad is None,
        )

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def test_scalar_and_empty_dimension_metadata_match_pytorch_2_13(self):
        cases = (
            ("scalar tuple", lambda module, options: module.empty((), **options)),
            ("scalar list", lambda module, options: module.empty([], **options)),
            ("one dimensional empty", lambda module, options: module.empty(0, **options)),
            (
                "multidimensional empty",
                lambda module, options: module.empty((2, 0, 3), **options),
            ),
        )
        metadata_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"dtype": module.float},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"device": module.device("cpu", 2)},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
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

        for case, factory in cases:
            for metadata_factory in metadata_factories:
                actual_options = metadata_factory(torch)
                expected_options = metadata_factory(reference_torch)
                with self.subTest(case=case, options=actual_options):
                    actual = factory(torch, actual_options)
                    expected = factory(reference_torch, expected_options)
                    self.assertEqual(
                        self.tensor_metadata(torch, actual),
                        self.tensor_metadata(reference_torch, expected),
                    )

    def test_variadic_list_tuple_and_index_protocol_sizes_match_pytorch_2_13(self):
        actual_dynamic = IndexDimension(2)
        expected_dynamic = IndexDimension(2)
        cases = (
            ("single int", lambda module: module.empty(2)),
            ("single index", lambda module: module.empty(IndexDimension(2))),
            ("variadic", lambda module: module.empty(2, 3)),
            ("variadic empty", lambda module: module.empty(2, 0, 3)),
            ("variadic bool false", lambda module: module.empty(2, False)),
            ("variadic bool true", lambda module: module.empty(2, True)),
            ("tuple", lambda module: module.empty((2, 3))),
            ("list", lambda module: module.empty([2, 3])),
            (
                "integer protocol list",
                lambda module: module.empty(
                    [
                        actual_dynamic if module is torch else expected_dynamic,
                        np.int64(3),
                        IntSubclass(1),
                    ]
                ),
            ),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                expected = factory(reference_torch)
                self.assertEqual(
                    self.tensor_metadata(torch, actual),
                    self.tensor_metadata(reference_torch, expected),
                )
        self.assertEqual(actual_dynamic.calls, expected_dynamic.calls)

    def test_variadic_leading_index_provider_calls_match_pytorch_2_13(self):
        actual_dimension = StatefulIndexDimension((2, 3, 4))
        expected_dimension = StatefulIndexDimension((2, 3, 4))

        actual = torch.empty(actual_dimension, 3)
        expected = reference_torch.empty(expected_dimension, 3)

        self.assertEqual(
            self.tensor_metadata(torch, actual),
            self.tensor_metadata(reference_torch, expected),
        )
        self.assertEqual(actual_dimension.calls, expected_dimension.calls)

    def test_requires_grad_leaf_behavior_matches_pytorch_2_13_under_no_grad(self):
        with torch.no_grad():
            actual_default = torch.empty((2, 3))
            actual_tracked = torch.empty((2, 3), requires_grad=True)
        with reference_torch.no_grad():
            expected_default = reference_torch.empty((2, 3))
            expected_tracked = reference_torch.empty((2, 3), requires_grad=True)

        self.assertEqual(
            self.tensor_metadata(torch, actual_default),
            self.tensor_metadata(reference_torch, expected_default),
        )
        self.assertEqual(
            self.tensor_metadata(torch, actual_tracked),
            self.tensor_metadata(reference_torch, expected_tracked),
        )

    def test_out_none_results_and_fresh_storage_match_pytorch_2_13(self):
        cases = (
            ("scalar tensor", lambda module: module.empty((), out=None)),
            ("single dimension", lambda module: module.empty(2, out=None)),
            ("variadic", lambda module: module.empty(2, 3, out=None)),
            ("tuple", lambda module: module.empty((2, 3), out=None)),
            ("list", lambda module: module.empty([2, 3], out=None)),
            ("size keyword", lambda module: module.empty(size=(2,), out=None)),
            (
                "requires grad",
                lambda module: module.empty(2, 3, out=None, requires_grad=True),
            ),
            ("empty dimension", lambda module: module.empty((0,), out=None)),
            (
                "empty middle dimension",
                lambda module: module.empty((2, 0, 3), out=None),
            ),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                actual_peer = factory(torch)
                expected = factory(reference_torch)
                expected_peer = factory(reference_torch)
                self.assertEqual(
                    self.tensor_metadata(torch, actual),
                    self.tensor_metadata(reference_torch, expected),
                )
                self.assertEqual(
                    actual.is_set_to(actual_peer),
                    expected.is_set_to(expected_peer),
                )
                self.assertEqual(
                    actual.data_ptr() == actual_peer.data_ptr(),
                    expected.data_ptr() == expected_peer.data_ptr(),
                )

    def test_dimension_errors_match_pytorch_2_13(self):
        exact_cases = (
            -1,
            IndexDimension(-1),
            True,
            False,
            sys.maxsize,
            IndexDimension(sys.maxsize),
        )
        for dimension in exact_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.empty(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.empty(dimension)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        overflow_cases = (
            2**63,
            -(2**63) - 1,
            np.uint64(2**63),
            IndexDimension(2**63),
        )
        for dimension in overflow_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.empty(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.empty(dimension)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        sequence_exact_cases = (
            (-1,),
            [-1],
            (IndexDimension(-1),),
            [IndexDimension(-1)],
            (2, -1),
            [2, -1],
        )
        for dimensions in sequence_exact_cases:
            with self.subTest(dimensions=dimensions):
                actual_type, actual_message = self.capture_error(
                    lambda dimensions=dimensions: torch.empty(dimensions)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimensions=dimensions: reference_torch.empty(dimensions)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        sequence_overflow_cases = (
            ((2**63, 0), 1),
            ([2**63, 0], 1),
            ((0, 2**63), 2),
            ([0, 2**63], 2),
            ((np.uint64(2**63), 0), 1),
            ([np.uint64(2**63), 0], 1),
            ((IndexDimension(2**63), 0), 1),
            ([IndexDimension(2**63), 0], 1),
        )
        for dimensions, position in sequence_overflow_cases:
            with self.subTest(dimensions=dimensions):
                actual_type, actual_message = self.capture_error(
                    lambda dimensions=dimensions: torch.empty(dimensions)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimensions=dimensions: reference_torch.empty(dimensions)
                )
                self.assertIs(actual_type, expected_type)
                marker = f"failed to unpack the object at pos {position} with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

        variadic_exact_cases = (
            (2, -1),
            (2, IndexDimension(-1)),
            (sys.maxsize, 2),
        )
        for dimensions in variadic_exact_cases:
            with self.subTest(dimensions=dimensions):
                actual_type, actual_message = self.capture_error(
                    lambda dimensions=dimensions: torch.empty(*dimensions)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimensions=dimensions: reference_torch.empty(*dimensions)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

        variadic_overflow_cases = (
            (2, 2**63),
            (2, np.uint64(2**63)),
            (2, IndexDimension(2**63)),
        )
        for dimensions in variadic_overflow_cases:
            with self.subTest(dimensions=dimensions):
                actual_type, actual_message = self.capture_error(
                    lambda dimensions=dimensions: torch.empty(*dimensions)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimensions=dimensions: reference_torch.empty(*dimensions)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 2 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

    def test_unsupported_dtype_device_layout_out_and_pin_memory_boundaries_are_pinned(
        self,
    ):
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'dtype' must be torch\.dtype, not dtype$",
        ):
            torch.empty((1,), dtype=reference_torch.float64)
        self.assertIs(
            reference_torch.empty((1,), dtype=reference_torch.float64).dtype,
            reference_torch.float64,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")
        meta = reference_torch.empty((1,), device="meta")
        self.assertEqual(str(meta.device), "meta")

        for layout in (object(), reference_torch.strided, reference_torch.sparse_coo):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'layout' must be torch\.layout, not ",
                ):
                    torch.empty((1,), layout=layout)

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'out' must be Tensor, not list$",
        ):
            torch.empty((1,), out=[])

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=torch.empty((1,)))

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.empty((1,), pin_memory=pin_memory)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.empty((1,), pin_memory=True)


if __name__ == "__main__":
    unittest.main()
