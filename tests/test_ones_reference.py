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

    def __index__(self):
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class OnesReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("ones differentials require pinned PyTorch 2.13.0")

    def tensor_observation(self, module, tensor):
        return (
            tuple(tensor.shape),
            tensor.stride(),
            tensor.tolist(),
            str(tensor.dtype),
            tensor.dtype is module.float32,
            str(tensor.device),
            tensor.requires_grad,
            tensor.is_leaf,
        )

    def capture_error(self, call):
        with self.assertRaises(Exception) as raised:
            call()
        return type(raised.exception), str(raised.exception)

    def test_scalar_results_and_metadata_match_pytorch_2_13(self):
        dimension_factories = (
            lambda: 2,
            lambda: 0,
            lambda: IntSubclass(2),
            lambda: np.int64(2),
            lambda: np.uint32(2),
            lambda: IndexDimension(2),
        )
        metadata_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {
                "out": None,
                "dtype": module.float32,
                "device": module.device("cpu"),
                "requires_grad": True,
            },
        )

        for dimension_factory in dimension_factories:
            for metadata_factory in metadata_factories:
                actual_dimension = dimension_factory()
                expected_dimension = dimension_factory()
                actual_keywords = metadata_factory(torch)
                expected_keywords = metadata_factory(reference_torch)
                with self.subTest(
                    dimension=actual_dimension,
                    keywords=actual_keywords,
                ):
                    actual = torch.ones(actual_dimension, **actual_keywords)
                    expected = reference_torch.ones(
                        expected_dimension, **expected_keywords
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )

    def test_variadic_results_and_storage_match_pytorch_2_13(self):
        cases = (
            ("matrix", lambda module: module.ones(2, 3)),
            ("zero size", lambda module: module.ones(2, 3, 0)),
            (
                "integer protocol",
                lambda module: module.ones(
                    IntSubclass(2), np.int64(3), IndexDimension(0)
                ),
            ),
            (
                "requires grad",
                lambda module: module.ones(2, 3, requires_grad=True),
            ),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                actual_peer = factory(torch)
                expected = factory(reference_torch)
                expected_peer = factory(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
                )
                self.assertEqual(
                    actual.is_set_to(actual_peer),
                    expected.is_set_to(expected_peer),
                )
                self.assertEqual(
                    actual.data_ptr() == actual_peer.data_ptr(),
                    expected.data_ptr() == expected_peer.data_ptr(),
                )

    def test_variadic_no_grad_matches_pytorch_2_13(self):
        with torch.no_grad():
            actual_default = torch.ones(2, 3)
            actual_requires_grad = torch.ones(2, 3, requires_grad=True)
        with reference_torch.no_grad():
            expected_default = reference_torch.ones(2, 3)
            expected_requires_grad = reference_torch.ones(2, 3, requires_grad=True)

        self.assertEqual(
            self.tensor_observation(torch, actual_default),
            self.tensor_observation(reference_torch, expected_default),
        )
        self.assertEqual(
            self.tensor_observation(torch, actual_requires_grad),
            self.tensor_observation(reference_torch, expected_requires_grad),
        )

    def test_out_none_results_and_storage_freshness_match_pytorch_2_13(self):
        cases = (
            ("scalar", lambda module: module.ones(2, out=None)),
            ("tuple", lambda module: module.ones((2, 3), out=None)),
            ("variadic", lambda module: module.ones(2, 3, 0, out=None)),
            ("size keyword", lambda module: module.ones(size=(2,), out=None)),
            (
                "requires grad",
                lambda module: module.ones((2,), out=None, requires_grad=True),
            ),
            ("empty", lambda module: module.ones((0,), out=None)),
            ("scalar tensor", lambda module: module.ones((), out=None)),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                actual = factory(torch)
                actual_peer = factory(torch)
                expected = factory(reference_torch)
                expected_peer = factory(reference_torch)
                self.assertEqual(
                    self.tensor_observation(torch, actual),
                    self.tensor_observation(reference_torch, expected),
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
                    lambda dimension=dimension: torch.ones(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.ones(dimension)
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
                    lambda dimension=dimension: torch.ones(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.ones(dimension)
                )
                self.assertIs(actual_type, expected_type)
                marker = "failed to unpack the object at pos 1 with error"
                self.assertIn(marker, actual_message)
                self.assertIn(marker, expected_message)
                self.assertIn("Overflow when unpacking long long", actual_message)
                self.assertIn("Overflow when unpacking long long", expected_message)

    def test_mixed_invalid_scalar_validation_order_matches_pytorch_2_13(self):
        cases = (
            (
                "negative and invalid dtype",
                lambda module: module.ones(-1, dtype=object()),
            ),
            (
                "overflow and invalid dtype",
                lambda module: module.ones(2**63, dtype=object()),
            ),
            (
                "negative and invalid device",
                lambda module: module.ones(-1, device=object()),
            ),
            (
                "overflow and invalid device",
                lambda module: module.ones(2**63, device=object()),
            ),
            (
                "negative and invalid requires_grad",
                lambda module: module.ones(-1, requires_grad=1),
            ),
            (
                "index negative and invalid requires_grad",
                lambda module: module.ones(IndexDimension(-1), requires_grad=1),
            ),
            (
                "overflow and invalid requires_grad",
                lambda module: module.ones(2**63, requires_grad=1),
            ),
            (
                "negative and duplicate size",
                lambda module: module.ones(-1, size=(2,)),
            ),
            (
                "overflow and duplicate size",
                lambda module: module.ones(2**63, size=(2,)),
            ),
            (
                "negative and unknown keyword",
                lambda module: module.ones(-1, unexpected=True),
            ),
            (
                "index overflow and unknown keyword",
                lambda module: module.ones(
                    IndexDimension(2**63), unexpected=True
                ),
            ),
            (
                "boolean type before requires_grad",
                lambda module: module.ones(True, requires_grad=1),
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
            ("missing size", lambda module: module.ones(out=[])),
            ("negative size", lambda module: module.ones(-1, out=[])),
            ("invalid dtype", lambda module: module.ones(2, dtype=object(), out=[])),
            ("unknown keyword", lambda module: module.ones(2, unexpected=True, out=[])),
            ("duplicate size", lambda module: module.ones(2, size=(2,), out=[])),
            ("bool dimension", lambda module: module.ones(True, out=[])),
        )
        for case, call in cases:
            with self.subTest(case=case):
                actual_type, actual_message = self.capture_error(lambda: call(torch))
                expected_type, expected_message = self.capture_error(
                    lambda: call(reference_torch)
                )
                self.assertIs(actual_type, expected_type)
                self.assertEqual(actual_message, expected_message)

    @unittest.skipUnless(
        reference_torch is not None and reference_torch.cuda.is_available(),
        "requires a CUDA-capable PyTorch runtime",
    )
    def test_negative_size_is_validated_before_cuda_device_resolution(self):
        actual_type, actual_message = self.capture_error(
            lambda: torch.ones(-1, device="cuda")
        )
        expected_type, expected_message = self.capture_error(
            lambda: reference_torch.ones(-1, device="cuda")
        )
        self.assertIs(actual_type, expected_type)
        self.assertEqual(actual_message, expected_message)


if __name__ == "__main__":
    unittest.main()
