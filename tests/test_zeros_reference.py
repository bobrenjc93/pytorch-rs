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
class ZerosReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("zeros differentials require pinned PyTorch 2.13.0")

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
            lambda module: {"dtype": module.float32},
            lambda module: {"device": "cpu"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {
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
                    actual = torch.zeros(actual_dimension, **actual_keywords)
                    expected = reference_torch.zeros(
                        expected_dimension, **expected_keywords
                    )
                    self.assertEqual(
                        self.tensor_observation(torch, actual),
                        self.tensor_observation(reference_torch, expected),
                    )

    def test_dimension_errors_match_pytorch_2_13(self):
        exact_cases = (
            -1,
            IndexDimension(-1),
            True,
            False,
            np.bool_(True),
            sys.maxsize,
            IndexDimension(sys.maxsize),
        )
        for dimension in exact_cases:
            with self.subTest(dimension=dimension):
                actual_type, actual_message = self.capture_error(
                    lambda dimension=dimension: torch.zeros(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.zeros(dimension)
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
                    lambda dimension=dimension: torch.zeros(dimension)
                )
                expected_type, expected_message = self.capture_error(
                    lambda dimension=dimension: reference_torch.zeros(dimension)
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
                lambda module: module.zeros(-1, dtype=object()),
            ),
            (
                "overflow and invalid dtype",
                lambda module: module.zeros(2**63, dtype=object()),
            ),
            (
                "negative and invalid device",
                lambda module: module.zeros(-1, device=object()),
            ),
            (
                "overflow and invalid device",
                lambda module: module.zeros(2**63, device=object()),
            ),
            (
                "negative and invalid requires_grad",
                lambda module: module.zeros(-1, requires_grad=1),
            ),
            (
                "index negative and invalid requires_grad",
                lambda module: module.zeros(IndexDimension(-1), requires_grad=1),
            ),
            (
                "overflow and invalid requires_grad",
                lambda module: module.zeros(2**63, requires_grad=1),
            ),
            (
                "negative and duplicate size",
                lambda module: module.zeros(-1, size=(2,)),
            ),
            (
                "overflow and duplicate size",
                lambda module: module.zeros(2**63, size=(2,)),
            ),
            (
                "negative and unknown keyword",
                lambda module: module.zeros(-1, unexpected=True),
            ),
            (
                "index overflow and unknown keyword",
                lambda module: module.zeros(IndexDimension(2**63), unexpected=True),
            ),
            (
                "negative before device resolution",
                lambda module: module.zeros(-1, device="cuda"),
            ),
            (
                "boolean type before requires_grad",
                lambda module: module.zeros(True, requires_grad=1),
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


if __name__ == "__main__":
    unittest.main()
