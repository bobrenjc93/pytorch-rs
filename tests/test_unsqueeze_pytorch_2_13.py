import platform
import random
import sys
import unittest

import numpy as np
import torch_rs


PYTORCH_UNAVAILABLE = sys.platform == "darwin" and platform.machine() == "x86_64"
if PYTORCH_UNAVAILABLE:
    pytorch = None
else:
    import torch as pytorch


@unittest.skipIf(
    PYTORCH_UNAVAILABLE,
    "PyTorch 2.13.0 does not publish macOS x86_64 distributions",
)
class UnsqueezePytorch213DifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if pytorch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "unsqueeze differential expectations are pinned to PyTorch 2.13.0"
            )

    def assert_same_metadata(self, actual, expected):
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(str(actual.dtype), str(expected.dtype))
        self.assertEqual(str(actual.device), str(expected.device))

    def assert_same_result(self, actual, expected):
        self.assert_same_metadata(actual, expected)
        np.testing.assert_array_equal(np.asarray(actual), expected.numpy())

    def test_seeded_ranks_shapes_dimensions_and_entry_points(self):
        generator = random.Random(0x213_055)
        for rank in range(7):
            shapes = [tuple(generator.randrange(0, 4) for _ in range(rank)) for _ in range(8)]
            shapes.extend([(2,) * rank, (0,) + (1,) * (rank - 1)] if rank else [()])
            for shape in shapes:
                count = int(np.prod(shape, dtype=np.int64)) if shape else 1
                values = [generator.uniform(-10.0, 10.0) for _ in range(count)]
                if count == 0:
                    actual_source = torch_rs.zeros(shape)
                    expected_source = pytorch.zeros(shape, dtype=pytorch.float32)
                else:
                    nested = np.asarray(values, dtype=np.float32).reshape(shape).tolist()
                    actual_source = torch_rs.tensor(nested)
                    expected_source = pytorch.tensor(nested, dtype=pytorch.float32)

                for axis in range(rank + 1):
                    for dimension in (axis, axis - rank - 1):
                        for actual_operation, expected_operation in (
                            (actual_source.unsqueeze, expected_source.unsqueeze),
                            (
                                lambda dim: torch_rs.unsqueeze(actual_source, dim),
                                lambda dim: pytorch.unsqueeze(expected_source, dim),
                            ),
                        ):
                            with self.subTest(shape=shape, dimension=dimension):
                                self.assert_same_result(
                                    actual_operation(dimension),
                                    expected_operation(dimension),
                                )

    def test_indexed_offsets_empty_strides_and_extreme_metadata(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist()
        actual = torch_rs.tensor(values)[1, 2]
        expected = pytorch.tensor(values)[1, 2]
        for dimension in range(-3, 3):
            self.assert_same_result(
                actual.unsqueeze(dimension), expected.unsqueeze(dimension)
            )

        for shape in ((2, 0, 3), (0, 1), (1, 0), (0, sys.maxsize, 3)):
            actual = torch_rs.zeros((0,)).reshape(shape)
            expected = pytorch.zeros((0,)).reshape(shape)
            for dimension in range(len(shape) + 1):
                with self.subTest(shape=shape, dimension=dimension):
                    self.assert_same_metadata(
                        actual.unsqueeze(dimension), expected.unsqueeze(dimension)
                    )

        actual = torch_rs.zeros((0,)).reshape((1 << 62, 0, 2))
        expected = pytorch.zeros((0,)).reshape((1 << 62, 0, 2))
        with self.assertRaisesRegex(
            RuntimeError, "SymIntArrayRef expected to contain only concrete integers"
        ):
            actual.unsqueeze(0)
        with self.assertRaisesRegex(
            RuntimeError, "SymIntArrayRef expected to contain only concrete integers"
        ):
            expected.unsqueeze(0)

    def test_seeded_error_categories_and_messages(self):
        actual = torch_rs.tensor([1.0])
        expected = pytorch.tensor([1.0])

        for input_alias in ("a", "x"):
            for dimension_alias in ("dim", "axis"):
                with self.subTest(
                    input_alias=input_alias, dimension_alias=dimension_alias
                ):
                    actual_result = torch_rs.unsqueeze(
                        **{input_alias: actual, dimension_alias: 0}
                    )
                    expected_result = pytorch.unsqueeze(
                        **{input_alias: expected, dimension_alias: 0}
                    )
                    self.assert_same_result(actual_result, expected_result)

        class IntSubclass(int):
            pass

        cases = (
            IntSubclass(0),
            np.int8(-1),
            np.int64(1),
            True,
            False,
            0.0,
            None,
            1 << 100,
            -(1 << 100),
            -3,
            2,
        )
        for dimension in cases:
            for actual_operation, expected_operation in (
                (actual.unsqueeze, expected.unsqueeze),
                (
                    lambda dim: torch_rs.unsqueeze(actual, dim),
                    lambda dim: pytorch.unsqueeze(expected, dim),
                ),
            ):
                with self.subTest(dimension=dimension, operation=actual_operation):
                    try:
                        expected_result = expected_operation(dimension)
                    except Exception as expected_error:
                        with self.assertRaises(type(expected_error)) as actual_error:
                            actual_operation(dimension)
                        self.assertEqual(str(actual_error.exception), str(expected_error))
                    else:
                        self.assert_same_result(
                            actual_operation(dimension), expected_result
                        )

        call_pairs = (
            (
                lambda: actual.unsqueeze(1 << 100, dim=0),
                lambda: expected.unsqueeze(1 << 100, dim=0),
            ),
            (
                lambda: actual.unsqueeze(1 << 100, axis=0),
                lambda: expected.unsqueeze(1 << 100, axis=0),
            ),
            (
                lambda: actual.unsqueeze(dim=1 << 100, unexpected=0),
                lambda: expected.unsqueeze(dim=1 << 100, unexpected=0),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, 1 << 100, dim=0),
                lambda: pytorch.unsqueeze(expected, 1 << 100, dim=0),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, 1 << 100, axis=0),
                lambda: pytorch.unsqueeze(expected, 1 << 100, axis=0),
            ),
            (
                lambda: torch_rs.unsqueeze(
                    actual, np.uint64(2**64 - 1), input=actual
                ),
                lambda: pytorch.unsqueeze(
                    expected, np.uint64(2**64 - 1), input=expected
                ),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, input=actual),
                lambda: pytorch.unsqueeze(expected, input=expected),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, input=actual, dim=True),
                lambda: pytorch.unsqueeze(expected, input=expected, dim=True),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, input=actual, axis=True),
                lambda: pytorch.unsqueeze(expected, input=expected, axis=True),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, input=actual, dim=0),
                lambda: pytorch.unsqueeze(expected, input=expected, dim=0),
            ),
            (
                lambda: actual.unsqueeze(True, dim=0),
                lambda: expected.unsqueeze(True, dim=0),
            ),
            (
                lambda: actual.unsqueeze(dim=True, unexpected=0),
                lambda: expected.unsqueeze(dim=True, unexpected=0),
            ),
            (
                lambda: actual.unsqueeze(True, 0),
                lambda: expected.unsqueeze(True, 0),
            ),
            (
                lambda: torch_rs.unsqueeze([]),
                lambda: pytorch.unsqueeze([]),
            ),
            (
                lambda: torch_rs.unsqueeze([], True),
                lambda: pytorch.unsqueeze([], True),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, True, dim=0),
                lambda: pytorch.unsqueeze(expected, True, dim=0),
            ),
            (
                lambda: torch_rs.unsqueeze([], 0, dim=0),
                lambda: pytorch.unsqueeze([], 0, dim=0),
            ),
            (
                lambda: torch_rs.unsqueeze([], True, 0),
                lambda: pytorch.unsqueeze([], True, 0),
            ),
            (
                lambda: torch_rs.unsqueeze(input=[], dim=1 << 100),
                lambda: pytorch.unsqueeze(input=[], dim=1 << 100),
            ),
            (
                lambda: actual.unsqueeze(axis=0, unexpected=True),
                lambda: expected.unsqueeze(axis=0, unexpected=True),
            ),
            (
                lambda: torch_rs.unsqueeze(input=actual, axis=0, unexpected=True),
                lambda: pytorch.unsqueeze(input=expected, axis=0, unexpected=True),
            ),
            (
                lambda: torch_rs.unsqueeze(np.array([1.0]), 0),
                lambda: pytorch.unsqueeze(np.array([1.0]), 0),
            ),
            (
                lambda: actual.unsqueeze(np.array(0)),
                lambda: expected.unsqueeze(np.array(0)),
            ),
            (
                lambda: torch_rs.unsqueeze(actual, np.float32(0)),
                lambda: pytorch.unsqueeze(expected, np.float32(0)),
            ),
        )
        for actual_call, expected_call in call_pairs:
            with self.subTest(call=actual_call):
                with self.assertRaises(Exception) as expected_error:
                    expected_call()
                with self.assertRaises(type(expected_error.exception)) as actual_error:
                    actual_call()
                self.assertEqual(
                    str(actual_error.exception), str(expected_error.exception)
                )


if __name__ == "__main__":
    unittest.main()
