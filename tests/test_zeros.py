import re
import sys
import unittest
from collections import UserList
from collections.abc import Sequence

import numpy as np
import torch_rs as torch


class ZerosTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)

    def test_one_positional_integer_matches_singleton_size(self):
        metadata = (
            {},
            {"dtype": torch.float32},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "requires_grad": True,
            },
        )
        for keywords in metadata:
            with self.subTest(keywords=keywords):
                self.assert_tensor_matches(
                    torch.zeros(2, **keywords),
                    torch.zeros((2,), **keywords),
                )

    def test_one_positional_dimension_uses_the_index_protocol(self):
        class IntSubclass(int):
            pass

        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        custom = IndexDimension(2)
        dimensions = (IntSubclass(2), np.int64(2), np.uint32(2), custom)
        for dimension in dimensions:
            with self.subTest(dimension=dimension):
                self.assert_tensor_matches(
                    torch.zeros(dimension),
                    torch.zeros((2,)),
                )
        self.assertGreater(custom.calls, 0)

    def test_two_positional_dimensions_use_the_existing_creation_path(self):
        class IntSubclass(int):
            pass

        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        cases = (
            (2, 3),
            (0, 3),
            (2, 0),
            (IntSubclass(2), np.int64(3)),
            (IndexDimension(2), np.uint32(3)),
            (2, True),
            (2, False),
        )
        metadata = (
            {},
            {"dtype": torch.float32},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "requires_grad": True,
            },
        )
        for dimensions in cases:
            for keywords in metadata:
                with self.subTest(dimensions=dimensions, keywords=keywords):
                    self.assert_tensor_matches(
                        torch.zeros(*dimensions, **keywords),
                        torch.zeros(tuple(map(int, dimensions)), **keywords),
                    )

    def test_two_positional_dimension_errors_and_ordering(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        for dimensions in ((-1, 2), (2, -1), (IndexDimension(-1), 2)):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(
                RuntimeError,
                re.escape("zeros: Dimension size must be non-negative."),
            ):
                torch.zeros(*dimensions)

        for first in (True, False, np.bool_(True), 2.0, (2,)):
            with self.subTest(first=first), self.assertRaisesRegex(
                TypeError,
                re.escape("zeros() takes 1 positional argument but 2 were given"),
            ):
                torch.zeros(first, 2)

        with self.assertRaisesRegex(
            TypeError,
            r"failed to unpack the object at pos 2 .*numpy\.bool",
        ):
            torch.zeros(2, np.bool_(True))

        overflow_cases = (
            ((2**63, 2), 1),
            ((2, 2**63), 2),
            ((-1, 2**63), 2),
            ((2, IndexDimension(2**63)), 2),
        )
        for dimensions, position in overflow_cases:
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(
                TypeError,
                rf"failed to unpack the object at pos {position} .*Overflow when unpacking long long",
            ):
                torch.zeros(*dimensions)

        ordered_errors = (
            (
                lambda: torch.zeros(-1, 2, dtype=object()),
                "argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.zeros(2, 2**63, device=object()),
                "argument 'device' must be torch.device or str, not object",
            ),
            (
                lambda: torch.zeros(2, -1, requires_grad=1),
                "argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.zeros(2, -1, size=(2, 3)),
                "zeros() got multiple values for argument 'size'",
            ),
            (
                lambda: torch.zeros(2, 2.0, unexpected=True),
                "zeros() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in ordered_errors:
            with self.subTest(message=message), self.assertRaisesRegex(
                TypeError, re.escape(message)
            ):
                call()

        for dimensions in ((sys.maxsize, 1), (1, sys.maxsize)):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(
                RuntimeError,
                re.escape(
                    f"Storage size calculation overflowed with sizes={list(dimensions)}"
                ),
            ):
                torch.zeros(*dimensions)

    def test_zero_negative_boolean_and_overflowing_dimensions(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        empty = torch.zeros(0)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.stride(), (1,))
        self.assertEqual(empty.numel(), 0)
        self.assertEqual(empty.tolist(), [])

        for dimension in (-1, IndexDimension(-1)):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape("zeros: Dimension size must be non-negative."),
                ):
                    torch.zeros(dimension)

        for dimension, type_name in (
            (True, "bool"),
            (False, "bool"),
            (np.bool_(True), "numpy.bool"),
        ):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"must be tuple of ints, not {re.escape(type_name)}$",
                ):
                    torch.zeros(dimension)

        for dimension in (
            2**63,
            -(2**63) - 1,
            np.uint64(2**63),
            IndexDimension(2**63),
        ):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    TypeError,
                    "failed to unpack.*Overflow when unpacking long long",
                ):
                    torch.zeros(dimension)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}]"
            ),
        ):
            torch.zeros(sys.maxsize)

    def test_existing_sequence_and_keyword_forms_are_unchanged(self):
        class CustomSequence(Sequence):
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

        for size, expected_shape in (
            ((2,), (2,)),
            ([2], (2,)),
            (np.array([2]), (2,)),
            (range(2, 4), (2, 3)),
            (UserList([2]), (2,)),
            (CustomSequence([2]), (2,)),
        ):
            with self.subTest(size=size):
                tensor = torch.zeros(size)
                self.assertEqual(tensor.shape, expected_shape)
                self.assertEqual(tensor.numel(), int(np.prod(expected_shape)))

        self.assertEqual(torch.zeros(size=(2,)).tolist(), [0.0, 0.0])
        self.assertEqual(torch.zeros(shape=(2,)).tolist(), [0.0, 0.0])
        self.assertEqual(torch.zeros(size=np.array([2])).shape, (2,))
        self.assertEqual(torch.zeros(shape=UserList([2])).shape, (2,))
        self.assertEqual(torch.zeros(None, shape=(2,)).tolist(), [0.0, 0.0])

        with self.assertRaises(Exception) as direct_shape_error:
            torch.zeros(shape=2)
        with self.assertRaises(Exception) as omitted_positional_error:
            torch.zeros(None, shape=2)
        self.assertIs(
            type(direct_shape_error.exception),
            type(omitted_positional_error.exception),
        )
        self.assertEqual(
            str(direct_shape_error.exception),
            str(omitted_positional_error.exception),
        )

        for call in (
            lambda: torch.zeros(size=2),
            lambda: torch.zeros(1, 2, 3),
            lambda: torch.zeros(2, 3, layout=None),
            lambda: torch.zeros(2, 3, out=None),
            lambda: torch.zeros(2, 3, pin_memory=False),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()


if __name__ == "__main__":
    unittest.main()
