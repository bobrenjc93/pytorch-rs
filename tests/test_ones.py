import re
import sys
import unittest
from collections import UserList
from collections.abc import Sequence

import numpy as np
import torch_rs as torch


class OnesTests(unittest.TestCase):
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
            {"dtype": None, "device": None, "requires_grad": False},
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
                    torch.ones(2, **keywords),
                    torch.ones((2,), **keywords),
                )

    def test_two_positional_integers_match_pair_size(self):
        class IntSubclass(int):
            pass

        class IndexDimension:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return self.value

        custom = IndexDimension(3)
        dimension_pairs = (
            (2, 3),
            (2, 0),
            (0, 3),
            (IntSubclass(2), np.int64(3)),
            (np.uint32(2), custom),
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

        for dimensions in dimension_pairs:
            for keywords in metadata:
                with self.subTest(dimensions=dimensions, keywords=keywords):
                    self.assert_tensor_matches(
                        torch.ones(*dimensions, **keywords),
                        torch.ones(tuple(map(int, dimensions)), **keywords),
                    )
        self.assertGreater(custom.calls, 0)

    def test_two_positional_calls_return_fresh_storage(self):
        first = torch.ones(2, 3)
        second = torch.ones(2, 3)

        self.assertFalse(first.is_set_to(second))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

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
                    torch.ones(dimension),
                    torch.ones((2,)),
                )
        self.assertGreater(custom.calls, 0)

    def test_zero_negative_boolean_and_overflowing_dimensions(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        empty = torch.ones(0)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.stride(), (1,))
        self.assertEqual(empty.numel(), 0)
        self.assertEqual(empty.tolist(), [])

        for dimension in (-1, IndexDimension(-1)):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape(
                        "Trying to create tensor with negative dimension -1: [-1]"
                    ),
                ):
                    torch.ones(dimension)

        for dimensions, message in (
            (
                (-1, 3),
                "Trying to create tensor with negative dimension -1: [-1, 3]",
            ),
            (
                (2, -3),
                "Trying to create tensor with negative dimension -3: [2, -3]",
            ),
            (
                (IndexDimension(-1), 3),
                "Trying to create tensor with negative dimension -1: [-1, 3]",
            ),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(RuntimeError, re.escape(message)):
                    torch.ones(*dimensions)

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
                    torch.ones(dimension)

        for dimensions in (
            (True, 3),
            (np.bool_(True), 3),
            ((2,), 3),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    re.escape("ones() takes 1 positional argument but 2 were given"),
                ):
                    torch.ones(*dimensions)

        for dimensions, position in (
            ((2, np.bool_(True)), 2),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"pos {position}.*type must be tuple of ints",
                ):
                    torch.ones(*dimensions)

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
                    torch.ones(dimension)

        for dimensions, position in (
            ((2**63, 3), 1),
            ((2, 2**63), 2),
            ((-1, 2**63), 2),
            ((IndexDimension(2**63), -1), 1),
        ):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"pos {position}.*Overflow when unpacking long long",
                ):
                    torch.ones(*dimensions)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}]"
            ),
        ):
            torch.ones(sys.maxsize)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}, {sys.maxsize}]"
            ),
        ):
            torch.ones(sys.maxsize, sys.maxsize)

    def test_invalid_second_positional_dimension_waits_for_keyword_validation(self):
        cases = (
            (
                lambda: torch.ones(2, object(), dtype=object()),
                r"ones\(\): argument 'dtype' must be torch\.dtype, not object",
            ),
            (
                lambda: torch.ones(2, object(), device=object()),
                r"ones\(\): argument 'device' must be torch\.device.*not object",
            ),
            (
                lambda: torch.ones(2, object(), requires_grad=1),
                r"ones\(\): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.ones(2, object(), size=(4,)),
                r"ones\(\) got multiple values for argument 'size'",
            ),
            (
                lambda: torch.ones(2, object(), unexpected=True),
                r"ones\(\) got an unexpected keyword argument 'unexpected'",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
                    call()

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
                tensor = torch.ones(size)
                self.assertEqual(tensor.shape, expected_shape)
                self.assertEqual(tensor.numel(), int(np.prod(expected_shape)))
                self.assertEqual(tensor.sum().item(), float(tensor.numel()))

        self.assertEqual(torch.ones(size=(2,)).tolist(), [1.0, 1.0])
        self.assertEqual(torch.ones(shape=(2,)).tolist(), [1.0, 1.0])
        self.assertEqual(torch.ones(size=np.array([2])).shape, (2,))
        self.assertEqual(torch.ones(shape=UserList([2])).shape, (2,))
        self.assertEqual(torch.ones(None, shape=(2,)).tolist(), [1.0, 1.0])

        with self.assertRaises(Exception) as direct_shape_error:
            torch.ones(shape=2)
        with self.assertRaises(Exception) as omitted_positional_error:
            torch.ones(None, shape=2)
        self.assertIs(
            type(direct_shape_error.exception),
            type(omitted_positional_error.exception),
        )
        self.assertEqual(
            str(direct_shape_error.exception),
            str(omitted_positional_error.exception),
        )

        for call in (
            lambda: torch.ones(size=2),
            lambda: torch.ones(2, 3, 4),
            lambda: torch.ones(2, 3, layout=torch.strided),
            lambda: torch.ones(2, 3, out=torch.ones((2, 3))),
            lambda: torch.ones(2, 3, pin_memory=False),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()


if __name__ == "__main__":
    unittest.main()
