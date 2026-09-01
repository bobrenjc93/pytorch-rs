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

    def test_positional_integer_sizes_match_tuple_size(self):
        metadata = (
            {},
            {"out": None},
            {"dtype": torch.float32},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {
                "out": None,
                "dtype": torch.float32,
                "device": torch.device("cpu"),
                "requires_grad": True,
            },
        )
        for size in ((2,), (2, 3), (2, 3, 0)):
            for keywords in metadata:
                with self.subTest(size=size, keywords=keywords):
                    self.assert_tensor_matches(
                        torch.zeros(*size, **keywords),
                        torch.zeros(size, **keywords),
                    )

    def test_variadic_requires_grad_is_explicit_under_no_grad(self):
        with torch.no_grad():
            default = torch.zeros(2, 3)
            leaf = torch.zeros(2, 3, requires_grad=True)
        self.assertFalse(default.requires_grad)
        self.assertTrue(leaf.requires_grad)
        self.assertTrue(leaf.is_leaf)
        self.assert_tensor_matches(
            leaf,
            torch.zeros((2, 3), requires_grad=True),
        )

    def test_out_none_uses_default_fresh_allocation(self):
        cases = (
            ("scalar", lambda keywords: torch.zeros(2, **keywords)),
            ("variadic", lambda keywords: torch.zeros(2, 3, 0, **keywords)),
            ("tuple", lambda keywords: torch.zeros((2, 3), **keywords)),
            ("size keyword", lambda keywords: torch.zeros(size=(2,), **keywords)),
            (
                "shape alias",
                lambda keywords: torch.zeros(None, shape=(2,), **keywords),
            ),
            (
                "requires grad",
                lambda keywords: torch.zeros((2,), requires_grad=True, **keywords),
            ),
            ("empty", lambda keywords: torch.zeros((0,), **keywords)),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                baseline = factory({})
                with_out_none = factory({"out": None})
                self.assert_tensor_matches(with_out_none, baseline)
                self.assertFalse(with_out_none.is_set_to(baseline))

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
        self.assert_tensor_matches(
            torch.zeros(IntSubclass(2), np.int64(3), custom),
            torch.zeros((2, 3, 2)),
        )
        self.assertGreater(custom.calls, 0)

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

        variadic_empty = torch.zeros(2, 3, 0)
        self.assertEqual(variadic_empty.shape, (2, 3, 0))
        self.assertEqual(variadic_empty.stride(), (3, 1, 1))
        self.assertEqual(variadic_empty.numel(), 0)
        self.assertEqual(variadic_empty.tolist(), [[[], [], []], [[], [], []]])

        for dimension in (-1, IndexDimension(-1)):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape("zeros: Dimension size must be non-negative."),
                ):
                    torch.zeros(dimension)

        for call in (
            lambda: torch.zeros(2, -1),
            lambda: torch.zeros(2, IndexDimension(-1)),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape("zeros: Dimension size must be non-negative."),
                ):
                    call()

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

        for call in (
            lambda: torch.zeros(2, True),
            lambda: torch.zeros(2, np.bool_(True)),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

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
            TypeError,
            "pos 2.*Overflow when unpacking long long",
        ):
            torch.zeros(2, 2**63)

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

        for call in (lambda: torch.zeros(size=2),):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_out_tensor_layout_and_pin_memory_remain_unsupported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("zeros(): the 'out' argument is not supported"),
        ):
            torch.zeros(2, 3, out=torch.zeros((2, 3)))

        for call, message in (
            (
                lambda: torch.zeros(2, 3, size=(2, 3)),
                "zeros() got multiple values for argument 'size'",
            ),
            (
                lambda: torch.zeros(2, 3, shape=(2, 3)),
                "zeros() received both 'size' and its compatibility alias 'shape'",
            ),
            (
                lambda: torch.zeros(2, 3, dtype=object()),
                "zeros(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.zeros(2, 3, device=object()),
                "zeros(): argument 'device' must be torch.device or str, not object",
            ),
            (
                lambda: torch.zeros(2, 3, layout=torch.strided, out=None),
                "zeros() got an unexpected keyword argument 'layout'",
            ),
            (
                lambda: torch.zeros(2, 3, pin_memory=False, out=None),
                "zeros() got an unexpected keyword argument 'pin_memory'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
