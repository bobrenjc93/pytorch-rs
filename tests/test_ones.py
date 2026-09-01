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
        for keywords in metadata:
            with self.subTest(keywords=keywords):
                self.assert_tensor_matches(
                    torch.ones(2, **keywords),
                    torch.ones((2,), **keywords),
                )

    def test_variadic_integer_dimensions_create_multidimensional_tensors(self):
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
        cases = (
            ("plain", (2, 3), (2, 3)),
            ("zero dimension", (2, 0, 3), (2, 0, 3)),
            (
                "integer protocol",
                (custom, np.int64(3), np.uint32(1), IntSubclass(2)),
                (2, 3, 1, 2),
            ),
            (
                "requires grad",
                (2, 3),
                (2, 3),
            ),
        )
        for case, dimensions, expected_shape in cases:
            with self.subTest(case=case):
                keywords = {"requires_grad": True} if case == "requires grad" else {}
                tensor = torch.ones(*dimensions, **keywords)
                self.assert_tensor_matches(tensor, torch.ones(expected_shape, **keywords))
                self.assertEqual(tensor.numel(), int(np.prod(expected_shape)))
                self.assertEqual(tensor.sum().item(), float(tensor.numel()))
        self.assertGreater(custom.calls, 0)

    def test_out_none_uses_default_fresh_allocation(self):
        cases = (
            ("scalar", lambda keywords: torch.ones(2, **keywords)),
            ("variadic", lambda keywords: torch.ones(2, 3, **keywords)),
            ("variadic empty", lambda keywords: torch.ones(2, 0, 3, **keywords)),
            ("tuple", lambda keywords: torch.ones((2, 3), **keywords)),
            ("size keyword", lambda keywords: torch.ones(size=(2,), **keywords)),
            (
                "shape alias",
                lambda keywords: torch.ones(None, shape=(2,), **keywords),
            ),
            (
                "requires grad",
                lambda keywords: torch.ones((2,), requires_grad=True, **keywords),
            ),
            ("empty", lambda keywords: torch.ones((0,), **keywords)),
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

        variadic_failures = (
            (
                lambda: torch.ones(2, -1),
                RuntimeError,
                re.escape(
                    "Trying to create tensor with negative dimension -1: [2, -1]"
                ),
            ),
            (
                lambda: torch.ones(2, IndexDimension(-1)),
                RuntimeError,
                re.escape(
                    "Trying to create tensor with negative dimension -1: [2, -1]"
                ),
            ),
            (
                lambda: torch.ones(True, 2),
                TypeError,
                r"pos 1.*bool",
            ),
            (
                lambda: torch.ones(2, False),
                TypeError,
                r"pos 2.*bool",
            ),
            (
                lambda: torch.ones(2, np.bool_(True)),
                TypeError,
                r"pos 2.*numpy\.bool",
            ),
            (
                lambda: torch.ones(2, 2**63),
                TypeError,
                r"pos 2.*Overflow when unpacking long long",
            ),
        )
        for call, error_type, message in variadic_failures:
            with self.subTest(call=call):
                with self.assertRaisesRegex(error_type, message):
                    call()

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
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}, 2]"
            ),
        ):
            torch.ones(sys.maxsize, 2)

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
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        with self.assertRaisesRegex(
            TypeError,
            re.escape("ones() got multiple values for argument 'size'"),
        ):
            torch.ones(2, 3, size=(2, 3))

    def test_out_tensor_layout_and_pin_memory_remain_unsupported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("ones(): the 'out' argument is not supported"),
        ):
            torch.ones(2, out=torch.ones(2))
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("ones(): the 'out' argument is not supported"),
        ):
            torch.ones(2, 3, out=torch.ones(2, 3))

        for call, error_type, message in (
            (
                lambda: torch.ones(2, 3, dtype=object()),
                TypeError,
                "ones(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.ones(2, 3, device="cuda"),
                RuntimeError,
                "ones(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.ones(2, layout=torch.strided, out=None),
                TypeError,
                "ones() got an unexpected keyword argument 'layout'",
            ),
            (
                lambda: torch.ones(2, pin_memory=False, out=None),
                TypeError,
                "ones() got an unexpected keyword argument 'pin_memory'",
            ),
            (
                lambda: torch.ones(2, 3, layout=torch.strided, out=None),
                TypeError,
                "ones() got an unexpected keyword argument 'layout'",
            ),
            (
                lambda: torch.ones(2, 3, pin_memory=False, out=None),
                TypeError,
                "ones() got an unexpected keyword argument 'pin_memory'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
