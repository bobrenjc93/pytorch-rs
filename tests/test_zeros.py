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
            {"out": None},
            {"dtype": torch.float32},
            {"layout": None},
            {"layout": torch.strided},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
            {"pin_memory": None},
            {"pin_memory": False},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )
        for keywords in metadata:
            with self.subTest(keywords=keywords):
                self.assert_tensor_matches(
                    torch.zeros(2, **keywords),
                    torch.zeros((2,), **keywords),
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

        class StatefulIndexDimension:
            def __init__(self, values):
                self.values = values
                self.calls = 0

            def __index__(self):
                value = self.values[min(self.calls, len(self.values) - 1)]
                self.calls += 1
                return value

        custom = IndexDimension(2)
        cases = (
            ("plain", (2, 3), (2, 3)),
            ("zero dimension", (2, 0, 3), (2, 0, 3)),
            ("python bool false dimension", (2, False), (2, 0)),
            ("python bool true dimension", (2, True), (2, 1)),
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
                tensor = torch.zeros(*dimensions, **keywords)
                self.assert_tensor_matches(tensor, torch.zeros(expected_shape, **keywords))
                self.assertEqual(tensor.numel(), int(np.prod(expected_shape)))
        self.assertGreater(custom.calls, 0)

        stateful = StatefulIndexDimension((2, 3, 4))
        stateful_tensor = torch.zeros(stateful, 3)
        self.assertEqual(stateful_tensor.shape, (4, 3))
        self.assertEqual(stateful.calls, 3)

    def test_out_none_uses_default_fresh_allocation(self):
        cases = (
            ("scalar", lambda keywords: torch.zeros(2, **keywords)),
            ("variadic", lambda keywords: torch.zeros(2, 3, **keywords)),
            ("variadic empty", lambda keywords: torch.zeros(2, 0, 3, **keywords)),
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

    def test_default_layout_and_unpinned_keywords_cover_supported_size_forms(self):
        class IntSubclass(int):
            pass

        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        cases = (
            ("scalar tuple", lambda keywords: torch.zeros((), **keywords)),
            ("scalar list", lambda keywords: torch.zeros([], **keywords)),
            ("empty tuple", lambda keywords: torch.zeros((0,), **keywords)),
            ("empty variadic", lambda keywords: torch.zeros(2, 0, 3, **keywords)),
            ("variadic", lambda keywords: torch.zeros(2, 3, **keywords)),
            ("tuple", lambda keywords: torch.zeros((2, 3), **keywords)),
            ("list", lambda keywords: torch.zeros([2, 3], **keywords)),
            (
                "integer protocol",
                lambda keywords: torch.zeros(
                    [IndexDimension(2), np.int64(3), IntSubclass(1)],
                    **keywords,
                ),
            ),
        )
        option_cases = (
            {"layout": None},
            {"layout": torch.strided},
            {"pin_memory": None},
            {"pin_memory": False},
            {"out": None, "layout": torch.strided, "pin_memory": False},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )

        def flattened_values(value):
            if isinstance(value, list):
                for item in value:
                    yield from flattened_values(item)
            else:
                yield value

        for case, factory in cases:
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    tensor = factory(options)
                    self.assertIs(tensor.dtype, torch.float32)
                    self.assertEqual(tensor.device, torch.device("cpu"))
                    self.assertIs(tensor.layout, torch.strided)
                    self.assertFalse(tensor.is_pinned())
                    self.assertEqual(
                        tensor.requires_grad,
                        bool(options.get("requires_grad", False)),
                    )
                    self.assertTrue(
                        all(value == 0.0 for value in flattened_values(tensor.tolist()))
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

        variadic_failures = (
            (
                lambda: torch.zeros(2, -1),
                RuntimeError,
                re.escape("zeros: Dimension size must be non-negative."),
            ),
            (
                lambda: torch.zeros(2, IndexDimension(-1)),
                RuntimeError,
                re.escape("zeros: Dimension size must be non-negative."),
            ),
            (
                lambda: torch.zeros(True, 2),
                TypeError,
                re.escape("zeros() takes 1 positional argument but 2 were given"),
            ),
            (
                lambda: torch.zeros(2, np.bool_(True)),
                TypeError,
                r"pos 2.*numpy\.bool",
            ),
            (
                lambda: torch.zeros(2, 2**63),
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
            torch.zeros(sys.maxsize)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}, 2]"
            ),
        ):
            torch.zeros(sys.maxsize, 2)

    def test_existing_sequence_and_keyword_forms_are_unchanged(self):
        class CustomSequence(Sequence):
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

        class TupleIndex(tuple):
            def __new__(cls, values):
                instance = super().__new__(cls, values)
                instance.calls = 0
                return instance

            def __index__(self):
                self.calls += 1
                return 2

        class ListIndex(list):
            def __init__(self, values):
                super().__init__(values)
                self.calls = 0

            def __index__(self):
                self.calls += 1
                return 2

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
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        with self.assertRaisesRegex(
            TypeError,
            re.escape("zeros() got multiple values for argument 'size'"),
        ):
            torch.zeros(2, 3, size=(2, 3))

        for size in (TupleIndex((4,)), ListIndex([4])):
            with self.subTest(size=type(size).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    re.escape("zeros() takes 1 positional argument but 2 were given"),
                ):
                    torch.zeros(size, 3)
                self.assertEqual(size.calls, 1)

        for size in ((1,), [1], range(1)):
            for competing_keyword in ({"wat": 1}, {"size": (1,)}, {"requires_grad": 1}):
                with self.subTest(size=size, competing_keyword=competing_keyword):
                    with self.assertRaisesRegex(
                        TypeError,
                        re.escape("zeros() takes 1 positional argument but 2 were given"),
                    ):
                        torch.zeros(size, True, **competing_keyword)

    def test_out_tensor_nondefault_layout_and_pinned_memory_remain_unsupported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("zeros(): the 'out' argument is not supported"),
        ):
            torch.zeros(2, out=torch.zeros(2))
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("zeros(): the 'out' argument is not supported"),
        ):
            torch.zeros(2, 3, out=torch.zeros(2, 3))

        for call, error_type, message in (
            (
                lambda: torch.zeros(2, 3, dtype=object()),
                TypeError,
                "zeros(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.zeros(2, 3, device="cuda"),
                RuntimeError,
                "zeros(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.zeros(2, layout=object(), out=None),
                TypeError,
                "zeros(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.zeros(2, pin_memory=0, out=None),
                TypeError,
                "zeros(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.zeros(2, pin_memory=True, out=None),
                RuntimeError,
                "zeros(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: torch.zeros(2, 3, pin_memory=True, out=None),
                RuntimeError,
                "zeros(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()


if __name__ == "__main__":
    unittest.main()
