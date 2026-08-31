import copy
import pickle
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
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.tolist(), expected.tolist())
        self.assertIs(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertIs(actual.layout, expected.layout)
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        self.assertIsNone(actual.grad)

    def test_one_positional_integer_matches_singleton_size(self):
        metadata = (
            {},
            {"out": None},
            {"dtype": torch.float32},
            {"layout": None},
            {"layout": torch.strided},
            {"device": "cpu"},
            {"device": torch.device("cpu")},
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

    def test_out_none_uses_default_fresh_allocation(self):
        cases = (
            ("scalar", lambda keywords: torch.zeros(2, **keywords)),
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

    def test_default_layout_and_pin_memory_keywords_use_default_fresh_allocation(self):
        cases = (
            ("scalar", lambda keywords: torch.zeros(2, **keywords)),
            ("tuple", lambda keywords: torch.zeros((2, 3), **keywords)),
            ("size keyword", lambda keywords: torch.zeros(size=(2,), **keywords)),
            (
                "shape alias",
                lambda keywords: torch.zeros(None, shape=(2,), **keywords),
            ),
            ("scalar tensor", lambda keywords: torch.zeros((), **keywords)),
            ("empty", lambda keywords: torch.zeros((0,), **keywords)),
        )
        metadata = (
            {"layout": None},
            {"layout": torch.strided},
            {"pin_memory": None},
            {"pin_memory": False},
            {"layout": torch.strided, "pin_memory": False},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )

        for case, factory in cases:
            for keywords in metadata:
                with self.subTest(case=case, keywords=keywords):
                    baseline_keywords = {
                        key: value
                        for key, value in keywords.items()
                        if key not in {"layout", "pin_memory"}
                    }
                    baseline = factory(baseline_keywords)
                    actual = factory(keywords)
                    self.assert_tensor_matches(actual, baseline)
                    self.assertIs(actual.layout, torch.strided)
                    self.assertFalse(actual.is_set_to(baseline))

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
            lambda: torch.zeros(2, 3),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_out_tensor_nondefault_layout_and_pinned_memory_remain_unsupported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("zeros(): the 'out' argument is not supported"),
        ):
            torch.zeros(
                2,
                out=torch.zeros(2),
                layout=torch.strided,
                pin_memory=False,
            )

        for call, error_type, message in (
            (
                lambda: torch.zeros(2, layout=object()),
                TypeError,
                "zeros(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.zeros(2, pin_memory=0),
                TypeError,
                "zeros(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.zeros(2, pin_memory=True),
                RuntimeError,
                "zeros(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_callable_import_wildcard_copy_and_pickle_behavior(self):
        function = torch.zeros
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        from torch_rs import zeros as imported

        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "zeros")
        self.assertIs(imported, function)
        self.assertIs(torch._C.zeros, function)
        self.assertEqual(torch.__all__.count("zeros"), 1)
        self.assertIs(wildcard_namespace["zeros"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)


if __name__ == "__main__":
    unittest.main()
