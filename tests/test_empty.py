import copy
import math
import pickle
import re
import unittest
from collections import UserList
from collections.abc import Sequence

import numpy as np
import torch_rs as torch


class IntSubclass(int):
    pass


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class EmptyTests(unittest.TestCase):
    def assert_metadata(
        self,
        tensor,
        shape,
        stride,
        *,
        requires_grad=False,
    ):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), math.prod(shape))
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertFalse(tensor.is_pinned())
        self.assertEqual(tensor.requires_grad, requires_grad)
        self.assertTrue(tensor.is_leaf)

    def test_supported_shapes_and_metadata(self):
        cases = (
            ("scalar tuple", lambda: torch.empty(()), (), ()),
            ("scalar list", lambda: torch.empty([]), (), ()),
            ("single integer", lambda: torch.empty(2), (2,), (1,)),
            ("empty vector", lambda: torch.empty((0,)), (0,), (1,)),
            ("empty middle", lambda: torch.empty((2, 0, 3)), (2, 0, 3), (3, 3, 1)),
            ("multidimensional", lambda: torch.empty((2, 3)), (2, 3), (3, 1)),
            ("size keyword", lambda: torch.empty(size=(2,)), (2,), (1,)),
        )
        for case, create, shape, stride in cases:
            with self.subTest(case=case):
                self.assert_metadata(create(), shape, stride)

    def test_integer_protocol_size_dimensions(self):
        dynamic = IndexDimension(2)
        tensor = torch.empty([dynamic, np.int64(0), IntSubclass(3)])

        self.assert_metadata(tensor, (2, 0, 3), (3, 3, 1))
        self.assertEqual(dynamic.calls, 1)

    def test_default_equivalent_factory_options(self):
        option_sets = (
            {"out": None},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"device": torch.device("cpu", 2)},
            {"pin_memory": None},
            {"pin_memory": False},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
            },
        )
        for options in option_sets:
            with self.subTest(options=options):
                with torch.no_grad():
                    tensor = torch.empty((2, 3), **options)
                self.assert_metadata(
                    tensor,
                    (2, 3),
                    (3, 1),
                    requires_grad=options.get("requires_grad") is True,
                )

    def test_empty_returns_fresh_storage(self):
        for shape in ((), (2, 3), (2, 0, 3)):
            with self.subTest(shape=shape):
                first = torch.empty(shape)
                second = torch.empty(shape)
                self.assertFalse(first.is_set_to(second))
                if first.numel():
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_unsupported_boundaries(self):
        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=out)
        self.assertEqual(out.tolist(), [0.0])

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.empty((1,), pin_memory=True)

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.empty((1,), pin_memory=pin_memory)

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'layout' must be torch\.layout, not ",
        ):
            torch.empty((1,), layout=object())

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'dtype' must be torch\.dtype, not ",
        ):
            torch.empty((1,), dtype=object())

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'requires_grad' must be bool, not ",
        ):
            torch.empty((1,), requires_grad=1)

        with self.assertRaisesRegex(
            TypeError,
            r'^empty\(\) missing 1 required positional arguments: "size"$',
        ):
            torch.empty(shape=(1,))

        self.assertFalse(hasattr(torch, "empty_like"))

    def test_sequence_size_forms(self):
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
                tensor = torch.empty(size)
                self.assertEqual(tensor.shape, expected_shape)
                self.assertEqual(tensor.numel(), int(np.prod(expected_shape)))

    def test_callable_import_and_wildcard_exports(self):
        function = torch.empty
        import_namespace = {}
        wildcard_namespace = {}
        exec("from torch_rs import empty as imported_empty", import_namespace)
        exec("from torch_rs import *", wildcard_namespace)

        self.assertTrue(callable(function))
        self.assertEqual(function.__name__, "empty")
        self.assertEqual(torch.__all__.count("empty"), 1)
        self.assertIs(import_namespace["imported_empty"], function)
        self.assertIs(wildcard_namespace["empty"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )


if __name__ == "__main__":
    unittest.main()
