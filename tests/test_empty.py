import copy
import inspect
import pickle
import re
import sys
import types
import unittest
from collections import UserList
from collections.abc import Sequence

import numpy as np
import torch_rs as torch


class EmptyTests(unittest.TestCase):
    def assert_empty_metadata(
        self,
        tensor,
        shape,
        stride,
        *,
        numel,
        requires_grad=False,
    ):
        self.assertEqual(tensor.shape, shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), numel)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertFalse(tensor.is_pinned())
        self.assertEqual(tensor.requires_grad, requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertIsNone(tensor.grad)

    def test_supported_shapes_and_metadata_options(self):
        cases = (
            ("scalar tuple", lambda: torch.empty(()), (), (), 1, False),
            ("scalar list", lambda: torch.empty([]), (), (), 1, False),
            ("single int", lambda: torch.empty(2), (2,), (1,), 2, False),
            (
                "empty middle",
                lambda: torch.empty([2, 0, 3]),
                (2, 0, 3),
                (3, 3, 1),
                0,
                False,
            ),
            (
                "multidimensional",
                lambda: torch.empty((2, 3)),
                (2, 3),
                (3, 1),
                6,
                False,
            ),
            (
                "keyword metadata",
                lambda: torch.empty(
                    size=[2],
                    out=None,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu", 2),
                    pin_memory=False,
                    memory_format=torch.contiguous_format,
                    requires_grad=True,
                ),
                (2,),
                (1,),
                2,
                True,
            ),
            (
                "none metadata",
                lambda: torch.empty(
                    (2,),
                    out=None,
                    dtype=None,
                    layout=None,
                    device=None,
                    pin_memory=None,
                    memory_format=None,
                    requires_grad=None,
                ),
                (2,),
                (1,),
                2,
                False,
            ),
        )
        for case, create, shape, stride, numel, requires_grad in cases:
            with self.subTest(case=case):
                self.assert_empty_metadata(
                    create(),
                    shape,
                    stride,
                    numel=numel,
                    requires_grad=requires_grad,
                )

    def test_default_equivalent_memory_format_creates_contiguous_tensor(self):
        for memory_format in (None, torch.contiguous_format):
            with self.subTest(memory_format=memory_format):
                self.assert_empty_metadata(
                    torch.empty((2, 3), memory_format=memory_format),
                    (2, 3),
                    (3, 1),
                    numel=6,
                )

    def test_requires_grad_creates_leaf_inside_no_grad(self):
        with torch.no_grad():
            tensor = torch.empty((2, 3), requires_grad=True)
        self.assert_empty_metadata(
            tensor,
            (2, 3),
            (3, 1),
            numel=6,
            requires_grad=True,
        )

    def test_out_none_uses_default_fresh_allocation(self):
        for size in ((), (2,), (2, 0, 3), (2, 3)):
            with self.subTest(size=size):
                first = torch.empty(size, out=None)
                second = torch.empty(size, out=None)
                self.assertEqual(first.shape, second.shape)
                self.assertEqual(first.stride(), second.stride())
                self.assertFalse(first.is_set_to(second))
                if first.numel() > 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                else:
                    self.assertEqual(first.data_ptr(), 0)
                    self.assertEqual(second.data_ptr(), 0)

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
                self.assert_empty_metadata(
                    torch.empty(dimension),
                    (2,),
                    (1,),
                    numel=2,
                )
        self.assertEqual(custom.calls, 1)

    def test_existing_sequence_and_keyword_forms_are_supported(self):
        class CustomSequence(Sequence):
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

        for size, expected_shape, expected_stride in (
            ((2,), (2,), (1,)),
            ([2], (2,), (1,)),
            (np.array([2]), (2,), (1,)),
            (range(2, 4), (2, 3), (3, 1)),
            (UserList([2]), (2,), (1,)),
            (CustomSequence([2]), (2,), (1,)),
        ):
            with self.subTest(size=size):
                tensor = torch.empty(size)
                self.assertEqual(tensor.shape, expected_shape)
                self.assertEqual(tensor.stride(), expected_stride)
                self.assertEqual(tensor.numel(), int(np.prod(expected_shape)))

        self.assertEqual(torch.empty(size=(2,)).shape, (2,))

        for call in (
            lambda: torch.empty(size=2),
            lambda: torch.empty(2, 3),
            lambda: torch.empty(shape=(2,)),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_rejects_invalid_dimensions_before_allocation(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        empty = torch.empty(0)
        self.assert_empty_metadata(empty, (0,), (1,), numel=0)

        for dimension in (-1, IndexDimension(-1)):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape(
                        "Trying to create tensor with negative dimension -1: [-1]"
                    ),
                ):
                    torch.empty(dimension)

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
                    torch.empty(dimension)

        for size, type_name in (
            ([True], "bool"),
            ([False], "bool"),
            ((True,), "bool"),
            ([np.bool_(True)], "numpy.bool"),
        ):
            with self.subTest(size=size):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"argument 'size' \(position 1\) must be tuple of ints, "
                    rf"but found element of type {re.escape(type_name)} at pos 0$",
                ):
                    torch.empty(size)

        for size, message in (
            ([-1], "Trying to create tensor with negative dimension -1: [-1]"),
            ([1, -2], "Trying to create tensor with negative dimension -2: [1, -2]"),
            ([np.int64(-1)], "Trying to create tensor with negative dimension -1: [-1]"),
        ):
            with self.subTest(size=size):
                with self.assertRaisesRegex(RuntimeError, re.escape(message)):
                    torch.empty(size)

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
                    torch.empty(dimension)

        for size in (
            [2**63],
            [-(2**63) - 1],
            [np.uint64(2**63)],
            [IndexDimension(2**63)],
        ):
            with self.subTest(size=size):
                with self.assertRaisesRegex(
                    TypeError,
                    "argument 'size' failed to unpack the object at pos 1 "
                    "with error .*Overflow when unpacking long long",
                ):
                    torch.empty(size)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}]"
            ),
        ):
            torch.empty(sys.maxsize)

    def test_rejects_unsupported_options_and_empty_like(self):
        out = torch.zeros((1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((1,), out=out)
        self.assertEqual(out.tolist(), [0.0])

        for dtype in (
            "float32",
            np.dtype("float32"),
            np.float32,
            float,
            object(),
            torch.device("cpu"),
        ):
            with self.subTest(argument="dtype", value=dtype):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'dtype' must be torch\.dtype, not ",
                ):
                    torch.empty((1,), dtype=dtype)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((1,), device="meta")

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'layout' must be torch\.layout, not object$",
        ):
            torch.empty((1,), layout=object())

        for pin_memory in (0, 1, "false", object()):
            with self.subTest(pin_memory=pin_memory):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'pin_memory' must be bool, not ",
                ):
                    torch.empty((1,), pin_memory=pin_memory)

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): pin_memory=True is not supported; only unpinned CPU storage is implemented$",
        ):
            torch.empty((1,), pin_memory=True)

        for memory_format in (0, "contiguous", object()):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^empty\(\): argument 'memory_format' must be torch\.memory_format, not ",
                ):
                    torch.empty((1,), memory_format=memory_format)

        for memory_format in (
            torch.preserve_format,
            torch.channels_last,
            torch.channels_last_3d,
        ):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    r"^empty\(\): only contiguous memory_format is supported$",
                ):
                    torch.empty((2, 3), memory_format=memory_format)

        self.assertFalse(hasattr(torch, "empty_like"))
        with self.assertRaises(AttributeError):
            torch.empty_like(torch.empty((1,)))

    def test_callable_import_wildcard_copy_and_pickle(self):
        function = torch.empty
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "empty")
        self.assertEqual(
            str(inspect.signature(function)),
            "(size, *, out=None, dtype=None, layout=None, device=None, "
            "pin_memory=False, memory_format=None, requires_grad=False)",
        )
        self.assertEqual(torch.__all__.count("empty"), 1)
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
