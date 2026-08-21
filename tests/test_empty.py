import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class IndexDimension:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


class BadIndexDimension:
    def __index__(self):
        return 1.5


class EmptyTests(unittest.TestCase):
    def assert_metadata(self, tensor, shape, stride, requires_grad=False):
        self.assertEqual(tuple(tensor.shape), shape)
        self.assertEqual(tensor.stride(), stride)
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), int(np.prod(shape, dtype=np.int64)))
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertEqual(tensor.requires_grad, requires_grad)
        self.assertTrue(tensor.is_leaf)

    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def test_integer_sequence_and_default_metadata(self):
        cases = (
            (2, (2,), (1,)),
            ((2, 3, 4), (2, 3, 4), (12, 4, 1)),
            ([2, 3], (2, 3), (3, 1)),
            (torch.Size([2, 3]), (2, 3), (3, 1)),
            ((), (), ()),
            ([], (), ()),
            (0, (0,), (1,)),
            ((2, 0, 3), (2, 0, 3), (3, 3, 1)),
            ((sys.maxsize, 0, 2), (sys.maxsize, 0, 2), (2, 2, 1)),
        )
        for size, shape, stride in cases:
            with self.subTest(size=size):
                self.assert_metadata(torch.empty(size), shape, stride)

        for options in (
            {"dtype": None},
            {"dtype": torch.float32},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"requires_grad": None},
            {"requires_grad": False},
        ):
            with self.subTest(options=options):
                self.assert_metadata(torch.empty((2, 3), **options), (2, 3), (3, 1))

        self.assert_metadata(torch.empty(size=(2, 3)), (2, 3), (3, 1))

    def test_each_call_has_fresh_storage_and_contents_are_not_assumed(self):
        first = torch.empty((8,))
        second = torch.empty((8,))

        self.assertNotEqual(first.data_ptr(), 0)
        self.assertNotEqual(second.data_ptr(), 0)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_index_protocol_is_used_for_scalar_and_sequence_dimensions(self):
        class IntSubclass(int):
            pass

        scalar_index = IndexDimension(2)
        tuple_index = IndexDimension(3)
        list_index = IndexDimension(4)
        cases = (
            (IntSubclass(2), (2,)),
            (np.int64(2), (2,)),
            (np.uint32(2), (2,)),
            (scalar_index, (2,)),
            ((tuple_index,), (3,)),
            ([list_index], (4,)),
        )
        for size, expected_shape in cases:
            with self.subTest(size=size):
                self.assertEqual(tuple(torch.empty(size).shape), expected_shape)

        self.assertGreater(scalar_index.calls, 0)
        self.assertGreater(tuple_index.calls, 0)
        self.assertGreater(list_index.calls, 0)

    def test_requires_grad_creates_an_independent_leaf(self):
        leaf = torch.empty((2, 3), requires_grad=True)
        self.assert_metadata(leaf, (2, 3), (3, 1), requires_grad=True)
        self.assertIsNone(leaf.grad)

        leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        self.assertNotEqual(leaf.data_ptr(), leaf.grad.data_ptr())

    def test_supported_form_errors_match_pytorch(self):
        cases = (
            (
                lambda: torch.empty(),
                TypeError,
                'empty() missing 1 required positional arguments: "size"',
            ),
            (
                lambda: torch.empty(None),
                TypeError,
                "empty(): argument 'size' (position 1) must be tuple of ints, not NoneType",
            ),
            (
                lambda: torch.empty(size=2),
                TypeError,
                "empty(): argument 'size' must be tuple of ints, not int",
            ),
            (
                lambda: torch.empty(-1),
                RuntimeError,
                "Trying to create tensor with negative dimension -1: [-1]",
            ),
            (
                lambda: torch.empty((2, -1)),
                RuntimeError,
                "Trying to create tensor with negative dimension -1: [2, -1]",
            ),
            (
                lambda: torch.empty(True),
                TypeError,
                "empty(): argument 'size' (position 1) must be tuple of ints, not bool",
            ),
            (
                lambda: torch.empty((True,)),
                TypeError,
                "empty(): argument 'size' (position 1) must be tuple of ints, but found element of type bool at pos 0",
            ),
            (
                lambda: torch.empty((1.5,)),
                TypeError,
                "empty(): argument 'size' (position 1) must be tuple of ints, but found element of type float at pos 0",
            ),
            (
                lambda: torch.empty(range(2)),
                TypeError,
                "empty(): argument 'size' (position 1) must be tuple of ints, not range",
            ),
            (
                lambda: torch.empty(BadIndexDimension()),
                TypeError,
                "empty(): argument 'size' (position 1) must be tuple of ints, not BadIndexDimension",
            ),
            (
                lambda: torch.empty((2,), dtype=object()),
                TypeError,
                "empty(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.empty((2,), device=object()),
                TypeError,
                "empty(): argument 'device' must be torch.device, not object",
            ),
            (
                lambda: torch.empty((2,), requires_grad=1),
                TypeError,
                "empty(): argument 'requires_grad' must be bool, not int",
            ),
            (
                lambda: torch.empty((2,), size=(3,)),
                TypeError,
                "empty() got multiple values for argument 'size'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                self.assert_error(call, error_type, message)

        for size in (2**63, -(2**63) - 1, IndexDimension(2**63)):
            with self.subTest(size=size):
                with self.assertRaisesRegex(
                    TypeError,
                    "failed to unpack.*Overflow when unpacking long long",
                ):
                    torch.empty(size)

        self.assert_error(
            lambda: torch.empty(sys.maxsize),
            RuntimeError,
            f"Storage size calculation overflowed with sizes=[{sys.maxsize}]",
        )
        self.assert_error(
            lambda: torch.empty((sys.maxsize, 2)),
            RuntimeError,
            f"Storage size calculation overflowed with sizes=[{sys.maxsize}, 2]",
        )

    def test_unsupported_extensions_remain_unavailable(self):
        calls = (
            lambda: torch.empty(2, 3),
            lambda: torch.empty((2,), out=None),
            lambda: torch.empty((2,), pin_memory=False),
            lambda: torch.empty((2,), layout=torch.strided),
            lambda: torch.empty((2,), memory_format=torch.contiguous_format),
            lambda: torch.empty((2,), shape=(3,)),
            lambda: torch.empty((2,), device="cuda"),
        )
        for call in calls:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, RuntimeError)):
                    call()

        self.assertFalse(hasattr(torch, "float64"))

    def test_callable_metadata_and_exports_match_pytorch(self):
        function = torch.empty
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "empty")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.empty")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.empty, function)
        self.assertEqual(torch.__all__.count("empty"), 1)
        self.assertIs(wildcard_namespace["empty"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)


if __name__ == "__main__":
    unittest.main()
