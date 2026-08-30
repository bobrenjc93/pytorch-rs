import copy
import importlib
import inspect
import pickle
import re
import sys
import types
import unittest

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
    def tensor_metadata(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is torch.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "layout_identity": tensor.layout is torch.strided,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "data_ptr": tensor.data_ptr(),
        }

    def assert_empty_metadata(self, tensor, shape, stride, *, requires_grad=False):
        self.assertEqual(
            self.tensor_metadata(tensor),
            {
                "shape": shape,
                "stride": stride,
                "storage_offset": 0,
                "numel": 0,
                "dtype": "torch.float32",
                "dtype_identity": True,
                "device": "cpu",
                "layout": "torch.strided",
                "layout_identity": True,
                "requires_grad": requires_grad,
                "is_leaf": True,
                "grad_is_none": True,
                "data_ptr": 0,
            },
        )

    def test_zero_element_shapes_use_row_major_metadata(self):
        cases = (
            (lambda: torch.empty(0), (0,), (1,)),
            (lambda: torch.empty((0,)), (0,), (1,)),
            (lambda: torch.empty([2, 0, 3]), (2, 0, 3), (3, 3, 1)),
            (
                lambda: torch.empty((0, sys.maxsize)),
                (0, sys.maxsize),
                (sys.maxsize, 1),
            ),
            (
                lambda: torch.empty((sys.maxsize, 0, sys.maxsize)),
                (sys.maxsize, 0, sys.maxsize),
                (sys.maxsize, sys.maxsize, 1),
            ),
        )
        for create, shape, stride in cases:
            with self.subTest(shape=shape):
                self.assert_empty_metadata(create(), shape, stride)

    def test_supported_keyword_forms_create_leaf_tensors(self):
        option_cases = (
            {"out": None},
            {"dtype": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"device": None},
            {"device": "cpu"},
            {"device": "cpu:0"},
            {"device": torch.device("cpu")},
            {"device": torch.device("cpu", 2)},
            {"requires_grad": None},
            {"requires_grad": False},
            {"requires_grad": True},
            {
                "out": None,
                "dtype": torch.float,
                "device": torch.device("cpu"),
                "requires_grad": True,
            },
        )
        for options in option_cases:
            with self.subTest(options=options):
                with torch.no_grad():
                    tensor = torch.empty((2, 0, 3), **options)
                self.assert_empty_metadata(
                    tensor,
                    (2, 0, 3),
                    (3, 3, 1),
                    requires_grad=options.get("requires_grad") is True,
                )

        self.assert_empty_metadata(torch.empty(size=(0,)), (0,), (1,))
        self.assert_empty_metadata(torch.empty(size=[2, 0]), (2, 0), (1, 1))

    def test_integer_protocol_dimensions_are_supported(self):
        dynamic = IndexDimension(0)
        tensor = torch.empty(dynamic)
        self.assert_empty_metadata(tensor, (0,), (1,))
        self.assertEqual(dynamic.calls, 1)

        for dimension in (IntSubclass(0), np.int64(0), np.uint32(0)):
            with self.subTest(dimension=dimension):
                self.assert_empty_metadata(torch.empty(dimension), (0,), (1,))

        tuple_dimension = IndexDimension(0)
        self.assert_empty_metadata(torch.empty((tuple_dimension,)), (0,), (1,))
        self.assertEqual(tuple_dimension.calls, 1)

    def test_fresh_empty_storage(self):
        first = torch.empty((2, 0, 3))
        second = torch.empty((2, 0, 3))

        self.assertEqual(first.data_ptr(), 0)
        self.assertEqual(second.data_ptr(), 0)
        self.assertFalse(first.is_set_to(second))

    def test_nonzero_element_allocation_remains_unsupported(self):
        cases = (
            lambda: torch.empty(1),
            lambda: torch.empty(IntSubclass(1)),
            lambda: torch.empty(np.int64(1)),
            lambda: torch.empty(IndexDimension(1)),
            lambda: torch.empty(()),
            lambda: torch.empty([]),
            lambda: torch.empty((1,)),
            lambda: torch.empty([1, 1]),
        )
        for call in cases:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    re.escape(
                        "empty(): nonzero-element uninitialized allocation is not supported"
                    ),
                ):
                    call()

    def test_dimension_and_overflow_errors_are_reported_before_allocation(self):
        for call in (
            lambda: torch.empty(-1),
            lambda: torch.empty(IndexDimension(-1)),
            lambda: torch.empty((-1, 0)),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^Trying to create tensor with negative dimension -1:",
                ):
                    call()

        for call, type_name in (
            (lambda: torch.empty(True), "bool"),
            (lambda: torch.empty(False), "bool"),
            (lambda: torch.empty(np.bool_(True)), "numpy.bool"),
        ):
            with self.subTest(type_name=type_name):
                with self.assertRaisesRegex(
                    TypeError,
                    rf"must be tuple of ints, not {re.escape(type_name)}$",
                ):
                    call()

        with self.assertRaisesRegex(
            TypeError,
            r"empty\(\): argument 'size' failed to unpack the object at pos 1",
        ):
            torch.empty(2**63)
        with self.assertRaisesRegex(
            TypeError,
            r"empty\(\): argument 'size' failed to unpack the object at pos 1",
        ):
            torch.empty((2**63,))

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}]"
            ),
        ):
            torch.empty(sys.maxsize)
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("Stride calculation overflowed"),
        ):
            torch.empty((0, sys.maxsize, sys.maxsize))

    def test_unsupported_keywords_metadata_and_like_factory(self):
        out = torch.empty((0,))
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("empty(): the 'out' argument is not supported"),
        ):
            torch.empty((0,), out=out)

        for call, message in (
            (
                lambda: torch.empty((0,), out=[]),
                "empty(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.empty((0,), layout=torch.strided),
                "empty() got an unexpected keyword argument 'layout'",
            ),
            (
                lambda: torch.empty((0,), pin_memory=False),
                "empty() got an unexpected keyword argument 'pin_memory'",
            ),
            (
                lambda: torch.empty((0,), memory_format=torch.contiguous_format),
                "empty() got an unexpected keyword argument 'memory_format'",
            ),
            (
                lambda: torch.empty((0,), shape=(0,)),
                "empty() got an unexpected keyword argument 'shape'",
            ),
            (
                lambda: torch.empty(size=0),
                "empty(): argument 'size' must be tuple of ints, not int",
            ),
            (
                lambda: torch.empty((0,), dtype=object()),
                "empty(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.empty((0,), requires_grad=1),
                "empty(): argument 'requires_grad' must be bool, not int",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(Exception, f"^{re.escape(message)}$"):
                    call()

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("empty(): device 'meta' is not supported; only 'cpu' is implemented"),
        ):
            torch.empty((0,), device="meta")
        self.assertFalse(hasattr(torch, "empty_like"))

    def test_extra_positional_dimensions_are_not_supported(self):
        with self.assertRaisesRegex(
            TypeError,
            re.escape("empty() takes 1 positional argument but 2 were given"),
        ):
            torch.empty(2, 0)

    def test_callable_metadata_imports_copy_pickle_and_reload(self):
        function = torch.empty
        owner = function.__reduce__()[1][0]

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "empty")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.empty")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method empty of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.empty, function)
        self.assertEqual(torch.__all__.count("empty"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)

        namespace = {}
        exec("from torch_rs import empty", namespace)
        self.assertIs(namespace["empty"], function)
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["empty"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        self.assertIs(reloaded.empty, function)
        self.assertIs(reloaded._C._VariableFunctionsClass.empty, function)


if __name__ == "__main__":
    unittest.main()
