import copy
import importlib
import inspect
import pickle
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
            "dtype": tensor.dtype,
            "device": tensor.device,
            "layout": tensor.layout,
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "grad_is_none": tensor.grad is None,
            "data_ptr": tensor.data_ptr(),
            "nbytes": tensor.nbytes,
            "is_pinned": tensor.is_pinned(),
        }

    def test_zero_element_shapes_use_row_major_metadata(self):
        cases = (
            (0, (0,), (1,)),
            ((0,), (0,), (1,)),
            ([0], (0,), (1,)),
            (torch.Size([0]), (0,), (1,)),
            ((2, 0, 3), (2, 0, 3), (3, 3, 1)),
            ((0, 2), (0, 2), (2, 1)),
            ((1, 0, 1), (1, 0, 1), (1, 1, 1)),
            ((sys.maxsize, 0), (sys.maxsize, 0), (1, 1)),
            ((0, sys.maxsize, 1), (0, sys.maxsize, 1), (sys.maxsize, 1, 1)),
        )
        for size, expected_shape, expected_stride in cases:
            with self.subTest(size=size):
                tensor = torch.empty(size)
                self.assertEqual(tuple(tensor.shape), expected_shape)
                self.assertEqual(tensor.stride(), expected_stride)
                self.assertEqual(tensor.storage_offset(), 0)
                self.assertEqual(tensor.numel(), 0)
                self.assertIs(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertIs(tensor.layout, torch.strided)
                self.assertFalse(tensor.requires_grad)
                self.assertTrue(tensor.is_leaf)
                self.assertIsNone(tensor.grad)
                self.assertEqual(tensor.data_ptr(), 0)
                self.assertEqual(tensor.nbytes, 0)
                self.assertFalse(tensor.is_pinned())

    def test_variadic_zero_element_shapes_use_row_major_metadata(self):
        cases = (
            (lambda: torch.empty(0, 0), (0, 0), (1, 1)),
            (lambda: torch.empty(2, 0, 3), (2, 0, 3), (3, 3, 1)),
            (
                lambda: torch.empty(np.uint32(2), IntSubclass(0), np.int64(3)),
                (2, 0, 3),
                (3, 3, 1),
            ),
            (lambda: torch.empty(0, True), (0, 1), (1, 1)),
        )
        for create, expected_shape, expected_stride in cases:
            with self.subTest(expected_shape=expected_shape):
                tensor = create()
                self.assertEqual(tuple(tensor.shape), expected_shape)
                self.assertEqual(tensor.stride(), expected_stride)
                self.assertEqual(tensor.numel(), 0)
                self.assertIs(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertIs(tensor.layout, torch.strided)

    def test_supported_options_and_integer_protocol_dimensions(self):
        custom_scalar = IndexDimension(0)
        custom_sequence = IndexDimension(0)
        cases = (
            lambda: torch.empty((0,), dtype=None),
            lambda: torch.empty((0,), dtype=torch.float32),
            lambda: torch.empty((0,), dtype=torch.float),
            lambda: torch.empty((0,), device=None),
            lambda: torch.empty((0,), device="cpu"),
            lambda: torch.empty((0,), device="cpu:0"),
            lambda: torch.empty((0,), device=torch.device("cpu")),
            lambda: torch.empty((0,), device=torch.device("cpu", 2)),
            lambda: torch.empty((0,), layout=None),
            lambda: torch.empty((0,), layout=torch.strided),
            lambda: torch.empty((0,), pin_memory=None),
            lambda: torch.empty((0,), pin_memory=False),
            lambda: torch.empty((0,), memory_format=None),
            lambda: torch.empty((0,), memory_format=torch.contiguous_format),
            lambda: torch.empty((0,), requires_grad=None),
            lambda: torch.empty((0,), requires_grad=False),
            lambda: torch.empty((0,), requires_grad=True),
            lambda: torch.empty(size=(0,), out=None),
            lambda: torch.empty(custom_scalar),
            lambda: torch.empty((IntSubclass(0), np.uint32(3))),
            lambda: torch.empty([custom_sequence, np.int64(2)]),
        )
        for create in cases:
            with self.subTest(create=create):
                with torch.no_grad():
                    tensor = create()
                self.assertEqual(tensor.numel(), 0)
                self.assertIs(tensor.dtype, torch.float32)
                self.assertEqual(tensor.device, torch.device("cpu"))
                self.assertIs(tensor.layout, torch.strided)
                self.assertTrue(tensor.is_leaf)
        self.assertEqual(custom_scalar.calls, 3)
        self.assertEqual(custom_sequence.calls, 2)

    def test_empty_returns_fresh_storage_even_with_zero_data_ptr(self):
        first = torch.empty((2, 0, 3))
        second = torch.empty((2, 0, 3))
        self.assertEqual(first.data_ptr(), 0)
        self.assertEqual(second.data_ptr(), 0)
        self.assertFalse(first.is_set_to(second))

    def test_nonempty_shapes_are_rejected_before_any_uninitialized_allocation(self):
        for size in ((), (1,), (2, 3), 1):
            with self.subTest(size=size):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    "nonzero-element uninitialized allocation is not supported",
                ):
                    torch.empty(size)
        with self.assertRaisesRegex(
            NotImplementedError,
            "nonzero-element uninitialized allocation is not supported",
        ):
            torch.empty(1, 1)

    def test_invalid_sizes_report_factory_errors(self):
        for size in (-1, IndexDimension(-1), (-1,), (1, -2, 0)):
            with self.subTest(size=size):
                with self.assertRaises(RuntimeError):
                    torch.empty(size)
        for create in (
            lambda: torch.empty(-1, 0),
            lambda: torch.empty(1, -2, 0),
        ):
            with self.subTest(create=create):
                with self.assertRaises(RuntimeError):
                    create()

        for size in (True, False, np.bool_(True), (True,), (np.bool_(True),), None):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    torch.empty(size)
        with self.assertRaises(TypeError):
            torch.empty(size=None)

        for size in (2**63, -(2**63) - 1, (2**63,), (np.uint64(2**63),)):
            with self.subTest(size=size):
                with self.assertRaises(TypeError):
                    torch.empty(size)
        for create in (
            lambda: torch.empty(2**63, 0),
            lambda: torch.empty(0, 2**63),
        ):
            with self.subTest(create=create):
                with self.assertRaises(TypeError):
                    create()

        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            torch.empty((0, 2**62, 4))
        with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
            torch.empty(0, 2**62, 4)

        with self.assertRaisesRegex(RuntimeError, "Storage size calculation overflowed"):
            torch.empty((sys.maxsize,))

    def test_unsupported_keywords_and_nondefault_metadata_are_rejected(self):
        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\) got an unexpected keyword argument 'shape'$",
        ):
            torch.empty((0,), shape=(0,))

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): the 'out' argument is not supported$",
        ):
            torch.empty((0,), out=torch.zeros((0,)))

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'dtype' must be torch\.dtype, not object$",
        ):
            torch.empty((0,), dtype=object())

        with self.assertRaisesRegex(
            RuntimeError,
            r"^empty\(\): device 'meta' is not supported; only 'cpu' is implemented$",
        ):
            torch.empty((0,), device="meta")

        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'layout' must be torch\.layout, not object$",
        ):
            torch.empty((0,), layout=object())

        with self.assertRaisesRegex(
            RuntimeError,
            r"pin_memory=True is not supported",
        ):
            torch.empty((0,), pin_memory=True)

        for memory_format in (
            torch.preserve_format,
            torch.channels_last,
            torch.channels_last_3d,
        ):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"only torch\.contiguous_format is implemented",
                ):
                    torch.empty((0,), memory_format=memory_format)

        self.assertFalse(hasattr(torch, "empty_like"))

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        function = torch.empty
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

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
        self.assertIs(wildcard_namespace["empty"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)

        native = torch._C
        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, function)
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.empty, function)

    def test_torch_function_mode_dispatch_for_factory_calls(self):
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func,
                        types,
                        args,
                        kwargs,
                        tuple(torch.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.empty(size=(0,), dtype=torch.float32), marker)
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [mode])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
        self.assertIs(function, torch.empty)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"size": (0,), "dtype": torch.float32})
        self.assertEqual(handler_stack, ())


if __name__ == "__main__":
    unittest.main()
