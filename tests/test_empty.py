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
            {"memory_format": None},
            {"memory_format": torch.contiguous_format},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": True,
                "memory_format": torch.contiguous_format,
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

        forwarded = torch.empty(
            (2, 3),
            **torch.nn.factory_kwargs(
                {"memory_format": torch.contiguous_format}
            ),
        )
        self.assert_metadata(forwarded, (2, 3), (3, 1))
        scalar_forwarded = torch.empty(
            2,
            **torch.nn.factory_kwargs(
                {"memory_format": torch.contiguous_format}
            ),
        )
        self.assert_metadata(scalar_forwarded, (2,), (1,))

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
            r"^empty\(\): argument 'memory_format' must be torch\.memory_format, not ",
        ):
            torch.empty((1,), memory_format=object())

        for memory_format in (
            torch.preserve_format,
            torch.channels_last,
            torch.channels_last_3d,
        ):
            with self.subTest(memory_format=memory_format):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^empty\(\): only torch\.contiguous_format memory_format is supported$",
                ):
                    torch.empty((1, 2, 3, 4, 5), memory_format=memory_format)

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

    def test_sequence_size_rejects_invalid_dimensions_before_allocation(self):
        invalid_cases = (
            (
                (True,),
                TypeError,
                r"^empty\(\): argument 'size' \(position 1\) must be tuple of ints, but found element of type bool at pos 0$",
            ),
            (
                (np.bool_(True),),
                TypeError,
                r"^empty\(\): argument 'size' \(position 1\) must be tuple of ints, but found element of type .*bool.* at pos 0$",
            ),
            (
                (-1,),
                RuntimeError,
                r"^Trying to create tensor with negative dimension -1: \[-1\]$",
            ),
            (
                (2, -1, 3),
                RuntimeError,
                r"^Trying to create tensor with negative dimension -1: \[2, -1, 3\]$",
            ),
            (
                (2**63, 0),
                TypeError,
                r"^empty\(\): argument 'size' failed to unpack the object at pos 1 with error \"Overflow when unpacking long long\"$",
            ),
        )
        for size, exception, message in invalid_cases:
            with self.subTest(size=size):
                with self.assertRaisesRegex(exception, message):
                    torch.empty(size)

        dynamic = IndexDimension(2**63)
        with self.assertRaisesRegex(
            TypeError,
            r"^empty\(\): argument 'size' failed to unpack the object at pos 1 with error \"Overflow when unpacking long long\"$",
        ):
            torch.empty((dynamic, 0))
        self.assertEqual(dynamic.calls, 1)

        tensor = torch.empty((2**63 - 1, 0))
        self.assert_metadata(tensor, (2**63 - 1, 0), (1, 1))

    def test_torch_function_mode_dispatches_and_restores_stack(self):
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

        cases = (
            (lambda: torch.empty((2, 3)), ((2, 3),), None),
            (
                lambda: torch.empty(
                    size=(2, 3),
                    memory_format=torch.contiguous_format,
                ),
                (),
                {"size": (2, 3), "memory_format": torch.contiguous_format},
            ),
        )
        for call, expected_args, expected_kwargs in cases:
            mode = RecordingMode()
            with self.subTest(args=expected_args, kwargs=expected_kwargs):
                with mode:
                    self.assertIs(call(), marker)
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
                self.assertEqual(len(mode.calls), 1)
                function, dispatch_types, args, kwargs, handler_stack = mode.calls[0]
                self.assertIs(function, torch.empty)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)
                self.assertEqual(handler_stack, ())

        events = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append(
                    (
                        self.label,
                        tuple(
                            mode.label
                            for mode in torch.overrides._get_current_function_mode_stack()
                        ),
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                result = torch.empty(size=(2, 3))
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, upper]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assert_metadata(result, (2, 3), (3, 1))
        self.assertEqual(events, [("upper", ("lower",)), ("lower", ())])

        expected = ValueError("empty mode failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected

        raising = RaisingMode()
        with lower:
            with raising:
                with self.assertRaises(ValueError) as raised:
                    torch.empty((2, 3))
                self.assertIs(raised.exception, expected)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, raising]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            with self.assertRaisesRegex(
                TypeError,
                r"^Multiple dispatch failed for 'torch\.empty'; all "
                r"__torch_function__ handlers returned NotImplemented:",
            ):
                torch.empty((2, 3))
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [declining]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

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
