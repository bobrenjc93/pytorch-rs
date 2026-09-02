import copy
import importlib
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
    def contiguous_stride(self, shape):
        stride = []
        running = 1
        for dimension in reversed(shape):
            stride.append(running)
            running *= max(dimension, 1)
        return tuple(reversed(stride))

    def assert_empty_metadata(
        self,
        tensor,
        expected_shape,
        *,
        requires_grad=False,
    ):
        self.assertIs(type(tensor), torch.Tensor)
        self.assertEqual(tensor.shape, expected_shape)
        self.assertEqual(tensor.stride(), self.contiguous_stride(expected_shape))
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertFalse(tensor.is_pinned())
        self.assertEqual(tensor.requires_grad, requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.numel(), int(np.prod(expected_shape, dtype=np.int64)))

    def test_supported_size_and_metadata_forms(self):
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
            ("one positional integer", lambda options: torch.empty(2, **options), (2,)),
            ("zero scalar dimension", lambda options: torch.empty(0, **options), (0,)),
            ("scalar tuple", lambda options: torch.empty((), **options), ()),
            ("scalar list", lambda options: torch.empty([], **options), ()),
            ("empty tuple", lambda options: torch.empty((0,), **options), (0,)),
            (
                "empty variadic",
                lambda options: torch.empty(2, 0, 3, **options),
                (2, 0, 3),
            ),
            ("variadic", lambda options: torch.empty(2, 3, **options), (2, 3)),
            ("tuple", lambda options: torch.empty((2, 3), **options), (2, 3)),
            ("list", lambda options: torch.empty([2, 3], **options), (2, 3)),
            (
                "tuple non-leading bool dimension",
                lambda options: torch.empty((2, True), **options),
                (2, 1),
            ),
            (
                "list non-leading bool dimension",
                lambda options: torch.empty([2, False], **options),
                (2, 0),
            ),
            (
                "integer protocol sequence",
                lambda options: torch.empty(
                    [IndexDimension(2), np.int64(3), IntSubclass(1)], **options
                ),
                (2, 3, 1),
            ),
            (
                "integer protocol variadic",
                lambda options: torch.empty(
                    custom, np.int64(3), np.uint32(1), IntSubclass(2), **options
                ),
                (2, 3, 1, 2),
            ),
            ("size keyword", lambda options: torch.empty(size=(2,), **options), (2,)),
        )
        option_cases = (
            {},
            {"out": None},
            {"dtype": torch.float32},
            {"dtype": torch.float},
            {"layout": None},
            {"layout": torch.strided},
            {"device": None},
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

        for case, factory, expected_shape in cases:
            for options in option_cases:
                with self.subTest(case=case, options=options):
                    tensor = factory(options)
                    self.assert_empty_metadata(
                        tensor,
                        expected_shape,
                        requires_grad=bool(options.get("requires_grad", False)),
                    )
        self.assertGreater(custom.calls, 0)

    def test_out_none_and_repeated_calls_use_fresh_storage(self):
        cases = (
            ("scalar", lambda options: torch.empty(2, **options)),
            ("variadic", lambda options: torch.empty(2, 3, **options)),
            ("variadic empty", lambda options: torch.empty(2, 0, 3, **options)),
            ("tuple", lambda options: torch.empty((2, 3), **options)),
            ("list", lambda options: torch.empty([2, 3], **options)),
            ("size keyword", lambda options: torch.empty(size=(2,), **options)),
            (
                "requires grad",
                lambda options: torch.empty((2,), requires_grad=True, **options),
            ),
            ("empty", lambda options: torch.empty((0,), **options)),
            ("scalar tensor", lambda options: torch.empty((), **options)),
        )

        for case, factory in cases:
            with self.subTest(case=case):
                first = factory({})
                with_out_none = factory({"out": None})
                second = factory({})
                self.assertEqual(with_out_none.shape, first.shape)
                self.assertEqual(with_out_none.stride(), first.stride())
                self.assertEqual(with_out_none.requires_grad, first.requires_grad)
                self.assertFalse(with_out_none.is_set_to(first))
                self.assertFalse(first.is_set_to(second))
                if first.numel() > 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_requires_grad_and_no_grad_match_factory_semantics(self):
        default = torch.empty((2, 3))
        self.assert_empty_metadata(default, (2, 3))

        tracked = torch.empty((2, 3), requires_grad=True)
        self.assert_empty_metadata(tracked, (2, 3), requires_grad=True)

        with torch.no_grad():
            no_grad_default = torch.empty((2, 3))
            no_grad_tracked = torch.empty((2, 3), requires_grad=True)
        self.assert_empty_metadata(no_grad_default, (2, 3))
        self.assert_empty_metadata(no_grad_tracked, (2, 3), requires_grad=True)

        tracked.sum().backward()
        self.assertEqual(tracked.grad.tolist(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    def test_existing_sequence_and_keyword_forms_are_supported(self):
        class CustomSequence(Sequence):
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __getitem__(self, index):
                return self.values[index]

        class StatefulIndexDimension:
            def __init__(self, values):
                self.values = values
                self.calls = 0

            def __index__(self):
                value = self.values[min(self.calls, len(self.values) - 1)]
                self.calls += 1
                return value

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
                tensor = torch.empty(size)
                self.assert_empty_metadata(tensor, expected_shape)

        self.assert_empty_metadata(torch.empty(size=(2,)), (2,))
        self.assert_empty_metadata(torch.empty(size=(2, True)), (2, 1))
        self.assert_empty_metadata(torch.empty(size=np.array([2])), (2,))

        stateful = StatefulIndexDimension((2, 3, 4))
        stateful_tensor = torch.empty(stateful, 3)
        self.assertEqual(stateful_tensor.shape, (4, 3))
        self.assertEqual(stateful.calls, 3)

        with self.assertRaisesRegex(
            TypeError,
            re.escape("empty(): argument 'size' must be tuple of ints, not int"),
        ):
            torch.empty(size=2)
        with self.assertRaisesRegex(
            TypeError,
            'empty\\(\\) missing 1 required positional arguments: "size"',
        ):
            torch.empty(shape=(2,))

        with self.assertRaisesRegex(
            TypeError,
            re.escape("empty() got multiple values for argument 'size'"),
        ):
            torch.empty(2, 3, size=(2, 3))

        for size in (TupleIndex((4,)), ListIndex([4])):
            with self.subTest(size=type(size).__name__):
                with self.assertRaisesRegex(
                    TypeError,
                    re.escape("empty() takes 1 positional argument but 2 were given"),
                ):
                    torch.empty(size, 3)
                self.assertEqual(size.calls, 1)

        for size in ((1,), [1], range(1)):
            for competing_keyword in ({"wat": 1}, {"size": (1,)}, {"requires_grad": 1}):
                with self.subTest(size=size, competing_keyword=competing_keyword):
                    with self.assertRaisesRegex(
                        TypeError,
                        re.escape("empty() takes 1 positional argument but 2 were given"),
                    ):
                        torch.empty(size, True, **competing_keyword)

    def test_invalid_dimensions_and_allocation_overflow(self):
        class IndexDimension:
            def __init__(self, value):
                self.value = value

            def __index__(self):
                return self.value

        for dimension in (-1, IndexDimension(-1)):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(
                    RuntimeError,
                    re.escape("Trying to create tensor with negative dimension -1: [-1]"),
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

        variadic_failures = (
            (
                lambda: torch.empty(2, -1),
                RuntimeError,
                re.escape("Trying to create tensor with negative dimension -1: [2, -1]"),
            ),
            (
                lambda: torch.empty(2, IndexDimension(-1)),
                RuntimeError,
                re.escape("Trying to create tensor with negative dimension -1: [2, -1]"),
            ),
            (
                lambda: torch.empty(True, 2),
                TypeError,
                re.escape("empty() takes 1 positional argument but 2 were given"),
            ),
            (
                lambda: torch.empty(2, np.bool_(True)),
                TypeError,
                r"pos 2.*numpy\.bool",
            ),
            (
                lambda: torch.empty(2, 2**63),
                TypeError,
                r"pos 2.*Overflow when unpacking long long",
            ),
        )
        for call, error_type, message in variadic_failures:
            with self.subTest(call=call):
                with self.assertRaisesRegex(error_type, message):
                    call()

        sequence_failures = (
            (
                lambda: torch.empty((True, 2)),
                TypeError,
                re.escape(
                    "empty(): argument 'size' (position 1) must be tuple of ints, "
                    "but found element of type bool at pos 0"
                ),
            ),
            (
                lambda: torch.empty([True, 2]),
                TypeError,
                re.escape(
                    "empty(): argument 'size' (position 1) must be tuple of ints, "
                    "but found element of type bool at pos 0"
                ),
            ),
            (
                lambda: torch.empty((np.bool_(True), 2)),
                TypeError,
                re.escape(
                    "empty(): argument 'size' (position 1) must be tuple of ints, "
                    "but found element of type numpy.bool at pos 0"
                ),
            ),
            (
                lambda: torch.empty(size=(True,)),
                TypeError,
                re.escape("empty(): argument 'size' must be tuple of ints, not tuple"),
            ),
            (
                lambda: torch.empty(size=[True]),
                TypeError,
                re.escape("empty(): argument 'size' must be tuple of ints, not list"),
            ),
            (
                lambda: torch.empty((-1,)),
                RuntimeError,
                re.escape("Trying to create tensor with negative dimension -1: [-1]"),
            ),
            (
                lambda: torch.empty([2, -1]),
                RuntimeError,
                re.escape("Trying to create tensor with negative dimension -1: [2, -1]"),
            ),
            (
                lambda: torch.empty((2, IndexDimension(-1))),
                RuntimeError,
                re.escape("Trying to create tensor with negative dimension -1: [2, -1]"),
            ),
            (
                lambda: torch.empty((2**63, 0)),
                TypeError,
                r"pos 1.*Overflow when unpacking long long",
            ),
            (
                lambda: torch.empty([2**63, 0]),
                TypeError,
                r"pos 1.*Overflow when unpacking long long",
            ),
            (
                lambda: torch.empty([2, np.uint64(2**63)]),
                TypeError,
                r"pos 2.*Overflow when unpacking long long",
            ),
        )
        for call, error_type, message in sequence_failures:
            with self.subTest(call=call):
                with self.assertRaisesRegex(error_type, message):
                    call()

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}]"
            ),
        ):
            torch.empty(sys.maxsize)

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"Storage size calculation overflowed with sizes=[{sys.maxsize}, 2]"
            ),
        ):
            torch.empty(sys.maxsize, 2)

    def test_torch_function_mode_intercepts_before_native_validation(self):
        marker = object()
        negative_size = [-1]
        overflow_size = [2**63, 0]

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
            (lambda: torch.empty(2), (2,), None),
            (lambda: torch.empty(negative_size), (negative_size,), None),
            (lambda: torch.empty(overflow_size), (overflow_size,), None),
            (lambda: torch.empty(2, device="cuda"), (2,), {"device": "cuda"}),
            (lambda: torch.empty(2, pin_memory=True), (2,), {"pin_memory": True}),
            (
                lambda: torch.empty(2, requires_grad=True),
                (2,),
                {"requires_grad": True},
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

    def test_torch_function_mode_does_not_intercept_binding_type_errors(self):
        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return object()

        cases = (
            (
                lambda: torch.empty([True, 2]),
                TypeError,
                re.escape(
                    "empty(): argument 'size' (position 1) must be tuple of ints, "
                    "but found element of type bool at pos 0"
                ),
            ),
            (
                lambda: torch.empty(size=(True,)),
                TypeError,
                re.escape("empty(): argument 'size' must be tuple of ints, not tuple"),
            ),
            (
                lambda: torch.empty(size=2),
                TypeError,
                re.escape("empty(): argument 'size' must be tuple of ints, not int"),
            ),
            (
                lambda: torch.empty(1.2),
                TypeError,
                re.escape(
                    "empty(): argument 'size' (position 1) must be tuple of ints, "
                    "not float"
                ),
            ),
            (
                lambda: torch.empty(2, dtype=object()),
                TypeError,
                re.escape("empty(): argument 'dtype' must be torch.dtype, not object"),
            ),
            (
                lambda: torch.empty(2, requires_grad=1),
                TypeError,
                re.escape("empty(): argument 'requires_grad' must be bool, not int"),
            ),
        )
        for call, error_type, message in cases:
            mode = RecordingMode()
            with self.subTest(call=call):
                with mode:
                    with self.assertRaisesRegex(error_type, message):
                        call()
                    self.assertEqual(
                        torch.overrides._get_current_function_mode_stack(), [mode]
                    )
                self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
                self.assertEqual(mode.calls, [])

    def test_torch_function_mode_forwards_and_restores_the_stack(self):
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
                        func,
                        types,
                        args,
                        kwargs,
                    )
                )
                return func(*args, **(kwargs or {}))

        lower = ForwardingMode("lower")
        upper = ForwardingMode("upper")
        with lower:
            with upper:
                result = torch.empty(2, True)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, upper]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assert_empty_metadata(result, (2, 1))
        self.assertEqual(
            [
                (label, stack, function is torch.empty, types, args, kwargs)
                for label, stack, function, types, args, kwargs in events
            ],
            [
                ("upper", ("lower",), True, (), (2, True), None),
                ("lower", (), True, (), (2, True), {}),
            ],
        )

        expected = ValueError("handler failed")

        class RaisingMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected

        raising = RaisingMode()
        with lower:
            with raising:
                with self.assertRaises(ValueError) as raised:
                    torch.empty(2)
                self.assertIs(raised.exception, expected)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, raising]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        forwarding = ForwardingMode("native-error")
        with forwarding:
            with self.assertRaisesRegex(
                TypeError,
                re.escape(
                    "empty(): argument 'size' (position 1) must be tuple of ints, "
                    "but found element of type bool at pos 0"
                ),
            ):
                torch.empty([True, 2])
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [forwarding]
            )
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
                torch.empty(2)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [declining]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_out_pinned_layout_dtype_device_and_memory_format_boundaries(self):
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("empty(): the 'out' argument is not supported"),
        ):
            torch.empty(2, out=torch.empty(2))
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("empty(): the 'out' argument is not supported"),
        ):
            torch.empty(2, 3, out=torch.empty(2, 3))

        for call, error_type, message in (
            (
                lambda: torch.empty(2, 3, dtype=object()),
                TypeError,
                "empty(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: torch.empty(2, 3, device="cuda"),
                RuntimeError,
                "empty(): device 'cuda' is not supported; only 'cpu' is implemented",
            ),
            (
                lambda: torch.empty(2, layout=object(), out=None),
                TypeError,
                "empty(): argument 'layout' must be torch.layout, not object",
            ),
            (
                lambda: torch.empty(2, pin_memory=0, out=None),
                TypeError,
                "empty(): argument 'pin_memory' must be bool, not int",
            ),
            (
                lambda: torch.empty(2, pin_memory=True, out=None),
                RuntimeError,
                "empty(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: torch.empty(2, 3, pin_memory=True, out=None),
                RuntimeError,
                "empty(): pin_memory=True is not supported; only unpinned CPU storage is implemented",
            ),
            (
                lambda: torch.empty(2, memory_format=torch.contiguous_format),
                TypeError,
                "empty() got an unexpected keyword argument 'memory_format'",
            ),
            (
                lambda: torch.empty(2, memory_format=None),
                TypeError,
                "empty() got an unexpected keyword argument 'memory_format'",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
                    call()

    def test_callable_metadata_exports_copy_pickle_and_reload(self):
        package = importlib.import_module("torch_rs")
        native = package._C
        function = package.empty
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "empty")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.empty")
        self.assertEqual(function.__module__, "torch")
        self.assertIn(
            "empty(*size, *, out=None, dtype=None, layout=torch.strided, "
            "device=None, requires_grad=False, pin_memory=False, "
            "memory_format=torch.contiguous_format) -> Tensor",
            function.__doc__,
        )
        self.assertIsNone(function.__text_signature__)
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertIs(owner, package._C._VariableFunctionsClass)
        self.assertIs(owner.empty, function)
        self.assertIs(native.empty, function)
        self.assertEqual(package.__all__.count("empty"), 1)
        self.assertIs(wildcard_namespace["empty"], function)
        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

        self.assertIs(importlib.reload(native), native)
        self.assertIs(native.empty, function)
        self.assertIs(importlib.reload(package), package)
        self.assertIs(package.empty, function)
        self.assertEqual(package.__all__.count("empty"), 1)


if __name__ == "__main__":
    unittest.main()
