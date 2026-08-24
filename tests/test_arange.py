import inspect
import math
import pickle
import re
import struct
import types
import unittest

import numpy as np
import torch_rs as torch


class FloatSubclass(float):
    pass


class ArangeTests(unittest.TestCase):
    def assert_default_tensor(self, tensor, values):
        self.assertEqual(tuple(tensor.shape), (len(values),))
        self.assertEqual(tensor.stride(), (1,))
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertEqual(tensor.numel(), len(values))
        self.assertEqual(tensor.tolist(), values)
        self.assertEqual(
            tuple(struct.pack("=f", value) for value in tensor.tolist()),
            tuple(struct.pack("=f", value) for value in values),
        )
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)

    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

    def two_bound_calls(self, start, end):
        return (
            ("positional", lambda: torch.arange(start, end)),
            ("keyword", lambda: torch.arange(start=start, end=end)),
            (
                "keyword-reversed",
                lambda: torch.arange(**{"end": end, "start": start}),
            ),
            ("mixed", lambda: torch.arange(start, end=end)),
        )

    def test_exact_float_endpoint_supports_positional_and_keyword_forms(self):
        cases = (
            (0.0, []),
            (-0.0, []),
            (math.nextafter(0.0, 1.0), [0.0]),
            (0.25, [0.0]),
            (1.0, [0.0]),
            (math.nextafter(1.0, 2.0), [0.0, 1.0]),
            (2.5, [0.0, 1.0, 2.0]),
            (4.0, [0.0, 1.0, 2.0, 3.0]),
        )
        for end, expected in cases:
            for form, call in (
                ("positional", lambda end=end: torch.arange(end)),
                ("keyword", lambda end=end: torch.arange(end=end)),
            ):
                with self.subTest(end=end, form=form):
                    self.assert_default_tensor(call(), expected)

    def test_two_bound_exact_floats_support_all_argument_forms_and_rounding(self):
        cases = (
            (-2.5, 2.5, [-2.5, -1.5, -0.5, 0.5, 1.5]),
            (-0.25, 2.25, [-0.25, 0.75, 1.75]),
            (0.25, 3.25, [0.25, 1.25, 2.25]),
            (-math.nextafter(0.0, 1.0), 2.0, [-0.0, 1.0]),
            (
                -math.nextafter(0.0, 1.0),
                16.0,
                [-0.0, *[float(index) for index in range(1, 16)]],
            ),
            (math.nextafter(1.0, 0.0), 3.0, [1.0, 2.0]),
            (math.nextafter(1.0, 2.0), 3.0, [1.0, 2.0]),
            (1.0, math.nextafter(3.0, 0.0), [1.0, 2.0]),
            (1.0, math.nextafter(3.0, math.inf), [1.0, 2.0, 3.0]),
            (
                16_777_216.5,
                16_777_220.5,
                [16_777_216.0, 16_777_218.0, 16_777_218.0, 16_777_220.0],
            ),
            (
                -16_777_220.5,
                -16_777_216.5,
                [-16_777_220.0, -16_777_220.0, -16_777_218.0, -16_777_218.0],
            ),
            (
                16_777_216.5,
                16_777_232.5,
                [
                    16_777_216.0,
                    16_777_216.0,
                    16_777_218.0,
                    16_777_220.0,
                    16_777_220.0,
                    16_777_220.0,
                    16_777_222.0,
                    16_777_224.0,
                    16_777_224.0,
                    16_777_224.0,
                    16_777_226.0,
                    16_777_228.0,
                    16_777_228.0,
                    16_777_228.0,
                    16_777_230.0,
                    16_777_232.0,
                ],
            ),
        )
        for start, end, expected in cases:
            for form, call in self.two_bound_calls(start, end):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_default_tensor(call(), expected)

    def test_default_equivalent_metadata_is_accepted(self):
        option_cases = (
            {},
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
            {"pin_memory": None},
            {"pin_memory": False},
            {"requires_grad": None},
            {"requires_grad": False},
            {
                "out": None,
                "dtype": torch.float32,
                "layout": torch.strided,
                "device": torch.device("cpu"),
                "pin_memory": False,
                "requires_grad": False,
            },
        )
        for options in option_cases:
            calls = (
                ("one-bound", lambda: torch.arange(end=2.5, **options)),
                ("positional", lambda: torch.arange(-0.5, 2.5, **options)),
                (
                    "keyword",
                    lambda: torch.arange(start=-0.5, end=2.5, **options),
                ),
                ("mixed", lambda: torch.arange(-0.5, end=2.5, **options)),
            )
            for form, call in calls:
                with self.subTest(options=options, form=form):
                    expected = (
                        [0.0, 1.0, 2.0]
                        if form == "one-bound"
                        else [-0.5, 0.5, 1.5]
                    )
                    self.assert_default_tensor(call(), expected)

    def test_each_result_owns_fresh_storage(self):
        first = torch.arange(8.5)
        second = torch.arange(8.5)
        self.assertNotEqual(first.data_ptr(), 0)
        self.assertNotEqual(second.data_ptr(), 0)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertFalse(first.is_set_to(second))

        empty_first = torch.arange(0.0)
        empty_second = torch.arange(-0.0)
        self.assertEqual(empty_first.data_ptr(), 0)
        self.assertEqual(empty_second.data_ptr(), 0)
        self.assertFalse(empty_first.is_set_to(empty_second))

        bounded_first = torch.arange(-2.5, 3.0)
        bounded_second = torch.arange(start=-2.5, end=3.0)
        self.assertNotEqual(bounded_first.data_ptr(), bounded_second.data_ptr())
        self.assertFalse(bounded_first.is_set_to(bounded_second))

        bounded_empty_first = torch.arange(2.5, 2.5)
        bounded_empty_second = torch.arange(2.5, end=2.5)
        self.assertEqual(bounded_empty_first.data_ptr(), 0)
        self.assertEqual(bounded_empty_second.data_ptr(), 0)
        self.assertFalse(bounded_empty_first.is_set_to(bounded_empty_second))

    def test_negative_and_nonfinite_endpoints_match_pytorch_errors(self):
        for end in (-math.nextafter(0.0, 1.0), -0.25, -1.0):
            for form, call in (
                ("positional", lambda end=end: torch.arange(end)),
                ("keyword", lambda end=end: torch.arange(end=end)),
            ):
                with self.subTest(end=end, form=form):
                    self.assert_error(
                        call,
                        RuntimeError,
                        "upper bound and lower bound inconsistent with step sign",
                    )

        for end, rendered in (
            (float("nan"), "nan"),
            (float("-nan"), "-nan"),
            (float("inf"), "inf"),
            (float("-inf"), "-inf"),
        ):
            with self.subTest(end=rendered):
                self.assert_error(
                    lambda end=end: torch.arange(end),
                    RuntimeError,
                    f"unsupported range: 0 -> {rendered}",
                )

    def test_two_bound_equal_reversed_and_nonfinite_ranges_match_pytorch(self):
        for start, end in ((0.0, -0.0), (-2.5, -2.5), (1.0e100, 1.0e100)):
            for form, call in self.two_bound_calls(start, end):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_default_tensor(call(), [])

        for start, end in ((0.25, -0.25), (-2.5, -3.0), (1.0e100, -1.0e100)):
            for form, call in self.two_bound_calls(start, end):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_error(
                        call,
                        RuntimeError,
                        "upper bound and lower bound inconsistent with step sign",
                    )

        cases = (
            (float("nan"), 3.0, "unsupported range: nan -> 3"),
            (float("-nan"), -2.5, "unsupported range: -nan -> -2.5"),
            (-0.25, float("inf"), "unsupported range: -0.25 -> inf"),
            (1.0e6, float("-inf"), "unsupported range: 1e+06 -> -inf"),
        )
        for start, end, message in cases:
            for form, call in self.two_bound_calls(start, end):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_error(call, RuntimeError, message)

    def test_oversized_endpoints_fail_before_allocation(self):
        cases = (
            (
                math.nextafter(float(2**63), 0.0),
                "Storage size calculation overflowed with sizes=[9223372036854774784]",
            ),
            (
                float(2**63),
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (math.nextafter(float(2**63), math.inf), "invalid size, possible overflow?"),
            (1.0e100, "invalid size, possible overflow?"),
        )
        for end, message in cases:
            with self.subTest(end=end):
                self.assert_error(
                    lambda end=end: torch.arange(end), RuntimeError, message
                )

    def test_oversized_two_bound_ranges_fail_before_allocation(self):
        cases = (
            (
                -math.nextafter(float(2**63), 0.0),
                0.0,
                "Storage size calculation overflowed with sizes=[9223372036854774784]",
            ),
            (
                -float(2**63),
                0.0,
                "IntArrayRef contains an int that cannot be represented as a SymInt: -9223372036854775808",
            ),
            (
                -math.nextafter(float(2**63), math.inf),
                0.0,
                "invalid size, possible overflow?",
            ),
            (-1.0e308, 1.0e308, "invalid size, possible overflow?"),
        )
        for start, end, message in cases:
            for form, call in self.two_bound_calls(start, end):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_error(call, RuntimeError, message)

    def test_integer_bounds_steps_outputs_and_nondefault_options_remain_unsupported(self):
        for call in (lambda: torch.arange(), lambda: torch.arange(end=None)):
            with self.subTest(call=call):
                self.assert_error(
                    call,
                    TypeError,
                    "arange() missing required argument 'end'",
                )

        for endpoint in (3, True, FloatSubclass(3.0), np.float64(3.0)):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(TypeError):
                    torch.arange(endpoint)

        bounds = (
            lambda: torch.arange(0, 3.0),
            lambda: torch.arange(0.0, 3),
            lambda: torch.arange(start=0, end=3.0),
            lambda: torch.arange(0.0, end=np.float64(3.0)),
            lambda: torch.arange(FloatSubclass(0.0), 3.0),
        )
        for call in bounds:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        overloads = (
            lambda: torch.arange(0.0, 3.0, 1.0),
            lambda: torch.arange(0.0, 3.0, step=1.0),
            lambda: torch.arange(start=0.0, end=3.0, step=1.0),
            lambda: torch.arange(3.0, step=1.0),
        )
        for call in overloads:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^arange\(\): explicit steps and malformed overloads are not supported",
                ):
                    call()

        destination = torch.full((3,), 9.0)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^arange\(\): the 'out' argument is not supported$",
        ):
            torch.arange(2.5, out=destination)
        self.assertEqual(destination.tolist(), [9.0, 9.0, 9.0])

        unsupported_options = (
            lambda: torch.arange(2.5, dtype=object()),
            lambda: torch.arange(2.5, layout=object()),
            lambda: torch.arange(2.5, device="cuda"),
            lambda: torch.arange(2.5, pin_memory=True),
            lambda: torch.arange(2.5, requires_grad=True),
            lambda: torch.arange(0.0, 2.5, dtype=object()),
            lambda: torch.arange(0.0, 2.5, layout=object()),
            lambda: torch.arange(0.0, 2.5, device="cuda"),
            lambda: torch.arange(0.0, 2.5, pin_memory=True),
            lambda: torch.arange(0.0, 2.5, requires_grad=True),
        )
        for call in unsupported_options:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, RuntimeError)):
                    call()

    def test_torch_function_mode_intercepts_raw_calls_before_native_validation(self):
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
            (lambda: torch.arange(2.5), (2.5,), None),
            (lambda: torch.arange(end=2.5), (), {"end": 2.5}),
            (lambda: torch.arange(-0.5, 2.5), (-0.5, 2.5), None),
            (
                lambda: torch.arange(start=-0.5, end=2.5),
                (),
                {"start": -0.5, "end": 2.5},
            ),
            (
                lambda: torch.arange(-0.5, end=2.5),
                (-0.5,),
                {"end": 2.5},
            ),
            (lambda: torch.arange(3), (3,), None),
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
                self.assertIs(function, torch.arange)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)
                self.assertEqual(handler_stack, ())

    def test_torch_function_mode_forwards_nested_calls_and_restores_the_stack(self):
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
                result = torch.arange(-0.5, end=2.5)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, upper]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])
        self.assert_default_tensor(result, [-0.5, 0.5, 1.5])
        self.assertEqual(
            [
                (label, stack, function is torch.arange, types, args, kwargs)
                for label, stack, function, types, args, kwargs in events
            ],
            [
                ("upper", ("lower",), True, (), (-0.5,), {"end": 2.5}),
                ("lower", (), True, (), (-0.5,), {"end": 2.5}),
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
                    torch.arange(2.5)
                self.assertIs(raised.exception, expected)
                self.assertEqual(
                    torch.overrides._get_current_function_mode_stack(), [lower, raising]
                )
            self.assertEqual(torch.overrides._get_current_function_mode_stack(), [lower])
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

        forwarding = ForwardingMode("native-error")
        with forwarding:
            with self.assertRaisesRegex(
                RuntimeError,
                "^upper bound and lower bound inconsistent with step sign$",
            ):
                torch.arange(-1.0)
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
                r"^Multiple dispatch failed for 'torch\.arange'; all "
                r"__torch_function__ handlers returned NotImplemented:",
            ):
                torch.arange(2.5)
            self.assertEqual(
                torch.overrides._get_current_function_mode_stack(), [declining]
            )
        self.assertEqual(torch.overrides._get_current_function_mode_stack(), [])

    def test_callable_metadata_exports_and_pickling_match_generated_builtins(self):
        function = torch.arange
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)

        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "arange")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.arange")
        self.assertEqual(function.__module__, "torch")
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method arange of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.arange, function)
        self.assertEqual(torch.__all__.count("arange"), 1)
        self.assertIs(wildcard_namespace["arange"], function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(function, protocol)), function)


if __name__ == "__main__":
    unittest.main()
