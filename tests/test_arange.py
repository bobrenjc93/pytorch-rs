import inspect
import math
import pickle
import re
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
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertIs(tensor.layout, torch.strided)
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)

    def assert_error(self, call, error_type, message):
        with self.assertRaisesRegex(error_type, f"^{re.escape(message)}$"):
            call()

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
            with self.subTest(options=options):
                self.assert_default_tensor(
                    torch.arange(end=2.5, **options),
                    [0.0, 1.0, 2.0],
                )

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

    def test_int_overloads_outputs_and_nondefault_options_remain_unsupported(self):
        for endpoint in (3, True, FloatSubclass(3.0), np.float64(3.0)):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(TypeError):
                    torch.arange(endpoint)

        overloads = (
            lambda: torch.arange(0.0, 3.0),
            lambda: torch.arange(0.0, 3.0, 1.0),
            lambda: torch.arange(2.5, end=3.0),
            lambda: torch.arange(start=0.0, end=3.0),
            lambda: torch.arange(3.0, step=1.0),
        )
        for call in overloads:
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^arange\(\): start and step overloads are not supported",
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
        )
        for call in unsupported_options:
            with self.subTest(call=call):
                with self.assertRaises((TypeError, RuntimeError)):
                    call()

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
