import inspect
import math
import pickle
import re
import struct
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ArangeReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("arange differentials require pinned PyTorch 2.13.0")

    def tensor_contract(self, module, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "values": tensor.tolist(),
            "value_bits": tuple(
                struct.pack("=f", value) for value in tensor.tolist()
            ),
            "dtype": str(tensor.dtype),
            "dtype_identity": tensor.dtype is module.float32,
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
        }

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def two_bound_calls(self, module, start, end):
        return (
            ("positional", lambda: module.arange(start, end)),
            ("keyword", lambda: module.arange(start=start, end=end)),
            (
                "keyword-reversed",
                lambda: module.arange(**{"end": end, "start": start}),
            ),
            ("mixed", lambda: module.arange(start, end=end)),
        )

    def test_values_shapes_and_default_metadata_match_pytorch_2_13(self):
        endpoints = (
            0.0,
            -0.0,
            math.nextafter(0.0, 1.0),
            0.25,
            math.nextafter(1.0, 0.0),
            1.0,
            math.nextafter(1.0, 2.0),
            2.5,
            8.0,
        )
        for end in endpoints:
            for form in ("positional", "keyword"):
                with self.subTest(end=end, form=form):
                    if form == "positional":
                        actual = torch.arange(end)
                        expected = reference_torch.arange(end)
                    else:
                        actual = torch.arange(end=end)
                        expected = reference_torch.arange(end=end)
                    self.assertEqual(
                        self.tensor_contract(torch, actual),
                        self.tensor_contract(reference_torch, expected),
                    )

    def test_two_bound_values_shapes_and_rounding_match_pytorch_2_13(self):
        cases = (
            (-2.5, 2.5),
            (-0.25, 2.25),
            (0.25, 3.25),
            (-math.nextafter(0.0, 1.0), 2.0),
            (-math.nextafter(0.0, 1.0), 16.0),
            (math.nextafter(1.0, 0.0), 3.0),
            (math.nextafter(1.0, 2.0), 3.0),
            (1.0, math.nextafter(3.0, 0.0)),
            (1.0, math.nextafter(3.0, math.inf)),
            (16_777_216.5, 16_777_220.5),
            (-16_777_220.5, -16_777_216.5),
            (16_777_216.5, 16_777_232.5),
        )
        for start, end in cases:
            actual_calls = self.two_bound_calls(torch, start, end)
            expected_calls = self.two_bound_calls(reference_torch, start, end)
            for (form, actual_call), (_, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                with self.subTest(start=start, end=end, form=form):
                    self.assertEqual(
                        self.tensor_contract(torch, actual_call()),
                        self.tensor_contract(reference_torch, expected_call()),
                    )

    def test_default_equivalent_options_match_pytorch_2_13(self):
        option_factories = (
            lambda module: {},
            lambda module: {"out": None},
            lambda module: {"dtype": None},
            lambda module: {"dtype": module.float32},
            lambda module: {"layout": None},
            lambda module: {"layout": module.strided},
            lambda module: {"device": None},
            lambda module: {"device": "cpu"},
            lambda module: {"device": "cpu:0"},
            lambda module: {"device": module.device("cpu")},
            lambda module: {"pin_memory": None},
            lambda module: {"pin_memory": False},
            lambda module: {"requires_grad": None},
            lambda module: {"requires_grad": False},
        )
        for option_factory in option_factories:
            actual_options = option_factory(torch)
            expected_options = option_factory(reference_torch)
            calls = (
                (
                    "one-bound",
                    lambda: torch.arange(2.5, **actual_options),
                    lambda: reference_torch.arange(2.5, **expected_options),
                ),
                (
                    "positional",
                    lambda: torch.arange(-0.5, 2.5, **actual_options),
                    lambda: reference_torch.arange(-0.5, 2.5, **expected_options),
                ),
                (
                    "keyword",
                    lambda: torch.arange(
                        start=-0.5, end=2.5, **actual_options
                    ),
                    lambda: reference_torch.arange(
                        start=-0.5, end=2.5, **expected_options
                    ),
                ),
                (
                    "mixed",
                    lambda: torch.arange(-0.5, end=2.5, **actual_options),
                    lambda: reference_torch.arange(
                        -0.5, end=2.5, **expected_options
                    ),
                ),
            )
            for form, actual_call, expected_call in calls:
                with self.subTest(options=actual_options, form=form):
                    self.assertEqual(
                        self.tensor_contract(torch, actual_call()),
                        self.tensor_contract(reference_torch, expected_call()),
                    )

    def test_fresh_storage_matches_pytorch_2_13(self):
        actual_first = torch.arange(8.5)
        actual_second = torch.arange(8.5)
        expected_first = reference_torch.arange(8.5)
        expected_second = reference_torch.arange(8.5)
        self.assertEqual(
            actual_first.data_ptr() != actual_second.data_ptr(),
            expected_first.data_ptr() != expected_second.data_ptr(),
        )
        self.assertEqual(
            actual_first.is_set_to(actual_second),
            expected_first.is_set_to(expected_second),
        )

        actual_empty_first = torch.arange(0.0)
        actual_empty_second = torch.arange(-0.0)
        expected_empty_first = reference_torch.arange(0.0)
        expected_empty_second = reference_torch.arange(-0.0)
        self.assertEqual(
            actual_empty_first.is_set_to(actual_empty_second),
            expected_empty_first.is_set_to(expected_empty_second),
        )

        actual_bounded_first = torch.arange(-2.5, 3.0)
        actual_bounded_second = torch.arange(start=-2.5, end=3.0)
        expected_bounded_first = reference_torch.arange(-2.5, 3.0)
        expected_bounded_second = reference_torch.arange(start=-2.5, end=3.0)
        self.assertEqual(
            actual_bounded_first.data_ptr() != actual_bounded_second.data_ptr(),
            expected_bounded_first.data_ptr() != expected_bounded_second.data_ptr(),
        )
        self.assertEqual(
            actual_bounded_first.is_set_to(actual_bounded_second),
            expected_bounded_first.is_set_to(expected_bounded_second),
        )

        actual_bounded_empty_first = torch.arange(2.5, 2.5)
        actual_bounded_empty_second = torch.arange(2.5, end=2.5)
        expected_bounded_empty_first = reference_torch.arange(2.5, 2.5)
        expected_bounded_empty_second = reference_torch.arange(2.5, end=2.5)
        self.assertEqual(
            actual_bounded_empty_first.is_set_to(actual_bounded_empty_second),
            expected_bounded_empty_first.is_set_to(expected_bounded_empty_second),
        )

    def test_negative_and_nonfinite_errors_match_pytorch_2_13(self):
        endpoints = (
            -math.nextafter(0.0, 1.0),
            -0.25,
            -1.0,
            float("nan"),
            float("-nan"),
            float("inf"),
            float("-inf"),
        )
        for end in endpoints:
            for form in ("positional", "keyword"):
                with self.subTest(end=end, form=form):
                    if form == "positional":
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end),
                            lambda end=end: reference_torch.arange(end),
                        )
                    else:
                        self.assert_error_matches(
                            lambda end=end: torch.arange(end=end),
                            lambda end=end: reference_torch.arange(end=end),
                        )

    def test_two_bound_equal_reversed_and_nonfinite_ranges_match_pytorch_2_13(self):
        equal_bounds = ((0.0, -0.0), (-2.5, -2.5), (1.0e100, 1.0e100))
        for start, end in equal_bounds:
            actual_calls = self.two_bound_calls(torch, start, end)
            expected_calls = self.two_bound_calls(reference_torch, start, end)
            for (form, actual_call), (_, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                with self.subTest(start=start, end=end, form=form):
                    self.assertEqual(
                        self.tensor_contract(torch, actual_call()),
                        self.tensor_contract(reference_torch, expected_call()),
                    )

        error_bounds = (
            (0.25, -0.25),
            (-2.5, -3.0),
            (1.0e100, -1.0e100),
            (float("nan"), 3.0),
            (float("-nan"), -2.5),
            (-0.25, float("inf")),
            (1.0e6, float("-inf")),
        )
        for start, end in error_bounds:
            actual_calls = self.two_bound_calls(torch, start, end)
            expected_calls = self.two_bound_calls(reference_torch, start, end)
            for (form, actual_call), (_, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_error_matches(actual_call, expected_call)

    def test_oversized_endpoint_errors_match_pytorch_2_13(self):
        endpoints = (
            math.nextafter(float(2**63), 0.0),
            float(2**63),
            math.nextafter(float(2**63), math.inf),
            1.0e100,
        )
        for end in endpoints:
            with self.subTest(end=end):
                self.assert_error_matches(
                    lambda end=end: torch.arange(end),
                    lambda end=end: reference_torch.arange(end),
                )

    def test_oversized_two_bound_errors_match_pytorch_2_13(self):
        bounds = (
            (-math.nextafter(float(2**63), 0.0), 0.0),
            (-float(2**63), 0.0),
            (-math.nextafter(float(2**63), math.inf), 0.0),
            (-1.0e308, 1.0e308),
        )
        for start, end in bounds:
            actual_calls = self.two_bound_calls(torch, start, end)
            expected_calls = self.two_bound_calls(reference_torch, start, end)
            for (form, actual_call), (_, expected_call) in zip(
                actual_calls, expected_calls, strict=True
            ):
                with self.subTest(start=start, end=end, form=form):
                    self.assert_error_matches(actual_call, expected_call)

    def mode_dispatch_observation(self, module):
        function = module.arange
        marker = object()
        intercepted = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append(
                    (
                        func is function,
                        types,
                        args,
                        kwargs,
                        len(module.overrides._get_current_function_mode_stack()),
                    )
                )
                return marker

        for call in (
            lambda: function(2.5),
            lambda: function(end=2.5),
            lambda: function(-0.5, 2.5),
            lambda: function(start=-0.5, end=2.5),
            lambda: function(-0.5, end=2.5),
            lambda: function(3),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
                restored_inside = (
                    module.overrides._get_current_function_mode_stack() == [mode]
                )
            intercepted.append(
                (
                    result is marker,
                    mode.calls,
                    restored_inside,
                    module.overrides._get_current_function_mode_stack() == [],
                )
            )

        forwarding_events = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_events.append(
                    (
                        self.label,
                        tuple(
                            mode.label
                            for mode in module.overrides._get_current_function_mode_stack()
                        ),
                        func is function,
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
                forwarded = function(-0.5, end=2.5)
                nested_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, upper]
                )
            lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )
        stack_empty = module.overrides._get_current_function_mode_stack() == []

        expected_error = ValueError("handler failed")

        class RaisingMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                raise expected_error

        raising = RaisingMode()
        with lower:
            with raising:
                try:
                    function(2.5)
                except Exception as error:
                    handler_error = (
                        type(error).__name__,
                        str(error),
                        error.args,
                        error is expected_error,
                    )
                else:
                    handler_error = None
                handler_error_restored = (
                    module.overrides._get_current_function_mode_stack()
                    == [lower, raising]
                )
            handler_lower_restored = (
                module.overrides._get_current_function_mode_stack() == [lower]
            )

        native_error_mode = ForwardingMode("native-error")
        with native_error_mode:
            try:
                function(-1.0)
            except Exception as error:
                native_error = (type(error).__name__, str(error), error.args)
            else:
                native_error = None
            native_error_restored = (
                module.overrides._get_current_function_mode_stack()
                == [native_error_mode]
            )

        class DecliningMode(module.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return NotImplemented

        declining = DecliningMode()
        with declining:
            try:
                function(2.5)
            except Exception as error:
                declining_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                    error.args[1:] if len(error.args) > 1 else (),
                )
            else:
                declining_error = None
            declining_restored = (
                module.overrides._get_current_function_mode_stack() == [declining]
            )

        return (
            intercepted,
            forwarding_events,
            self.tensor_contract(module, forwarded),
            nested_restored,
            lower_restored,
            stack_empty,
            handler_error,
            handler_error_restored,
            handler_lower_restored,
            native_error,
            native_error_restored,
            declining_error,
            declining_restored,
            module.overrides._get_current_function_mode_stack() == [],
        )

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation(torch),
            self.mode_dispatch_observation(reference_torch),
        )

    def callable_contract(self, module):
        function = module.arange
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.arange is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("arange"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "wildcard_identity": wildcard_namespace["arange"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
