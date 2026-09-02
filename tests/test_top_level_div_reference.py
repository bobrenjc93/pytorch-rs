import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelDivisionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.div differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_values_layouts_ieee_empties_and_argument_forms_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected_right = reference_torch.tensor([[2.0], [-0.0], [float("inf")]])

        for name in ("div", "divide"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            cases = (
                (
                    "positional tensors",
                    lambda actual_function=actual_function: actual_function(
                        actual_left, actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left, expected_right
                    ),
                ),
                (
                    "canonical keywords",
                    lambda actual_function=actual_function: actual_function(
                        input=actual_left, other=actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        input=expected_left, other=expected_right
                    ),
                ),
                (
                    "aliases",
                    lambda actual_function=actual_function: actual_function(
                        x1=actual_left, x2=actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        x1=expected_left, x2=expected_right
                    ),
                ),
                (
                    "a alias",
                    lambda actual_function=actual_function: actual_function(
                        a=actual_left, other=actual_right
                    ),
                    lambda expected_function=expected_function: expected_function(
                        a=expected_left, other=expected_right
                    ),
                ),
                (
                    "rounding mode none",
                    lambda actual_function=actual_function: actual_function(
                        actual_left, actual_right, rounding_mode=None
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left, expected_right, rounding_mode=None
                    ),
                ),
                (
                    "out none",
                    lambda actual_function=actual_function: actual_function(
                        actual_left, actual_right, out=None
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left, expected_right, out=None
                    ),
                ),
                (
                    "tensor/scalar",
                    lambda actual_function=actual_function: actual_function(
                        actual_left[1], np.float32(-0.0)
                    ),
                    lambda expected_function=expected_function: expected_function(
                        expected_left[1], np.float32(-0.0)
                    ),
                ),
                (
                    "scalar/tensor",
                    lambda actual_function=actual_function: actual_function(
                        np.int64(3), actual_left[1]
                    ),
                    lambda expected_function=expected_function: expected_function(
                        np.int64(3), expected_left[1]
                    ),
                ),
            )
            for case, actual_call, expected_call in cases:
                self.assert_matches(actual_call(), expected_call(), case=(name, case))

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        actual_broadcast = torch.ones((1, 1, 2))
        expected_broadcast = reference_torch.ones((1, 1, 2))
        for name in ("div", "divide"):
            self.assert_matches(
                getattr(torch, name)(actual_empty, actual_broadcast),
                getattr(reference_torch, name)(expected_empty, expected_broadcast),
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        actual_special = torch.tensor(values)
        expected_special = reference_torch.tensor(values)
        actual_divisors = torch.tensor(
            [1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0]
        )
        expected_divisors = reference_torch.tensor(
            [1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0]
        )
        for name in ("div", "divide"):
            self.assert_matches(
                getattr(torch, name)(actual_special, actual_divisors),
                getattr(reference_torch, name)(expected_special, expected_divisors),
                case=(name, "signed zero nan infinity"),
            )

    def test_no_grad_matches_pytorch_2_13_for_autograd_operands(self):
        for name in ("div", "divide"):
            actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            actual_right = torch.tensor([[5.0], [7.0]], requires_grad=True)
            expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
            expected_right = reference_torch.tensor([[5.0], [7.0]], requires_grad=True)
            with torch.no_grad():
                actual_output = getattr(torch, name)(
                    actual_left.transpose(0, 1), actual_right.transpose(0, 1)
                )
                actual_scalar = getattr(torch, name)(np.float32(2.0), actual_left)
            with reference_torch.no_grad():
                expected_output = getattr(reference_torch, name)(
                    expected_left.transpose(0, 1), expected_right.transpose(0, 1)
                )
                expected_scalar = getattr(reference_torch, name)(
                    np.float32(2.0), expected_left
                )
            self.assert_matches(actual_output, expected_output, case=(name, "no_grad tensor"))
            self.assert_matches(actual_scalar, expected_scalar, case=(name, "no_grad scalar"))

    @staticmethod
    def dispatch_observation(module, function_name):
        left = module.tensor([2.0])
        right = module.tensor([4.0])
        destination = module.tensor([0.0])
        function = getattr(module, function_name)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
            (lambda: function(2, 3), None),
            (lambda: function(input=left, other=right, out=destination), ("input", "other", "out")),
            (lambda: function(left, right, rounding_mode="floor"), ("rounding_mode",)),
        )
        for call, keywords in calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == keywords,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keywords in (
            (lambda value: function(value, right), None),
            (lambda value: function(left, value), None),
            (lambda value: function(input=left, other=value, out=destination), ("input", "other", "out")),
            (lambda value: function(left, right, rounding_mode=value), ("rounding_mode",)),
            (lambda value: function(left, right, out=value), ("out",)),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    dispatch_types == (Override,),
                    len(args),
                    kwargs is None,
                    kwargs is not None and tuple(kwargs) == keywords,
                )
            )

        order = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("left", tuple(item.__name__ for item in types)))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                order.append(("right", tuple(item.__name__ for item in types)))
                return marker

        both_result = function(LeftOverride(), RightOverride())

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(input=left, other=FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, rounding_mode=1),
            lambda: function(left, right, unexpected=True),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            both_result is marker,
            order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )

    def test_supported_binding_errors_and_callable_metadata_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        for name in ("div", "divide"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            cases = (
                (lambda: actual_function(), lambda: expected_function()),
                (lambda: actual_function(actual), lambda: expected_function(expected)),
                (
                    lambda: actual_function(actual, actual, actual),
                    lambda: expected_function(expected, expected, expected),
                ),
                (
                    lambda: actual_function(actual, actual, input=actual),
                    lambda: expected_function(expected, expected, input=expected),
                ),
                (
                    lambda: actual_function(input=actual, x=actual, other=actual),
                    lambda: expected_function(input=expected, x=expected, other=expected),
                ),
                (
                    lambda: actual_function(input=actual, other=actual, x2=actual),
                    lambda: expected_function(input=expected, other=expected, x2=expected),
                ),
                (
                    lambda: actual_function(actual, actual, rounding_mode=1),
                    lambda: expected_function(expected, expected, rounding_mode=1),
                ),
                (
                    lambda: actual_function(actual, actual, unexpected=True),
                    lambda: expected_function(expected, expected, unexpected=True),
                ),
                (
                    lambda: actual_function(torch.zeros((2, 3)), torch.zeros((4, 2))),
                    lambda: expected_function(
                        reference_torch.zeros((2, 3)),
                        reference_torch.zeros((4, 2)),
                    ),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(name=name, case=case):
                    self.assert_error_matches(actual_call, expected_call)

            actual_owner = actual_function.__reduce__()[1][0]
            expected_owner = expected_function.__reduce__()[1][0]
            wildcard_namespace = {}
            exec("from torch_rs import *", wildcard_namespace)
            try:
                inspect.signature(actual_function)
            except Exception as error:
                actual_signature_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                )
            else:
                actual_signature_error = None
            try:
                inspect.signature(expected_function)
            except Exception as error:
                expected_signature_error = (
                    type(error).__name__,
                    re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
                )
            else:
                expected_signature_error = None
            self.assertEqual(
                {
                    "type": type(actual_function).__name__,
                    "is_builtin": type(actual_function) is types.BuiltinFunctionType,
                    "name": actual_function.__name__,
                    "qualname": actual_function.__qualname__,
                    "module": actual_function.__module__,
                    "owner_name": actual_owner.__name__,
                    "owner_qualname": actual_owner.__qualname__,
                    "owner_module": actual_owner.__module__.replace("torch_rs._C", "torch._C"),
                    "owner_path_identity": actual_owner is torch._C._VariableFunctionsClass,
                    "owner_callable_identity": getattr(actual_owner, name) is actual_function,
                    "doc": actual_function.__doc__,
                    "text_signature": actual_function.__text_signature__,
                    "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(actual_function)),
                    "signature_error": actual_signature_error,
                    "all_count": torch.__all__.count(name),
                    "owner_not_in_all": "_VariableFunctionsClass" not in torch.__all__,
                    "owner_not_top_level": not hasattr(torch, "_VariableFunctionsClass"),
                    "wildcard_identity": wildcard_namespace[name] is actual_function,
                    "pickle_identities": tuple(
                        pickle.loads(pickle.dumps(actual_function, protocol=protocol))
                        is actual_function
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                    ),
                },
                {
                    "type": type(expected_function).__name__,
                    "is_builtin": type(expected_function) is types.BuiltinFunctionType,
                    "name": expected_function.__name__,
                    "qualname": expected_function.__qualname__,
                    "module": expected_function.__module__,
                    "owner_name": expected_owner.__name__,
                    "owner_qualname": expected_owner.__qualname__,
                    "owner_module": expected_owner.__module__,
                    "owner_path_identity": expected_owner is reference_torch._C._VariableFunctionsClass,
                    "owner_callable_identity": getattr(expected_owner, name) is expected_function,
                    "doc": expected_function.__doc__,
                    "text_signature": expected_function.__text_signature__,
                    "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(expected_function)),
                    "signature_error": expected_signature_error,
                    "all_count": reference_torch.__all__.count(name),
                    "owner_not_in_all": "_VariableFunctionsClass" not in reference_torch.__all__,
                    "owner_not_top_level": not hasattr(reference_torch, "_VariableFunctionsClass"),
                    "wildcard_identity": getattr(reference_torch, name) is expected_function,
                    "pickle_identities": tuple(
                        pickle.loads(pickle.dumps(expected_function, protocol=protocol))
                        is expected_function
                        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
                    ),
                },
            )


if __name__ == "__main__":
    unittest.main()
