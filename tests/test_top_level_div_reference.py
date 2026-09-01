import copy
import importlib
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
                    lambda: actual_function(actual_left, actual_right),
                    lambda: expected_function(expected_left, expected_right),
                ),
                (
                    "canonical keywords",
                    lambda: actual_function(input=actual_left, other=actual_right),
                    lambda: expected_function(input=expected_left, other=expected_right),
                ),
                (
                    "x aliases",
                    lambda: actual_function(x=actual_left, x2=actual_right),
                    lambda: expected_function(x=expected_left, x2=expected_right),
                ),
                (
                    "x1 aliases",
                    lambda: actual_function(x1=actual_left, x2=actual_right),
                    lambda: expected_function(x1=expected_left, x2=expected_right),
                ),
                (
                    "explicit true division",
                    lambda: actual_function(
                        actual_left, actual_right, rounding_mode=None
                    ),
                    lambda: expected_function(
                        expected_left, expected_right, rounding_mode=None
                    ),
                ),
                (
                    "out none",
                    lambda: actual_function(actual_left, actual_right, out=None),
                    lambda: expected_function(expected_left, expected_right, out=None),
                ),
                (
                    "tensor/scalar",
                    lambda: actual_function(actual_left[1], np.float32(-0.0)),
                    lambda: expected_function(expected_left[1], np.float32(-0.0)),
                ),
                (
                    "scalar/tensor",
                    lambda: actual_function(np.int64(3), actual_left[1]),
                    lambda: expected_function(np.int64(3), expected_left[1]),
                ),
                (
                    "keyword scalar/tensor",
                    lambda: actual_function(input=-2.5, other=actual_left[1]),
                    lambda: expected_function(input=-2.5, other=expected_left[1]),
                ),
                (
                    "numpy bool scalar",
                    lambda: actual_function(actual_left[1], np.bool_(True)),
                    lambda: expected_function(expected_left[1], np.bool_(True)),
                ),
            )
            for case, actual_call, expected_call in cases:
                self.assert_matches(
                    actual_call(), expected_call(), case=(name, case)
                )

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
        actual_special = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected_special = reference_torch.tensor(memoryview(special_bits.view(np.float32)))
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
                actual_scalar = getattr(torch, name)(2.0, actual_left)
            with reference_torch.no_grad():
                expected_output = getattr(reference_torch, name)(
                    expected_left.transpose(0, 1), expected_right.transpose(0, 1)
                )
                expected_scalar = getattr(reference_torch, name)(2.0, expected_left)
            self.assert_matches(actual_output, expected_output, case=(name, "no_grad tensor"))
            self.assert_matches(actual_scalar, expected_scalar, case=(name, "no_grad scalar"))

    @staticmethod
    def dispatch_observation(module, function_name):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        destination = module.tensor([0.0])
        function = getattr(module, function_name)
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
            (lambda: function(input=4.0, other=left, rounding_mode="floor"), ("input", "other", "rounding_mode")),
            (lambda: function(left, right, out=destination), ("out",)),
        )
        for call, keyword_names in mode_calls:
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
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value, right), None),
            (lambda value: function(left, value), None),
            (lambda value: function(input=left, other=value), "other"),
            (lambda value: function(left, right, rounding_mode=value), "rounding_mode"),
            (lambda value: function(left, right, out=value), "out"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
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

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(("base", tuple(item.__name__ for item in types)))
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), DerivedOverride())

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(input=left, other=FallbackOverride())

        wide_mode = RecordingMode()
        with wide_mode:
            wide_result = function(left, np.uint64(2**63))

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
            subclass_result is marker,
            subclass_order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            wide_result is marker,
            len(wide_mode.calls),
            invalid_observations,
        )

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        for function_name in ("div", "divide"):
            with self.subTest(function=function_name):
                self.assertEqual(
                    self.dispatch_observation(torch, function_name),
                    self.dispatch_observation(reference_torch, function_name),
                )

    @staticmethod
    def callable_contract(module, function_name):
        function = getattr(module, function_name)
        owner = function.__reduce__()[1][0]
        direct_namespace = {}
        exec(f"from {module.__name__} import {function_name} as imported", direct_namespace)
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
            "distinct_from_div": function is not module.div
            if function_name == "divide"
            else True,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": getattr(owner, function_name) is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count(function_name),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "direct_import_identity": direct_namespace["imported"] is function,
            "wildcard_identity": wildcard_namespace[function_name] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        for function_name in ("div", "divide"):
            with self.subTest(function=function_name):
                self.assertEqual(
                    self.callable_contract(torch, function_name),
                    self.callable_contract(reference_torch, function_name),
                )

        actual_descriptors = {
            name: inspect.getattr_static(torch.Tensor, name)
            for name in ("div", "divide")
        }
        actual_reloaded = importlib.reload(torch)
        self.assertIs(actual_reloaded, torch)
        for name, descriptor in actual_descriptors.items():
            self.assertIs(inspect.getattr_static(torch.Tensor, name), descriptor)

    def test_binding_type_and_shape_errors_match_pytorch_2_13(self):
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
                    lambda: actual_function([], actual),
                    lambda: expected_function([], expected),
                ),
                (
                    lambda: actual_function(actual, []),
                    lambda: expected_function(expected, []),
                ),
                (
                    lambda: actual_function(input=None, other=actual),
                    lambda: expected_function(input=None, other=expected),
                ),
                (
                    lambda: actual_function(actual, actual, input=actual),
                    lambda: expected_function(expected, expected, input=expected),
                ),
                (
                    lambda: actual_function(actual, actual, x2=actual),
                    lambda: expected_function(expected, expected, x2=expected),
                ),
                (
                    lambda: actual_function(foo=actual),
                    lambda: expected_function(foo=expected),
                ),
                (
                    lambda: actual_function(actual, actual, extra=True),
                    lambda: expected_function(expected, expected, extra=True),
                ),
                (
                    lambda: actual_function(actual, actual, rounding_mode=1),
                    lambda: expected_function(expected, expected, rounding_mode=1),
                ),
                (
                    lambda: actual_function(torch.zeros((2, 3)), torch.zeros((4, 2))),
                    lambda: expected_function(
                        reference_torch.zeros((2, 3)),
                        reference_torch.zeros((4, 2)),
                    ),
                ),
            )
            for index, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(function=name, index=index):
                    self.assert_error_matches(actual_call, expected_call)

        for name in ("div", "divide"):
            self.assertFalse(hasattr(torch, f"{name}_"))
            self.assertFalse(hasattr(reference_torch, f"{name}_"))


if __name__ == "__main__":
    unittest.main()
