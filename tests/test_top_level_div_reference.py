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
class TopLevelDivReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.div differentials require pinned PyTorch 2.13.0")

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
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32).tolist()
            expected_bits = (
                expected.detach()
                .cpu()
                .reshape(-1)
                .contiguous()
                .view(reference_torch.int32)
                .tolist()
            )
            expected_bits = [value & 0xFFFF_FFFF for value in expected_bits]
            self.assertEqual(actual_bits, expected_bits)

    def test_values_layouts_empty_ieee_and_argument_forms_match_pytorch_2_13(self):
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
            calls = (
                (
                    "positional tensors",
                    lambda: actual_function(actual_left, actual_right),
                    lambda: expected_function(expected_left, expected_right),
                ),
                (
                    "canonical keywords",
                    lambda: actual_function(input=actual_left, other=actual_right),
                    lambda: expected_function(
                        input=expected_left, other=expected_right
                    ),
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
                    "a alias",
                    lambda: actual_function(a=actual_left, other=actual_right),
                    lambda: expected_function(a=expected_left, other=expected_right),
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
            )
            for case, actual_call, expected_call in calls:
                self.assert_matches(actual_call(), expected_call(), case=(name, case))

            actual_offset = torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            ).transpose(0, 1)[1]
            expected_offset = reference_torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
            ).transpose(0, 1)[1]
            self.assert_matches(
                actual_function(actual_offset, torch.tensor([2.0, -0.0])),
                expected_function(
                    expected_offset, reference_torch.tensor([2.0, -0.0])
                ),
                case=(name, "offset noncontiguous tensor"),
            )

            for case, scalar in (
                ("python bool", True),
                ("python int", -2),
                ("python float", 2.5),
                ("numpy bool", np.bool_(True)),
                ("numpy int", np.int64(3)),
                ("numpy float signed zero", np.float32(-0.0)),
                ("python inf", float("inf")),
                ("python nan", float("nan")),
            ):
                self.assert_matches(
                    actual_function(actual_offset, scalar),
                    expected_function(expected_offset, scalar),
                    case=(name, "positional scalar", case),
                )
                self.assert_matches(
                    actual_function(input=actual_offset, other=scalar),
                    expected_function(input=expected_offset, other=scalar),
                    case=(name, "keyword scalar", case),
                )
                self.assert_matches(
                    actual_function(scalar, actual_offset),
                    expected_function(scalar, expected_offset),
                    case=(name, "scalar-left", case),
                )
                self.assert_matches(
                    actual_function(input=scalar, other=actual_offset),
                    expected_function(input=scalar, other=expected_offset),
                    case=(name, "keyword scalar-left", case),
                )

            actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
            expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
            self.assert_matches(
                actual_function(actual_empty, torch.ones((1, 1, 2))),
                expected_function(
                    expected_empty, reference_torch.ones((1, 1, 2))
                ),
                case=(name, "strided broadcast empty"),
            )
            self.assert_matches(
                actual_function(2.0, actual_empty),
                expected_function(2.0, expected_empty),
                case=(name, "scalar-left strided broadcast empty"),
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
            expected_special = reference_torch.tensor(
                memoryview(special_bits.view(np.float32))
            )
            actual_divisors = torch.tensor(
                [1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0]
            )
            expected_divisors = reference_torch.tensor(
                [1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0]
            )
            self.assert_matches(
                actual_function(actual_special, actual_divisors),
                expected_function(expected_special, expected_divisors),
                case=(name, "signed zero nan infinity"),
            )
            for scalar in (0.0, -0.0, float("inf"), float("-inf"), float("nan")):
                self.assert_matches(
                    actual_function(scalar, actual_divisors),
                    expected_function(scalar, expected_divisors),
                    case=(name, "scalar-left signed zero nan infinity", scalar),
                )

    def test_no_grad_operands_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            actual_right = torch.tensor([[5.0], [7.0]], requires_grad=True)
            expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
            expected_right = reference_torch.tensor(
                [[5.0], [7.0]], requires_grad=True
            )
            with torch.no_grad():
                actual_tensor = actual_function(
                    actual_left.transpose(0, 1), actual_right.transpose(0, 1)
                )
                actual_scalar = actual_function(actual_left, 2.0)
                actual_reflected_scalar = actual_function(2.0, actual_left)
            with reference_torch.no_grad():
                expected_tensor = expected_function(
                    expected_left.transpose(0, 1),
                    expected_right.transpose(0, 1),
                )
                expected_scalar = expected_function(expected_left, 2.0)
                expected_reflected_scalar = expected_function(2.0, expected_left)

            self.assert_matches(actual_tensor, expected_tensor, case=(name, "tensor"))
            self.assert_matches(actual_scalar, expected_scalar, case=(name, "scalar"))
            self.assert_matches(
                actual_reflected_scalar,
                expected_reflected_scalar,
                case=(name, "reflected scalar"),
            )

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
        mode_calls = (
            (lambda: function(left, right), None),
            (lambda: function(left, 4.0), None),
            (lambda: function(4.0, left), None),
            (lambda: function(4.0, 2.0), None),
            (lambda: function(input=left, other=right), ("input", "other")),
            (lambda: function(x1=left, x2=right), ("x1", "x2")),
            (lambda: function(left, right, rounding_mode="floor"), ("rounding_mode",)),
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

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(input=left, x=left, other=right),
            lambda: function(input=left, other=right, x2=right),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, len(invalid_mode.calls))
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

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=left, other=right, rounding_mode=None)

        return (
            mode_observations,
            invalid_observations,
            override_observations,
            both_result is marker,
            order,
            subclass_result is marker,
            subclass_order,
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            forwarding_order,
            tuple(np.asarray(forwarded).reshape(-1).view(np.uint32)),
        )

    def test_torch_function_dispatch_boundaries_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )

    @staticmethod
    def callable_contract(module, function_name):
        function = getattr(module, function_name)
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
            "owner_callable_identity": getattr(owner, function_name) is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count(function_name),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace[function_name] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

    def test_reload_preserves_top_level_division_callables(self):
        functions = {name: getattr(torch, name) for name in ("div", "divide")}
        reloaded = importlib.reload(torch)
        self.assertIs(reloaded, torch)
        for name, function in functions.items():
            self.assertIs(getattr(torch, name), function)


if __name__ == "__main__":
    unittest.main()
