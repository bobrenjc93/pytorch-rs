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

    def test_values_layouts_empties_and_real_scalars_match_pytorch_2_13(self):
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
                    lambda: expected_function(input=expected_left, other=expected_right),
                ),
                (
                    "x aliases",
                    lambda: actual_function(x=actual_left, x2=actual_right),
                    lambda: expected_function(x=expected_left, x2=expected_right),
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
                    "offset tensor scalar",
                    lambda: actual_function(actual_left[1], np.float32(-0.0)),
                    lambda: expected_function(expected_left[1], np.float32(-0.0)),
                ),
                (
                    "scalar tensor",
                    lambda: actual_function(np.int64(3), actual_left[1]),
                    lambda: expected_function(np.int64(3), expected_left[1]),
                ),
                (
                    "keyword scalar tensor",
                    lambda: actual_function(input=-2.5, other=actual_left[1]),
                    lambda: expected_function(input=-2.5, other=expected_left[1]),
                ),
            )
            for case, actual_call, expected_call in calls:
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

    def test_no_grad_matches_and_active_autograd_boundary_is_explicit(self):
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

            expected_tracked = getattr(reference_torch, name)(
                expected_left.transpose(0, 1), expected_right.transpose(0, 1)
            )
            self.assertTrue(expected_tracked.requires_grad)
            with self.assertRaisesRegex(
                RuntimeError,
                rf"^{name}\(\): autograd recording is not supported$",
            ):
                getattr(torch, name)(
                    actual_left.transpose(0, 1), actual_right.transpose(0, 1)
                )

    def dispatch_observation(self, module, name):
        left = module.tensor([2.0])
        right = module.tensor([4.0])
        function = getattr(module, name)
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
            (lambda: function(input=4.0, other=left), ("input", "other")),
            (
                lambda: function(left, right, rounding_mode="floor"),
                ("rounding_mode",),
            ),
            (lambda: function(left, right, out=left), ("out",)),
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
                    None if kwargs is None else tuple(kwargs),
                    keywords,
                )
            )

        override_events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_events.append(("left", func, types, len(args), kwargs))
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_events.append(("right", func, types, len(args), kwargs))
                return marker

        override_result = function(LeftOverride(), RightOverride())
        normalized_override_events = tuple(
            (
                label,
                func is function,
                tuple(item.__name__ for item in types),
                arg_count,
                kwargs is None,
            )
            for label, func, types, arg_count, kwargs in override_events
        )

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append((self.label, func is function))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=4.0, other=left)

        return (
            mode_observations,
            override_result is marker,
            normalized_override_events,
            order,
            tuple(forwarded.tolist()),
        )

    def test_modes_overrides_and_forwarding_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )

    def callable_contract(self, module, name):
        function = getattr(module, name)
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
            "owner_callable_identity": getattr(owner, name) is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count(name),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace[name] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_exports_and_reload_match_pytorch_2_13(self):
        for name in ("div", "divide"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

        old_div = torch.div
        old_divide = torch.divide
        self.assertIs(importlib.reload(torch), torch)
        self.assertIs(torch.div, old_div)
        self.assertIs(torch.divide, old_divide)


if __name__ == "__main__":
    unittest.main()
