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
class TopLevelMultiplyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.multiply differentials require pinned PyTorch 2.13.0"
            )

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

    def test_supported_values_layouts_scalars_and_autograd_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])
        calls = (
            (
                "positional tensors",
                lambda: torch.multiply(actual_left, actual_right),
                lambda: reference_torch.multiply(expected_left, expected_right),
            ),
            (
                "canonical keywords",
                lambda: torch.multiply(input=actual_left, other=actual_right),
                lambda: reference_torch.multiply(
                    input=expected_left, other=expected_right
                ),
            ),
            (
                "legacy aliases",
                lambda: torch.multiply(x1=actual_left, x2=actual_right),
                lambda: reference_torch.multiply(x1=expected_left, x2=expected_right),
            ),
            (
                "tensor scalar",
                lambda: torch.multiply(actual_left[1], np.float32(-0.0)),
                lambda: reference_torch.multiply(
                    expected_left[1], np.float32(-0.0)
                ),
            ),
            (
                "scalar tensor",
                lambda: torch.multiply(np.int64(3), actual_left[1]),
                lambda: reference_torch.multiply(np.int64(3), expected_left[1]),
            ),
            (
                "keyword scalar tensor",
                lambda: torch.multiply(input=-2.5, other=actual_left[1]),
                lambda: reference_torch.multiply(
                    input=-2.5, other=expected_left[1]
                ),
            ),
        )
        for case, actual_call, expected_call in calls:
            self.assert_matches(actual_call(), expected_call(), case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        self.assert_matches(
            torch.multiply(actual_empty, torch.ones((1, 1, 2))),
            reference_torch.multiply(
                expected_empty, reference_torch.ones((1, 1, 2))
            ),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.multiply(-0.0, torch.tensor(values)),
            reference_torch.multiply(-0.0, reference_torch.tensor(values)),
            case="signed zero and non-finites",
        )

        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )
        actual_output = torch.multiply(
            actual_left.transpose(0, 1), actual_right.transpose(0, 1)
        )
        expected_output = reference_torch.multiply(
            expected_left.transpose(0, 1), expected_right.transpose(0, 1)
        )
        self.assert_matches(actual_output, expected_output, case="tracked views")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        torch.multiply(actual_shared, actual_shared).sum().backward()
        reference_torch.multiply(expected_shared, expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_no_grad = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_no_grad = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        with torch.no_grad():
            actual_untracked = torch.multiply(4.0, actual_no_grad)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.multiply(4.0, expected_no_grad)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad")

    def dispatch_observation(self, module):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        function = module.multiply
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        for call in (
            lambda: function(left, right),
            lambda: function(left, 4.0),
            lambda: function(4.0, left),
            lambda: function(input=4.0, other=left),
            lambda: function(x1=left, x2=4.0),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
                    func is not module.mul,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
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

        invalid_observations = []
        for call in (
            lambda: function([], right),
            lambda: function(left, []),
            lambda: function(left, right, unexpected=True),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_result is marker,
            normalized_override_events,
            order,
            tuple(forwarded.tolist()),
            invalid_observations,
        )

    def test_modes_overrides_and_forwarding_match_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )

    def test_binding_type_shape_and_scalar_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.multiply(), lambda: reference_torch.multiply()),
            (lambda: torch.multiply(actual), lambda: reference_torch.multiply(expected)),
            (lambda: torch.multiply(2), lambda: reference_torch.multiply(2)),
            (
                lambda: torch.multiply(actual, actual, actual),
                lambda: reference_torch.multiply(expected, expected, expected),
            ),
            (
                lambda: torch.multiply([], actual),
                lambda: reference_torch.multiply([], expected),
            ),
            (
                lambda: torch.multiply(actual, []),
                lambda: reference_torch.multiply(expected, []),
            ),
            (
                lambda: torch.multiply(input=None, other=actual),
                lambda: reference_torch.multiply(input=None, other=expected),
            ),
            (
                lambda: torch.multiply(actual, x2=[]),
                lambda: reference_torch.multiply(expected, x2=[]),
            ),
            (
                lambda: torch.multiply([], actual, extra=True),
                lambda: reference_torch.multiply([], expected, extra=True),
            ),
            (
                lambda: torch.multiply(actual, actual, input=actual),
                lambda: reference_torch.multiply(expected, expected, input=expected),
            ),
            (
                lambda: torch.multiply(actual, actual, x2=actual),
                lambda: reference_torch.multiply(expected, expected, x2=expected),
            ),
            (
                lambda: torch.multiply(foo=actual),
                lambda: reference_torch.multiply(foo=expected),
            ),
            (
                lambda: torch.multiply(actual, actual, extra=True),
                lambda: reference_torch.multiply(expected, expected, extra=True),
            ),
            (
                lambda: torch.multiply(actual, np.uint64(2**63)),
                lambda: reference_torch.multiply(expected, np.uint64(2**63)),
            ),
            (
                lambda: torch.multiply(2**64, actual),
                lambda: reference_torch.multiply(2**64, expected),
            ),
            (
                lambda: torch.multiply(actual, -(2**63) - 1),
                lambda: reference_torch.multiply(expected, -(2**63) - 1),
            ),
            (
                lambda: torch.multiply(torch.zeros((2, 3)), torch.zeros((4, 2))),
                lambda: reference_torch.multiply(
                    reference_torch.zeros((2, 3)),
                    reference_torch.zeros((4, 2)),
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def callable_contract(self, module):
        function = module.multiply
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
            "distinct_from_mul": function is not module.mul,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.multiply is function,
            "owner_distinct_from_mul": owner.multiply is not owner.mul,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("multiply"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["multiply"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_and_exports_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
