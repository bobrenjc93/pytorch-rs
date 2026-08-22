import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AllcloseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.allclose differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    @staticmethod
    def make_layout_cases(module):
        strided = module.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
        ).transpose(0, 1)
        offset = module.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]], dtype=module.float32
        ).transpose(0, 1)[1]
        strided_empty = module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)
        return (
            (
                "contiguous close",
                module.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=module.float32
                ),
                module.tensor(
                    [[1.0, 2.00001], [3.0, 4.0]], dtype=module.float32
                ),
            ),
            (
                "strided outside",
                strided,
                module.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0002]],
                    dtype=module.float32,
                ),
            ),
            (
                "offset close",
                offset,
                module.tensor([20.0, 3.0, 4.00002], dtype=module.float32),
            ),
            (
                "contiguous empty",
                module.zeros((2, 0, 3), dtype=module.float32),
                module.ones((2, 0, 3), dtype=module.float32),
            ),
            (
                "strided empty",
                strided_empty,
                module.ones((3, 0, 2), dtype=module.float32),
            ),
            (
                "offset empty",
                strided_empty[1],
                module.ones((0, 2), dtype=module.float32),
            ),
        )

    def test_layout_results_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_input, actual_other = actual_case
            expected_name, expected_input, expected_other = expected_case
            self.assertEqual(case, expected_name)
            with self.subTest(case=case):
                self.assertEqual(actual_input.shape, tuple(expected_input.shape))
                self.assertEqual(actual_input.stride(), expected_input.stride())
                self.assertEqual(
                    actual_input.storage_offset(), expected_input.storage_offset()
                )
                actual = torch.allclose(actual_input, actual_other)
                expected = reference_torch.allclose(expected_input, expected_other)
                self.assertIs(type(actual), bool)
                self.assertEqual(actual, expected)

    @staticmethod
    def numerical_cases():
        smallest_subnormal = float.fromhex("0x1p-149")
        maximum = 3.4028234663852886e38
        return (
            ([10000.0, 1.0e-7], [10000.1, 1.0e-8], {}),
            ([10000.0, 1.0e-8], [10000.1, 1.0e-9], {}),
            ([0.0, -0.0], [-0.0, 0.0], {"rtol": 0.0, "atol": 0.0}),
            (
                [float("inf"), -float("inf")],
                [float("inf"), -float("inf")],
                {},
            ),
            ([float("inf")], [-float("inf")], {"atol": float("inf")}),
            ([float("inf")], [1.0], {"atol": float("inf")}),
            ([maximum], [-maximum], {"atol": float("inf")}),
            ([maximum], [0.0], {"atol": float("inf")}),
            ([float("nan")], [float("nan")], {}),
            ([float("nan")], [float("nan")], {"equal_nan": True}),
            (
                [float("nan")],
                [1.0],
                {"atol": float("inf"), "equal_nan": True},
            ),
            ([1.0], [2.0], {"rtol": 0.5, "atol": 0.0}),
            ([2.0], [1.0], {"rtol": 0.5, "atol": 0.0}),
            ([1.5], [1.0], {"rtol": 0.0, "atol": 0.5}),
            ([1.5], [1.0], {"rtol": 0.0, "atol": 0.499}),
            (
                [smallest_subnormal],
                [0.0],
                {"rtol": 0.0, "atol": smallest_subnormal * 0.5},
            ),
            (
                [smallest_subnormal],
                [0.0],
                {"rtol": 0.0, "atol": smallest_subnormal * 0.50000001},
            ),
        )

    def test_numerical_semantics_match_pytorch_2_13(self):
        for case, (input_values, other_values, kwargs) in enumerate(
            self.numerical_cases()
        ):
            actual_input = torch.tensor(input_values, dtype=torch.float32)
            actual_other = torch.tensor(other_values, dtype=torch.float32)
            expected_input = reference_torch.tensor(
                input_values, dtype=reference_torch.float32
            )
            expected_other = reference_torch.tensor(
                other_values, dtype=reference_torch.float32
            )
            with self.subTest(case=case):
                actual = torch.allclose(actual_input, actual_other, **kwargs)
                expected = reference_torch.allclose(
                    expected_input, expected_other, **kwargs
                )
                self.assertIs(type(actual), bool)
                self.assertEqual(actual, expected)

    @staticmethod
    def autograd_observation(module):
        leaf = module.tensor(
            [1.0, 2.0], dtype=module.float32, requires_grad=True
        )
        input = leaf * 2.0
        other = module.tensor([2.0, 4.0], dtype=module.float32)
        before = (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad is None,
            input.requires_grad,
            input.is_leaf,
            input.data_ptr(),
            input.storage_offset(),
            input.stride(),
        )
        result = module.allclose(input, other)
        unchanged = before == (
            leaf.requires_grad,
            leaf.is_leaf,
            leaf.grad is None,
            input.requires_grad,
            input.is_leaf,
            input.data_ptr(),
            input.storage_offset(),
            input.stride(),
        )
        input.sum().backward()
        return result, unchanged, leaf.grad.tolist()

    def test_autograd_state_matches_pytorch_2_13(self):
        self.assertEqual(
            self.autograd_observation(torch),
            self.autograd_observation(reference_torch),
        )

    def test_callable_metadata_matches_pytorch_2_13(self):
        actual = torch.allclose
        expected = reference_torch.allclose
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        for attribute in (
            "__name__",
            "__qualname__",
            "__module__",
            "__text_signature__",
            "__doc__",
            "__self__",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(getattr(actual, attribute), getattr(expected, attribute))
        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)
        self.assertIs(pickle.loads(pickle.dumps(actual)), actual)
        self.assertIs(pickle.loads(pickle.dumps(expected)), expected)
        self.assertEqual(torch.__all__.count("allclose"), 1)
        self.assertEqual(reference_torch.__all__.count("allclose"), 1)

    def test_binding_and_tolerance_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (lambda: torch.allclose(), lambda: reference_torch.allclose()),
            (lambda: torch.allclose(actual), lambda: reference_torch.allclose(expected)),
            (lambda: torch.allclose(None), lambda: reference_torch.allclose(None)),
            (
                lambda: torch.allclose(actual, actual, 1.0, 1.0, False, None),
                lambda: reference_torch.allclose(
                    expected, expected, 1.0, 1.0, False, None
                ),
            ),
            (
                lambda: torch.allclose([], actual),
                lambda: reference_torch.allclose([], expected),
            ),
            (
                lambda: torch.allclose(actual, []),
                lambda: reference_torch.allclose(expected, []),
            ),
            (
                lambda: torch.allclose(actual, actual, "bad"),
                lambda: reference_torch.allclose(expected, expected, "bad"),
            ),
            (
                lambda: torch.allclose(actual, actual, atol=None),
                lambda: reference_torch.allclose(expected, expected, atol=None),
            ),
            (
                lambda: torch.allclose(actual, actual, equal_nan=1),
                lambda: reference_torch.allclose(expected, expected, equal_nan=1),
            ),
            (
                lambda: torch.allclose(actual, actual, extra=True),
                lambda: reference_torch.allclose(expected, expected, extra=True),
            ),
            (
                lambda: torch.allclose(actual, actual, other=actual),
                lambda: reference_torch.allclose(expected, expected, other=expected),
            ),
            (
                lambda: torch.allclose(actual, actual, 0.0, rtol=0.0),
                lambda: reference_torch.allclose(
                    expected, expected, 0.0, rtol=0.0
                ),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=-1.0e-5),
                lambda: reference_torch.allclose(expected, expected, rtol=-1.0e-5),
            ),
            (
                lambda: torch.allclose(actual, actual, atol=float("nan")),
                lambda: reference_torch.allclose(
                    expected, expected, atol=float("nan")
                ),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=2**2000),
                lambda: reference_torch.allclose(expected, expected, rtol=2**2000),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=torch.ones((1,))),
                lambda: reference_torch.allclose(
                    expected, expected, rtol=reference_torch.ones((1,))
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        valid_cases = (
            (
                lambda: torch.allclose(input=actual, other=actual),
                lambda: reference_torch.allclose(input=expected, other=expected),
            ),
            (
                lambda: torch.allclose(x=actual, x2=actual),
                lambda: reference_torch.allclose(x=expected, x2=expected),
            ),
            (
                lambda: torch.allclose(actual, actual, 0.0, 0.0, True),
                lambda: reference_torch.allclose(expected, expected, 0.0, 0.0, True),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=np.float32(0.0)),
                lambda: reference_torch.allclose(
                    expected, expected, rtol=np.float32(0.0)
                ),
            ),
            (
                lambda: torch.allclose(actual, actual, rtol=torch.tensor(0.0)),
                lambda: reference_torch.allclose(
                    expected, expected, rtol=reference_torch.tensor(0.0)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(valid_cases):
            with self.subTest(valid_case=case):
                self.assertEqual(actual_call(), expected_call())

    @staticmethod
    def dispatch_observation(module):
        tensor = module.tensor([1.0], dtype=module.float32)
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = module.allclose(
                input=tensor,
                other=tensor,
                rtol=-1.0,
                equal_nan=True,
            )
        function, dispatch_types, args, kwargs = mode.calls[0]
        mode_record = (
            result is marker,
            function is module.allclose,
            dispatch_types == (),
            args == (),
            tuple(kwargs) == ("input", "other", "rtol", "equal_nan"),
            kwargs["input"] is tensor,
            kwargs["other"] is tensor,
        )

        events = []

        class LeftOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "left",
                        func is module.allclose,
                        types == (LeftOverride, RightOverride),
                        len(args),
                        kwargs is None,
                    )
                )
                return NotImplemented

        class RightOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append(
                    (
                        "right",
                        func is module.allclose,
                        types == (LeftOverride, RightOverride),
                        len(args),
                        kwargs is None,
                    )
                )
                return marker

        override_result = module.allclose(
            LeftOverride(), RightOverride(), 0.0, 0.0, True
        )

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = module.allclose(tensor, tensor)

        invalid_mode = RecordingMode(marker)
        try:
            with invalid_mode:
                module.allclose(tensor, [], rtol="bad")
        except TypeError:
            invalid_call_count = len(invalid_mode.calls)
        else:
            invalid_call_count = None

        return (
            mode_record,
            override_result is marker,
            events,
            forwarding_order,
            forwarded,
            invalid_call_count,
        )

    def test_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.dispatch_observation(torch),
            self.dispatch_observation(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
