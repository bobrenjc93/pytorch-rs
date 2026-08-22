import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorTrueDivideReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("true_divide differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(str(actual.layout), str(expected.layout))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def tensor_cases(module):
        base = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = base.transpose(0, 2)
        return (
            (module.tensor(-0.0, dtype=module.float32), module.tensor(2.0)),
            (
                noncontiguous,
                module.tensor([[2.0], [4.0], [8.0]], dtype=module.float32),
            ),
            (
                noncontiguous[1],
                module.tensor([[2.0], [4.0], [8.0]], dtype=module.float32),
            ),
            (noncontiguous, module.full((4, 3, 2), -2.0)),
            (
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
                module.ones((1, 2), dtype=module.float32),
            ),
        )

    def test_tensor_layouts_broadcasting_and_empty_outputs_match_pytorch_2_13(self):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for case, ((actual_left, actual_right), (expected_left, expected_right)) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for binding in ("positional", "keyword"):
                actual = (
                    actual_left.true_divide(actual_right)
                    if binding == "positional"
                    else actual_left.true_divide(other=actual_right)
                )
                expected = (
                    expected_left.true_divide(expected_right)
                    if binding == "positional"
                    else expected_left.true_divide(other=expected_right)
                )
                self.assert_tensor_matches(
                    actual, expected, case=(case, binding)
                )

        actual_left, actual_right = actual_cases[1]
        expected_left, expected_right = expected_cases[1]
        self.assert_tensor_matches(
            actual_left.true_divide(x2=actual_right),
            expected_left.true_divide(x2=expected_right),
            case="x2 keyword",
        )

    def test_real_scalar_values_and_bits_match_pytorch_2_13(self):
        actual_input = torch.tensor([1.0, -2.0, 0.0, -0.0])
        expected_input = reference_torch.tensor(
            [1.0, -2.0, 0.0, -0.0], dtype=reference_torch.float32
        )
        for scalar in (
            True,
            False,
            -3,
            2**64 - 1,
            2.5,
            -0.0,
            float("inf"),
            np.bool_(True),
            np.int64(-3),
            np.uint64(2**63 - 1),
            np.float16(2.0),
            np.float32(-0.0),
            np.float64(2.5),
        ):
            with self.subTest(scalar=repr(scalar)):
                self.assert_tensor_matches(
                    actual_input.true_divide(other=scalar),
                    expected_input.true_divide(other=scalar),
                    case=repr(scalar),
                )

        pairs = (
            (0x3F800000, 0x00000000),
            (0xBF800000, 0x00000000),
            (0x3F800000, 0x80000000),
            (0xBF800000, 0x80000000),
            (0x00000000, 0x40000000),
            (0x80000000, 0x40000000),
            (0x00000000, 0xC0000000),
            (0x80000000, 0xC0000000),
            (0x00000000, 0x00000000),
            (0x7F800000, 0x40000000),
            (0xFF800000, 0x40000000),
            (0x3F800000, 0x7F800000),
            (0xBF800000, 0x7F800000),
            (0x7F800000, 0x7F800000),
            (0x7FC12345, 0x3F800000),
            (0xFFC54321, 0x3F800000),
            (0x7F812345, 0x3F800000),
            (0xFF812345, 0x3F800000),
        )
        left = memoryview(
            np.asarray([value for value, _ in pairs], dtype=np.uint32).view(np.float32)
        )
        right = memoryview(
            np.asarray([value for _, value in pairs], dtype=np.uint32).view(np.float32)
        )
        self.assert_tensor_matches(
            torch.tensor(left).true_divide(torch.tensor(right)),
            reference_torch.tensor(left).true_divide(reference_torch.tensor(right)),
            case="IEEE edge bits",
        )

    def test_disabled_torch_function_scalar_is_native_and_absent_from_mode_types(self):
        disabled_handler = reference_torch._C._disabled_torch_function_impl

        class DisabledInt(int):
            __torch_function__ = disabled_handler

        value = DisabledInt(2)
        actual_input = torch.tensor([8.0, -4.0])
        expected_input = reference_torch.tensor(
            [8.0, -4.0], dtype=reference_torch.float32
        )
        for binding in ("positional", "other", "x2"):
            with self.subTest(binding=binding):
                if binding == "positional":
                    actual = actual_input.true_divide(value)
                    expected = expected_input.true_divide(value)
                else:
                    actual = actual_input.true_divide(**{binding: value})
                    expected = expected_input.true_divide(**{binding: value})
                self.assert_tensor_matches(actual, expected, case=binding)

        def exercise_mode(module, *, forward):
            input = module.tensor([8.0])
            descriptor = inspect.getattr_static(module.Tensor, "true_divide")

            class Mode(module.overrides.TorchFunctionMode):
                def __init__(self):
                    self.calls = []

                def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                    self.calls.append((func, dispatch_types, args, kwargs))
                    if forward:
                        return func(*args, **(kwargs or {}))
                    return "marker"

            mode = Mode()
            with mode:
                result = input.true_divide(other=value)
            func, dispatch_types, args, kwargs = mode.calls[0]
            trace = (
                len(mode.calls),
                func is descriptor,
                func.__qualname__,
                tuple(item.__name__ for item in dispatch_types),
                len(args),
                args[0] is input,
                tuple(kwargs),
                kwargs["other"] is value,
            )
            return result, trace

        actual_marker, actual_trace = exercise_mode(torch, forward=False)
        expected_marker, expected_trace = exercise_mode(reference_torch, forward=False)
        self.assertEqual(actual_marker, expected_marker)
        self.assertEqual(actual_trace, expected_trace)
        self.assertEqual(actual_trace[3], ())

        actual_forwarded, actual_trace = exercise_mode(torch, forward=True)
        expected_forwarded, expected_trace = exercise_mode(
            reference_torch, forward=True
        )
        self.assertEqual(actual_trace, expected_trace)
        self.assertEqual(actual_trace[3], ())
        self.assert_tensor_matches(
            actual_forwarded,
            expected_forwarded,
            case="disabled handler mode forwarding",
        )

    def test_failed_scalar_override_probe_order_matches_pytorch_2_13(self):
        def exercise(module, scalar_base, binding, mode_behavior):
            descriptor_events = []
            mode_calls = []

            class StatefulDescriptor:
                def __init__(self):
                    self.lookups = 0

                def __get__(self, instance, owner):
                    self.lookups += 1
                    resolution = self.lookups
                    descriptor_events.append(("lookup", resolution))
                    if resolution == 1:
                        raise RuntimeError("transient probe failure")

                    def handler(func, dispatch_types, args=(), kwargs=None):
                        descriptor_events.append(
                            (
                                "handler",
                                resolution,
                                func.__qualname__,
                                tuple(value.__name__ for value in dispatch_types),
                            )
                        )
                        return f"override-{resolution}"

                    return handler

            class Scalar(scalar_base):
                __torch_function__ = StatefulDescriptor()

            scalar = Scalar(2)
            tensor = module.tensor([8.0])

            def invoke():
                if binding == "positional":
                    return tensor.true_divide(scalar)
                return tensor.true_divide(**{binding: scalar})

            if mode_behavior == "direct":
                result = invoke()
            else:
                class Mode(module.overrides.TorchFunctionMode):
                    def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                        mode_calls.append(
                            (
                                func.__qualname__,
                                tuple(value.__name__ for value in dispatch_types),
                                len(args),
                                None if kwargs is None else tuple(kwargs),
                            )
                        )
                        if mode_behavior == "forward":
                            return func(*args, **(kwargs or {}))
                        return "mode-marker"

                with Mode():
                    result = invoke()

            outcome = (
                ("tensor", result.tolist())
                if hasattr(result, "tolist")
                else ("value", result)
            )
            return outcome, tuple(descriptor_events), tuple(mode_calls)

        for scalar_base in (int, float):
            for binding in ("positional", "other", "x2"):
                for mode_behavior in ("direct", "intercept", "forward"):
                    with self.subTest(
                        scalar_base=scalar_base.__name__,
                        binding=binding,
                        mode_behavior=mode_behavior,
                    ):
                        actual = exercise(
                            torch, scalar_base, binding, mode_behavior
                        )
                        expected = exercise(
                            reference_torch, scalar_base, binding, mode_behavior
                        )
                        self.assertEqual(actual, expected)
                        if mode_behavior != "forward":
                            self.assertEqual(actual[1], (("lookup", 1),))
                        if mode_behavior != "direct":
                            self.assertEqual(actual[2][0][1], ())

    def test_no_grad_outputs_match_while_recording_remains_explicitly_unsupported(self):
        actual_left = torch.tensor([[2.0, 4.0]], requires_grad=True).transpose(0, 1)
        expected_left = reference_torch.tensor(
            [[2.0, 4.0]], requires_grad=True
        ).transpose(0, 1)
        actual_right = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[1.0, 2.0]], requires_grad=True
        )

        with self.assertRaisesRegex(
            RuntimeError, r"^true_divide\(\): autograd recording is not supported$"
        ):
            actual_left.true_divide(actual_right)
        self.assertTrue(expected_left.true_divide(expected_right).requires_grad)

        with torch.no_grad():
            actual_tensor = actual_left.true_divide(other=actual_right)
            actual_scalar = actual_left.true_divide(-2.0)
        with reference_torch.no_grad():
            expected_tensor = expected_left.true_divide(other=expected_right)
            expected_scalar = expected_left.true_divide(-2.0)
        self.assert_tensor_matches(
            actual_tensor, expected_tensor, case="no_grad tensor operands"
        )
        self.assert_tensor_matches(
            actual_scalar, expected_scalar, case="no_grad scalar operand"
        )

    def test_descriptor_metadata_and_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_other = torch.tensor([2.0])
        expected_other = reference_torch.tensor([2.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "true_divide")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "true_divide"
        )

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "true_divide")
            self.assertIsNone(descriptor.__text_signature__)
            self.assertFalse(hasattr(descriptor, "__module__"))
            with self.assertRaises(ValueError):
                inspect.signature(descriptor)
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertEqual(
            actual_descriptor.__qualname__, expected_descriptor.__qualname__
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))
        for bound in (actual.true_divide, expected.true_divide):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "true_divide")
            self.assertIsNone(bound.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(bound)
        self.assertEqual(
            actual.true_divide.__qualname__, expected.true_divide.__qualname__
        )

        self.assert_tensor_matches(
            actual_descriptor(actual, other=actual_other),
            expected_descriptor(expected, other=expected_other),
            case="unbound call",
        )
        cases = (
            (lambda: actual.true_divide(), lambda: expected.true_divide()),
            (
                lambda: actual.true_divide(actual_other, actual_other),
                lambda: expected.true_divide(expected_other, expected_other),
            ),
            (
                lambda: actual.true_divide(actual_other, other=actual_other),
                lambda: expected.true_divide(expected_other, other=expected_other),
            ),
            (
                lambda: actual.true_divide(actual_other, out=actual),
                lambda: expected.true_divide(expected_other, out=expected),
            ),
            (lambda: actual.true_divide(wat=actual), lambda: expected.true_divide(wat=expected)),
            (lambda: actual.true_divide([]), lambda: expected.true_divide([])),
            (
                lambda: actual.true_divide(other=None),
                lambda: expected.true_divide(other=None),
            ),
            (
                lambda: actual.true_divide(x2=[]),
                lambda: expected.true_divide(x2=[]),
            ),
            (
                lambda: actual.true_divide(np.uint64(2**63)),
                lambda: expected.true_divide(np.uint64(2**63)),
            ),
            (lambda: actual.true_divide(2**64), lambda: expected.true_divide(2**64)),
            (
                lambda: actual.true_divide(-(2**63) - 1),
                lambda: expected.true_divide(-(2**63) - 1),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(self=actual, other=actual_other),
                lambda: expected_descriptor(self=expected, other=expected_other),
            ),
            (
                lambda: actual_descriptor(1, actual_other),
                lambda: expected_descriptor(1, expected_other),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_torch_function_mode_calls_and_forwarding_match_pytorch_2_13(self):
        def exercise(module):
            tensor = module.tensor([8.0])
            other = module.tensor([2.0])
            descriptor = inspect.getattr_static(module.Tensor, "true_divide")
            marker = object()

            class RecordingMode(module.overrides.TorchFunctionMode):
                def __init__(self, result):
                    self.result = result
                    self.calls = []

                def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                    self.calls.append((func, dispatch_types, args, kwargs))
                    return self.result

            mode = RecordingMode(marker)
            with mode:
                result = tensor.true_divide(other=other)
            func, dispatch_types, args, kwargs = mode.calls[0]
            recorded = (
                result is marker,
                func is descriptor,
                func.__qualname__,
                tuple(value.__name__ for value in dispatch_types),
                len(args),
                args[0] is tensor,
                tuple(kwargs),
                kwargs["other"] is other,
            )

            order = []

            class ForwardingMode(module.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                    order.append(self.label)
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = tensor.true_divide(other=other)

            declining = RecordingMode(NotImplemented)
            try:
                with declining:
                    tensor.true_divide(other)
            except Exception as error:
                decline = type(error).__name__, str(error).splitlines()[0]
            else:
                raise AssertionError("a declining true_divide mode was accepted")
            return recorded, tuple(order), tuple(forwarded.shape), forwarded.item(), decline

        self.assertEqual(exercise(torch), exercise(reference_torch))


if __name__ == "__main__":
    unittest.main()
