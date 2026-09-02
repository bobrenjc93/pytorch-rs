import inspect
import re
import sys
import types
import unittest

import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorToReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.to differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_identity_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        leaf.sum().backward()
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            module.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ],
                dtype=module.float32,
            ).transpose(0, 1)[1],
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
            leaf,
            tracked,
            leaf.grad,
        )

    def identity_calls(self, module, tensor, other):
        return (
            lambda: tensor.to(),
            lambda: tensor.to(None),
            lambda: tensor.to(module.float32),
            lambda: tensor.to(module.float),
            lambda: tensor.to(dtype=module.float32),
            lambda: tensor.to(dtype=module.float),
            lambda: tensor.to(dtype=None),
            lambda: tensor.to("cpu"),
            lambda: tensor.to(module.device("cpu")),
            lambda: tensor.to(device="cpu"),
            lambda: tensor.to(device=module.device("cpu")),
            lambda: tensor.to(device=None),
            lambda: tensor.to(None, module.float32),
            lambda: tensor.to("cpu", module.float32, True, False),
            lambda: tensor.to(device="cpu", dtype=module.float32),
            lambda: tensor.to(module.float32, non_blocking=True, copy=False),
            lambda: tensor.to(memory_format=None),
            lambda: tensor.to(memory_format=module.preserve_format),
            lambda: tensor.to(memory_format=module.contiguous_format),
            lambda: tensor.to(other),
            lambda: tensor.to(tensor=other),
        )

    def test_default_equivalent_requests_match_pytorch_2_13(self):
        actual_cases = self.make_identity_cases(torch)
        expected_cases = self.make_identity_cases(reference_torch)
        actual_other = torch.tensor([1.0])
        expected_other = reference_torch.tensor([1.0], dtype=reference_torch.float32)

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            actual_calls = self.identity_calls(torch, actual, actual_other)
            expected_calls = self.identity_calls(
                reference_torch, expected, expected_other
            )
            for call_case, (actual_call, expected_call) in enumerate(
                zip(actual_calls, expected_calls, strict=True)
            ):
                with self.subTest(case=case, call_case=call_case):
                    actual_result = actual_call()
                    expected_result = expected_call()
                    self.assertEqual(
                        actual_result is actual, expected_result is expected
                    )
                    self.assertEqual(actual_result.shape, tuple(expected_result.shape))
                    self.assertEqual(actual_result.stride(), expected_result.stride())
                    self.assertEqual(
                        actual_result.storage_offset(),
                        expected_result.storage_offset(),
                    )
                    self.assertEqual(actual_result.dtype is torch.float32, True)
                    self.assertEqual(str(expected_result.dtype), "torch.float32")
                    self.assertEqual(
                        actual_result.requires_grad,
                        expected_result.requires_grad,
                    )
                    self.assertEqual(actual_result.is_leaf, expected_result.is_leaf)

    def test_existing_channel_last_identity_matches_pytorch_2_13(self):
        cases = (
            (
                (2, 3, 4, 5),
                "channels_last",
                torch.channels_last,
                reference_torch.channels_last,
            ),
            (
                (2, 3, 4, 5, 6),
                "channels_last_3d",
                torch.channels_last_3d,
                reference_torch.channels_last_3d,
            ),
        )
        for shape, name, actual_format, expected_format in cases:
            with self.subTest(memory_format=name):
                actual = torch.ones(shape).clone(memory_format=actual_format)
                expected = reference_torch.ones(
                    shape, dtype=reference_torch.float32
                ).clone(memory_format=expected_format)
                actual_result = actual.to(memory_format=actual_format)
                expected_result = expected.to(memory_format=expected_format)
                self.assertEqual(
                    actual_result is actual, expected_result is expected
                )
                self.assertEqual(actual_result.stride(), expected_result.stride())

    def test_descriptor_documentation_and_binding_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "to")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "to")

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual.to, expected.to, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__qualname__, expected_callable.__qualname__
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            with self.assertRaises(ValueError):
                inspect.signature(actual_callable)
            with self.assertRaises(ValueError):
                inspect.signature(expected_callable)

        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(
            hasattr(actual_descriptor, "__module__"),
            hasattr(expected_descriptor, "__module__"),
        )
        self.assertIs(actual_descriptor(actual), actual)
        self.assertIs(expected_descriptor(expected), expected)

        call_pairs = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
            (
                lambda: actual.to(object()),
                lambda: expected.to(object()),
            ),
            (
                lambda: actual.to(non_blocking=0),
                lambda: expected.to(non_blocking=0),
            ),
            (
                lambda: actual.to(copy=0),
                lambda: expected.to(copy=0),
            ),
            (
                lambda: actual.to(memory_format=1),
                lambda: expected.to(memory_format=1),
            ),
            (
                lambda: actual.to(unexpected=True),
                lambda: expected.to(unexpected=True),
            ),
            (
                lambda: actual.to(torch.float32, "cpu"),
                lambda: expected.to(reference_torch.float32, "cpu"),
            ),
            (
                lambda: actual.to(None, None, False, False, False),
                lambda: expected.to(None, None, False, False, False),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def mode_contract(self, module):
        tensor = module.tensor([1.0], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "to")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        recording = RecordingMode(marker)
        with recording:
            intercepted = tensor.to(module.float32, copy=True)
        function, dispatch_types, args, kwargs = recording.calls[0]

        rejected_before_dispatch = RecordingMode(marker)
        try:
            with rejected_before_dispatch:
                tensor.to(non_blocking=0)
        except Exception as error:
            strict_bool_error = (type(error).__name__, str(error))
        else:
            strict_bool_error = None

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.to("cpu", module.float32, True, False)

        declining = RecordingMode(NotImplemented)
        lower = RecordingMode(marker)
        try:
            with lower:
                with declining:
                    tensor.to()
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            declining_error = None

        return {
            "intercepted": intercepted is marker,
            "call_count": len(recording.calls),
            "function_type": type(function).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_is_descriptor": function is descriptor,
            "types_empty": dispatch_types == (),
            "args": len(args) == 2 and args[0] is tensor and args[1] is module.float32,
            "kwargs": (
                None
                if kwargs is None
                else tuple((key, repr(value)) for key, value in kwargs.items())
            ),
            "strict_bool_error": strict_bool_error,
            "strict_bool_call_count": len(rejected_before_dispatch.calls),
            "forwarding_order": order,
            "forwarded_is_receiver": forwarded is tensor,
            "declining_error": declining_error,
            "declining_calls": len(declining.calls),
            "lower_calls": len(lower.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch),
            self.mode_contract(reference_torch),
        )


if __name__ == "__main__":
    unittest.main()
