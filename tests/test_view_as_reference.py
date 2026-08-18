import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorViewAsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("view_as differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        noncontiguous = base.transpose(0, 1)
        return (
            (
                "scalar",
                module.tensor(-0.0, dtype=module.float32),
                module.tensor(8.0, dtype=module.float32),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
                module.zeros((2, 0), dtype=module.float32),
            ),
            (
                "empty-same-shape",
                module.zeros((0, 1), dtype=module.float32) + 1,
                module.zeros((0, 1), dtype=module.float32),
            ),
            (
                "contiguous-with-strided-other",
                base,
                module.zeros((4, 6), dtype=module.float32).transpose(0, 1),
            ),
            (
                "contiguous-offset",
                base[1],
                module.zeros((2, 6), dtype=module.float32),
            ),
            (
                "noncontiguous-same-shape",
                noncontiguous,
                module.zeros((3, 2, 4), dtype=module.float32),
            ),
            (
                "noncontiguous-compatible-split",
                noncontiguous,
                module.zeros((3, 2, 2, 2), dtype=module.float32),
            ),
        )

    def tensor_array(self, tensor, module):
        detached = tensor.detach()
        if module is reference_torch:
            return detached.cpu().numpy()
        return np.asarray(detached)

    def test_layout_stride_offset_aliasing_and_values_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_other = actual_case
            expected_name, expected_source, expected_other = expected_case
            self.assertEqual(case, expected_name)
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    if keyword:
                        actual = actual_source.view_as(other=actual_other)
                        expected = expected_source.view_as(other=expected_other)
                    else:
                        actual = actual_source.view_as(actual_other)
                        expected = expected_source.view_as(expected_other)

                    actual_direct = actual_source.reshape(actual_other.shape)
                    expected_direct = expected_source.reshape(expected_other.shape)
                    self.assertIsNot(actual, actual_source)
                    self.assertIsNot(expected, expected_source)
                    self.assertEqual(actual.shape, tuple(expected.shape))
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )
                    self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
                    self.assertEqual(actual.requires_grad, expected.requires_grad)
                    self.assertEqual(actual.is_leaf, expected.is_leaf)
                    np.testing.assert_array_equal(
                        self.tensor_array(actual, torch),
                        self.tensor_array(expected, reference_torch),
                    )
                    self.assertEqual(
                        actual.data_ptr() == actual_source.data_ptr(),
                        expected.data_ptr() == expected_source.data_ptr(),
                    )
                    self.assertEqual(
                        actual.is_set_to(actual_direct),
                        expected.is_set_to(expected_direct),
                    )

    def test_extreme_empty_and_incompatible_errors_match_pytorch_2_13(self):
        maximum = sys.maxsize
        actual_source = torch.zeros((0,))
        expected_source = reference_torch.zeros(
            (0,), dtype=reference_torch.float32
        )
        actual_other = actual_source.reshape((0, maximum, maximum))
        expected_other = expected_source.reshape((0, maximum, maximum))

        actual = actual_source.view_as(actual_other)
        expected = expected_source.view_as(expected_other)
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.numel(), expected.numel())
        self.assertEqual(
            actual.data_ptr() == actual_source.data_ptr(),
            expected.data_ptr() == expected_source.data_ptr(),
        )

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_noncontiguous = torch.tensor(values.tolist()).transpose(0, 1)
        expected_noncontiguous = reference_torch.tensor(
            values.tolist(), dtype=reference_torch.float32
        ).transpose(0, 1)
        self.assert_error_matches(
            lambda: actual_noncontiguous.view_as(torch.zeros((6, 4))),
            lambda: expected_noncontiguous.view_as(
                reference_torch.zeros((6, 4), dtype=reference_torch.float32)
            ),
        )
        self.assert_error_matches(
            lambda: torch.zeros((6,)).view_as(torch.zeros((2, 2))),
            lambda: reference_torch.zeros(
                (6,), dtype=reference_torch.float32
            ).view_as(
                reference_torch.zeros((2, 2), dtype=reference_torch.float32)
            ),
        )

    def autograd_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        other = module.zeros((3, 2), dtype=module.float32, requires_grad=True)
        result = source.view_as(other=other)
        metadata = (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
        )
        weights = module.tensor(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            dtype=module.float32,
        )
        loss = (result * weights).sum()
        loss.backward()
        return metadata, self.tensor_array(leaf.grad, module).copy(), other.grad

    def repeated_backward_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        other = module.zeros((3, 2), dtype=module.float32, requires_grad=True)
        loss = leaf.transpose(0, 1).view_as(other).sum()
        loss.backward()
        loss.backward()
        return self.tensor_array(leaf.grad, module).copy(), other.grad

    def no_grad_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        source = leaf.transpose(0, 1)
        other = module.zeros((3, 2), dtype=module.float32, requires_grad=True)
        with module.no_grad():
            result = source.view_as(other)
        return (
            tuple(result.shape),
            result.stride(),
            result.storage_offset(),
            result.requires_grad,
            result.is_leaf,
            result.data_ptr() == source.data_ptr(),
            leaf.grad,
            other.grad,
        )

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        actual_metadata, actual_grad, actual_other_grad = self.autograd_outcome(torch)
        expected_metadata, expected_grad, expected_other_grad = self.autograd_outcome(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        np.testing.assert_array_equal(actual_grad, expected_grad)
        self.assertIsNone(actual_other_grad)
        self.assertIsNone(expected_other_grad)
        actual_repeated, actual_repeated_other = self.repeated_backward_outcome(
            torch
        )
        expected_repeated, expected_repeated_other = self.repeated_backward_outcome(
            reference_torch
        )
        np.testing.assert_array_equal(actual_repeated, expected_repeated)
        self.assertIsNone(actual_repeated_other)
        self.assertIsNone(expected_repeated_other)
        self.assertEqual(
            self.no_grad_outcome(torch), self.no_grad_outcome(reference_torch)
        )

    def descriptor_contract(self, module):
        tensor = module.tensor([1.0, 2.0], dtype=module.float32)
        other = module.zeros((2, 1), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "view_as")
        bound = tensor.view_as
        contract = []
        for callable_object, expected_type in (
            (descriptor, types.MethodDescriptorType),
            (bound, types.BuiltinMethodType),
        ):
            try:
                inspect.signature(callable_object)
            except Exception as error:
                signature_error = type(error).__name__
            else:
                signature_error = None
            contract.append(
                (
                    type(callable_object) is expected_type,
                    callable_object.__name__,
                    callable_object.__qualname__,
                    callable_object.__doc__,
                    callable_object.__text_signature__,
                    getattr(callable_object, "__module__", "missing"),
                    signature_error,
                )
            )
        return (
            tuple(contract),
            descriptor.__objclass__.__name__,
            descriptor.__objclass__.__module__,
            repr(descriptor),
            descriptor is module.Tensor.view_as,
            descriptor.__get__(None, module.Tensor) is descriptor,
            tuple(descriptor(tensor, other).shape),
            tuple(descriptor(tensor, other=other).shape),
        )

    def test_tensorbase_descriptor_and_documentation_match_pytorch_2_13(self):
        self.assertEqual(
            self.descriptor_contract(torch),
            self.descriptor_contract(reference_torch),
        )

    def mode_contract(self, module):
        tensor = module.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=module.float32
        )
        other = module.zeros((2, 3), dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "view_as")
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        def normalize_call(call):
            function, dispatch_types, args, kwargs = call

            def normalize(value):
                if value is tensor:
                    return "self"
                if value is other:
                    return "other"
                return value

            return (
                function is descriptor,
                function.__qualname__,
                dispatch_types,
                tuple(normalize(argument) for argument in args),
                {key: normalize(value) for key, value in kwargs.items()}
                if kwargs is not None
                else None,
            )

        records = []
        for call in (
            lambda: tensor.view_as(other),
            lambda: tensor.view_as(other=other),
        ):
            mode = RecordingMode(marker)
            with mode:
                result = call()
            records.append((result is marker, tuple(map(normalize_call, mode.calls))))

        invalid = RecordingMode(marker)
        try:
            with invalid:
                tensor.view_as(1)
        except Exception as error:
            invalid_error = type(error).__name__, str(error)
        else:
            self.fail(f"{module.__name__} accepted a non-Tensor other")

        order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append(
                    (self.label, func, dispatch_types, args, kwargs)
                )
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.view_as(other=other)

        declining = RecordingMode(NotImplemented)
        try:
            with declining:
                tensor.view_as(other)
        except Exception as error:
            declining_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x<address>", str(error)),
            )
        else:
            self.fail(f"{module.__name__} accepted a declining mode")

        return {
            "records": tuple(records),
            "invalid": invalid_error,
            "invalid_calls": len(invalid.calls),
            "forwarding": tuple(
                (label, normalize_call((func, dispatch_types, args, kwargs)))
                for label, func, dispatch_types, args, kwargs in order
            ),
            "forwarded": (
                tuple(forwarded.shape),
                forwarded.stride(),
                forwarded.storage_offset(),
                forwarded.data_ptr() == tensor.data_ptr(),
            ),
            "declining": declining_error,
            "declining_calls": len(declining.calls),
            "stack_depth": len(module.overrides._get_current_function_mode_stack()),
        }

    def test_torch_function_modes_match_pytorch_2_13(self):
        self.assertEqual(
            self.mode_contract(torch), self.mode_contract(reference_torch)
        )

    def test_binding_and_type_error_precedence_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        actual_other = torch.tensor([2.0])
        expected_other = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32
        )
        actual_descriptor = inspect.getattr_static(torch.Tensor, "view_as")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "view_as"
        )
        array = np.zeros((2, 3), dtype=np.float32)
        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(self=actual, other=actual_other),
                lambda: expected_descriptor(self=expected, other=expected_other),
            ),
            (
                lambda: actual_descriptor(1, actual_other),
                lambda: expected_descriptor(1, expected_other),
            ),
            (lambda: actual.view_as(), lambda: expected.view_as()),
            (
                lambda: actual.view_as(actual_other, actual_other),
                lambda: expected.view_as(expected_other, expected_other),
            ),
            (
                lambda: actual.view_as(actual_other, other=actual_other),
                lambda: expected.view_as(expected_other, other=expected_other),
            ),
            (
                lambda: actual.view_as(foo=actual_other),
                lambda: expected.view_as(foo=expected_other),
            ),
            (
                lambda: actual.view_as(actual_other, extra=True),
                lambda: expected.view_as(expected_other, extra=True),
            ),
            (lambda: actual.view_as(1), lambda: expected.view_as(1)),
            (lambda: actual.view_as(None), lambda: expected.view_as(None)),
            (lambda: actual.view_as([]), lambda: expected.view_as([])),
            (lambda: actual.view_as(array), lambda: expected.view_as(array)),
            (
                lambda: actual.view_as(other=1),
                lambda: expected.view_as(other=1),
            ),
            (
                lambda: actual.view_as(other=None),
                lambda: expected.view_as(other=None),
            ),
            (
                lambda: actual.view_as(other=[]),
                lambda: expected.view_as(other=[]),
            ),
            (
                lambda: actual.view_as(**{"other": 1, "extra": True}),
                lambda: expected.view_as(**{"other": 1, "extra": True}),
            ),
            (
                lambda: actual.view_as(**{"extra": True, "other": 1}),
                lambda: expected.view_as(**{"extra": True, "other": 1}),
            ),
            (
                lambda: actual.view_as(1, other=actual_other),
                lambda: expected.view_as(1, other=expected_other),
            ),
            (
                lambda: actual.view_as(1, extra=True),
                lambda: expected.view_as(1, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
