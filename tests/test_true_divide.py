import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


TRUE_DIVIDE_DOC = """
true_divide(value) -> Tensor

See :func:`torch.true_divide`
"""


class TensorTrueDivideTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertIs(actual.layout, torch.strided)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_tensor_broadcast_empty_offset_and_noncontiguous_calls_reuse_division(self):
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        noncontiguous = base.transpose(0, 2)
        cases = (
            ("scalar tensor", torch.tensor(-0.0), torch.tensor(2.0)),
            (
                "broadcast noncontiguous",
                noncontiguous,
                torch.tensor([[2.0], [4.0], [8.0]]),
            ),
            (
                "offset noncontiguous",
                noncontiguous[1],
                torch.tensor([[2.0], [4.0], [8.0]]),
            ),
            (
                "same-shape noncontiguous",
                noncontiguous,
                torch.full((4, 3, 2), -2.0),
            ),
            (
                "empty offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                torch.ones((1, 2)),
            ),
        )

        for case, left, right in cases:
            expected = left / right
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    actual = (
                        left.true_divide(other=right)
                        if keyword
                        else left.true_divide(right)
                    )
                    self.assert_tensor_matches(actual, expected, case=case)
                    if actual.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), left.data_ptr())

        self.assert_tensor_matches(
            noncontiguous.true_divide(x2=torch.tensor([[2.0], [4.0], [8.0]])),
            noncontiguous / torch.tensor([[2.0], [4.0], [8.0]]),
            case="legacy x2 keyword",
        )

    def test_supported_python_and_numpy_real_scalars_reuse_scalar_division(self):
        source = torch.tensor([1.0, -2.0, 0.0, -0.0])
        scalars = (
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
        )
        for scalar in scalars:
            expected = source / scalar
            with self.subTest(scalar=repr(scalar), binding="positional"):
                self.assert_tensor_matches(
                    source.true_divide(scalar), expected, case=repr(scalar)
                )
            with self.subTest(scalar=repr(scalar), binding="keyword"):
                self.assert_tensor_matches(
                    source.true_divide(other=scalar), expected, case=repr(scalar)
                )

    def test_complex_scalars_are_rejected_only_after_mode_dispatch(self):
        tensor = torch.tensor([8.0])
        message = r"^true_divide\(\): complex scalar operands are not supported$"

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        for scalar in (2 + 1j, np.complex64(2 + 1j), np.complex128(2 + 1j)):
            for binding in ("positional", "other", "x2"):
                def invoke():
                    if binding == "positional":
                        return tensor.true_divide(scalar)
                    return tensor.true_divide(**{binding: scalar})

                with self.subTest(scalar=type(scalar).__name__, binding=binding):
                    with self.assertRaisesRegex(TypeError, message):
                        invoke()

                    mode = ForwardingMode()
                    with mode, self.assertRaisesRegex(TypeError, message):
                        invoke()
                    self.assertEqual(len(mode.calls), 1)
                    _, dispatch_types, _, _ = mode.calls[0]
                    self.assertEqual(dispatch_types, ())

    def test_ieee_edge_bits_are_preserved(self):
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
        expected = np.asarray(
            (
                0x7F800000,
                0xFF800000,
                0xFF800000,
                0x7F800000,
                0x00000000,
                0x80000000,
                0x80000000,
                0x00000000,
                0xFFC00000,
                0x7F800000,
                0xFF800000,
                0x00000000,
                0x80000000,
                0xFFC00000,
                0x7FC12345,
                0xFFC54321,
                0x7FC12345,
                0xFFC12345,
            ),
            dtype=np.uint32,
        )
        left_bits = np.asarray([left for left, _ in pairs], dtype=np.uint32)
        right_bits = np.asarray([right for _, right in pairs], dtype=np.uint32)
        left = torch.tensor(memoryview(left_bits.view(np.float32)))
        right = torch.tensor(memoryview(right_bits.view(np.float32)))

        result = left.true_divide(other=right)

        np.testing.assert_array_equal(
            np.asarray(result).view(np.uint32), expected
        )

    def test_active_autograd_is_rejected_and_no_grad_is_honored(self):
        left = torch.tensor([[2.0, 4.0]], requires_grad=True).transpose(0, 1)
        right = torch.tensor([[1.0, 2.0]], requires_grad=True)
        ordinary = torch.tensor([[1.0, 2.0]])
        message = r"^true_divide\(\): autograd recording is not supported$"

        for call in (
            lambda: left.true_divide(2.0),
            lambda: ordinary.true_divide(right),
            lambda: left.true_divide(other=right),
        ):
            with self.subTest(call=call), self.assertRaisesRegex(RuntimeError, message):
                call()

        with torch.no_grad():
            tensor_result = left.true_divide(other=right)
            scalar_result = left.true_divide(-2.0)
        self.assert_tensor_matches(
            tensor_result,
            left.detach() / right.detach(),
            case="no_grad tensor operands",
        )
        self.assert_tensor_matches(
            scalar_result,
            left.detach() / -2.0,
            case="no_grad scalar operand",
        )
        self.assertIsNone(left.grad)
        self.assertIsNone(right.grad)

    def test_tensorbase_descriptor_and_binding_errors(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "true_divide")
        bound = tensor.true_divide

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "true_divide")
        self.assertEqual(descriptor.__qualname__, "TensorBase.true_divide")
        self.assertEqual(bound.__name__, "true_divide")
        self.assertEqual(bound.__qualname__, "Tensor.true_divide")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor),
            "<method 'true_divide' of 'torch._C.TensorBase' objects>",
        )
        self.assertIs(torch.Tensor.true_divide, descriptor)
        self.assertIs(descriptor.__get__(None, torch.Tensor), descriptor)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__doc__, TRUE_DIVIDE_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assert_tensor_matches(
            descriptor(tensor, other=other), tensor / other, case="unbound call"
        )
        overloads = "but expected one of:\n * (Tensor other)\n * (Number other)\n"
        cases = (
            (
                lambda: tensor.true_divide(),
                "true_divide() received an invalid combination of arguments - "
                f"got (), {overloads}",
            ),
            (
                lambda: tensor.true_divide(other, other),
                "true_divide() received an invalid combination of arguments - "
                f"got (Tensor, Tensor), {overloads}",
            ),
            (
                lambda: tensor.true_divide(other, other=other),
                "true_divide() received an invalid combination of arguments - "
                f"got (Tensor, other=Tensor), {overloads}",
            ),
            (
                lambda: tensor.true_divide(other, out=tensor),
                "true_divide() received an invalid combination of arguments - "
                f"got (Tensor, out=Tensor), {overloads}",
            ),
            (
                lambda: tensor.true_divide([]),
                "true_divide() received an invalid combination of arguments - "
                "got (list), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!list of []!)\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!list of []!)\n",
            ),
            (
                lambda: tensor.true_divide(other=None),
                "true_divide() received an invalid combination of arguments - "
                "got (other=NoneType, ), but expected one of:\n"
                " * (Tensor other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!other=NoneType!, )\n"
                " * (Number other)\n"
                "      didn't match because some of the arguments have invalid "
                "types: (!other=NoneType!, )\n",
            ),
            (lambda: tensor.true_divide(np.uint64(2**63)), "an integer is required"),
            (lambda: tensor.true_divide(2**64), "int too big to convert"),
            (
                lambda: tensor.true_divide(-(2**63) - 1),
                "can't convert negative int to unsigned",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                Exception, f"^{re.escape(message)}$"
            ):
                call()

        descriptor_cases = (
            (
                lambda: descriptor(),
                "unbound method TensorBase.true_divide() needs an argument",
            ),
            (
                lambda: descriptor(self=tensor, other=other),
                "unbound method TensorBase.true_divide() needs an argument",
            ),
            (
                lambda: descriptor(1, other),
                "descriptor 'true_divide' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
        )
        for call, message in descriptor_cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                TypeError, f"^{re.escape(message)}$"
            ):
                call()

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([8.0], requires_grad=True)
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "true_divide")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        cases = (
            (lambda: tensor.true_divide(other), (tensor, other), None),
            (lambda: tensor.true_divide(other=other), (tensor,), {"other": other}),
            (lambda: tensor.true_divide(x2=2.0), (tensor,), {"x2": 2.0}),
        )
        for call, expected_args, expected_kwargs in cases:
            mode = RecordingMode(marker)
            with mode:
                result = call()
            self.assertIs(result, marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(len(args), len(expected_args))
            for actual, expected in zip(args, expected_args, strict=True):
                if isinstance(expected, torch.Tensor):
                    self.assertIs(actual, expected)
                else:
                    self.assertEqual(actual, expected)
            if expected_kwargs is None:
                self.assertIsNone(kwargs)
            else:
                self.assertEqual(tuple(kwargs), tuple(expected_kwargs))
                for key, value in expected_kwargs.items():
                    if isinstance(value, torch.Tensor):
                        self.assertIs(kwargs[key], value)
                    else:
                        self.assertEqual(kwargs[key], value)

        invalid = RecordingMode(marker)
        with invalid, self.assertRaises(TypeError):
            tensor.true_divide(object())
        self.assertEqual(invalid.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, func, dispatch_types, args, kwargs))
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([8.0])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.true_divide(other=other)
        self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
        for _, function, dispatch_types, args, kwargs in order:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, (plain,))
            self.assertEqual(tuple(kwargs), ("other",))
            self.assertIs(kwargs["other"], other)
        self.assertEqual(forwarded.tolist(), [4.0])

        declining = RecordingMode(NotImplemented)
        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.true_divide';",
        ):
            with declining:
                plain.true_divide(other)
        self.assertEqual(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_other_torch_function_override_is_forwarded_after_modes(self):
        tensor = torch.tensor([8.0])
        descriptor = inspect.getattr_static(torch.Tensor, "true_divide")
        marker = object()
        events = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                events.append(("override", func, dispatch_types, args, kwargs))
                return marker

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                events.append(("mode", func, dispatch_types, args, kwargs))
                return NotImplemented

        value = Override()
        with DecliningMode():
            result = tensor.true_divide(other=value)
        self.assertIs(result, marker)
        self.assertEqual([event[0] for event in events], ["mode", "override"])
        for _, function, dispatch_types, args, kwargs in events:
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (Override,))
            self.assertEqual(args, (tensor,))
            self.assertEqual(tuple(kwargs), ("other",))
            self.assertIs(kwargs["other"], value)

    def test_related_division_surfaces_remain_unsupported(self):
        tensor = torch.tensor([1.0])
        for name in ("div", "divide", "div_", "divide_", "true_divide_"):
            with self.subTest(owner="Tensor", name=name):
                self.assertFalse(hasattr(torch.Tensor, name))
                self.assertFalse(hasattr(tensor, name))
        for name in ("div", "divide", "true_divide"):
            with self.subTest(owner="module", name=name):
                self.assertFalse(hasattr(torch, name))


if __name__ == "__main__":
    unittest.main()
