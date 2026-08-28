import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = "\nsub(other, *, alpha=1) -> Tensor\n\nSee :func:`torch.sub`.\n"


class TensorSubTests(unittest.TestCase):
    def assert_same_tensor(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def test_tensor_scalar_broadcast_layout_empty_and_ieee_calls_reuse_operator(self):
        left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        right = torch.tensor([[10.0], [20.0], [30.0]])
        calls = (
            ("tensor positional", left.sub(right)),
            ("tensor keyword", left.sub(other=right)),
            ("x2 alias", left.sub(x2=right)),
            ("integer alpha", left.sub(right, alpha=1)),
            ("floating alpha", left.sub(other=right, alpha=1.0)),
            ("numpy integer alpha", left.sub(right, alpha=np.int64(1))),
            ("numpy floating alpha", left.sub(right, alpha=np.float32(1.0))),
            ("numpy boolean alpha", left.sub(right, alpha=np.bool_(True))),
            ("tensor scalar alpha", left.sub(right, alpha=torch.tensor(1.0))),
        )
        expected = left - right
        for case, actual in calls:
            self.assert_same_tensor(actual, expected, case=case)
            self.assertNotEqual(actual.data_ptr(), left.data_ptr())
            self.assertNotEqual(actual.data_ptr(), right.data_ptr())

        offset = left[1]
        for scalar in (-2, 2.5, np.int64(3), np.float32(-0.0)):
            self.assert_same_tensor(
                offset.sub(scalar), offset - scalar, case=("scalar positional", scalar)
            )
            self.assert_same_tensor(
                offset.sub(other=scalar),
                offset - scalar,
                case=("scalar keyword", scalar),
            )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_same_tensor(
            empty.sub(other=broadcast),
            empty - broadcast,
            case="strided broadcast empty",
        )

        left_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        right_bits = np.asarray(
            (
                0x8000_0000,
                0x0000_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x7FC6_789A,
                0xFFC7_89AB,
            ),
            dtype=np.uint32,
        )
        special_left = torch.tensor(memoryview(left_bits.view(np.float32)))
        special_right = torch.tensor(memoryview(right_bits.view(np.float32)))
        self.assert_same_tensor(
            special_left.sub(special_right),
            special_left - special_right,
            case="tensor IEEE edges",
        )
        self.assert_same_tensor(
            special_left.sub(-0.0),
            special_left - -0.0,
            case="scalar signed zero and non-finites",
        )

    def test_tensor_sub_autograd_broadcast_shared_empty_and_no_grad(self):
        left = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        right = torch.tensor([[10.0, 20.0]], requires_grad=True)
        weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        output = left.transpose(0, 1).sub(other=right, alpha=1)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.stride(), (1, 3))
        (output * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(left.grad), [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        )
        np.testing.assert_array_equal(np.asarray(right.grad), [[-9.0, -12.0]])

        shared = torch.tensor([2.0, -3.0], requires_grad=True)
        loss = shared.sub(shared).sum()
        loss.backward()
        loss.backward()
        np.testing.assert_array_equal(np.asarray(shared.grad), [0.0, 0.0])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        broadcast = torch.ones((1, 1, 3), requires_grad=True)
        empty.sub(other=broadcast).sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.numel(), 0)
        np.testing.assert_array_equal(
            np.asarray(broadcast.grad), np.zeros((1, 1, 3), dtype=np.float32)
        )

        tracked = torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            untracked_tensor = tracked.transpose(0, 1).sub(
                other=torch.tensor([[3.0, 4.0]])
            )
            untracked_scalar = tracked.sub(2.0, alpha=1)
        self.assertFalse(untracked_tensor.requires_grad)
        self.assertFalse(untracked_scalar.requires_grad)
        self.assertTrue(tracked.sub(2.0).requires_grad)

    def test_descriptor_binding_unit_alpha_and_unsupported_surface(self):
        tensor = torch.tensor([1.0])
        descriptor = inspect.getattr_static(torch.Tensor, "sub")
        bound = tensor.sub

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(descriptor.__name__, "sub")
        self.assertEqual(bound.__name__, "sub")
        self.assertEqual(descriptor.__qualname__, "TensorBase.sub")
        self.assertEqual(bound.__qualname__, "Tensor.sub")
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        self.assertEqual(
            repr(descriptor), "<method 'sub' of 'torch._C.TensorBase' objects>"
        )
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assert_same_tensor(
            descriptor(tensor, other=tensor, alpha=1),
            tensor - tensor,
            case="unbound descriptor call",
        )

        cases = (
            (
                lambda: tensor.sub(),
                TypeError,
                "sub() received an invalid combination of arguments - got (), but expected (Tensor other, *, Number alpha = 1)",
            ),
            (
                lambda: tensor.sub(tensor, tensor),
                TypeError,
                "sub() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.sub(tensor, other=tensor),
                TypeError,
                "sub() got multiple values for argument 'other'",
            ),
            (
                lambda: tensor.sub(tensor, out=tensor),
                TypeError,
                "sub() got an unexpected keyword argument 'out'",
            ),
            (
                lambda: tensor.sub([]),
                TypeError,
                "sub(): argument 'other' (position 1) must be Tensor, not list",
            ),
            (
                lambda: tensor.sub(other=None),
                TypeError,
                "sub(): argument 'other' must be Tensor, not NoneType",
            ),
            (
                lambda: tensor.sub(tensor, alpha=None),
                TypeError,
                "sub(): argument 'alpha' must be Number, not NoneType",
            ),
            (
                lambda: tensor.sub(tensor, alpha=True),
                RuntimeError,
                "Boolean alpha only supported for Boolean results.",
            ),
            (
                lambda: tensor.sub(np.uint64(2**63)),
                TypeError,
                "an integer is required",
            ),
            (
                lambda: tensor.sub(2**64),
                OverflowError,
                "int too big to convert",
            ),
            (
                lambda: tensor.sub(-(2**63) - 1),
                OverflowError,
                "can't convert negative int to unsigned",
            ),
            (
                lambda: descriptor(),
                TypeError,
                "unbound method TensorBase.sub() needs an argument",
            ),
            (
                lambda: descriptor(1, tensor),
                TypeError,
                "descriptor 'sub' for 'torch._C.TensorBase' objects doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor, other=tensor),
                TypeError,
                "unbound method TensorBase.sub() needs an argument",
            ),
        )
        for call, exception, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(exception, f"^{re.escape(message)}$"):
                    call()

        for alpha in (0, -1, 2, -0.0, np.float32(1.5), float("nan")):
            with self.subTest(alpha=alpha):
                with self.assertRaisesRegex(
                    RuntimeError, r"^Tensor\.sub only supports alpha=1$"
                ):
                    tensor.sub(tensor, alpha=alpha)

        with self.assertRaisesRegex(
            RuntimeError, r"^Tensor\.sub only supports alpha=1$"
        ):
            torch.zeros((2, 3)).sub(torch.ones((4,)), alpha=2)
        extreme = torch.zeros((0,)).reshape((0, 1, 1 << 62, 1 << 32))
        with self.assertRaisesRegex(RuntimeError, r"^Stride calculation overflowed$"):
            extreme.sub(torch.tensor(1.0), alpha=1)
        with self.assertRaisesRegex(
            RuntimeError, r"^Tensor\.sub only supports alpha=1$"
        ):
            extreme.sub(torch.tensor(1.0), alpha=2)

        self.assertFalse(hasattr(torch.Tensor, "sub_"))
        self.assertFalse(hasattr(torch.Tensor, "subtract"))
        self.assertFalse(hasattr(torch, "sub"))
        self.assertNotIn("sub", torch.__all__)

    def test_modes_and_operand_overrides_receive_original_calls(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "sub")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                self.calls.append((func, dispatch_types, args, kwargs))
                return self.result

        calls = (
            (lambda: tensor.sub(other), (tensor, other), None),
            (lambda: tensor.sub(other=other), (tensor,), {"other": other}),
            (lambda: tensor.sub(x2=other), (tensor,), {"x2": other}),
            (
                lambda: tensor.sub(other, alpha=1),
                (tensor, other),
                {"alpha": 1},
            ),
            (
                lambda: tensor.sub(other, alpha=2),
                (tensor, other),
                {"alpha": 2},
            ),
        )
        for call, expected_args, expected_kwargs in calls:
            mode = RecordingMode()
            with mode:
                self.assertIs(call(), marker)
            self.assertEqual(len(mode.calls), 1)
            function, dispatch_types, args, kwargs = mode.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, ())
            self.assertEqual(args, expected_args)
            self.assertEqual(kwargs, expected_kwargs)

        invalid_mode = RecordingMode()
        with invalid_mode, self.assertRaises(TypeError):
            tensor.sub([])
        self.assertEqual(invalid_mode.calls, [])

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, dispatch_types, args=(), kwargs=None):
                order.append((self.label, dispatch_types))
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.sub(other=other, alpha=1)
        self.assertEqual(order, [("upper", ()), ("lower", ())])
        self.assert_same_tensor(forwarded, tensor - other, case="forwarded modes")

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        for location, call in (
            ("other", lambda value: tensor.sub(value)),
            ("other", lambda value: tensor.sub(other=value, alpha=1)),
            ("alpha", lambda value: tensor.sub(other, alpha=value)),
        ):
            value = Override()
            Override.calls.clear()
            self.assertIs(call(value), marker)
            self.assertEqual(len(Override.calls), 1)
            function, dispatch_types, args, kwargs = Override.calls[0]
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (Override,))
            self.assertIs(args[0], tensor)
            if location == "alpha":
                self.assertIs(kwargs["alpha"], value)

        mode = RecordingMode(NotImplemented)
        Override.calls.clear()
        value = Override()
        with mode:
            self.assertIs(tensor.sub(value, alpha=1), marker)
        self.assertEqual(len(mode.calls), 1)
        self.assertEqual(mode.calls[0][1], (Override,))
        self.assertEqual(len(Override.calls), 1)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                return NotImplemented

        with self.assertRaisesRegex(
            TypeError,
            r"^Multiple dispatch failed for 'torch\.Tensor\.sub'; all __torch_function__ handlers returned NotImplemented:",
        ):
            tensor.sub(DecliningOverride())

        class NumericAlpha(float):
            calls = []

            @classmethod
            def __torch_function__(cls, func, dispatch_types, args=(), kwargs=None):
                cls.calls.append((func, dispatch_types, args, kwargs))
                return marker

        NumericAlpha.calls.clear()
        numeric = tensor.sub(other, alpha=NumericAlpha(1.0))
        self.assert_same_tensor(numeric, tensor - other, case="numeric alpha subclass")
        self.assertEqual(NumericAlpha.calls, [])


if __name__ == "__main__":
    unittest.main()
