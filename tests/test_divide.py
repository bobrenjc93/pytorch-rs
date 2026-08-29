import copy
import inspect
import math
import pickle
import re
import types
import unittest
from decimal import Decimal

import numpy as np
import torch_rs as torch


METHOD_DOCS = {
    "div": "\ndiv(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.div`\n",
    "divide": "\ndivide(value, *, rounding_mode=None) -> Tensor\n\nSee :func:`torch.divide`\n",
}


class TensorDivideMethodTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected).reshape(-1).view(np.uint32),
            )

    def assert_layout_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_tensor_bits(self, actual, expected_bits, *, case):
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                np.asarray(expected_bits, dtype=np.uint32),
            )

    def test_tensor_and_real_scalar_calls_reuse_operator_semantics(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [4.0], [8.0]])

        for name in ("div", "divide"):
            method = getattr(left, name)
            self.assert_tensor_matches(
                method(right), left / right, case=(name, "tensor positional")
            )
            self.assert_tensor_matches(
                method(other=right), left / right, case=(name, "tensor keyword")
            )
            self.assert_tensor_matches(
                method(x2=right), left / right, case=(name, "x2 keyword")
            )
            self.assert_tensor_matches(
                method(other=right, rounding_mode=None),
                left / right,
                case=(name, "rounding none"),
            )
            rounded_left = torch.tensor(
                [[1.5, -1.5], [5.0, -5.0]]
            ).transpose(0, 1)
            rounded_right = torch.tensor([[2.0, -2.0]])
            for rounding_mode, expected_bits in (
                ("floor", [0x0000_0000, 0xC040_0000, 0xBF80_0000, 0x4000_0000]),
                ("trunc", [0x0000_0000, 0xC000_0000, 0x8000_0000, 0x4000_0000]),
                (b"floor", [0x0000_0000, 0xC040_0000, 0xBF80_0000, 0x4000_0000]),
                (
                    np.bytes_(b"trunc"),
                    [0x0000_0000, 0xC000_0000, 0x8000_0000, 0x4000_0000],
                ),
            ):
                actual = getattr(rounded_left, name)(
                    rounded_right, rounding_mode=rounding_mode
                )
                self.assert_layout_matches(
                    actual,
                    rounded_left / rounded_right,
                    case=(name, "rounded tensor layout", rounding_mode),
                )
                self.assert_tensor_bits(
                    actual,
                    expected_bits,
                    case=(name, "rounded tensor values", rounding_mode),
                )

            offset_view = left[1]
            for scalar in (
                True,
                -2,
                2.5,
                np.bool_(True),
                np.int64(4),
                np.float32(-0.0),
            ):
                expected = offset_view / scalar
                self.assert_tensor_matches(
                    getattr(offset_view, name)(scalar),
                    expected,
                    case=(name, "scalar positional", scalar),
                )
                self.assert_tensor_matches(
                    getattr(offset_view, name)(other=scalar),
                    expected,
                    case=(name, "scalar keyword", scalar),
                )
            self.assert_tensor_matches(
                getattr(offset_view, name)(x2=np.float32(-2.5)),
                offset_view / np.float32(-2.5),
                case=(name, "scalar x2 keyword"),
            )
            rounded = torch.tensor([1.5, -1.5, 5.0, -5.0])
            self.assert_tensor_matches(
                getattr(rounded, name)(2.0, rounding_mode=np.str_("floor")),
                torch.tensor([0.0, -1.0, 2.0, -3.0]),
                case=(name, "scalar floor"),
            )
            self.assert_tensor_matches(
                getattr(rounded, name)(2.0, rounding_mode="trunc"),
                torch.tensor([0.0, -0.0, 2.0, -2.0]),
                case=(name, "scalar trunc"),
            )

            empty = torch.zeros((2, 0, 3)).transpose(0, 2)
            broadcast = torch.ones((1, 1, 2))
            self.assert_tensor_matches(
                getattr(empty, name)(other=broadcast),
                empty / broadcast,
                case=(name, "strided broadcast empty"),
            )
            self.assert_layout_matches(
                getattr(empty, name)(other=broadcast, rounding_mode="floor"),
                empty / broadcast,
                case=(name, "rounded strided broadcast empty"),
            )

            numerator = torch.tensor(
                [
                    math.nan,
                    math.inf,
                    -math.inf,
                    math.inf,
                    -math.inf,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    0.0,
                    -0.0,
                ]
            )
            denominator = torch.tensor(
                [
                    1.0,
                    math.inf,
                    -math.inf,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.0,
                    2.0,
                    2.0,
                    -2.0,
                    -2.0,
                ]
            )
            self.assert_tensor_matches(
                getattr(numerator, name)(denominator),
                numerator / denominator,
                case=(name, "ieee tensor values"),
            )
            self.assert_tensor_bits(
                getattr(numerator, name)(denominator, rounding_mode="floor"),
                [
                    0x7FC0_0000,
                    0xFFC0_0000,
                    0xFFC0_0000,
                    0xFFC0_0000,
                    0xFFC0_0000,
                    0x7F80_0000,
                    0xFF80_0000,
                    0xFF80_0000,
                    0x7F80_0000,
                    0x0000_0000,
                    0x8000_0000,
                    0x8000_0000,
                    0x0000_0000,
                ],
                case=(name, "ieee floor values"),
            )
            self.assert_tensor_bits(
                getattr(numerator, name)(denominator, rounding_mode="trunc"),
                [
                    0x7FC0_0000,
                    0xFFC0_0000,
                    0xFFC0_0000,
                    0x7F80_0000,
                    0xFF80_0000,
                    0x7F80_0000,
                    0xFF80_0000,
                    0xFF80_0000,
                    0x7F80_0000,
                    0x0000_0000,
                    0x8000_0000,
                    0x8000_0000,
                    0x0000_0000,
                ],
                case=(name, "ieee trunc values"),
            )

            scalar_bits = np.array([0xC25F_B64C], dtype=np.uint32)
            scalar = scalar_bits.view(np.float32)[0].item()
            self.assert_tensor_matches(
                torch.tensor([scalar]).__getattribute__(name)(-0.0),
                torch.tensor([scalar]) / -0.0,
                case=(name, "signed zero scalar"),
            )

    def test_autograd_boundary_matches_operator(self):
        for name in ("div", "divide"):
            method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            method_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
            operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

            method_output = getattr(method_left.transpose(0, 1), name)(
                other=method_right.transpose(0, 1)
            )
            operator_output = operator_left.transpose(0, 1) / operator_right.transpose(
                0, 1
            )
            self.assert_tensor_matches(
                method_output, operator_output, case=(name, "tracked views")
            )
            self.assertFalse(method_output.requires_grad)
            self.assertFalse(operator_output.requires_grad)
            self.assertIsNone(method_left.grad)
            self.assertIsNone(method_right.grad)

            with torch.no_grad():
                scalar_output = getattr(method_left, name)(other=2.0)
            self.assertFalse(scalar_output.requires_grad)

            rounded_output = getattr(method_left, name)(2.0, rounding_mode="floor")
            self.assertFalse(rounded_output.requires_grad)
            self.assertIsNone(method_left.grad)

    def test_descriptor_metadata_copy_and_pickle(self):
        tensor = torch.tensor([1.0])
        for name in ("div", "divide"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            bound = getattr(tensor, name)

            self.assertIs(torch.Tensor.__base__.__dict__[name], descriptor)
            self.assertIs(getattr(torch.Tensor, name), descriptor)
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(descriptor.__name__, name)
            self.assertEqual(bound.__name__, name)
            self.assertEqual(descriptor.__qualname__, f"TensorBase.{name}")
            self.assertEqual(bound.__qualname__, f"Tensor.{name}")
            self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
            self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
            self.assertFalse(hasattr(descriptor, "__module__"))
            self.assertIsNone(bound.__module__)
            self.assertIsNone(descriptor.__text_signature__)
            self.assertIsNone(bound.__text_signature__)
            self.assertEqual(descriptor.__doc__, METHOD_DOCS[name])
            self.assertEqual(bound.__doc__, METHOD_DOCS[name])
            with self.assertRaises(ValueError):
                inspect.signature(descriptor)
            with self.assertRaises(ValueError):
                inspect.signature(bound)
            self.assert_tensor_matches(
                descriptor(tensor, other=tensor),
                tensor / tensor,
                case=(name, "unbound call"),
            )
            self.assertIs(copy.copy(descriptor), descriptor)
            self.assertIs(copy.deepcopy(descriptor), descriptor)
            owner = descriptor.__reduce__()[1][0]
            self.assertEqual(owner.__name__, "TensorBase")
            self.assertEqual(owner.__module__, "torch._C")
            self.assertIs(getattr(owner, name), descriptor)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(descriptor, protocol=protocol)),
                        descriptor,
                    )

    def test_torch_function_modes_receive_original_calls_and_forward(self):
        tensor = torch.tensor([2.0])
        other = torch.tensor([4.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("div", "divide"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            for label, call, expected_args, expected_kwargs in (
                ("tensor", lambda: getattr(tensor, name)(other), (tensor, other), None),
                ("scalar", lambda: getattr(tensor, name)(2.0), (tensor, 2.0), None),
                (
                    "keyword",
                    lambda: getattr(tensor, name)(other=other, rounding_mode=None),
                    (tensor,),
                    {"other": other, "rounding_mode": None},
                ),
                (
                    "rounding floor",
                    lambda: getattr(tensor, name)(other, rounding_mode="floor"),
                    (tensor, other),
                    {"rounding_mode": "floor"},
                ),
            ):
                mode = RecordingMode()
                with self.subTest(name=name, label=label), mode:
                    self.assertIs(call(), marker)
                function, dispatch_types, args, kwargs = mode.calls[0]
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, expected_args)
                self.assertEqual(kwargs, expected_kwargs)

            for invalid_call in (
                lambda: getattr(tensor, name)([]),
                lambda: getattr(tensor, name)(other, out=tensor),
                lambda: getattr(tensor, name)(other, rounding_mode=1),
            ):
                mode = RecordingMode()
                with self.subTest(name=name, invalid_call=invalid_call):
                    with mode, self.assertRaises(TypeError):
                        invalid_call()
                    self.assertEqual(mode.calls, [])

            order = []

            class ForwardingMode(torch.overrides.TorchFunctionMode):
                def __init__(self, label):
                    self.label = label

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    order.append((self.label, func, types, args, kwargs))
                    return func(*args, **(kwargs or {}))

            with ForwardingMode("lower"):
                with ForwardingMode("upper"):
                    forwarded = getattr(tensor, name)(other=other, rounding_mode=None)
            self.assertEqual([entry[0] for entry in order], ["upper", "lower"])
            for _, function, dispatch_types, args, kwargs in order:
                self.assertIs(function, descriptor)
                self.assertEqual(dispatch_types, ())
                self.assertEqual(args, (tensor,))
                self.assertEqual(kwargs, {"other": other, "rounding_mode": None})
            self.assert_tensor_matches(forwarded, tensor / other, case=(name, "forwarded"))

            declining = RecordingMode(NotImplemented)
            with self.assertRaises(TypeError) as raised:
                with declining:
                    getattr(tensor, name)(other)
            self.assertTrue(
                str(raised.exception).startswith(
                    f"Multiple dispatch failed for 'torch.Tensor.{name}'; "
                    "all __torch_function__ handlers returned NotImplemented:"
                )
            )
            self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

    def test_other_torch_function_overrides_are_ordered_and_deferred(self):
        tensor = torch.tensor([2.0])
        marker = object()

        class BaseOverride:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append(("base", func, types, args, kwargs))
                return NotImplemented

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append(("derived", func, types, args, kwargs))
                return marker

        for name in ("div", "divide"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            value = DerivedOverride()
            BaseOverride.calls.clear()
            self.assertIs(getattr(tensor, name)(value), marker)
            label, function, dispatch_types, args, kwargs = BaseOverride.calls[0]
            self.assertEqual(label, "derived")
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (DerivedOverride,))
            self.assertEqual(args, (tensor, value))
            self.assertIsNone(kwargs)

            value = DerivedOverride()
            BaseOverride.calls.clear()
            self.assertIs(getattr(tensor, name)(other=value), marker)
            label, function, dispatch_types, args, kwargs = BaseOverride.calls[0]
            self.assertEqual(label, "derived")
            self.assertIs(function, descriptor)
            self.assertEqual(dispatch_types, (DerivedOverride,))
            self.assertEqual(args, (tensor,))
            self.assertEqual(kwargs, {"other": value})

            value = DerivedOverride()
            BaseOverride.calls.clear()
            self.assertIs(getattr(tensor, name)(value, rounding_mode="floor"), marker)
            self.assertEqual(BaseOverride.calls[0][4], {"rounding_mode": "floor"})

        class ScalarOverride(int):
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for name in ("div", "divide"):
            ScalarOverride.calls.clear()
            self.assertIs(getattr(tensor, name)(ScalarOverride(4)), marker)
            function, dispatch_types, args, kwargs = ScalarOverride.calls[0]
            self.assertIs(function, inspect.getattr_static(torch.Tensor, name))
            self.assertEqual(dispatch_types, (ScalarOverride,))
            self.assertEqual(args, (tensor, ScalarOverride(4)))
            self.assertIsNone(kwargs)

        class DecliningOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        for name in ("div", "divide"):
            with self.subTest(name=name, declined=True):
                with self.assertRaises(TypeError) as raised:
                    getattr(tensor, name)(DecliningOverride())
                self.assertIn(
                    f"Multiple dispatch failed for 'torch.Tensor.{name}'",
                    str(raised.exception),
                )
                self.assertIn("DecliningOverride", str(raised.exception))

    def test_rejects_unsupported_arguments_without_mutating_out(self):
        tensor = torch.tensor([1.0])
        other = torch.tensor([2.0])
        destination = torch.tensor([17.0])

        for name in ("div", "divide"):
            for rounding_mode, rendered in (
                ("bad", "bad"),
                ("", ""),
                (b"bad", "bad"),
                (b"", ""),
                (np.str_("bad"), "bad"),
                (np.bytes_(b"bad"), "bad"),
            ):
                with self.subTest(name=name, rounding_mode=rounding_mode):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "^div expected rounding_mode to be one of None, 'trunc', "
                        f"or 'floor' but found '{re.escape(rendered)}'$",
                    ):
                        getattr(tensor, name)(other, rounding_mode=rounding_mode)

            for call in (
                lambda: getattr(tensor, name)(other, out=destination),
                lambda: getattr(tensor, name)(other, out=None),
                lambda: getattr(tensor, name)(other, dtype=torch.float32),
                lambda: getattr(tensor, name)(other, device=torch.device("cpu")),
                lambda: getattr(tensor, name)(other, rounding_mode=1),
                lambda: getattr(tensor, name)(other, rounding_mode=True),
                lambda: getattr(tensor, name)(other, other=other),
                lambda: getattr(tensor, name)(other, x2=other),
                lambda: getattr(tensor, name)(other, other=other, rounding_mode=None),
                lambda: getattr(tensor, name)(),
                lambda: getattr(tensor, name)(other, other),
            ):
                with self.subTest(name=name, call=call):
                    with self.assertRaises(TypeError):
                        call()
                    self.assertEqual(destination.tolist(), [17.0])

            for value in (object(), Decimal("1.0"), 1 + 2j, [1.0]):
                with self.subTest(name=name, value=value):
                    with self.assertRaises(TypeError):
                        getattr(tensor, name)(value)

            with self.assertRaises(TypeError):
                inspect.getattr_static(torch.Tensor, name)()
            with self.assertRaises(TypeError):
                inspect.getattr_static(torch.Tensor, name)(2, 3)
            with self.assertRaises(TypeError):
                inspect.getattr_static(torch.Tensor, name)(self=tensor, other=other)

        with self.assertRaisesRegex(TypeError, "^an integer is required$"):
            tensor.div(np.uint64(2**63))
        with self.assertRaisesRegex(OverflowError, "^int too big to convert$"):
            tensor.divide(2**64)
        with self.assertRaisesRegex(
            OverflowError, "^can't convert negative int to unsigned$"
        ):
            tensor.div(-(2**63) - 1)


if __name__ == "__main__":
    unittest.main()
