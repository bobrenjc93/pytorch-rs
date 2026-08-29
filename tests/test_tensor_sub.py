import inspect
import math
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


class TensorSubtractionMethodTests(unittest.TestCase):
    def tensor_bits(self, tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    def assert_tensor_matches(self, actual, expected, *, case):
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
            np.testing.assert_array_equal(self.tensor_bits(actual), self.tensor_bits(expected))

    def assert_dispatch_call_matches(
        self, actual_args, expected_args, actual_kwargs, expected_kwargs
    ):
        self.assertEqual(len(actual_args), len(expected_args))
        for actual, expected in zip(actual_args, expected_args):
            self.assertIs(actual, expected)
        if expected_kwargs is None:
            self.assertIsNone(actual_kwargs)
            return
        self.assertIsNotNone(actual_kwargs)
        self.assertEqual(tuple(actual_kwargs), tuple(expected_kwargs))
        for key, expected in expected_kwargs.items():
            self.assertIs(actual_kwargs[key], expected)

    def test_tensor_real_scalar_broadcast_empty_and_ieee_values_match_operator(self):
        left = torch.tensor([[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]).transpose(
            0, 2
        )
        right = torch.tensor([[2.0], [3.0], [4.0]])

        calls = (
            ("sub tensor positional", lambda: left.sub(right), lambda: left - right),
            (
                "sub tensor keyword",
                lambda: left.sub(other=right, alpha=1),
                lambda: left - right,
            ),
            (
                "sub tensor x2 keyword",
                lambda: left.sub(x2=right, alpha=np.float32(1.0)),
                lambda: left - right,
            ),
            (
                "subtract tensor positional",
                lambda: left.subtract(right),
                lambda: left - right,
            ),
            (
                "subtract tensor keyword",
                lambda: left.subtract(other=right, alpha=1.0),
                lambda: left - right,
            ),
            (
                "subtract tensor x2 keyword",
                lambda: left.subtract(x2=right, alpha=np.uint64(1)),
                lambda: left - right,
            ),
        )
        for case, method_call, operator_call in calls:
            self.assert_tensor_matches(method_call(), operator_call(), case=case)

        offset_view = left[1]
        for scalar in (-2, 2.5, np.bool_(False), np.int64(3), np.float32(-0.0)):
            for method_name in ("sub", "subtract"):
                with self.subTest(method=method_name, scalar=scalar):
                    method = getattr(offset_view, method_name)
                    self.assert_tensor_matches(
                        method(other=scalar),
                        offset_view - scalar,
                        case=(method_name, scalar),
                    )

        empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        broadcast = torch.ones((1, 1, 2))
        self.assert_tensor_matches(
            empty.sub(other=broadcast),
            empty - broadcast,
            case="strided broadcast empty",
        )
        self.assert_tensor_matches(
            empty.subtract(x2=broadcast),
            empty - broadcast,
            case="subtract strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x7FC1_2345, 0x7F80_0000, 0xFF80_0000, 0x0000_0000, 0x8000_0000),
            dtype=np.uint32,
        )
        special = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected = special - torch.tensor([1.0, math.inf, -math.inf, -0.0, 0.0])
        self.assert_tensor_matches(
            special.sub(torch.tensor([1.0, math.inf, -math.inf, -0.0, 0.0])),
            expected,
            case="special tensor values",
        )
        self.assert_tensor_matches(
            special.subtract(-0.0),
            special - -0.0,
            case="special scalar values",
        )

    def test_autograd_shared_operands_and_no_grad_match_operator(self):
        method_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        method_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        operator_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        operator_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)

        method_output = method_left.transpose(0, 1).sub(
            other=method_right.transpose(0, 1)
        )
        operator_output = operator_left.transpose(0, 1) - operator_right.transpose(0, 1)
        self.assert_tensor_matches(method_output, operator_output, case="tracked views")
        method_output.sum().backward()
        operator_output.sum().backward()
        self.assert_tensor_matches(
            method_left.grad, operator_left.grad, case="left gradient"
        )
        self.assert_tensor_matches(
            method_right.grad, operator_right.grad, case="right gradient"
        )

        method_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        operator_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        method_shared.subtract(method_shared).sum().backward()
        (operator_shared - operator_shared).sum().backward()
        self.assert_tensor_matches(
            method_shared.grad, operator_shared.grad, case="shared operand gradient"
        )

        method_empty = torch.zeros((2, 0, 3), requires_grad=True)
        operator_empty = torch.zeros((2, 0, 3), requires_grad=True)
        method_empty.sub(other=torch.ones((1, 1, 3))).sum().backward()
        (operator_empty - torch.ones((1, 1, 3))).sum().backward()
        self.assert_tensor_matches(
            method_empty.grad, operator_empty.grad, case="empty gradient"
        )

        no_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
        no_grad_right = torch.tensor([[3.0], [4.0]], requires_grad=True)
        with torch.no_grad():
            tensor_output = no_grad_left.transpose(0, 1).subtract(
                no_grad_right.transpose(0, 1)
            )
            scalar_output = no_grad_left.sub(other=2.0)
        self.assertFalse(tensor_output.requires_grad)
        self.assertFalse(scalar_output.requires_grad)
        self.assertTrue(no_grad_left.sub(no_grad_right.transpose(0, 1)).requires_grad)

    def test_descriptor_metadata_unbound_calls_and_inplace_boundary(self):
        tensor = torch.tensor([1.0])
        for name, target in (("sub", "sub"), ("subtract", "subtract")):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            bound = getattr(tensor, name)
            with self.subTest(name=name):
                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(descriptor.__name__, name)
                self.assertEqual(bound.__name__, name)
                self.assertEqual(descriptor.__qualname__, f"TensorBase.{name}")
                self.assertEqual(bound.__qualname__, f"Tensor.{name}")
                self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
                self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
                self.assertIsNone(descriptor.__text_signature__)
                self.assertIsNone(bound.__text_signature__)
                self.assertEqual(
                    descriptor.__doc__,
                    f"\n{name}(other, *, alpha=1) -> Tensor\n\nSee :func:`torch.{target}`.\n",
                )
                with self.assertRaises(ValueError):
                    inspect.signature(descriptor)
                with self.assertRaises(ValueError):
                    inspect.signature(bound)
                self.assert_tensor_matches(
                    descriptor(tensor, other=tensor),
                    tensor - tensor,
                    case=("unbound", name),
                )

        self.assertFalse(hasattr(torch.Tensor, "sub_"))
        self.assertFalse(hasattr(torch.Tensor, "subtract_"))
        self.assertFalse(hasattr(tensor, "sub_"))
        self.assertFalse(hasattr(tensor, "subtract_"))
        before = self.tensor_bits(tensor).copy()
        with self.assertRaises(AttributeError):
            tensor.sub_(torch.tensor([1.0]))
        np.testing.assert_array_equal(self.tensor_bits(tensor), before)

    def test_unsupported_keywords_operands_and_alpha_fail_before_native_kernel(self):
        source = torch.tensor([1.0, 2.0], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for method_name in ("sub", "subtract"):
            method = getattr(source, method_name)
            with self.subTest(method=method_name, case="out"):
                with self.assertRaisesRegex(Exception, "out"):
                    method(torch.ones((2,)), out=destination)
                self.assertEqual(destination.tolist(), [17.0, 19.0])
                self.assertIsNone(source.grad)
            for keyword in ("dtype", "device"):
                with self.subTest(method=method_name, keyword=keyword):
                    with self.assertRaisesRegex(TypeError, keyword):
                        method(torch.ones((2,)), **{keyword: torch.float32})
                    self.assertIsNone(source.grad)
            for alpha in (2, 1.5, np.float32(2.0), np.bool_(False)):
                with self.subTest(method=method_name, alpha=alpha):
                    with self.assertRaisesRegex(
                        RuntimeError, "non-default alpha is not supported"
                    ):
                        method(torch.ones((2,)), alpha=alpha)
                    self.assertIsNone(source.grad)
            with self.subTest(method=method_name, alpha=True):
                with self.assertRaisesRegex(RuntimeError, "Boolean alpha"):
                    method(torch.ones((2,)), alpha=True)
                self.assertIsNone(source.grad)
            with self.subTest(method=method_name, alpha="list"):
                with self.assertRaisesRegex(TypeError, "alpha"):
                    method(torch.ones((2,)), alpha=[])
                self.assertIsNone(source.grad)
            with self.subTest(method=method_name, other="list"):
                with self.assertRaises(TypeError):
                    method([])
                self.assertIsNone(source.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape((0, sys.maxsize, 3))
        blocker = torch.ones((1, 1, 3))
        for method_name in ("sub", "subtract"):
            with self.subTest(method=method_name, case="alpha before allocation"):
                with self.assertRaisesRegex(
                    RuntimeError, "non-default alpha is not supported"
                ):
                    getattr(extreme, method_name)(blocker, alpha=2)

    def test_torch_function_modes_and_overrides_dispatch_before_native_limits(self):
        tensor = torch.tensor([2.0])
        other = torch.tensor([1.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __repr__(self):
                return "recording-sub-mode"

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("sub", "subtract"):
            descriptor = inspect.getattr_static(torch.Tensor, name)
            for label, call, expected_args, expected_kwargs in (
                ("tensor", lambda name=name: getattr(tensor, name)(other), (tensor, other), None),
                ("scalar", lambda name=name: getattr(tensor, name)(1.0), (tensor, 1.0), None),
                (
                    "keyword alpha",
                    lambda name=name: getattr(tensor, name)(other=other, alpha=1),
                    (tensor,),
                    {"other": other, "alpha": 1},
                ),
                (
                    "nondefault alpha",
                    lambda name=name: getattr(tensor, name)(other, alpha=2),
                    (tensor, other),
                    {"alpha": 2},
                ),
            ):
                mode = RecordingMode()
                with self.subTest(name=name, label=label):
                    with mode:
                        self.assertIs(call(), marker)
                    function, dispatch_types, args, kwargs = mode.calls[0]
                    self.assertIs(function, descriptor)
                    self.assertEqual(dispatch_types, ())
                    self.assert_dispatch_call_matches(
                        args, expected_args, kwargs, expected_kwargs
                    )

            invalid_mode = RecordingMode()
            with self.subTest(name=name, label="invalid schema"):
                with self.assertRaises(TypeError):
                    with invalid_mode:
                        getattr(tensor, name)(other, out=tensor)
                self.assertEqual(invalid_mode.calls, [])

            class RightOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    self.assertIs(func, descriptor)
                    self.assertEqual(types, (RightOverride,))
                    self.assertIs(args[0], tensor)
                    return marker

            with self.subTest(name=name, label="other override"):
                self.assertIs(getattr(tensor, name)(RightOverride()), marker)

            class AlphaOverride:
                @classmethod
                def __torch_function__(cls, func, types, args=(), kwargs=None):
                    self.assertIs(func, descriptor)
                    self.assertEqual(types, (AlphaOverride,))
                    self.assertEqual(len(args), 2)
                    self.assertIs(args[0], tensor)
                    self.assertIs(args[1], other)
                    self.assertEqual(tuple(kwargs), ("alpha",))
                    return marker

            with self.subTest(name=name, label="alpha override"):
                self.assertIs(getattr(tensor, name)(other, alpha=AlphaOverride()), marker)

            class DecliningMode(torch.overrides.TorchFunctionMode):
                def __repr__(self):
                    return f"declining-{name}-mode"

                def __torch_function__(self, func, types, args=(), kwargs=None):
                    return NotImplemented

            with self.subTest(name=name, label="declining mode"):
                with self.assertRaisesRegex(
                    TypeError,
                    re.escape(
                        f"Multiple dispatch failed for 'torch.Tensor.{name}'; "
                        "all __torch_function__ handlers returned NotImplemented"
                    ),
                ):
                    with DecliningMode():
                        getattr(tensor, name)(other)


if __name__ == "__main__":
    unittest.main()
