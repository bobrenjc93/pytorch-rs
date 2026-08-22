import inspect
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature


SQUARE_DOC = """
square() -> Tensor

See :func:`torch.square`
"""


class TensorSquareTests(unittest.TestCase):
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
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32),
            )

    @staticmethod
    def value_cases():
        base = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist()
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", torch.tensor(-0.0)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1]),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                torch.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def autograd_case(case):
        if case == "scalar":
            leaf = torch.tensor(-3.0, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = torch.zeros((2, 0, 3), requires_grad=True)
            return leaf, leaf.transpose(0, 2)[1], None

        leaf = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        if case == "offset":
            source = leaf[1]
            weights = torch.tensor(
                np.arange(1, 13, dtype=np.float32).reshape(3, 4).tolist()
            )
            return leaf, source, weights
        if case == "noncontiguous":
            source = leaf.transpose(0, 2)
            weights = torch.tensor(
                np.arange(1, 25, dtype=np.float32).reshape(4, 3, 2).tolist()
            )
            return leaf, source, weights
        raise AssertionError(f"unknown square autograd case: {case}")

    def test_values_layout_and_fresh_storage_reuse_shared_multiplication(self):
        expected_special_bits = np.asarray(
            (
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x3F80_0000,
                0x3F80_0000,
                0x7F80_0000,
                0x7F80_0000,
                0x7F80_0000,
                0x7F80_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        cases = self.value_cases()
        for case, source in cases:
            output = source.square()
            expected = source.mul(source)
            self.assert_tensor_matches(output, expected, case=case)
            self.assertFalse(output.is_set_to(source))
        np.testing.assert_array_equal(
            np.asarray(cases[-1][1].square(), dtype=np.float32)
            .reshape(-1)
            .view(np.uint32),
            expected_special_bits,
        )

    def test_shared_operand_gradients_cover_views_repeated_backward_and_no_grad(self):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            square_leaf, square_input, square_weights = self.autograd_case(case)
            mul_leaf, mul_input, mul_weights = self.autograd_case(case)
            square_output = square_input.square()
            mul_output = mul_input.mul(mul_input)
            self.assert_tensor_matches(
                square_output, mul_output, case=(case, "forward")
            )

            if square_weights is None:
                square_loss = (
                    square_output if case == "scalar" else square_output.sum()
                )
                mul_loss = mul_output if case == "scalar" else mul_output.sum()
            else:
                square_loss = (square_output * square_weights).sum()
                mul_loss = (mul_output * mul_weights).sum()
            square_loss.backward()
            mul_loss.backward()
            self.assert_tensor_matches(
                square_leaf.grad, mul_leaf.grad, case=(case, "gradient")
            )

        accumulated = torch.tensor([2.0, -3.0], requires_grad=True)
        accumulated.square().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(accumulated.grad), np.asarray([4.0, -6.0], dtype=np.float32)
        )
        accumulated.square().sum().backward()
        np.testing.assert_array_equal(
            np.asarray(accumulated.grad), np.asarray([8.0, -12.0], dtype=np.float32)
        )

        freed = torch.tensor([2.0, -3.0], requires_grad=True)
        loss = freed.square().sum()
        loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        for case in ("scalar", "empty", "offset", "noncontiguous"):
            _, source, _ = self.autograd_case(case)
            detached = source.detach()
            expected = detached.mul(detached)
            with torch.no_grad():
                actual = source.square()
            self.assert_tensor_matches(actual, expected, case=(case, "no_grad"))
            self.assertFalse(actual.is_set_to(source))

    def test_pow_backward_order_preserves_overflow_subnormal_and_nonfinite_bits(self):
        input_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x3F80_0000,
                0xBF80_0000,
                0x3F00_0000,
                0x3F00_0000,
                0x0000_0001,
                0x0000_0001,
                0x0000_0000,
                0x8000_0000,
                0x3E80_0000,
                0x3E80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC0_1234,
                0xFFC0_5678,
            ),
            dtype=np.uint32,
        )
        expected_gradient_bits = np.asarray(
            (
                0x0000_0000,
                0x0000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0x0000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0x7F80_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        leaf = torch.tensor(
            memoryview(input_bits.view(np.float32)), requires_grad=True
        )
        weights = torch.tensor(memoryview(weight_bits.view(np.float32)))
        output = leaf.square()
        (output * weights).sum().backward()

        np.testing.assert_array_equal(
            np.asarray(leaf.grad, dtype=np.float32).view(np.uint32),
            expected_gradient_bits,
        )

        probability = torch.tensor([2.0], requires_grad=True).square()
        with self.assertRaisesRegex(
            ValueError,
            (
                r"^dropout probability has to be between 0 and 1, but got "
                r"tensor\(\[4\.\], grad_fn=<PowBackward0>\)$"
            ),
        ):
            torch.nn.functional.dropout(
                torch.tensor([1.0]), p=probability, training=False
            )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "square")
        bound = tensor.square

        self.assertIs(torch.Tensor.square, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'square' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "square")
        self.assertEqual(descriptor.__qualname__, "TensorBase.square")
        self.assertEqual(bound.__name__, "square")
        self.assertEqual(bound.__qualname__, "Tensor.square")
        self.assertEqual(descriptor.__doc__, SQUARE_DOC)
        self.assertEqual(bound.__doc__, SQUARE_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.square(1), "TensorBase.square() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.square() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.square() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.square(1, 2),
                "TensorBase.square() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.square(input=tensor),
                (
                    "Tensor.square() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.square() takes no keyword arguments"
                ),
            ),
            (
                lambda: bound(unexpected=True),
                "Tensor.square() takes no keyword arguments",
            ),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.square() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.square() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'square' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.square() needs an argument",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_torch_function_modes_receive_descriptor_and_forward(self):
        tensor = torch.tensor([2.0, -3.0], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "square")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.square()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = tensor.square()
        self.assertEqual(order, ["upper", "lower"])
        self.assertEqual(forwarded.tolist(), [4.0, 9.0])
        forwarded.sum().backward()
        self.assertEqual(tensor.grad.tolist(), [4.0, -6.0])

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                tensor.square(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_inplace_dtype_and_device_extensions_remain_unsupported(self):
        tensor = torch.tensor([2.0])
        self.assertFalse(hasattr(torch, "square"))
        self.assertNotIn("square", torch.__all__)
        self.assertFalse(hasattr(torch.Tensor, "square_"))
        self.assertFalse(hasattr(tensor, "square_"))
        with self.assertRaises(TypeError):
            tensor.square(out=None)
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([2.0], device="cuda")


if __name__ == "__main__":
    unittest.main()
