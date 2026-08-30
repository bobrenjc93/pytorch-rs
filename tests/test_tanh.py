import copy
import inspect
import pickle
import re
import sys
import types
import unittest

import numpy as np
import torch_rs as torch


TANH_DOC = """
tanh() -> Tensor

See :func:`torch.tanh`
"""

TOP_LEVEL_TANH_DOC = """
tanh(input, *, out=None) -> Tensor

Returns a new tensor with the hyperbolic tangent of the elements
of :attr:`input`.

.. math::
    \\text{out}_{i} = \\tanh(\\text{input}_{i})

Args:
    input (Tensor): the input tensor.

Keyword args:
    out (Tensor, optional): the output tensor.

Example::

    >>> a = torch.randn(4)
    >>> a
    tensor([ 0.8986, -0.7279,  1.1745,  0.2611])
    >>> torch.tanh(a)
    tensor([ 0.7156, -0.6218,  0.8257,  0.2553])
"""

AUTOGRAD_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x4110_2C66,
        0xC110_2C66,
        0x4110_2C67,
        0xC110_2C67,
    ),
    dtype=np.uint32,
)
AUTOGRAD_WEIGHTS = np.asarray(
    (1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0), dtype=np.float32
)
AUTOGRAD_OUTPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x3EEC_9A9F,
        0xBEEC_9A9F,
        0x3F7F_FFFF,
        0xBF7F_FFFF,
        0x3F80_0000,
        0xBF80_0000,
    ),
    dtype=np.uint32,
)
AUTOGRAD_GRADIENT_BITS = np.asarray(
    (
        0x3F80_0000,
        0xC000_0000,
        0x3EC9_54A3,
        0xBE49_54A3,
        0x34C0_0000,
        0xB500_0000,
        0x0000_0000,
        0x8000_0000,
    ),
    dtype=np.uint32,
)
AUTOGRAD_ACCUMULATED_GRADIENT_BITS = np.asarray(
    (
        0x4000_0000,
        0xC080_0000,
        0x3F49_54A3,
        0xBEC9_54A3,
        0x3540_0000,
        0xB580_0000,
        0x0000_0000,
        0x8000_0000,
    ),
    dtype=np.uint32,
)


class TensorTanhTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor, dtype=np.float32).reshape(-1).view(np.uint32)

    def assert_result(self, actual, source, expected_stride, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, source.shape)
            self.assertEqual(actual.stride(), expected_stride)
            self.assertEqual(actual.storage_offset(), 0)
            self.assertFalse(actual.requires_grad)
            self.assertTrue(actual.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertFalse(actual.is_set_to(source))
            if source.numel():
                self.assertNotEqual(actual.data_ptr(), source.data_ptr())

    @staticmethod
    def make_cases():
        base = torch.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
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
                0x3EAA_AAAB,
                0xBEAA_AAAB,
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
        channels_last = torch.tensor(
            np.linspace(-3.0, 3.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-3.0, 3.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist()
        ).contiguous(memory_format=torch.channels_last_3d)
        return (
            ("scalar", torch.tensor(-0.0), ()),
            (
                "empty offset",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                (2, 1),
            ),
            ("empty singleton trailing", torch.zeros((0, 1)), (1, 1)),
            ("empty singleton middle", torch.zeros((0, 1, 2)), (2, 2, 1)),
            ("empty singleton surrounding", torch.zeros((1, 0, 1)), (1, 1, 1)),
            ("offset", strided[1], (1, 3)),
            ("noncontiguous", strided, (1, 4, 12)),
            ("channels last", channels_last, channels_last.stride()),
            (
                "channels last 3d",
                channels_last_3d,
                channels_last_3d.stride(),
            ),
            (
                "numerical edges",
                torch.tensor(memoryview(special_bits.view(np.float32))),
                (1,),
            ),
        )

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.tanh(source)),
            ("input", lambda: torch.tanh(input=source)),
            ("x", lambda: torch.tanh(x=source)),
            ("a", lambda: torch.tanh(a=source)),
            ("x1", lambda: torch.tanh(x1=source)),
            ("out none", lambda: torch.tanh(source, out=None)),
            ("alias and out none", lambda: torch.tanh(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        expected_special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EA4_9D52,
                0xBEA4_9D52,
                0x3F42_F7D6,
                0xBF42_F7D6,
                0x3F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7FC1_2345,
                0xFFC1_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        for case, source, expected_stride in self.make_cases():
            output = source.tanh()
            self.assert_result(output, source, expected_stride, case=case)
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_bits(output), expected_special_bits
                )
            else:
                np.testing.assert_allclose(
                    np.asarray(output, dtype=np.float32),
                    np.tanh(np.asarray(source, dtype=np.float32)),
                    rtol=3.0 * np.finfo(np.float32).eps,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                )

    def test_top_level_calls_reuse_tensor_tanh_values_layouts_and_storage(self):
        for case, source, expected_stride in self.make_cases():
            expected = source.tanh()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_result(
                    actual, source, expected_stride, case=(case, form)
                )
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    def test_finite_owned_scalar_autograd_matches_signed_zero_and_saturation(self):
        cases = (
            (0x0000_0000, 0x0000_0000, 0x3F80_0000),
            (0x8000_0000, 0x8000_0000, 0x3F80_0000),
            (0x0000_0001, 0x0000_0001, 0x3F80_0000),
            (0x8000_0001, 0x8000_0001, 0x3F80_0000),
            (0x3F00_0000, 0x3EEC_9A9F, 0x3F49_54A3),
            (0xBF00_0000, 0xBEEC_9A9F, 0x3F49_54A3),
            (0x4110_2C66, 0x3F7F_FFFF, 0x3400_0000),
            (0xC110_2C66, 0xBF7F_FFFF, 0x3400_0000),
            (0x4110_2C67, 0x3F80_0000, 0x0000_0000),
            (0xC110_2C67, 0xBF80_0000, 0x0000_0000),
        )
        for input_bits, output_bits, gradient_bits in cases:
            with self.subTest(input_bits=f"0x{input_bits:08x}"):
                value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
                leaf = torch.tensor(value, requires_grad=True)
                output = leaf.tanh()

                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertEqual(output.shape, ())
                self.assertEqual(output.stride(), ())
                self.assertEqual(output.storage_offset(), 0)
                self.assertFalse(output.is_set_to(leaf))
                self.assertEqual(self.tensor_bits(output).item(), output_bits)

                output.backward()
                self.assertEqual(self.tensor_bits(leaf.grad).item(), gradient_bits)
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    output.backward()

        output = torch.tensor(0.5, requires_grad=True).tanh()
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<TanhBackward0>",
        )

    def test_finite_owned_vectors_support_weighted_autograd_and_empty_inputs(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.tanh()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (8,))
        self.assertEqual(output.stride(), (1,))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<TanhBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(values, requires_grad=True)
        for _ in range(2):
            (accumulated.tanh() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad), AUTOGRAD_ACCUMULATED_GRADIENT_BITS
        )

        composed = torch.tensor([0.5, -0.5], requires_grad=True)
        composed.tanh().sin().sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(composed.grad),
            np.asarray((0x3F34_3692, 0x3F34_3692), dtype=np.uint32),
        )

        empty = torch.tensor([], requires_grad=True)
        empty_output = empty.tanh()
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, (0,))
        self.assertEqual(empty_output.stride(), (1,))
        self.assertFalse(empty_output.is_set_to(empty))
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_output),
            ", grad_fn=<TanhBackward0>",
        )
        empty_loss = empty_output.sum()
        empty_loss.backward()
        self.assertEqual(empty.grad.shape, (0,))
        self.assertEqual(empty.grad.stride(), (1,))
        self.assertEqual(empty.grad.tolist(), [])
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            empty_loss.backward()

        higher_order = torch.tensor([0.25, -0.25], requires_grad=True)
        higher_order_loss = higher_order.tanh().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_matrices_support_weighted_autograd_and_empty_shapes(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 4).tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.tanh()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.stride(), (4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<TanhBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (2, 4))
        self.assertEqual(leaf.grad.stride(), (4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(values, requires_grad=True)
        for _ in range(2):
            (accumulated.tanh() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape, expected_stride in (
            ((0, 0), (1, 1)),
            ((0, 3), (3, 1)),
            ((2, 0), (1, 1)),
        ):
            with self.subTest(shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.tanh()
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                self.assertEqual(empty_output.storage_offset(), 0)
                self.assertFalse(empty_output.is_set_to(empty))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        empty_output
                    ),
                    ", grad_fn=<TanhBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

    def test_finite_owned_rank_three_tensors_support_weighted_autograd_and_empty_shapes(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.tanh()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (2, 1, 4))
        self.assertEqual(output.stride(), (4, 4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<TanhBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (2, 1, 4))
        self.assertEqual(leaf.grad.stride(), (4, 4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(values, requires_grad=True)
        for _ in range(2):
            (accumulated.tanh() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape, expected_stride in (
            ((0, 1, 4), (4, 4, 1)),
            ((2, 0, 4), (4, 4, 1)),
            ((2, 1, 0), (1, 1, 1)),
            ((1, 0, 1), (1, 1, 1)),
        ):
            with self.subTest(shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.tanh()
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                self.assertEqual(empty_output.storage_offset(), 0)
                self.assertFalse(empty_output.is_set_to(empty))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        empty_output
                    ),
                    ", grad_fn=<TanhBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[[0.25, -0.25]]], requires_grad=True)
        higher_order_loss = higher_order.tanh().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_rank_four_tensors_support_weighted_autograd_and_empty_shapes(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 4).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4).tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.tanh()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (1, 2, 1, 4))
        self.assertEqual(output.stride(), (8, 4, 4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<TanhBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (1, 2, 1, 4))
        self.assertEqual(leaf.grad.stride(), (8, 4, 4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), AUTOGRAD_GRADIENT_BITS
        )
        gradient_before_repeated_backward = self.tensor_bits(leaf.grad).copy()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()
        np.testing.assert_array_equal(
            self.tensor_bits(leaf.grad), gradient_before_repeated_backward
        )

        accumulated = torch.tensor(values, requires_grad=True)
        for _ in range(2):
            (accumulated.tanh() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape, expected_stride in (
            ((0, 1, 1, 4), (4, 4, 4, 1)),
            ((2, 0, 1, 4), (4, 4, 4, 1)),
            ((2, 1, 0, 4), (4, 4, 4, 1)),
            ((2, 1, 1, 0), (1, 1, 1, 1)),
            ((1, 0, 1, 1), (1, 1, 1, 1)),
        ):
            with self.subTest(shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.tanh()
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                self.assertEqual(empty_output.storage_offset(), 0)
                self.assertFalse(empty_output.is_set_to(empty))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        empty_output
                    ),
                    ", grad_fn=<TanhBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[[[0.25, -0.25]]]], requires_grad=True)
        higher_order_loss = higher_order.tanh().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

        tracked = torch.tensor(values, requires_grad=True)
        with torch.no_grad():
            no_grad_output = tracked.tanh()
        self.assert_result(
            no_grad_output,
            tracked,
            (8, 4, 4, 1),
            case="rank-four no_grad",
        )
        self.assertFalse(no_grad_output.requires_grad)

        detached = tracked.detach()
        detached_output = detached.tanh()
        self.assert_result(
            detached_output,
            detached,
            (8, 4, 4, 1),
            case="rank-four detached",
        )
        np.testing.assert_array_equal(
            self.tensor_bits(no_grad_output), self.tensor_bits(detached_output)
        )

    def test_scalar_autograd_composes_accumulates_and_obeys_grad_mode(self):
        composed = torch.tensor(0.5, requires_grad=True)
        composed.tanh().sin().backward()
        tanh = np.tanh(np.float32(0.5))
        expected = np.cos(tanh) * (np.float32(1.0) - tanh * tanh)
        np.testing.assert_allclose(
            np.asarray(composed.grad), expected, rtol=2.0e-6, atol=0.0
        )

        accumulated = torch.tensor(-0.5, requires_grad=True)
        accumulated.tanh().backward()
        first = np.asarray(accumulated.grad).copy()
        accumulated.tanh().backward()
        np.testing.assert_array_equal(np.asarray(accumulated.grad), first * 2.0)

        higher_order = torch.tensor(0.25, requires_grad=True)
        loss = higher_order.tanh()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        loss.backward()
        self.assertIsNotNone(higher_order.grad)

        tracked = torch.tensor(-0.5, requires_grad=True)
        with torch.no_grad():
            no_grad_output = tracked.tanh()
        self.assert_result(no_grad_output, tracked, (), case="scalar no_grad")
        self.assertTrue(tracked.tanh().requires_grad)

        detached = tracked.detach()
        detached_output = detached.tanh()
        self.assert_result(detached_output, detached, (), case="scalar detached")
        np.testing.assert_array_equal(
            self.tensor_bits(no_grad_output), self.tensor_bits(detached_output)
        )

    def test_unsupported_tracked_inputs_fail_before_existing_graphs_or_layouts_change(self):
        message = r"^tanh\(\): autograd recording is not supported$"

        for bits in (0x7F80_0000, 0xFF80_0000, 0x7FC1_2345, 0xFFC5_4321):
            with self.subTest(nonfinite=f"0x{bits:08x}"):
                value = np.asarray(bits, dtype=np.uint32).view(np.float32).item()
                leaf = torch.tensor(value, requires_grad=True)
                for call in (
                    leaf.tanh,
                    lambda leaf=leaf: torch.tanh(leaf),
                    lambda leaf=leaf: torch.tanh(leaf, out=None),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(leaf.grad)
                leaf.sum().backward()
                self.assertEqual(leaf.grad.item(), 1.0)

                vector = torch.tensor([0.5, value], requires_grad=True)
                for call in (
                    vector.tanh,
                    lambda vector=vector: torch.tanh(vector),
                    lambda vector=vector: torch.tanh(vector, out=None),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(vector.grad)
                vector.sum().backward()
                self.assertEqual(vector.grad.tolist(), [1.0, 1.0])

                matrix = torch.tensor([[0.5, value]], requires_grad=True)
                for call in (
                    matrix.tanh,
                    lambda matrix=matrix: torch.tanh(matrix),
                    lambda matrix=matrix: torch.tanh(matrix, out=None),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(matrix.grad)
                matrix.sum().backward()
                self.assertEqual(matrix.grad.tolist(), [[1.0, 1.0]])

                rank_three = torch.tensor([[[0.5, value]]], requires_grad=True)
                for call in (
                    rank_three.tanh,
                    lambda rank_three=rank_three: torch.tanh(rank_three),
                    lambda rank_three=rank_three: torch.tanh(
                        rank_three, out=None
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(rank_three.grad)
                rank_three.sum().backward()
                self.assertEqual(rank_three.grad.tolist(), [[[1.0, 1.0]]])

                rank_four = torch.tensor([[[[0.5, value]]]], requires_grad=True)
                for call in (
                    rank_four.tanh,
                    lambda rank_four=rank_four: torch.tanh(rank_four),
                    lambda rank_four=rank_four: torch.tanh(
                        rank_four, out=None
                    ),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(rank_four.grad)
                rank_four.sum().backward()
                self.assertEqual(rank_four.grad.tolist(), [[[[1.0, 1.0]]]])

        rank_five = torch.tensor([[[[[0.5, -1.0]]]]], requires_grad=True)
        for call in (
            rank_five.tanh,
            lambda: torch.tanh(rank_five),
            lambda: torch.tanh(rank_five, out=None),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        self.assertIsNone(rank_five.grad)
        rank_five.sum().backward()
        self.assertEqual(rank_five.grad.tolist(), [[[[[1.0, 1.0]]]]])

        view_base = torch.tensor([0.5], requires_grad=True)
        scalar_view = view_base[0]
        self.assertFalse(scalar_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            scalar_view.tanh()
        scalar_view.backward()
        self.assertEqual(view_base.grad.tolist(), [1.0])

        vector_view_base = torch.tensor(
            [[0.5, -1.0], [2.0, -3.0]], requires_grad=True
        )
        vector_view = vector_view_base[0]
        self.assertEqual(vector_view.shape, (2,))
        self.assertFalse(vector_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            vector_view.tanh()
        vector_view.sum().backward()
        self.assertEqual(vector_view_base.grad.tolist(), [[1.0, 1.0], [0.0, 0.0]])

        matrix_view_base = torch.tensor(
            [
                [[0.5, -1.0], [2.0, -3.0]],
                [[4.0, -5.0], [6.0, -7.0]],
            ],
            requires_grad=True,
        )
        matrix_view = matrix_view_base[0]
        self.assertEqual(matrix_view.shape, (2, 2))
        self.assertFalse(matrix_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            matrix_view.tanh()
        matrix_view.sum().backward()
        self.assertEqual(
            matrix_view_base.grad.tolist(),
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        )

        rank_three_view_base = torch.tensor(
            [[[[0.5, -1.0]]], [[[2.0, -3.0]]]], requires_grad=True
        )
        rank_three_view = rank_three_view_base[0]
        self.assertEqual(rank_three_view.shape, (1, 1, 2))
        self.assertFalse(rank_three_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_three_view.tanh()
        rank_three_view.sum().backward()
        self.assertEqual(
            rank_three_view_base.grad.tolist(),
            [[[[1.0, 1.0]]], [[[0.0, 0.0]]]],
        )

        rank_four_view_base = torch.tensor(
            [[[[[0.5, -1.0]]]], [[[[2.0, -3.0]]]]], requires_grad=True
        )
        rank_four_view = rank_four_view_base[0]
        self.assertEqual(rank_four_view.shape, (1, 1, 1, 2))
        self.assertFalse(rank_four_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_four_view.tanh()
        rank_four_view.sum().backward()
        self.assertEqual(
            rank_four_view_base.grad.tolist(),
            [[[[[1.0, 1.0]]]], [[[[0.0, 0.0]]]]],
        )

        nonleaf_base = torch.tensor([[[0.5, -0.5]]], requires_grad=True)
        nonleaf = nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            nonleaf.tanh()
        nonleaf.sum().backward()
        np.testing.assert_allclose(
            np.asarray(nonleaf_base.grad),
            np.cos(np.asarray([[[0.5, -0.5]]], dtype=np.float32)),
        )

        with torch.no_grad():
            no_grad_view = matrix_view_base[0]
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        self.assertEqual(no_grad_view.shape, (2, 2))
        with self.assertRaisesRegex(RuntimeError, message):
            no_grad_view.tanh()

        with torch.no_grad():
            no_grad_rank_three_view = rank_three_view_base[0]
        self.assertTrue(no_grad_rank_three_view.requires_grad)
        self.assertTrue(no_grad_rank_three_view.is_leaf)
        self.assertEqual(no_grad_rank_three_view.shape, (1, 1, 2))
        with self.assertRaisesRegex(RuntimeError, message):
            no_grad_rank_three_view.tanh()

        with torch.no_grad():
            no_grad_rank_four_view = rank_four_view_base[0]
        self.assertTrue(no_grad_rank_four_view.requires_grad)
        self.assertTrue(no_grad_rank_four_view.is_leaf)
        self.assertEqual(no_grad_rank_four_view.shape, (1, 1, 1, 2))
        with self.assertRaisesRegex(RuntimeError, message):
            no_grad_rank_four_view.tanh()

        empty_view_base = torch.zeros((1, 0), requires_grad=True)
        with torch.no_grad():
            empty_view = empty_view_base[0]
        self.assertTrue(empty_view.requires_grad)
        self.assertTrue(empty_view.is_leaf)
        self.assertEqual(empty_view.shape, (0,))
        with self.assertRaisesRegex(RuntimeError, message):
            empty_view.tanh()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.tanh()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.tanh()

    def test_top_level_supports_finite_owned_rank_four_or_lower_autograd(self):
        scalar = torch.tensor(0.5, requires_grad=True)
        calls = self.top_level_calls(scalar)
        for form, call in calls:
            with self.subTest(form=form, mode="recording"):
                output = call()
                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertEqual(output.shape, ())
                output.backward()
        unit_gradient = np.asarray(0x3F49_54A3, dtype=np.uint32).view(np.float32)
        expected = np.float32(len(calls)) * unit_gradient
        np.testing.assert_allclose(scalar.grad.item(), expected, rtol=2.0e-6)

        vector = torch.tensor(
            AUTOGRAD_INPUT_BITS.view(np.float32).tolist(), requires_grad=True
        )
        vector_output = torch.tanh(vector, out=None)
        self.assertTrue(vector_output.requires_grad)
        self.assertFalse(vector_output.is_leaf)
        self.assertEqual(vector_output.shape, (8,))
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(vector_output),
            ", grad_fn=<TanhBackward0>",
        )
        (vector_output * torch.tensor(AUTOGRAD_WEIGHTS.tolist())).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(vector.grad), AUTOGRAD_GRADIENT_BITS
        )

        matrix = torch.tensor(
            AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist(),
            requires_grad=True,
        )
        matrix_output = torch.tanh(matrix, out=None)
        self.assertTrue(matrix_output.requires_grad)
        self.assertFalse(matrix_output.is_leaf)
        self.assertEqual(matrix_output.shape, (2, 4))
        self.assertEqual(matrix_output.stride(), (4, 1))
        self.assertEqual(matrix_output.storage_offset(), 0)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(matrix_output),
            ", grad_fn=<TanhBackward0>",
        )
        np.testing.assert_array_equal(
            self.tensor_bits(matrix_output), AUTOGRAD_OUTPUT_BITS
        )
        matrix_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 4).tolist())
        (matrix_output * matrix_weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(matrix.grad), AUTOGRAD_GRADIENT_BITS
        )

        rank_three = torch.tensor(
            AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4).tolist(),
            requires_grad=True,
        )
        rank_three_output = torch.tanh(rank_three, out=None)
        self.assertTrue(rank_three_output.requires_grad)
        self.assertFalse(rank_three_output.is_leaf)
        self.assertEqual(rank_three_output.shape, (2, 1, 4))
        self.assertEqual(rank_three_output.stride(), (4, 4, 1))
        self.assertEqual(rank_three_output.storage_offset(), 0)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                rank_three_output
            ),
            ", grad_fn=<TanhBackward0>",
        )
        np.testing.assert_array_equal(
            self.tensor_bits(rank_three_output), AUTOGRAD_OUTPUT_BITS
        )
        rank_three_weights = torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist()
        )
        (rank_three_output * rank_three_weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(rank_three.grad), AUTOGRAD_GRADIENT_BITS
        )

        rank_four_values = (
            AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 4).tolist()
        )
        rank_four_weights = torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4).tolist()
        )
        for form in (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        ):
            rank_four = torch.tensor(rank_four_values, requires_grad=True)
            if form == "positional":
                rank_four_output = torch.tanh(rank_four)
            elif form == "out none":
                rank_four_output = torch.tanh(rank_four, out=None)
            elif form == "alias and out none":
                rank_four_output = torch.tanh(x=rank_four, out=None)
            else:
                rank_four_output = torch.tanh(**{form: rank_four})
            self.assertTrue(rank_four_output.requires_grad)
            self.assertFalse(rank_four_output.is_leaf)
            self.assertEqual(rank_four_output.shape, (1, 2, 1, 4))
            self.assertEqual(rank_four_output.stride(), (8, 4, 4, 1))
            self.assertEqual(rank_four_output.storage_offset(), 0)
            self.assertEqual(
                torch._C._nn_functional_dropout_tensor_autograd_suffix(
                    rank_four_output
                ),
                ", grad_fn=<TanhBackward0>",
            )
            np.testing.assert_array_equal(
                self.tensor_bits(rank_four_output), AUTOGRAD_OUTPUT_BITS
            )
            (rank_four_output * rank_four_weights).sum().backward()
            np.testing.assert_array_equal(
                self.tensor_bits(rank_four.grad), AUTOGRAD_GRADIENT_BITS
            )

        leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)[1]

        for form, call in self.top_level_calls(source):
            with self.subTest(form=form, mode="recording"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, mode="no_grad"):
                with torch.no_grad():
                    actual = call()
                    expected = source.tanh()
                self.assert_result(actual, source, (1,), case=(form, "no_grad"))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        for form, call in self.top_level_calls(extreme):
            with self.subTest(form=form, metadata="autograd precedence"):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): autograd recording is not supported$",
                ):
                    call()
            with self.subTest(form=form, metadata="layout planning"):
                with torch.no_grad():
                    with self.assertRaisesRegex(
                        RuntimeError, "Stride calculation overflowed"
                    ):
                        call()

        detached = source.detach()
        expected = detached.tanh()
        for form, call in self.top_level_calls(detached):
            actual = call()
            self.assert_result(actual, detached, (1,), case=(form, "detached"))
            np.testing.assert_array_equal(
                self.tensor_bits(actual), self.tensor_bits(expected)
            )

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([0.5])
        descriptor = inspect.getattr_static(torch.Tensor, "tanh")
        bound = tensor.tanh

        self.assertIs(torch.Tensor.tanh, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'tanh' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "tanh")
        self.assertEqual(descriptor.__qualname__, "TensorBase.tanh")
        self.assertEqual(bound.__name__, "tanh")
        self.assertEqual(bound.__qualname__, "Tensor.tanh")
        self.assertEqual(descriptor.__doc__, TANH_DOC)
        self.assertEqual(bound.__doc__, TANH_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)

        for callable_object, expected_signature in (
            (descriptor, "(self, /)"),
            (bound, "()"),
        ):
            if sys.version_info >= (3, 13):
                self.assertEqual(callable_object.__text_signature__, "($self, /)")
                self.assertEqual(
                    str(inspect.signature(callable_object)), expected_signature
                )
            else:
                self.assertIsNone(callable_object.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        cases = (
            (lambda: tensor.tanh(1), "TensorBase.tanh() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.tanh() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.tanh() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.tanh(1, 2),
                "TensorBase.tanh() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.tanh(input=tensor),
                (
                    "Tensor.tanh() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.tanh() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.tanh() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.tanh() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.tanh() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'tanh' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.tanh() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        descriptor = inspect.getattr_static(torch.Tensor, "tanh")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return marker

        mode = RecordingMode()
        with mode:
            result = tensor.tanh()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tensor)
        self.assertIsNone(kwargs)

        extreme = torch.zeros((0,)).reshape((0, sys.maxsize, 3))
        bypass = RecordingMode()
        with bypass:
            self.assertIs(extreme.tanh(), marker)
        self.assertEqual(len(bypass.calls), 1)

        order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([0.5])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = plain.tanh()
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_allclose(forwarded.tolist(), [0.46211717], rtol=1.0e-6)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_output = tensor.tanh()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_output.requires_grad)
        self.assertFalse(tracked_output.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(tracked_output),
            ", grad_fn=<TanhBackward0>",
        )

        invalid_mode = RecordingMode()
        with self.assertRaises(TypeError):
            with invalid_mode:
                plain.tanh(1)
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_modes_and_overrides_dispatch_before_native_limits(self):
        tensor = torch.tensor([0.5], requires_grad=True)
        destination = torch.tensor([0.0])
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode()
        with mode:
            self.assertIs(torch.tanh(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.tanh)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.tanh(Override()), marker)
        self.assertIs(torch.tanh(torch.tensor([0.5]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for dispatched, types_, _, _ in override_calls:
            self.assertIs(dispatched, torch.tanh)
            self.assertEqual(types_, (Override,))

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("base")
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append("derived")
                return marker

        self.assertIs(torch.tanh(BaseOverride(), out=DerivedOverride()), marker)
        self.assertEqual(subclass_order, ["derived"])

        forwarding_order = []

        class ForwardingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        plain = torch.tensor([0.5])
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = torch.tanh(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        np.testing.assert_array_equal(
            self.tensor_bits(forwarded), self.tensor_bits(plain.tanh())
        )

        events = []

        class DecliningMode(torch.overrides.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                events.append("mode")
                return NotImplemented

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                events.append("override")
                return marker

        with DecliningMode():
            self.assertIs(torch.tanh(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid_mode = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^tanh\(\): argument 'out' must be Tensor, not list$",
        ):
            with invalid_mode:
                torch.tanh(tensor, out=[])
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([0.5, -0.5], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.tanh(source, out=destination)),
            ("keyword", lambda: torch.tanh(input=source, out=destination)),
            ("alias", lambda: torch.tanh(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^tanh\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])
                self.assertIsNone(source.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): the 'out' argument is not supported$",
        ):
            torch.tanh(extreme, out=destination)
        self.assertEqual(destination.tolist(), [17.0, 19.0])
        self.assertIsNone(source.grad)

        with torch.no_grad():
            actual = torch.tanh(source, out=None)
            expected = source.tanh()
        np.testing.assert_array_equal(
            self.tensor_bits(actual), self.tensor_bits(expected)
        )
        source.tanh().sum().backward()
        self.assertIsNotNone(source.grad)

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.tanh
        self.assertIs(function, torch._C.tanh)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "tanh")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.tanh")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_TANH_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function), r"^<built-in method tanh of type object at 0x[0-9a-f]+>$"
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.tanh, function)
        for action in (
            lambda: setattr(owner, "tanh", None),
            lambda: delattr(owner, "tanh"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.tanh, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("tanh"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["tanh"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([0.5])
        cases = (
            (
                lambda: torch.tanh(),
                'tanh() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.tanh(tensor, tensor),
                "tanh() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.tanh(tensor, input=tensor),
                "tanh() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.tanh(out=tensor),
                'tanh() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.tanh(extra=tensor),
                'tanh() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.tanh(1, extra=True),
                "tanh(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.tanh(input=[]),
                "tanh(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.tanh(tensor, out=[]),
                "tanh(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.tanh(tensor, extra=True, out=[]),
                "tanh(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.tanh(tensor, extra=True),
                "tanh() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.tanh(input=tensor, a=tensor),
                "tanh() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.tanh(a=tensor, x=tensor, out=None),
                "tanh() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.tanh(x=tensor, a=tensor, out=None),
                "tanh() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.tanh(np.zeros((2, 3), dtype=np.float32)),
                (
                    "tanh(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_functional_form_is_exposed_and_inplace_forms_remain_unsupported(self):
        tensor = torch.tensor([0.5])
        self.assertTrue(hasattr(torch.nn.functional, "tanh"))
        self.assertFalse(hasattr(torch.Tensor, "tanh_"))
        self.assertFalse(hasattr(tensor, "tanh_"))
        self.assertFalse(hasattr(torch, "tanh_"))
        self.assertNotIn("tanh_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.tanh(out=None)


if __name__ == "__main__":
    unittest.main()
