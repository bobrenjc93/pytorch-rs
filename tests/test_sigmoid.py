import copy
import inspect
import pickle
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


SIGMOID_DOC = """
sigmoid() -> Tensor

See :func:`torch.sigmoid`
"""

TOP_LEVEL_SIGMOID_DOC = """
sigmoid(input, *, out=None) -> Tensor

Alias for :func:`torch.special.expit`.
"""


SPECIAL_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x807F_FFFF,
        0x0080_0000,
        0x8080_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F7F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
        0xBF7F_FFFF,
        0xBF80_0000,
        0xBFC0_0000,
        0x3FC0_0000,
        0x42B0_0000,
        0x42B2_0000,
        0xC2B0_0000,
        0xC2B2_0000,
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

SPECIAL_OUTPUT_BITS = np.asarray(
    (
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F00_0000,
        0x3F1F_597F,
        0x3F1F_597F,
        0x3F3B_26A8,
        0x3F3B_26A8,
        0x3EC1_4D03,
        0x3E89_B2B1,
        0x3E89_B2B1,
        0x3E3A_CDC2,
        0x3F51_4C8F,
        0x3F80_0000,
        0x3F80_0000,
        0x0041_EDC4,
        0x0000_0000,
        0x3F80_0000,
        0x0000_0000,
        0x3F80_0000,
        0x0000_0000,
        0xFFC1_2345,
        0x7FC1_2345,
        0xFFC1_2345,
        0x7FC5_4321,
    ),
    dtype=np.uint32,
)

AUTOGRAD_INPUT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x3F00_0000,
        0xBF00_0000,
        0x4185_1591,
        0x4185_1592,
        0xC2B1_7217,
        0xC2B1_7218,
    ),
    dtype=np.uint32,
)
AUTOGRAD_WEIGHTS = np.asarray(
    (1.0, -2.0, 0.5, -0.25, 3.0, -4.0, 5.0, -6.0), dtype=np.float32
)
AUTOGRAD_OUTPUT_BITS = np.asarray(
    (
        0x3F00_0000,
        0x3F00_0000,
        0x3F1F_597F,
        0x3EC1_4D03,
        0x3F7F_FFFE,
        0x3F80_0000,
        0x0020_0010,
        0x0000_0000,
    ),
    dtype=np.uint32,
)
AUTOGRAD_GRADIENT_BITS = np.asarray(
    (
        0x3E80_0000,
        0xBF00_0000,
        0x3DF0_A4D0,
        0xBD70_A4D0,
        0x34BF_FFFE,
        0x8000_0000,
        0x00A0_0050,
        0x8000_0000,
    ),
    dtype=np.uint32,
)
AUTOGRAD_ACCUMULATED_GRADIENT_BITS = np.asarray(
    (
        0x3F00_0000,
        0xBF80_0000,
        0x3E70_A4D0,
        0xBDF0_A4D0,
        0x353F_FFFE,
        0x8000_0000,
        0x0120_0050,
        0x8000_0000,
    ),
    dtype=np.uint32,
)


def rank_preserving_nonleaf_parent_cases(module):
    return (
        ("clone", lambda input: input.clone()),
        ("add scalar", lambda input: input + 0.25),
        ("subtract scalar", lambda input: input - 0.25),
        ("reflected subtract", lambda input: 0.25 - input),
        ("multiply scalar", lambda input: input * 1.5),
        (
            "multiply tensor",
            lambda input: input * module.tensor(1.5, dtype=module.float32),
        ),
        ("self multiply", lambda input: input * input),
        ("negate", lambda input: -input),
        ("relu", lambda input: input.relu()),
        ("sin", lambda input: input.sin()),
        ("exp", lambda input: input.exp()),
        ("floor", lambda input: input.floor()),
        ("ceil", lambda input: input.ceil()),
        ("trunc", lambda input: input.trunc()),
        ("fix", lambda input: input.fix()),
        ("sigmoid", lambda input: input.sigmoid()),
        ("tanh", lambda input: input.tanh()),
        ("sqrt", lambda input: input.sqrt()),
        ("square", lambda input: input.square()),
    )


def scalar_nonleaf_parent_cases(module):
    return (*rank_preserving_nonleaf_parent_cases(module), ("sum", lambda input: input.sum()))


class TensorSigmoidTests(unittest.TestCase):
    @staticmethod
    def tensor_values(tensor):
        return np.asarray(tensor, dtype=np.float32)

    @classmethod
    def tensor_bits(cls, tensor):
        return cls.tensor_values(tensor).reshape(-1).view(np.uint32)

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
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        strided = base.transpose(0, 2)
        channels_last = torch.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_3d = torch.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
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
                torch.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
                (1,),
            ),
        )

    @staticmethod
    def top_level_calls(source):
        return (
            ("positional", lambda: torch.sigmoid(source)),
            ("input", lambda: torch.sigmoid(input=source)),
            ("x", lambda: torch.sigmoid(x=source)),
            ("a", lambda: torch.sigmoid(a=source)),
            ("x1", lambda: torch.sigmoid(x1=source)),
            ("out none", lambda: torch.sigmoid(source, out=None)),
            ("alias and out none", lambda: torch.sigmoid(x=source, out=None)),
        )

    def test_values_layouts_offsets_empty_tensors_and_fresh_storage(self):
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))
        for case, source, expected_stride in self.make_cases():
            output = source.sigmoid()
            self.assert_result(output, source, expected_stride, case=case)
            actual = self.tensor_values(output).reshape(-1)
            if case == "numerical edges":
                np.testing.assert_array_equal(actual.view(np.uint32), SPECIAL_OUTPUT_BITS)
            else:
                values = self.tensor_values(source).reshape(-1)
                with np.errstate(over="ignore", invalid="ignore"):
                    expected = np.float32(1.0) / (
                        np.float32(1.0) + np.exp(-values, dtype=np.float32)
                    )
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )

    def test_top_level_calls_reuse_tensor_sigmoid_values_layouts_and_storage(self):
        for case, source, expected_stride in self.make_cases():
            expected = source.sigmoid()
            for form, call in self.top_level_calls(source):
                actual = call()
                self.assert_result(actual, source, expected_stride, case=(case, form))
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )

    @staticmethod
    def make_tracked_cases():
        scalar = torch.tensor(-1.25, requires_grad=True)
        empty = torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1]
        leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        strided = leaf.transpose(0, 2)
        return scalar, empty, strided[1], strided

    def test_finite_owned_scalar_autograd_matches_signed_zero_and_saturation(self):
        cases = (
            (0x0000_0000, 0x3F00_0000, 0x3E80_0000),
            (0x8000_0000, 0x3F00_0000, 0x3E80_0000),
            (0x0000_0001, 0x3F00_0000, 0x3E80_0000),
            (0x8000_0001, 0x3F00_0000, 0x3E80_0000),
            (0x3F00_0000, 0x3F1F_597F, 0x3E70_A4D0),
            (0xBF00_0000, 0x3EC1_4D03, 0x3E70_A4D0),
            (0x4185_1591, 0x3F7F_FFFE, 0x33FF_FFFE),
            (0x4185_1592, 0x3F80_0000, 0x0000_0000),
            (0xC2B1_7217, 0x0020_0010, 0x0020_0010),
            (0xC2B1_7218, 0x0000_0000, 0x0000_0000),
        )
        for input_bits, output_bits, gradient_bits in cases:
            with self.subTest(input_bits=f"0x{input_bits:08x}"):
                value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
                leaf = torch.tensor(value, requires_grad=True)
                output = leaf.sigmoid()

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

        output = torch.tensor(0.5, requires_grad=True).sigmoid()
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

    def test_finite_owned_vectors_support_weighted_autograd_and_empty_inputs(self):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).tolist()
        weights = torch.tensor(AUTOGRAD_WEIGHTS.tolist())
        leaf = torch.tensor(values, requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (8,))
        self.assertEqual(output.stride(), (1,))
        self.assertEqual(output.storage_offset(), 0)
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
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
            (accumulated.sigmoid() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad), AUTOGRAD_ACCUMULATED_GRADIENT_BITS
        )

        empty = torch.tensor([], requires_grad=True)
        empty_output = empty.sigmoid()
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, (0,))
        self.assertEqual(empty_output.stride(), (1,))
        self.assertFalse(empty_output.is_set_to(empty))
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_output),
            ", grad_fn=<SigmoidBackward0>",
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
        higher_order_loss = higher_order.sigmoid().sum()
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
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (2, 4))
        self.assertEqual(output.stride(), (4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (2, 4))
        self.assertEqual(leaf.grad.stride(), (4, 1))
        self.assertEqual(leaf.grad.storage_offset(), 0)
        self.assertFalse(leaf.grad.requires_grad)
        self.assertTrue(leaf.grad.is_leaf)
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
            (accumulated.sigmoid() * weights).sum().backward()
        self.assertEqual(accumulated.grad.shape, (2, 4))
        self.assertEqual(accumulated.grad.stride(), (4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape, expected_stride in (
            ((0, 0), (1, 1)),
            ((0, 3), (3, 1)),
            ((2, 0), (1, 1)),
        ):
            with self.subTest(empty_shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.sigmoid()
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
                    ", grad_fn=<SigmoidBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.storage_offset(), 0)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[0.25, -0.25]], requires_grad=True)
        higher_order_loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_rank_three_supports_singletons_empty_shapes_and_composition(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4)
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist())
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (2, 1, 4))
        self.assertEqual(output.stride(), (4, 4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertFalse(output.is_set_to(leaf))
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (2, 1, 4))
        self.assertEqual(leaf.grad.stride(), (4, 4, 1))
        self.assertEqual(leaf.grad.storage_offset(), 0)
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

        accumulated = torch.tensor(values.tolist(), requires_grad=True)
        for _ in range(2):
            (accumulated.sigmoid() * weights).sum().backward()
        self.assertEqual(accumulated.grad.shape, (2, 1, 4))
        self.assertEqual(accumulated.grad.stride(), (4, 4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        composed = torch.tensor(values.tolist(), requires_grad=True)
        composed.sigmoid().sin().sum().backward()
        sigmoid_values = AUTOGRAD_OUTPUT_BITS.view(np.float32).reshape(2, 1, 4)
        expected_composed_gradient = (
            np.cos(sigmoid_values, dtype=np.float32)
            * (np.float32(1.0) - sigmoid_values)
            * sigmoid_values
        )
        np.testing.assert_allclose(
            np.asarray(composed.grad),
            expected_composed_gradient,
            rtol=2.0e-6,
            atol=0.0,
        )

        for shape, expected_stride in (
            ((0, 1, 3), (3, 3, 1)),
            ((1, 0, 3), (3, 3, 1)),
            ((2, 3, 0), (3, 1, 1)),
            ((0, 0, 0), (1, 1, 1)),
        ):
            with self.subTest(empty_shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.sigmoid()
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
                    ", grad_fn=<SigmoidBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.storage_offset(), 0)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[[0.25, -0.25]]], requires_grad=True)
        higher_order_loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_rank_four_supports_singletons_empty_shapes_and_composition(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 4)
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4).tolist())
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (1, 2, 1, 4))
        self.assertEqual(output.stride(), (8, 4, 4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertFalse(output.is_set_to(leaf))
        self.assertNotEqual(output.data_ptr(), leaf.data_ptr())
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (1, 2, 1, 4))
        self.assertEqual(leaf.grad.stride(), (8, 4, 4, 1))
        self.assertEqual(leaf.grad.storage_offset(), 0)
        self.assertIs(leaf.grad.dtype, torch.float32)
        self.assertEqual(leaf.grad.device, torch.device("cpu"))
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

        accumulated = torch.tensor(values.tolist(), requires_grad=True)
        for _ in range(2):
            (accumulated.sigmoid() * weights).sum().backward()
        self.assertEqual(accumulated.grad.shape, (1, 2, 1, 4))
        self.assertEqual(accumulated.grad.stride(), (8, 4, 4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        composed = torch.tensor(values.tolist(), requires_grad=True)
        composed.sigmoid().sin().sum().backward()
        sigmoid_values = AUTOGRAD_OUTPUT_BITS.view(np.float32).reshape(1, 2, 1, 4)
        expected_composed_gradient = (
            np.cos(sigmoid_values, dtype=np.float32)
            * (np.float32(1.0) - sigmoid_values)
            * sigmoid_values
        )
        np.testing.assert_allclose(
            np.asarray(composed.grad),
            expected_composed_gradient,
            rtol=2.0e-6,
            atol=0.0,
        )

        for shape, expected_stride in (
            ((0, 1, 2, 3), (6, 6, 3, 1)),
            ((1, 0, 2, 3), (6, 6, 3, 1)),
            ((1, 2, 0, 3), (6, 3, 3, 1)),
            ((1, 2, 3, 0), (6, 3, 1, 1)),
            ((0, 0, 0, 0), (1, 1, 1, 1)),
        ):
            with self.subTest(empty_shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.sigmoid()
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                self.assertEqual(empty_output.storage_offset(), 0)
                self.assertIs(empty_output.dtype, torch.float32)
                self.assertEqual(empty_output.device, torch.device("cpu"))
                self.assertFalse(empty_output.is_set_to(empty))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        empty_output
                    ),
                    ", grad_fn=<SigmoidBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.storage_offset(), 0)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[[[0.25, -0.25]]]], requires_grad=True)
        higher_order_loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_rank_five_supports_ncdhw_singletons_empty_shapes_and_composition(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 1, 4)
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 1, 4).tolist())
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, (1, 2, 1, 1, 4))
        self.assertEqual(output.stride(), (8, 4, 4, 4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertFalse(output.is_set_to(leaf))
        self.assertNotEqual(output.data_ptr(), leaf.data_ptr())
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
        )

        loss = (output * weights).sum()
        loss.backward()
        self.assertEqual(leaf.grad.shape, (1, 2, 1, 1, 4))
        self.assertEqual(leaf.grad.stride(), (8, 4, 4, 4, 1))
        self.assertEqual(leaf.grad.storage_offset(), 0)
        self.assertIs(leaf.grad.dtype, torch.float32)
        self.assertEqual(leaf.grad.device, torch.device("cpu"))
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

        accumulated = torch.tensor(values.tolist(), requires_grad=True)
        for _ in range(2):
            (accumulated.sigmoid() * weights).sum().backward()
        self.assertEqual(accumulated.grad.shape, (1, 2, 1, 1, 4))
        self.assertEqual(accumulated.grad.stride(), (8, 4, 4, 4, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        composed = torch.tensor(values.tolist(), requires_grad=True)
        composed.sigmoid().sin().sum().backward()
        sigmoid_values = AUTOGRAD_OUTPUT_BITS.view(np.float32).reshape(
            1, 2, 1, 1, 4
        )
        expected_composed_gradient = (
            np.cos(sigmoid_values, dtype=np.float32)
            * (np.float32(1.0) - sigmoid_values)
            * sigmoid_values
        )
        np.testing.assert_allclose(
            np.asarray(composed.grad),
            expected_composed_gradient,
            rtol=2.0e-6,
            atol=0.0,
        )

        for shape, expected_stride in (
            ((0, 1, 2, 3, 4), (24, 24, 12, 4, 1)),
            ((1, 0, 2, 3, 4), (24, 24, 12, 4, 1)),
            ((1, 2, 0, 3, 4), (24, 12, 12, 4, 1)),
            ((1, 2, 3, 0, 4), (24, 12, 4, 4, 1)),
            ((1, 2, 3, 4, 0), (24, 12, 4, 1, 1)),
            ((0, 0, 0, 0, 0), (1, 1, 1, 1, 1)),
        ):
            with self.subTest(empty_shape=shape):
                empty = torch.zeros(shape, requires_grad=True)
                empty_output = empty.sigmoid()
                self.assertTrue(empty_output.requires_grad)
                self.assertFalse(empty_output.is_leaf)
                self.assertEqual(empty_output.shape, shape)
                self.assertEqual(empty_output.stride(), expected_stride)
                self.assertEqual(empty_output.storage_offset(), 0)
                self.assertIs(empty_output.dtype, torch.float32)
                self.assertEqual(empty_output.device, torch.device("cpu"))
                self.assertFalse(empty_output.is_set_to(empty))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        empty_output
                    ),
                    ", grad_fn=<SigmoidBackward0>",
                )
                empty_loss = empty_output.sum()
                empty_loss.backward()
                self.assertEqual(empty.grad.shape, shape)
                self.assertEqual(empty.grad.stride(), expected_stride)
                self.assertEqual(empty.grad.storage_offset(), 0)
                self.assertEqual(empty.grad.tolist(), empty.tolist())
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    empty_loss.backward()

        higher_order = torch.tensor([[[[[0.25, -0.25]]]]], requires_grad=True)
        higher_order_loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_finite_owned_rank_six_and_high_rank_autograd(self):
        shape = (1, 2, 1, 1, 1, 4)
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(shape)
        weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(shape).tolist())
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        output = leaf.sigmoid()

        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.shape, shape)
        self.assertEqual(output.stride(), (8, 4, 4, 4, 4, 1))
        self.assertEqual(output.storage_offset(), 0)
        self.assertIs(output.dtype, torch.float32)
        self.assertEqual(output.device, torch.device("cpu"))
        self.assertFalse(output.is_set_to(leaf))
        self.assertNotEqual(output.data_ptr(), leaf.data_ptr())
        np.testing.assert_array_equal(self.tensor_bits(output), AUTOGRAD_OUTPUT_BITS)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
            ", grad_fn=<SigmoidBackward0>",
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

        accumulated = torch.tensor(values.tolist(), requires_grad=True)
        for _ in range(2):
            (accumulated.sigmoid() * weights).sum().backward()
        np.testing.assert_array_equal(
            self.tensor_bits(accumulated.grad),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        composed = torch.tensor(values.tolist(), requires_grad=True)
        composed.sigmoid().sin().sum().backward()
        sigmoid_values = AUTOGRAD_OUTPUT_BITS.view(np.float32).reshape(shape)
        expected_composed_gradient = (
            np.cos(sigmoid_values, dtype=np.float32)
            * (np.float32(1.0) - sigmoid_values)
            * sigmoid_values
        )
        np.testing.assert_allclose(
            np.asarray(composed.grad),
            expected_composed_gradient,
            rtol=2.0e-6,
            atol=0.0,
        )

        empty = torch.zeros((1, 2, 0, 1, 1, 4), requires_grad=True)
        empty_output = empty.sigmoid()
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, empty.shape)
        self.assertEqual(empty_output.stride(), empty.stride())
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(empty_output),
            ", grad_fn=<SigmoidBackward0>",
        )
        empty_loss = empty_output.sum()
        empty_loss.backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.stride(), empty.stride())
        self.assertEqual(empty.grad.tolist(), empty.tolist())
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            empty_loss.backward()

        high_rank_shape = (1,) * 65
        high_rank = torch.full(high_rank_shape, 0.5, requires_grad=True)
        high_rank_output = high_rank.sigmoid()
        self.assertTrue(high_rank_output.requires_grad)
        self.assertFalse(high_rank_output.is_leaf)
        self.assertEqual(high_rank_output.shape, high_rank_shape)
        self.assertEqual(high_rank_output.stride(), (1,) * 65)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(high_rank_output),
            ", grad_fn=<SigmoidBackward0>",
        )
        self.assertEqual(
            np.float32(high_rank_output.item()).view(np.uint32).item(),
            0x3F1F_597F,
        )
        high_rank_output.backward()
        self.assertEqual(
            np.float32(high_rank.grad.item()).view(np.uint32).item(),
            0x3E70_A4D0,
        )
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            high_rank_output.backward()

        high_rank_accumulated = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        high_rank_accumulated.sigmoid().backward()
        first_high_rank_gradient = np.float32(high_rank_accumulated.grad.item())
        high_rank_accumulated.sigmoid().backward()
        self.assertEqual(
            np.float32(high_rank_accumulated.grad.item()).view(np.uint32).item(),
            np.float32(first_high_rank_gradient * np.float32(2.0))
            .view(np.uint32)
            .item(),
        )

        high_rank_composed = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        high_rank_composed.sigmoid().sin().backward()
        sigmoid = np.float32(high_rank_output.item())
        expected_composed = (
            np.cos(sigmoid, dtype=np.float32)
            * (np.float32(1.0) - sigmoid)
            * sigmoid
        )
        np.testing.assert_allclose(
            np.float32(high_rank_composed.grad.item()),
            expected_composed,
            rtol=2.0e-6,
            atol=0.0,
        )

        high_rank_empty_shape = (1,) * 32 + (0,) + (1,) * 32
        high_rank_empty = torch.zeros(high_rank_empty_shape, requires_grad=True)
        high_rank_empty_output = high_rank_empty.sigmoid()
        self.assertTrue(high_rank_empty_output.requires_grad)
        self.assertFalse(high_rank_empty_output.is_leaf)
        self.assertEqual(high_rank_empty_output.shape, high_rank_empty_shape)
        self.assertEqual(high_rank_empty_output.stride(), high_rank_empty.stride())
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                high_rank_empty_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        high_rank_empty_loss = high_rank_empty_output.sum()
        high_rank_empty_loss.backward()
        self.assertEqual(high_rank_empty.grad.shape, high_rank_empty_shape)
        self.assertEqual(high_rank_empty.grad.stride(), high_rank_empty.stride())
        self.assertEqual(high_rank_empty.grad.numel(), 0)
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            high_rank_empty_loss.backward()

        higher_order = torch.full(high_rank_shape, 0.25, requires_grad=True)
        higher_order_loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order.grad)

    def test_scalar_autograd_composes_accumulates_and_obeys_grad_mode(self):
        composed = torch.tensor(0.5, requires_grad=True)
        composed.sigmoid().sin().backward()
        sigmoid = np.float32(1.0) / (
            np.float32(1.0) + np.exp(np.float32(-0.5), dtype=np.float32)
        )
        expected = np.cos(sigmoid) * (np.float32(1.0) - sigmoid) * sigmoid
        np.testing.assert_allclose(
            np.asarray(composed.grad), expected, rtol=2.0e-6, atol=0.0
        )

        accumulated = torch.tensor(-0.5, requires_grad=True)
        accumulated.sigmoid().backward()
        first = np.asarray(accumulated.grad).copy()
        accumulated.sigmoid().backward()
        np.testing.assert_array_equal(np.asarray(accumulated.grad), first * 2.0)

        higher_order = torch.tensor(0.25, requires_grad=True)
        loss = higher_order.sigmoid()
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
            no_grad_output = tracked.sigmoid()
        self.assert_result(no_grad_output, tracked, (), case="scalar no_grad")
        self.assertTrue(tracked.sigmoid().requires_grad)

        detached = tracked.detach()
        detached_output = detached.sigmoid()
        self.assert_result(detached_output, detached, (), case="scalar detached")
        np.testing.assert_array_equal(
            self.tensor_bits(no_grad_output), self.tensor_bits(detached_output)
        )

    def test_finite_owned_scalar_nonleaves_compose_accumulate_and_free_graphs(self):
        sigmoid_forms = (
            ("method", lambda input: input.sigmoid()),
            ("functional", torch.nn.functional.sigmoid),
            ("top-level", torch.sigmoid),
        )
        for parent_case, make_parent in scalar_nonleaf_parent_cases(torch):
            for form, apply_sigmoid in sigmoid_forms:
                with self.subTest(parent=parent_case, form=form):
                    leaf = torch.tensor(0.5, requires_grad=True)
                    parent = make_parent(leaf)
                    self.assertTrue(parent.requires_grad)
                    self.assertFalse(parent.is_leaf)
                    self.assertEqual(parent.shape, ())
                    self.assertEqual(parent.storage_offset(), 0)
                    self.assertFalse(parent.is_set_to(leaf))

                    expected_bits = self.tensor_bits(
                        parent.detach().sigmoid()
                    ).item()
                    output = apply_sigmoid(parent)
                    self.assertTrue(output.requires_grad)
                    self.assertFalse(output.is_leaf)
                    self.assertEqual(output.shape, ())
                    self.assertEqual(output.stride(), ())
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertFalse(output.is_set_to(parent))
                    self.assertEqual(self.tensor_bits(output).item(), expected_bits)
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            output
                        ),
                        ", grad_fn=<SigmoidBackward0>",
                    )

                    output.backward()
                    gradient_before_repeated_backward = np.asarray(
                        leaf.grad, dtype=np.float32
                    ).copy()
                    with self.assertRaisesRegex(
                        RuntimeError, "backward through the graph a second time"
                    ):
                        output.backward()
                    np.testing.assert_array_equal(
                        np.asarray(leaf.grad, dtype=np.float32),
                        gradient_before_repeated_backward,
                    )

                    accumulated = torch.tensor(0.5, requires_grad=True)
                    apply_sigmoid(make_parent(accumulated)).backward()
                    first = np.asarray(accumulated.grad, dtype=np.float32).copy()
                    apply_sigmoid(make_parent(accumulated)).backward()
                    np.testing.assert_array_equal(
                        np.asarray(accumulated.grad, dtype=np.float32),
                        first + first,
                    )

        higher_order_leaf = torch.tensor(0.25, requires_grad=True)
        higher_order_loss = torch.nn.functional.sigmoid(higher_order_leaf.sin())
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order_leaf.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order_leaf.grad)

    def test_finite_owned_rank_one_nonleaves_compose_accumulate_and_free_graphs(
        self,
    ):
        sigmoid_forms = (
            ("method", lambda input: input.sigmoid()),
            ("functional", torch.nn.functional.sigmoid),
            ("top-level", torch.sigmoid),
        )
        values = [0.25, 0.5]
        weights = torch.tensor([1.25, -0.75])
        for parent_case, make_parent in rank_preserving_nonleaf_parent_cases(torch):
            for form, apply_sigmoid in sigmoid_forms:
                with self.subTest(parent=parent_case, form=form):
                    leaf = torch.tensor(values, requires_grad=True)
                    parent = make_parent(leaf)
                    self.assertTrue(parent.requires_grad)
                    self.assertFalse(parent.is_leaf)
                    self.assertEqual(parent.shape, (2,))
                    self.assertEqual(parent.stride(), (1,))
                    self.assertEqual(parent.storage_offset(), 0)
                    self.assertFalse(parent.is_set_to(leaf))
                    self.assertNotEqual(parent.data_ptr(), leaf.data_ptr())

                    expected_bits = self.tensor_bits(parent.detach().sigmoid())
                    output = apply_sigmoid(parent)
                    self.assertTrue(output.requires_grad)
                    self.assertFalse(output.is_leaf)
                    self.assertEqual(output.shape, (2,))
                    self.assertEqual(output.stride(), (1,))
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertFalse(output.is_set_to(parent))
                    self.assertNotEqual(output.data_ptr(), parent.data_ptr())
                    np.testing.assert_array_equal(
                        self.tensor_bits(output), expected_bits
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            output
                        ),
                        ", grad_fn=<SigmoidBackward0>",
                    )

                    loss = (output * weights).sum()
                    loss.backward()
                    gradient_before_repeated_backward = np.asarray(
                        leaf.grad, dtype=np.float32
                    ).copy()
                    with self.assertRaisesRegex(
                        RuntimeError, "backward through the graph a second time"
                    ):
                        loss.backward()
                    np.testing.assert_array_equal(
                        np.asarray(leaf.grad, dtype=np.float32),
                        gradient_before_repeated_backward,
                    )

                    accumulated = torch.tensor(values, requires_grad=True)
                    (
                        apply_sigmoid(make_parent(accumulated)) * weights
                    ).sum().backward()
                    first = np.asarray(accumulated.grad, dtype=np.float32).copy()
                    (
                        apply_sigmoid(make_parent(accumulated)) * weights
                    ).sum().backward()
                    np.testing.assert_array_equal(
                        np.asarray(accumulated.grad, dtype=np.float32),
                        first + first,
                    )

                    empty = torch.tensor([], requires_grad=True)
                    empty_parent = make_parent(empty)
                    self.assertTrue(empty_parent.requires_grad)
                    self.assertFalse(empty_parent.is_leaf)
                    self.assertEqual(empty_parent.shape, (0,))
                    self.assertEqual(empty_parent.stride(), (1,))
                    self.assertEqual(empty_parent.storage_offset(), 0)
                    self.assertFalse(empty_parent.is_set_to(empty))
                    empty_output = apply_sigmoid(empty_parent)
                    self.assertTrue(empty_output.requires_grad)
                    self.assertFalse(empty_output.is_leaf)
                    self.assertEqual(empty_output.shape, (0,))
                    self.assertEqual(empty_output.stride(), (1,))
                    self.assertEqual(empty_output.storage_offset(), 0)
                    self.assertFalse(empty_output.is_set_to(empty_parent))
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            empty_output
                        ),
                        ", grad_fn=<SigmoidBackward0>",
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

        higher_order_leaf = torch.tensor([0.25, -0.25], requires_grad=True)
        higher_order_loss = torch.nn.functional.sigmoid(higher_order_leaf.sin()).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order_leaf.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order_leaf.grad)

    def test_finite_owned_rank_two_nonleaves_compose_accumulate_and_free_graphs(
        self,
    ):
        sigmoid_forms = (
            ("method", lambda input: input.sigmoid()),
            ("functional", torch.nn.functional.sigmoid),
            ("top-level", torch.sigmoid),
        )
        values = [[0.25, 0.5], [0.5, 0.25]]
        weights = torch.tensor([[1.25, -0.75], [0.5, -1.5]])
        for parent_case, make_parent in rank_preserving_nonleaf_parent_cases(torch):
            for form, apply_sigmoid in sigmoid_forms:
                with self.subTest(parent=parent_case, form=form):
                    leaf = torch.tensor(values, requires_grad=True)
                    parent = make_parent(leaf)
                    self.assertTrue(parent.requires_grad)
                    self.assertFalse(parent.is_leaf)
                    self.assertEqual(parent.shape, (2, 2))
                    self.assertEqual(parent.stride(), (2, 1))
                    self.assertEqual(parent.storage_offset(), 0)
                    self.assertFalse(parent.is_set_to(leaf))
                    self.assertNotEqual(parent.data_ptr(), leaf.data_ptr())

                    expected_bits = self.tensor_bits(parent.detach().sigmoid())
                    output = apply_sigmoid(parent)
                    self.assertTrue(output.requires_grad)
                    self.assertFalse(output.is_leaf)
                    self.assertEqual(output.shape, (2, 2))
                    self.assertEqual(output.stride(), (2, 1))
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertFalse(output.is_set_to(parent))
                    self.assertNotEqual(output.data_ptr(), parent.data_ptr())
                    np.testing.assert_array_equal(
                        self.tensor_bits(output), expected_bits
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            output
                        ),
                        ", grad_fn=<SigmoidBackward0>",
                    )

                    loss = (output * weights).sum()
                    loss.backward()
                    gradient_before_repeated_backward = np.asarray(
                        leaf.grad, dtype=np.float32
                    ).copy()
                    with self.assertRaisesRegex(
                        RuntimeError, "backward through the graph a second time"
                    ):
                        loss.backward()
                    np.testing.assert_array_equal(
                        np.asarray(leaf.grad, dtype=np.float32),
                        gradient_before_repeated_backward,
                    )

                    accumulated = torch.tensor(values, requires_grad=True)
                    (
                        apply_sigmoid(make_parent(accumulated)) * weights
                    ).sum().backward()
                    first = np.asarray(accumulated.grad, dtype=np.float32).copy()
                    (
                        apply_sigmoid(make_parent(accumulated)) * weights
                    ).sum().backward()
                    np.testing.assert_array_equal(
                        np.asarray(accumulated.grad, dtype=np.float32),
                        first + first,
                    )

                    for empty_shape in ((0, 0), (0, 3), (2, 0)):
                        empty = torch.zeros(empty_shape, requires_grad=True)
                        empty_parent = make_parent(empty)
                        self.assertTrue(empty_parent.requires_grad)
                        self.assertFalse(empty_parent.is_leaf)
                        self.assertEqual(empty_parent.shape, empty_shape)
                        self.assertEqual(empty_parent.stride(), empty.stride())
                        self.assertEqual(empty_parent.storage_offset(), 0)
                        self.assertFalse(empty_parent.is_set_to(empty))
                        empty_output = apply_sigmoid(empty_parent)
                        self.assertTrue(empty_output.requires_grad)
                        self.assertFalse(empty_output.is_leaf)
                        self.assertEqual(empty_output.shape, empty_shape)
                        self.assertEqual(empty_output.stride(), empty.stride())
                        self.assertEqual(empty_output.storage_offset(), 0)
                        self.assertFalse(empty_output.is_set_to(empty_parent))
                        self.assertEqual(
                            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                                empty_output
                            ),
                            ", grad_fn=<SigmoidBackward0>",
                        )
                        empty_loss = empty_output.sum()
                        empty_loss.backward()
                        self.assertEqual(empty.grad.shape, empty_shape)
                        self.assertEqual(empty.grad.stride(), empty.stride())
                        self.assertEqual(empty.grad.tolist(), empty.tolist())
                        with self.assertRaisesRegex(
                            RuntimeError, "backward through the graph a second time"
                        ):
                            empty_loss.backward()

        higher_order_leaf = torch.tensor(
            [[0.25, -0.25], [0.5, -0.5]], requires_grad=True
        )
        higher_order_loss = torch.nn.functional.sigmoid(
            higher_order_leaf.sin()
        ).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order_leaf.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order_leaf.grad)

    def test_finite_owned_rank_three_nonleaves_compose_accumulate_and_free_graphs(
        self,
    ):
        sigmoid_forms = (
            ("method", lambda input: input.sigmoid()),
            ("functional", torch.nn.functional.sigmoid),
            ("top-level", torch.sigmoid),
        )
        values = [[[0.25, 0.5]], [[0.5, 0.25]]]
        weights = torch.tensor([[[1.25, -0.75]], [[0.5, -1.5]]])
        for parent_case, make_parent in rank_preserving_nonleaf_parent_cases(torch):
            for form, apply_sigmoid in sigmoid_forms:
                with self.subTest(parent=parent_case, form=form):
                    leaf = torch.tensor(values, requires_grad=True)
                    parent = make_parent(leaf)
                    self.assertTrue(parent.requires_grad)
                    self.assertFalse(parent.is_leaf)
                    self.assertEqual(parent.shape, (2, 1, 2))
                    self.assertEqual(parent.stride(), (2, 2, 1))
                    self.assertEqual(parent.storage_offset(), 0)
                    self.assertFalse(parent.is_set_to(leaf))
                    self.assertNotEqual(parent.data_ptr(), leaf.data_ptr())

                    expected_bits = self.tensor_bits(parent.detach().sigmoid())
                    output = apply_sigmoid(parent)
                    self.assertTrue(output.requires_grad)
                    self.assertFalse(output.is_leaf)
                    self.assertEqual(output.shape, (2, 1, 2))
                    self.assertEqual(output.stride(), (2, 2, 1))
                    self.assertEqual(output.storage_offset(), 0)
                    self.assertFalse(output.is_set_to(parent))
                    self.assertNotEqual(output.data_ptr(), parent.data_ptr())
                    np.testing.assert_array_equal(
                        self.tensor_bits(output), expected_bits
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            output
                        ),
                        ", grad_fn=<SigmoidBackward0>",
                    )

                    loss = (output * weights).sum()
                    loss.backward()
                    gradient_before_repeated_backward = np.asarray(
                        leaf.grad, dtype=np.float32
                    ).copy()
                    with self.assertRaisesRegex(
                        RuntimeError, "backward through the graph a second time"
                    ):
                        loss.backward()
                    np.testing.assert_array_equal(
                        np.asarray(leaf.grad, dtype=np.float32),
                        gradient_before_repeated_backward,
                    )

                    accumulated = torch.tensor(values, requires_grad=True)
                    (
                        apply_sigmoid(make_parent(accumulated)) * weights
                    ).sum().backward()
                    first = np.asarray(accumulated.grad, dtype=np.float32).copy()
                    (
                        apply_sigmoid(make_parent(accumulated)) * weights
                    ).sum().backward()
                    np.testing.assert_array_equal(
                        np.asarray(accumulated.grad, dtype=np.float32),
                        first + first,
                    )

                    for empty_shape in (
                        (0, 1, 3),
                        (1, 0, 3),
                        (2, 3, 0),
                        (0, 0, 0),
                    ):
                        empty = torch.zeros(empty_shape, requires_grad=True)
                        empty_parent = make_parent(empty)
                        self.assertTrue(empty_parent.requires_grad)
                        self.assertFalse(empty_parent.is_leaf)
                        self.assertEqual(empty_parent.shape, empty_shape)
                        self.assertEqual(empty_parent.storage_offset(), 0)
                        self.assertFalse(empty_parent.is_set_to(empty))
                        empty_output = apply_sigmoid(empty_parent)
                        self.assertTrue(empty_output.requires_grad)
                        self.assertFalse(empty_output.is_leaf)
                        self.assertEqual(empty_output.shape, empty_shape)
                        self.assertEqual(empty_output.stride(), empty.stride())
                        self.assertEqual(empty_output.storage_offset(), 0)
                        self.assertFalse(empty_output.is_set_to(empty_parent))
                        self.assertEqual(
                            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                                empty_output
                            ),
                            ", grad_fn=<SigmoidBackward0>",
                        )
                        empty_loss = empty_output.sum()
                        empty_loss.backward()
                        self.assertEqual(empty.grad.shape, empty_shape)
                        self.assertEqual(empty.grad.stride(), empty.stride())
                        self.assertEqual(empty.grad.tolist(), empty.tolist())
                        with self.assertRaisesRegex(
                            RuntimeError, "backward through the graph a second time"
                        ):
                            empty_loss.backward()

        higher_order_leaf = torch.tensor(
            [[[0.25, -0.25]], [[0.5, -0.5]]], requires_grad=True
        )
        higher_order_loss = torch.nn.functional.sigmoid(
            higher_order_leaf.sin()
        ).sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            higher_order_loss.backward(create_graph=True)
        self.assertIsNone(higher_order_leaf.grad)
        higher_order_loss.backward()
        self.assertIsNotNone(higher_order_leaf.grad)

    def test_unsupported_tracked_inputs_fail_before_existing_graphs_or_layouts_change(self):
        message = r"^sigmoid\(\): autograd recording is not supported$"

        for bits in (0x7F80_0000, 0xFF80_0000, 0x7FC1_2345, 0xFFC5_4321):
            with self.subTest(nonfinite=f"0x{bits:08x}"):
                value = np.asarray(bits, dtype=np.uint32).view(np.float32).item()
                leaf = torch.tensor(value, requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    leaf.sigmoid()
                self.assertIsNone(leaf.grad)
                leaf.sum().backward()
                self.assertEqual(leaf.grad.item(), 1.0)

                nonleaf_base = torch.tensor(0.5, requires_grad=True)
                nonleaf = nonleaf_base + value
                self.assertTrue(nonleaf.requires_grad)
                self.assertFalse(nonleaf.is_leaf)
                for call in (
                    nonleaf.sigmoid,
                    lambda: torch.nn.functional.sigmoid(nonleaf),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(nonleaf_base.grad)
                nonleaf.backward()
                self.assertEqual(nonleaf_base.grad.item(), 1.0)

                vector_nonleaf_base = torch.tensor(
                    [0.5, 0.25], requires_grad=True
                )
                vector_nonleaf = vector_nonleaf_base + value
                self.assertTrue(vector_nonleaf.requires_grad)
                self.assertFalse(vector_nonleaf.is_leaf)
                for call in (
                    vector_nonleaf.sigmoid,
                    lambda: torch.nn.functional.sigmoid(vector_nonleaf),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(vector_nonleaf_base.grad)
                vector_nonleaf.sum().backward()
                self.assertEqual(vector_nonleaf_base.grad.tolist(), [1.0, 1.0])

                matrix_nonleaf_base = torch.tensor(
                    [[0.5, 0.25]], requires_grad=True
                )
                matrix_nonleaf = matrix_nonleaf_base + value
                self.assertTrue(matrix_nonleaf.requires_grad)
                self.assertFalse(matrix_nonleaf.is_leaf)
                for call in (
                    matrix_nonleaf.sigmoid,
                    lambda: torch.nn.functional.sigmoid(matrix_nonleaf),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(matrix_nonleaf_base.grad)
                matrix_nonleaf.sum().backward()
                self.assertEqual(matrix_nonleaf_base.grad.tolist(), [[1.0, 1.0]])

                rank_three_nonleaf_base = torch.tensor(
                    [[[0.5, 0.25]]], requires_grad=True
                )
                rank_three_nonleaf = rank_three_nonleaf_base + value
                self.assertTrue(rank_three_nonleaf.requires_grad)
                self.assertFalse(rank_three_nonleaf.is_leaf)
                for call in (
                    rank_three_nonleaf.sigmoid,
                    lambda: torch.nn.functional.sigmoid(rank_three_nonleaf),
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
                self.assertIsNone(rank_three_nonleaf_base.grad)
                rank_three_nonleaf.sum().backward()
                self.assertEqual(
                    rank_three_nonleaf_base.grad.tolist(), [[[1.0, 1.0]]]
                )

                vector = torch.tensor([0.5, value], requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    vector.sigmoid()
                self.assertIsNone(vector.grad)
                vector.sum().backward()
                self.assertEqual(vector.grad.tolist(), [1.0, 1.0])

                matrix = torch.tensor([[0.5, value]], requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    matrix.sigmoid()
                self.assertIsNone(matrix.grad)
                matrix.sum().backward()
                self.assertEqual(matrix.grad.tolist(), [[1.0, 1.0]])

                rank_three = torch.tensor([[[0.5, value]]], requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    rank_three.sigmoid()
                self.assertIsNone(rank_three.grad)
                rank_three.sum().backward()
                self.assertEqual(rank_three.grad.tolist(), [[[1.0, 1.0]]])

                rank_four = torch.tensor([[[[0.5, value]]]], requires_grad=True)
                with self.assertRaisesRegex(RuntimeError, message):
                    rank_four.sigmoid()
                self.assertIsNone(rank_four.grad)
                rank_four.sum().backward()
                self.assertEqual(rank_four.grad.tolist(), [[[[1.0, 1.0]]]])

                rank_five = torch.tensor(
                    [[[[[0.5, value]]]]], requires_grad=True
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    rank_five.sigmoid()
                self.assertIsNone(rank_five.grad)
                rank_five.sum().backward()
                self.assertEqual(rank_five.grad.tolist(), [[[[[1.0, 1.0]]]]])

                rank_six = torch.tensor(
                    [[[[[[0.5, value]]]]]], requires_grad=True
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    rank_six.sigmoid()
                self.assertIsNone(rank_six.grad)
                rank_six.sum().backward()
                self.assertEqual(rank_six.grad.tolist(), [[[[[[1.0, 1.0]]]]]])

        high_rank_nonfinite = torch.full(
            (1,) * 65, float("inf"), requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, message):
            high_rank_nonfinite.sigmoid()
        self.assertIsNone(high_rank_nonfinite.grad)
        high_rank_nonfinite.sum().backward()
        self.assertEqual(high_rank_nonfinite.grad.item(), 1.0)

        view_base = torch.tensor([0.5], requires_grad=True)
        scalar_view = view_base[0]
        self.assertFalse(scalar_view.is_leaf)
        for call in (
            scalar_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(scalar_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        scalar_view.backward()
        self.assertEqual(view_base.grad.tolist(), [1.0])

        full_vector_view_base = torch.tensor([0.5, -0.5], requires_grad=True)
        full_vector_view = full_vector_view_base.view((2,))
        self.assertFalse(full_vector_view.is_leaf)
        self.assertTrue(full_vector_view.is_set_to(full_vector_view_base))
        for call in (
            full_vector_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(full_vector_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        full_vector_view.sum().backward()
        self.assertEqual(full_vector_view_base.grad.tolist(), [1.0, 1.0])

        full_matrix_view_base = torch.tensor(
            [[0.5, -0.5], [1.0, -1.0]], requires_grad=True
        )
        full_matrix_view = full_matrix_view_base.view((2, 2))
        self.assertFalse(full_matrix_view.is_leaf)
        self.assertTrue(full_matrix_view.is_set_to(full_matrix_view_base))
        for call in (
            full_matrix_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(full_matrix_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        full_matrix_view.sum().backward()
        self.assertEqual(
            full_matrix_view_base.grad.tolist(), [[1.0, 1.0], [1.0, 1.0]]
        )

        full_rank_three_view_base = torch.tensor(
            [[[0.5, -0.5]], [[1.0, -1.0]]], requires_grad=True
        )
        full_rank_three_view = full_rank_three_view_base.view((2, 1, 2))
        self.assertFalse(full_rank_three_view.is_leaf)
        self.assertTrue(full_rank_three_view.is_set_to(full_rank_three_view_base))
        for call in (
            full_rank_three_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(full_rank_three_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        full_rank_three_view.sum().backward()
        self.assertEqual(
            full_rank_three_view_base.grad.tolist(),
            [[[1.0, 1.0]], [[1.0, 1.0]]],
        )

        vector_view_base = torch.tensor(
            [[0.5, -1.0], [2.0, -3.0]], requires_grad=True
        )
        vector_view = vector_view_base[0]
        self.assertEqual(vector_view.shape, (2,))
        self.assertFalse(vector_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            vector_view.sigmoid()
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
            matrix_view.sigmoid()
        matrix_view.sum().backward()
        self.assertEqual(
            matrix_view_base.grad.tolist(),
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        )

        rank_three_view_base = torch.tensor(
            np.linspace(-2.0, 2.0, 16, dtype=np.float32)
            .reshape(2, 2, 2, 2)
            .tolist(),
            requires_grad=True,
        )
        rank_three_view = rank_three_view_base[0]
        self.assertEqual(rank_three_view.shape, (2, 2, 2))
        self.assertFalse(rank_three_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_three_view.sigmoid()
        rank_three_view.sum().backward()
        self.assertEqual(
            rank_three_view_base.grad.tolist(),
            [[[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]],
             [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]],
        )

        rank_four_view_base = torch.tensor(
            [[[[[0.5, -1.0]]]], [[[[2.0, -3.0]]]]], requires_grad=True
        )
        rank_four_view = rank_four_view_base[0]
        self.assertEqual(rank_four_view.shape, (1, 1, 1, 2))
        self.assertFalse(rank_four_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_four_view.sigmoid()
        rank_four_view.sum().backward()
        self.assertEqual(
            rank_four_view_base.grad.tolist(),
            [[[[[1.0, 1.0]]]], [[[[0.0, 0.0]]]]],
        )

        rank_five_view_base = torch.tensor(
            [[[[[[0.5, -1.0]]]]], [[[[[2.0, -3.0]]]]]], requires_grad=True
        )
        rank_five_view = rank_five_view_base[0]
        self.assertEqual(rank_five_view.shape, (1, 1, 1, 1, 2))
        self.assertFalse(rank_five_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_five_view.sigmoid()
        rank_five_view.sum().backward()
        self.assertEqual(
            rank_five_view_base.grad.tolist(),
            [[[[[[1.0, 1.0]]]]], [[[[[0.0, 0.0]]]]]],
        )

        rank_six_view_base = torch.full(
            (2,) + (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        rank_six_view = rank_six_view_base[0]
        self.assertEqual(rank_six_view.shape, (1, 1, 1, 1, 1, 2))
        self.assertFalse(rank_six_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_six_view.sigmoid()
        rank_six_view.sum().backward()
        self.assertEqual(rank_six_view_base.grad.sum().item(), 2.0)

        high_rank_view_base = torch.full(
            (2,) + (1,) * 65, 0.5, requires_grad=True
        )
        high_rank_view = high_rank_view_base[0]
        self.assertEqual(high_rank_view.shape, (1,) * 65)
        self.assertFalse(high_rank_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            high_rank_view.sigmoid()
        high_rank_view.backward()
        self.assertEqual(high_rank_view_base.grad.sum().item(), 1.0)

        rank_four_nonleaf_base = torch.tensor(
            [[[[0.5, -0.5]]]], requires_grad=True
        )
        rank_four_nonleaf = rank_four_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            rank_four_nonleaf.sigmoid()
        rank_four_nonleaf.sum().backward()
        np.testing.assert_allclose(
            np.asarray(rank_four_nonleaf_base.grad),
            np.cos(np.asarray([[[[0.5, -0.5]]]], dtype=np.float32)),
        )

        rank_five_nonleaf_base = torch.tensor(
            [[[[[0.5, -0.5]]]]], requires_grad=True
        )
        rank_five_nonleaf = rank_five_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            rank_five_nonleaf.sigmoid()
        rank_five_nonleaf.sum().backward()
        np.testing.assert_allclose(
            np.asarray(rank_five_nonleaf_base.grad),
            np.cos(np.asarray([[[[[0.5, -0.5]]]]], dtype=np.float32)),
        )

        rank_six_nonleaf_base = torch.full(
            (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        rank_six_nonleaf = rank_six_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            rank_six_nonleaf.sigmoid()
        rank_six_nonleaf.sum().backward()
        self.assertIsNotNone(rank_six_nonleaf_base.grad)

        high_rank_nonleaf_base = torch.full(
            (1,) * 65, 0.5, requires_grad=True
        )
        high_rank_nonleaf = high_rank_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            high_rank_nonleaf.sigmoid()
        high_rank_nonleaf.backward()
        self.assertIsNotNone(high_rank_nonleaf_base.grad)

        with torch.no_grad():
            no_grad_view = vector_view_base[0]
        self.assertTrue(no_grad_view.requires_grad)
        self.assertTrue(no_grad_view.is_leaf)
        self.assertEqual(no_grad_view.shape, (2,))
        with self.assertRaisesRegex(RuntimeError, message):
            no_grad_view.sigmoid()

        empty_view_base = torch.zeros((1, 0), requires_grad=True)
        with torch.no_grad():
            empty_view = empty_view_base[0]
        self.assertTrue(empty_view.requires_grad)
        self.assertTrue(empty_view.is_leaf)
        self.assertEqual(empty_view.shape, (0,))
        with self.assertRaisesRegex(RuntimeError, message):
            empty_view.sigmoid()

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 1, 1, 1, 3)
        )
        with self.assertRaisesRegex(RuntimeError, message):
            extreme.sigmoid()
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "Stride calculation overflowed"):
                extreme.sigmoid()

    def test_detached_and_no_grad_inputs_use_the_inference_path(self):
        for case, source in enumerate(self.make_tracked_cases()):
            detached = source.detach()
            expected = detached.sigmoid()
            with torch.no_grad():
                actual = source.sigmoid()
            with self.subTest(case=case, mode="no_grad"):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertFalse(actual.is_set_to(source))
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                )
            with self.subTest(case=case, mode="detached"):
                self.assertFalse(expected.is_set_to(detached))
                if detached.numel():
                    self.assertNotEqual(expected.data_ptr(), detached.data_ptr())

    def test_tensorbase_descriptor_metadata_and_no_argument_errors(self):
        tensor = torch.tensor([1.25])
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        bound = tensor.sigmoid

        self.assertIs(torch.Tensor.sigmoid, descriptor)
        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        self.assertEqual(
            repr(descriptor), "<method 'sigmoid' of 'torch._C.TensorBase' objects>"
        )
        self.assertEqual(descriptor.__name__, "sigmoid")
        self.assertEqual(descriptor.__qualname__, "TensorBase.sigmoid")
        self.assertEqual(bound.__name__, "sigmoid")
        self.assertEqual(bound.__qualname__, "Tensor.sigmoid")
        self.assertEqual(descriptor.__doc__, SIGMOID_DOC)
        self.assertEqual(bound.__doc__, SIGMOID_DOC)
        self.assertEqual(descriptor.__objclass__.__name__, "TensorBase")
        self.assertEqual(descriptor.__objclass__.__module__, "torch._C")
        self.assertFalse(hasattr(descriptor, "__module__"))
        self.assertIsNone(bound.__module__)
        assert_no_argument_signature(self, descriptor, "(self, /)")
        assert_no_argument_signature(self, bound, "()")

        cases = (
            (lambda: tensor.sigmoid(1), "TensorBase.sigmoid() takes no arguments (1 given)"),
            (lambda: bound(1), "Tensor.sigmoid() takes no arguments (1 given)"),
            (
                lambda: descriptor(tensor, 1),
                "TensorBase.sigmoid() takes no arguments (1 given)",
            ),
            (
                lambda: tensor.sigmoid(1, 2),
                "TensorBase.sigmoid() takes no arguments (2 given)",
            ),
            (
                lambda: tensor.sigmoid(input=tensor),
                (
                    "Tensor.sigmoid() takes no keyword arguments"
                    if sys.version_info < (3, 11)
                    else "TensorBase.sigmoid() takes no keyword arguments"
                ),
            ),
            (lambda: bound(unexpected=True), "Tensor.sigmoid() takes no keyword arguments"),
            (
                lambda: descriptor(tensor, unexpected=True),
                "TensorBase.sigmoid() takes no keyword arguments",
            ),
            (lambda: descriptor(), "unbound method TensorBase.sigmoid() needs an argument"),
            (
                lambda: descriptor(1),
                "descriptor 'sigmoid' for 'torch._C.TensorBase' objects "
                "doesn't apply to a 'int' object",
            ),
            (
                lambda: descriptor(self=tensor),
                "unbound method TensorBase.sigmoid() needs an argument",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

    def test_torch_function_modes_dispatch_before_native_execution(self):
        tracked = torch.tensor([1.25], requires_grad=True)
        plain = tracked.detach()
        descriptor = inspect.getattr_static(torch.Tensor, "sigmoid")
        marker = object()

        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result):
                self.result = result
                self.calls = []

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode = RecordingMode(marker)
        with mode:
            result = tracked.sigmoid()
        self.assertIs(result, marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, descriptor)
        self.assertEqual(dispatch_types, (torch.Tensor,))
        self.assertEqual(len(args), 1)
        self.assertIs(args[0], tracked)
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
                forwarded = plain.sigmoid()
        self.assertEqual(order, ["upper", "lower"])
        np.testing.assert_allclose(forwarded.tolist(), [0.7772999], rtol=1.0e-6)

        order.clear()
        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                tracked_output = tracked.sigmoid()
        self.assertEqual(order, ["upper", "lower"])
        self.assertTrue(tracked_output.requires_grad)
        self.assertFalse(tracked_output.is_leaf)
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(tracked_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        old_recursion_limit = sys.getrecursionlimit()
        declining = RecordingMode(NotImplemented)
        try:
            sys.setrecursionlimit(80)
            with declining:
                with self.assertRaises(RecursionError):
                    plain.sigmoid()
                self.assertEqual(
                    len(torch.overrides._get_current_function_mode_stack()), 1
                )
        finally:
            sys.setrecursionlimit(old_recursion_limit)
        self.assertGreater(len(declining.calls), 1)
        self.assertEqual(len(torch.overrides._get_current_function_mode_stack()), 0)

        invalid = RecordingMode(marker)
        with self.assertRaises(TypeError):
            with invalid:
                plain.sigmoid(1)
        self.assertEqual(invalid.calls, [])

    def test_top_level_supports_current_sigmoid_autograd_boundary(self):
        scalar = torch.tensor(0.5, requires_grad=True)
        calls = self.top_level_calls(scalar)
        for form, call in calls:
            with self.subTest(form=form, mode="scalar leaf"):
                output = call()
                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertEqual(output.shape, ())
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
                    ", grad_fn=<SigmoidBackward0>",
                )
                output.backward()
        unit_gradient = np.asarray(0x3E70_A4D0, dtype=np.uint32).view(np.float32)
        expected = np.float32(len(calls)) * unit_gradient
        np.testing.assert_allclose(scalar.grad.item(), expected, rtol=2.0e-6)

        for shape in (
            (8,),
            (2, 4),
            (2, 1, 4),
            (1, 2, 1, 4),
            (1, 2, 1, 1, 4),
            (1, 2, 1, 1, 1, 4),
        ):
            with self.subTest(shape=shape, mode="finite owned leaf"):
                values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(shape).tolist()
                leaf = torch.tensor(values, requires_grad=True)
                output = torch.sigmoid(leaf, out=None)
                self.assertTrue(output.requires_grad)
                self.assertFalse(output.is_leaf)
                self.assertEqual(output.shape, shape)
                self.assertEqual(output.stride(), leaf.stride())
                self.assertEqual(output.storage_offset(), 0)
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(output),
                    ", grad_fn=<SigmoidBackward0>",
                )
                output.sum().backward()
                self.assertIsNotNone(leaf.grad)

        high_rank_shape = (1,) * 65
        high_rank = torch.full(high_rank_shape, 0.5, requires_grad=True)
        high_rank_output = torch.sigmoid(input=high_rank, out=None)
        self.assertTrue(high_rank_output.requires_grad)
        self.assertFalse(high_rank_output.is_leaf)
        self.assertEqual(high_rank_output.shape, high_rank_shape)
        self.assertEqual(high_rank_output.stride(), (1,) * 65)
        high_rank_output.backward()
        self.assertIsNotNone(high_rank.grad)

        empty = torch.zeros((1, 2, 0, 1), requires_grad=True)
        empty_output = torch.sigmoid(input=empty, out=None)
        self.assertTrue(empty_output.requires_grad)
        self.assertFalse(empty_output.is_leaf)
        self.assertEqual(empty_output.shape, (1, 2, 0, 1))
        self.assertEqual(empty_output.stride(), empty.stride())
        empty_output.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.numel(), 0)

        message = r"^sigmoid\(\): autograd recording is not supported$"
        nonfinite = torch.tensor(float("inf"), requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, message):
            torch.sigmoid(nonfinite)
        self.assertIsNone(nonfinite.grad)

        view_base = torch.tensor([0.5], requires_grad=True)
        view = view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            torch.sigmoid(input=view)
        view.backward()
        self.assertEqual(view_base.grad.tolist(), [1.0])

        rank_four_base = torch.tensor([[[[0.5, -0.5]]]], requires_grad=True)
        rank_four_nonleaf = rank_four_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            torch.sigmoid(rank_four_nonleaf, out=None)
        self.assertIsNone(rank_four_base.grad)
        rank_four_nonleaf.sum().backward()
        self.assertIsNotNone(rank_four_base.grad)

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
            self.assertIs(torch.sigmoid(input=tensor, out=destination), marker)
        self.assertEqual(len(mode.calls), 1)
        function, dispatch_types, args, kwargs = mode.calls[0]
        self.assertIs(function, torch.sigmoid)
        self.assertEqual(dispatch_types, ())
        self.assertEqual(args, ())
        self.assertEqual(kwargs, {"input": tensor, "out": destination})

        override_calls = []

        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                override_calls.append((func, types, args, kwargs))
                return marker

        self.assertIs(torch.sigmoid(Override()), marker)
        self.assertIs(torch.sigmoid(torch.tensor([0.5]), out=Override()), marker)
        self.assertEqual(len(override_calls), 2)
        for dispatched, types_, _, _ in override_calls:
            self.assertIs(dispatched, torch.sigmoid)
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

        self.assertIs(torch.sigmoid(BaseOverride(), out=DerivedOverride()), marker)
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
                forwarded = torch.sigmoid(input=plain, out=None)
        self.assertEqual(forwarding_order, ["upper", "lower"])
        np.testing.assert_array_equal(
            self.tensor_bits(forwarded), self.tensor_bits(plain.sigmoid())
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
            self.assertIs(torch.sigmoid(FallbackOverride()), marker)
        self.assertEqual(events, ["mode", "override"])

        invalid_mode = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^sigmoid\(\): argument 'out' must be Tensor, not list$",
        ):
            with invalid_mode:
                torch.sigmoid(tensor, out=[])
        self.assertEqual(invalid_mode.calls, [])

    def test_top_level_concrete_out_is_rejected_without_mutation(self):
        source = torch.tensor([0.5, -0.5], requires_grad=True)
        destination = torch.tensor([17.0, 19.0])
        for form, call in (
            ("positional", lambda: torch.sigmoid(source, out=destination)),
            ("keyword", lambda: torch.sigmoid(input=source, out=destination)),
            ("alias", lambda: torch.sigmoid(x=source, out=destination)),
        ):
            with self.subTest(form=form):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^sigmoid\(\): the 'out' argument is not supported$",
                ):
                    call()
                self.assertEqual(destination.tolist(), [17.0, 19.0])
                self.assertIsNone(source.grad)

        extreme = torch.zeros((0,), requires_grad=True).reshape(
            (0, sys.maxsize, 1, 1, 1, 3)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): the 'out' argument is not supported$",
        ):
            torch.sigmoid(extreme, out=destination)
        self.assertEqual(destination.tolist(), [17.0, 19.0])
        self.assertIsNone(source.grad)

        with torch.no_grad():
            actual = torch.sigmoid(source, out=None)
            expected = source.sigmoid()
        np.testing.assert_array_equal(
            self.tensor_bits(actual), self.tensor_bits(expected)
        )
        source.sigmoid().sum().backward()
        self.assertIsNotNone(source.grad)

    def test_top_level_builtin_metadata_exports_copying_and_pickling(self):
        function = torch.sigmoid
        self.assertIs(function, torch._C.sigmoid)
        self.assertIs(type(function), types.BuiltinFunctionType)
        self.assertEqual(function.__name__, "sigmoid")
        self.assertEqual(function.__qualname__, "_VariableFunctionsClass.sigmoid")
        self.assertEqual(function.__module__, "torch")
        self.assertEqual(function.__doc__, TOP_LEVEL_SIGMOID_DOC)
        self.assertIsNone(function.__text_signature__)
        self.assertRegex(
            repr(function),
            r"^<built-in method sigmoid of type object at 0x[0-9a-f]+>$",
        )
        with self.assertRaises(ValueError):
            inspect.signature(function)

        owner = function.__reduce__()[1][0]
        self.assertEqual(owner.__name__, "_VariableFunctionsClass")
        self.assertEqual(owner.__qualname__, "_VariableFunctionsClass")
        self.assertEqual(owner.__module__, "torch_rs._C")
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.sigmoid, function)
        for action in (
            lambda: setattr(owner, "sigmoid", None),
            lambda: delattr(owner, "sigmoid"),
        ):
            with self.assertRaises(TypeError):
                action()
            self.assertIs(owner.sigmoid, function)

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)), function
                )

        self.assertEqual(torch.__all__.count("sigmoid"), 1)
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))
        wildcard_namespace = {}
        exec("from torch_rs import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["sigmoid"], function)

    def test_top_level_binding_and_type_error_precedence(self):
        tensor = torch.tensor([0.5])
        cases = (
            (
                lambda: torch.sigmoid(),
                'sigmoid() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sigmoid(tensor, tensor),
                "sigmoid() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: torch.sigmoid(tensor, input=tensor),
                "sigmoid() got multiple values for argument 'input'",
            ),
            (
                lambda: torch.sigmoid(out=tensor),
                'sigmoid() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sigmoid(extra=tensor),
                'sigmoid() missing 1 required positional arguments: "input"',
            ),
            (
                lambda: torch.sigmoid(1, extra=True),
                "sigmoid(): argument 'input' (position 1) must be Tensor, not int",
            ),
            (
                lambda: torch.sigmoid(input=[]),
                "sigmoid(): argument 'input' must be Tensor, not list",
            ),
            (
                lambda: torch.sigmoid(tensor, out=[]),
                "sigmoid(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sigmoid(tensor, extra=True, out=[]),
                "sigmoid(): argument 'out' must be Tensor, not list",
            ),
            (
                lambda: torch.sigmoid(tensor, extra=True),
                "sigmoid() got an unexpected keyword argument 'extra'",
            ),
            (
                lambda: torch.sigmoid(input=tensor, a=tensor),
                "sigmoid() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sigmoid(a=tensor, x=tensor, out=None),
                "sigmoid() got an unexpected keyword argument 'a'",
            ),
            (
                lambda: torch.sigmoid(x=tensor, a=tensor, out=None),
                "sigmoid() got an unexpected keyword argument 'x'",
            ),
            (
                lambda: torch.sigmoid(np.zeros((2, 3), dtype=np.float32)),
                (
                    "sigmoid(): argument 'input' (position 1) must be Tensor, "
                    "not numpy.ndarray"
                ),
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_top_level_is_exposed_and_inplace_forms_remain_unsupported_without_mutation(self):
        tensor = torch.tensor([1.25])
        self.assertTrue(hasattr(torch, "sigmoid"))
        self.assertIn("sigmoid", torch.__all__)
        self.assertTrue(hasattr(torch.nn.functional, "sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(tensor, "sigmoid_"))
        self.assertFalse(hasattr(torch, "sigmoid_"))
        self.assertNotIn("sigmoid_", torch.__all__)
        with self.assertRaises(TypeError):
            tensor.sigmoid(out=None)

        tracked = torch.tensor(0.5, requires_grad=True)
        before = self.tensor_bits(tracked).copy()
        with self.assertRaises(AttributeError):
            tracked.sigmoid_()
        np.testing.assert_array_equal(self.tensor_bits(tracked), before)
        self.assertIsNone(tracked.grad)
        tracked.sigmoid().backward()
        self.assertEqual(self.tensor_bits(tracked.grad).item(), 0x3E70_A4D0)


if __name__ == "__main__":
    unittest.main()
