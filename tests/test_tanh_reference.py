import copy
import ctypes
import inspect
import json
import pickle
import re
import subprocess
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

if __package__:
    from .test_tanh import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
    )
else:
    from test_tanh import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
    )

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorTanhReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.tanh differentials require pinned PyTorch 2.13.0")

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    def assert_tensor_matches(self, actual, expected, *, case, exact_bits=False):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        actual_values = self.tensor_values(actual)
        expected_values = self.tensor_values(expected)
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual_values,
                expected_values,
                rtol=3.0 * np.finfo(np.float32).eps,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )
            zero_mask = expected_values == 0
            np.testing.assert_array_equal(
                actual_values[zero_mask].view(np.uint32),
                expected_values[zero_mask].view(np.uint32),
            )
            np.testing.assert_array_equal(
                np.isnan(actual_values), np.isnan(expected_values)
            )
            if exact_bits:
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32),
                    expected_values.reshape(-1).view(np.uint32),
                )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
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
        channels_last = module.tensor(
            np.linspace(-3.0, 3.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-3.0, 3.0, 720, dtype=np.float32)
            .reshape(2, 3, 4, 5, 6)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last_3d)
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            ("empty singleton trailing", module.zeros((0, 1), dtype=module.float32)),
            ("empty singleton middle", module.zeros((0, 1, 2), dtype=module.float32)),
            (
                "empty singleton surrounding",
                module.zeros((1, 0, 1), dtype=module.float32),
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            ("channels last", channels_last),
            ("channels last 3d", channels_last_3d),
            (
                "numerical edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_top_level_tanh(module, tensor, form):
        if form == "positional":
            return module.tanh(tensor)
        if form == "out none":
            return module.tanh(tensor, out=None)
        if form == "alias and out none":
            return module.tanh(x=tensor, out=None)
        return module.tanh(**{form: tensor})

    def test_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual = actual_input.tanh()
            expected = expected_input.tanh()
            self.assert_tensor_matches(
                actual,
                expected,
                case=case,
                exact_bits=case == "numerical edges",
            )
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def test_near_zero_normal_values_match_pytorch_bits(self):
        values = np.asarray(
            (
                1.0e-7,
                -1.0e-7,
                3.0e-7,
                -3.0e-7,
                1.0e-6,
                -1.0e-6,
                1.0e-5,
                -1.0e-5,
                1.0e-4,
                -1.0e-4,
                2.0e-4,
                -2.0e-4,
                5.0e-4,
                -5.0e-4,
                1.0e-3,
                -1.0e-3,
                2.0e-3,
                -2.0e-3,
                5.0e-3,
                -5.0e-3,
                1.0e-2,
                -1.0e-2,
            ),
            dtype=np.float32,
        )
        actual = torch.tensor(memoryview(values)).tanh()
        expected = reference_torch.tensor(
            values.copy(), dtype=reference_torch.float32
        ).tanh()

        np.testing.assert_array_equal(
            self.tensor_values(actual).view(np.uint32),
            self.tensor_values(expected).view(np.uint32),
        )

    def test_top_level_values_layouts_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            for form in forms:
                actual = self.call_top_level_tanh(torch, actual_input, form)
                expected = self.call_top_level_tanh(
                    reference_torch, expected_input, form
                )
                self.assert_tensor_matches(
                    actual,
                    expected,
                    case=(case, form),
                    exact_bits=case == "numerical edges",
                )
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def test_seeded_random_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x7A4E_213)
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(24):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 9, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(0.0, 5.0, size=elements).astype(np.float32)
            if elements:
                values[::7] = np.float32(0.0)
                values[1::11] = np.float32(-0.0)
                values[2::13] = np.float32(np.inf)
                values[3::17] = np.float32(-np.inf)
                values[4::19] = np.float32(np.nan)
            values = values.reshape(shape)

            actual_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            expected_input = reference_torch.tensor(
                values, dtype=reference_torch.float32
            )
            self.assert_tensor_matches(
                actual_input.tanh(),
                expected_input.tanh(),
                case=(case, shape),
            )

    def test_finite_owned_scalar_autograd_bits_match_pytorch_2_13(self):
        cases = (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x3F00_0000,
            0xBF00_0000,
            0x4110_2C66,
            0xC110_2C66,
            0x4110_2C67,
            0xC110_2C67,
        )

        for input_bits in cases:
            value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
            for form in ("method", "top level", "top level out none"):
                actual_leaf = torch.tensor(value, requires_grad=True)
                expected_leaf = reference_torch.tensor(
                    value, dtype=reference_torch.float32, requires_grad=True
                )
                if form == "method":
                    actual_output = actual_leaf.tanh()
                    expected_output = expected_leaf.tanh()
                elif form == "top level":
                    actual_output = torch.tanh(actual_leaf)
                    expected_output = reference_torch.tanh(expected_leaf)
                else:
                    actual_output = torch.tanh(actual_leaf, out=None)
                    expected_output = reference_torch.tanh(expected_leaf, out=None)

                actual_output.backward()
                expected_output.backward()
                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(f"0x{input_bits:08x}", form, "forward"),
                    exact_bits=True,
                )
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(f"0x{input_bits:08x}", form, "gradient"),
                    exact_bits=True,
                )

    def test_rank_one_weighted_autograd_empty_and_graph_lifetime_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).tolist()
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.tolist(), dtype=reference_torch.float32
        )
        actual_output = torch.tanh(actual_leaf, out=None)
        expected_output = reference_torch.tanh(expected_leaf, out=None)

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-one forward",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).view(np.uint32), AUTOGRAD_OUTPUT_BITS
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "TanhBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<TanhBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="weighted gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(self.error(actual_loss.backward), self.error(expected_loss.backward))
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values, requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (actual_accumulated.tanh() * actual_weights).sum().backward()
            (expected_accumulated.tanh() * expected_weights).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="accumulated gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad).view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        actual_composed = torch.tensor([0.5, -0.5], requires_grad=True)
        expected_composed = reference_torch.tensor(
            [0.5, -0.5], dtype=reference_torch.float32, requires_grad=True
        )
        actual_composed.tanh().sin().sum().backward()
        expected_composed.tanh().sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="composed gradient",
            exact_bits=True,
        )

        actual_empty = torch.tensor([], requires_grad=True)
        expected_empty = reference_torch.tensor(
            [], dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_output = torch.tanh(actual_empty)
        expected_empty_output = reference_torch.tanh(expected_empty)
        self.assert_tensor_matches(
            actual_empty_output,
            expected_empty_output,
            case="empty forward",
            exact_bits=True,
        )
        self.assertEqual(type(expected_empty_output.grad_fn).__name__, "TanhBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_empty_output),
            ", grad_fn=<TanhBackward0>",
        )
        actual_empty_loss = actual_empty_output.sum()
        expected_empty_loss = expected_empty_output.sum()
        actual_empty_loss.backward()
        expected_empty_loss.backward()
        self.assert_tensor_matches(
            actual_empty.grad,
            expected_empty.grad,
            case="empty gradient",
            exact_bits=True,
        )
        self.assertEqual(
            self.error(actual_empty_loss.backward),
            self.error(expected_empty_loss.backward),
        )

    def test_rank_two_weighted_autograd_empty_and_graph_lifetime_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist()
        weight_values = AUTOGRAD_WEIGHTS.reshape(2, 4).tolist()

        for form in ("method", "top level out none"):
            with self.subTest(form=form):
                actual_leaf = torch.tensor(values, requires_grad=True)
                expected_leaf = reference_torch.tensor(
                    values, dtype=reference_torch.float32, requires_grad=True
                )
                actual_weights = torch.tensor(weight_values)
                expected_weights = reference_torch.tensor(
                    weight_values, dtype=reference_torch.float32
                )
                if form == "method":
                    actual_output = actual_leaf.tanh()
                    expected_output = expected_leaf.tanh()
                else:
                    actual_output = torch.tanh(actual_leaf, out=None)
                    expected_output = reference_torch.tanh(expected_leaf, out=None)

                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(form, "forward"),
                    exact_bits=True,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(actual_output).reshape(-1).view(np.uint32),
                    AUTOGRAD_OUTPUT_BITS,
                )
                self.assertFalse(actual_output.is_set_to(actual_leaf))
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        actual_output
                    ),
                    ", grad_fn=<TanhBackward0>",
                )

                actual_loss = (actual_output * actual_weights).sum()
                expected_loss = (expected_output * expected_weights).sum()
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(form, "weighted gradient"),
                    exact_bits=True,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(actual_leaf.grad)
                    .reshape(-1)
                    .view(np.uint32),
                    AUTOGRAD_GRADIENT_BITS,
                )
                actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
                expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
                self.assertEqual(
                    self.error(actual_loss.backward),
                    self.error(expected_loss.backward),
                )
                np.testing.assert_array_equal(
                    self.tensor_values(actual_leaf.grad), actual_gradient_before
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected_leaf.grad), expected_gradient_before
                )

        actual_accumulated = torch.tensor(values, requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(weight_values)
        expected_weights = reference_torch.tensor(
            weight_values, dtype=reference_torch.float32
        )
        for _ in range(2):
            (
                torch.tanh(actual_accumulated, out=None) * actual_weights
            ).sum().backward()
            (
                reference_torch.tanh(expected_accumulated, out=None)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-two accumulated gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape in ((0, 0), (0, 3), (2, 0)):
            for form in ("method", "top level out none"):
                with self.subTest(shape=shape, form=form):
                    actual_empty = torch.zeros(shape, requires_grad=True)
                    expected_empty = reference_torch.zeros(
                        shape,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    if form == "method":
                        actual_output = actual_empty.tanh()
                        expected_output = expected_empty.tanh()
                    else:
                        actual_output = torch.tanh(actual_empty, out=None)
                        expected_output = reference_torch.tanh(
                            expected_empty, out=None
                        )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(shape, form, "empty forward"),
                        exact_bits=True,
                    )
                    actual_loss = actual_output.sum()
                    expected_loss = expected_output.sum()
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assert_tensor_matches(
                        actual_empty.grad,
                        expected_empty.grad,
                        case=(shape, form, "empty gradient"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        self.error(actual_loss.backward),
                        self.error(expected_loss.backward),
                    )

    def test_rank_three_weighted_autograd_empty_and_graph_lifetime_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4).tolist()
        weight_values = AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist()

        for form in ("method", "top level out none"):
            with self.subTest(form=form):
                actual_leaf = torch.tensor(values, requires_grad=True)
                expected_leaf = reference_torch.tensor(
                    values, dtype=reference_torch.float32, requires_grad=True
                )
                actual_weights = torch.tensor(weight_values)
                expected_weights = reference_torch.tensor(
                    weight_values, dtype=reference_torch.float32
                )
                if form == "method":
                    actual_output = actual_leaf.tanh()
                    expected_output = expected_leaf.tanh()
                else:
                    actual_output = torch.tanh(actual_leaf, out=None)
                    expected_output = reference_torch.tanh(expected_leaf, out=None)

                self.assert_tensor_matches(
                    actual_output,
                    expected_output,
                    case=(form, "forward"),
                    exact_bits=True,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(actual_output).reshape(-1).view(np.uint32),
                    AUTOGRAD_OUTPUT_BITS,
                )
                self.assertFalse(actual_output.is_set_to(actual_leaf))
                self.assertEqual(type(expected_output.grad_fn).__name__, "TanhBackward0")
                self.assertEqual(
                    torch._C._nn_functional_dropout_tensor_autograd_suffix(
                        actual_output
                    ),
                    ", grad_fn=<TanhBackward0>",
                )

                actual_loss = (actual_output * actual_weights).sum()
                expected_loss = (expected_output * expected_weights).sum()
                actual_loss.backward()
                expected_loss.backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(form, "weighted gradient"),
                    exact_bits=True,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(actual_leaf.grad)
                    .reshape(-1)
                    .view(np.uint32),
                    AUTOGRAD_GRADIENT_BITS,
                )
                actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
                expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
                self.assertEqual(
                    self.error(actual_loss.backward),
                    self.error(expected_loss.backward),
                )
                np.testing.assert_array_equal(
                    self.tensor_values(actual_leaf.grad), actual_gradient_before
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected_leaf.grad), expected_gradient_before
                )

        actual_accumulated = torch.tensor(values, requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(weight_values)
        expected_weights = reference_torch.tensor(
            weight_values, dtype=reference_torch.float32
        )
        for _ in range(2):
            (
                torch.tanh(actual_accumulated, out=None) * actual_weights
            ).sum().backward()
            (
                reference_torch.tanh(expected_accumulated, out=None)
                * expected_weights
            ).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-three accumulated gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        for shape in ((0, 1, 4), (2, 0, 4), (2, 1, 0), (1, 0, 1)):
            for form in ("method", "top level out none"):
                with self.subTest(shape=shape, form=form):
                    actual_empty = torch.zeros(shape, requires_grad=True)
                    expected_empty = reference_torch.zeros(
                        shape,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    if form == "method":
                        actual_output = actual_empty.tanh()
                        expected_output = expected_empty.tanh()
                    else:
                        actual_output = torch.tanh(actual_empty, out=None)
                        expected_output = reference_torch.tanh(
                            expected_empty, out=None
                        )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(shape, form, "empty forward"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        type(expected_output.grad_fn).__name__, "TanhBackward0"
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual_output
                        ),
                        ", grad_fn=<TanhBackward0>",
                    )
                    actual_loss = actual_output.sum()
                    expected_loss = expected_output.sum()
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assert_tensor_matches(
                        actual_empty.grad,
                        expected_empty.grad,
                        case=(shape, form, "empty gradient"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        self.error(actual_loss.backward),
                        self.error(expected_loss.backward),
                    )

    def test_scalar_composition_accumulation_and_freed_graph_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            composed = module.tensor(
                0.5, dtype=module.float32, requires_grad=True
            )
            module.tanh(composed, out=None).sin().backward()
            composed_gradient = self.tensor_values(composed.grad).copy()

            accumulated = module.tensor(
                -0.5, dtype=module.float32, requires_grad=True
            )
            accumulated.tanh().backward()
            first = self.tensor_values(accumulated.grad).copy()
            module.tanh(accumulated, out=None).backward()
            second = self.tensor_values(accumulated.grad).copy()

            freed = module.tensor(
                0.25, dtype=module.float32, requires_grad=True
            )
            loss = freed.tanh()
            loss.backward()
            repeated_backward = self.error(loss.backward)
            snapshots.append(
                (composed_gradient, first, second, repeated_backward)
            )

        for index in range(3):
            np.testing.assert_allclose(
                snapshots[0][index],
                snapshots[1][index],
                rtol=2.0e-6,
                atol=0.0,
            )
        self.assertEqual(snapshots[0][3], snapshots[1][3])

    def test_scalar_backward_uses_the_saved_output_across_rounding_modes(self):
        try:
            runtime = ctypes.CDLL(None)
            fegetround = runtime.fegetround
            fesetround = runtime.fesetround
        except (AttributeError, OSError):
            self.skipTest("the platform C runtime does not expose fenv controls")
        fegetround.argtypes = []
        fegetround.restype = ctypes.c_int
        fesetround.argtypes = [ctypes.c_int]
        fesetround.restype = ctypes.c_int

        original_rounding = fegetround()
        input_value = np.asarray(0xC015_F5AC, dtype=np.uint32).view(np.float32).item()
        try:
            if fesetround(0) != 0 or fegetround() != 0:
                self.skipTest("the platform does not expose round-to-nearest as zero")

            downward_rounding = None
            for candidate in (0x400, 0x80_0000):
                if fesetround(candidate) != 0 or fegetround() != candidate:
                    continue
                probe = torch.tensor(input_value).tanh()
                if self.tensor_values(probe).view(np.uint32).item() == 0xBF7B_5263:
                    downward_rounding = candidate
                    break
            if downward_rounding is None:
                self.skipTest("native tanh is not sensitive to available fenv modes")

            self.assertEqual(fesetround(0), 0)
            actual_leaf = torch.tensor(input_value, requires_grad=True)
            expected_leaf = reference_torch.tensor(
                input_value,
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            actual_output = actual_leaf.tanh()
            expected_output = expected_leaf.tanh()

            self.assertEqual(fesetround(downward_rounding), 0)
            actual_output.backward()
            expected_output.backward()
        finally:
            self.assertEqual(fesetround(original_rounding), 0)

        self.assertEqual(
            self.tensor_values(actual_output).view(np.uint32).item(),
            self.tensor_values(expected_output).view(np.uint32).item(),
        )
        self.assertEqual(
            self.tensor_values(actual_leaf.grad).view(np.uint32).item(),
            self.tensor_values(expected_leaf.grad).view(np.uint32).item(),
        )

    def test_tanh_backward_node_identity_matches_pytorch_2_13(self):
        actual = torch.tensor(0.5, requires_grad=True).tanh()
        expected = reference_torch.tensor(
            0.5, dtype=reference_torch.float32, requires_grad=True
        ).tanh()
        self.assertEqual(type(expected.grad_fn).__name__, "TanhBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual),
            f", grad_fn=<{type(expected.grad_fn).__name__}>",
        )

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.tanh unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "tanh")
        bound = tensor.tanh
        return {
            "descriptor_type": type(descriptor).__name__,
            "bound_type": type(bound).__name__,
            "descriptor_repr": repr(descriptor),
            "descriptor_name": descriptor.__name__,
            "descriptor_qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "descriptor_text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signatures": (
                self.signature_outcome(descriptor),
                self.signature_outcome(bound),
            ),
            "owner_name": descriptor.__objclass__.__name__,
            "owner_module": descriptor.__objclass__.__module__,
            "descriptor_has_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "types_match": (
                type(descriptor) is types.MethodDescriptorType,
                type(bound) is types.BuiltinMethodType,
            ),
            "errors": tuple(
                self.error(call)
                for call in (
                    lambda: tensor.tanh(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.tanh(1, 2),
                    lambda: tensor.tanh(input=tensor),
                    lambda: bound(unexpected=True),
                    lambda: descriptor(tensor, unexpected=True),
                    lambda: descriptor(),
                    lambda: descriptor(1),
                    lambda: descriptor(self=tensor),
                )
            ),
        }

    def test_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch), self.callable_contract(reference_torch)
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([0.5], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "tanh")
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
    intercepted = tensor.tanh()
function, dispatch_types, args, kwargs = recording.calls[0]

order = []
class ForwardingMode(module.overrides.TorchFunctionMode):
    def __init__(self, label):
        self.label = label

    def __torch_function__(self, func, types, args=(), kwargs=None):
        order.append(self.label)
        return func(*args, **(kwargs or {}))

with ForwardingMode("lower"):
    with ForwardingMode("upper"):
        forwarded = tensor.tanh()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.tanh()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.tanh(1)
except Exception as error:
    invalid_error = [type(error).__name__, str(error)]
else:
    invalid_error = None

print(json.dumps({
    "intercepted": intercepted is marker,
    "call_count": len(recording.calls),
    "function_type": type(function).__name__,
    "function_name": function.__name__,
    "function_qualname": function.__qualname__,
    "function_is_descriptor": function is descriptor,
    "types": dispatch_types == (module.Tensor,),
    "args": len(args) == 1 and args[0] is tensor,
    "kwargs_is_none": kwargs is None,
    "forwarding_order": order,
    "forwarded": forwarded.tolist(),
    "declining_error": declining_error,
    "declining_calls": len(declining.calls),
    "invalid_error": invalid_error,
    "invalid_calls": len(invalid.calls),
    "stack_depth": len(module.overrides._get_current_function_mode_stack()),
}, sort_keys=True))
'''
        result = subprocess.run(
            [sys.executable, "-c", f"MODULE = {module_name!r}\n" + source],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_torch_function_mode_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.mode_dispatch_observation("torch_rs"),
            self.mode_dispatch_observation("torch"),
        )

    def top_level_callable_contract(self, module):
        function = module.tanh
        owner = function.__reduce__()[1][0]
        wildcard_namespace = {}
        exec(f"from {module.__name__} import *", wildcard_namespace)
        try:
            inspect.signature(function)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "type": type(function).__name__,
            "is_builtin": type(function) is types.BuiltinFunctionType,
            "name": function.__name__,
            "qualname": function.__qualname__,
            "module": function.__module__,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.tanh is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
            "all_count": module.__all__.count("tanh"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["tanh"] is function,
            "pickle_identities": tuple(
                pickle.loads(pickle.dumps(function, protocol=protocol)) is function
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_top_level_callable_contract_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_callable_contract(torch),
            self.top_level_callable_contract(reference_torch),
        )

    @staticmethod
    def top_level_dispatch_observation(module):
        tensor = module.tensor([0.5], dtype=module.float32)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.tanh
        marker = object()

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_observations = []
        mode_calls = (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tensor, out=None), ("input", "out")),
            (lambda: function(tensor, out=destination), ("out",)),
        )
        for call, keyword_names in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            dispatched, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: function(value), None),
            (lambda value: function(input=value), "input"),
            (lambda value: function(tensor, out=value), "out"),
            (lambda value: function(x=value, out=None), "x"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            dispatched, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    dispatched is function,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        subclass_order = []

        class BaseOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("base", tuple(item.__name__ for item in types))
                )
                return marker

        class DerivedOverride(BaseOverride):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                subclass_order.append(
                    ("derived", tuple(item.__name__ for item in types))
                )
                return marker

        subclass_result = function(BaseOverride(), out=DerivedOverride())

        forwarding_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forwarding_order.append(self.label)
                return func(*args, **(kwargs or {}))

        with ForwardingMode("lower"):
            with ForwardingMode("upper"):
                forwarded = function(input=tensor, out=None)

        fallback_events = []

        class FallbackOverride:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                fallback_events.append("override")
                return marker

        declining_mode = RecordingMode(NotImplemented)
        with declining_mode:
            fallback_result = function(FallbackOverride())

        invalid_observations = []
        for call in (
            lambda: function(),
            lambda: function([], out=destination),
            lambda: function(tensor, out=[]),
            lambda: function(tensor, extra=True),
            lambda: function(tensor, tensor),
        ):
            mode = RecordingMode()
            try:
                with mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(mode.calls))
                )

        return (
            mode_observations,
            override_observations,
            subclass_result is marker,
            subclass_order,
            forwarding_order,
            tuple(np.asarray(forwarded).reshape(-1)),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_top_level_modes_and_subclass_dispatch_match_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_dispatch_observation(torch),
            self.top_level_dispatch_observation(reference_torch),
        )

    def test_top_level_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([0.5])
        expected = reference_torch.tensor([0.5])
        cases = (
            (lambda: torch.tanh(), lambda: reference_torch.tanh()),
            (
                lambda: torch.tanh(actual, actual),
                lambda: reference_torch.tanh(expected, expected),
            ),
            (
                lambda: torch.tanh(actual, input=actual),
                lambda: reference_torch.tanh(expected, input=expected),
            ),
            (
                lambda: torch.tanh(out=actual),
                lambda: reference_torch.tanh(out=expected),
            ),
            (
                lambda: torch.tanh(1, extra=True),
                lambda: reference_torch.tanh(1, extra=True),
            ),
            (lambda: torch.tanh(input=[]), lambda: reference_torch.tanh(input=[])),
            (
                lambda: torch.tanh(actual, out=[]),
                lambda: reference_torch.tanh(expected, out=[]),
            ),
            (
                lambda: torch.tanh(actual, extra=True, out=[]),
                lambda: reference_torch.tanh(expected, extra=True, out=[]),
            ),
            (
                lambda: torch.tanh(actual, extra=True),
                lambda: reference_torch.tanh(expected, extra=True),
            ),
            (
                lambda: torch.tanh(input=actual, a=actual),
                lambda: reference_torch.tanh(input=expected, a=expected),
            ),
            (
                lambda: torch.tanh(a=actual, x=actual, out=None),
                lambda: reference_torch.tanh(a=expected, x=expected, out=None),
            ),
            (
                lambda: torch.tanh(x=actual, a=actual, out=None),
                lambda: reference_torch.tanh(x=expected, a=expected, out=None),
            ),
            (
                lambda: torch.tanh(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.tanh(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_top_level_declining_override_diagnostics_match_pytorch_2_13(self):
        class Override:
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        self.assertEqual(
            self.error(lambda: torch.tanh(Override())),
            self.error(lambda: reference_torch.tanh(Override())),
        )
        self.assertEqual(
            self.error(lambda: torch.tanh(torch.tensor([0.5]), out=Override())),
            self.error(
                lambda: reference_torch.tanh(
                    reference_torch.tensor([0.5]), out=Override()
                )
            ),
        )

    def test_concrete_out_support_boundary_is_explicit_and_nonmutating(self):
        actual_input = torch.tensor([0.5, -0.5], requires_grad=True)
        actual_out = torch.tensor([17.0, 19.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tanh\(\): the 'out' argument is not supported$",
        ):
            torch.tanh(actual_input, out=actual_out)
        self.assertEqual(actual_out.tolist(), [17.0, 19.0])
        self.assertIsNone(actual_input.grad)

        expected_input = reference_torch.tensor(
            [0.5, -0.5], dtype=reference_torch.float32
        )
        expected_out = reference_torch.tensor(
            [17.0, 19.0], dtype=reference_torch.float32
        )
        self.assertIs(
            reference_torch.tanh(expected_input, out=expected_out), expected_out
        )
        self.assert_tensor_matches(
            actual_input.detach().tanh(),
            expected_out,
            case="reference concrete out values",
        )
        actual_input.tanh().sum().backward()
        self.assertIsNotNone(actual_input.grad)

    def test_autograd_modes_and_unsupported_boundaries_remain_explicit(self):
        actual_scalar = torch.tensor(0.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(
            0.5, dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_no_grad_scalar = torch.tanh(actual_scalar, out=None)
        with reference_torch.no_grad():
            expected_no_grad_scalar = reference_torch.tanh(
                expected_scalar, out=None
            )
        self.assert_tensor_matches(
            actual_no_grad_scalar,
            expected_no_grad_scalar,
            case="scalar no_grad",
            exact_bits=True,
        )
        self.assert_tensor_matches(
            actual_scalar.detach().tanh(),
            expected_scalar.detach().tanh(),
            case="scalar detached",
            exact_bits=True,
        )

        higher_order = torch.tensor([[[0.25, -0.25]]], requires_grad=True)
        loss = higher_order.tanh().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        loss.backward()
        self.assertIsNotNone(higher_order.grad)

        message = r"^tanh\(\): autograd recording is not supported$"
        for bits in (0x7F80_0000, 0xFF80_0000, 0x7FC1_2345, 0xFFC5_4321):
            value = np.asarray(bits, dtype=np.uint32).view(np.float32).item()
            actual = torch.tensor(value, requires_grad=True)
            for call in (
                actual.tanh,
                lambda actual=actual: torch.tanh(actual),
                lambda actual=actual: torch.tanh(actual, out=None),
            ):
                with self.subTest(nonfinite=f"0x{bits:08x}"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual.grad)
            actual.sum().backward()
            self.assertEqual(actual.grad.item(), 1.0)

            actual_vector = torch.tensor([0.5, value], requires_grad=True)
            for call in (
                actual_vector.tanh,
                lambda actual_vector=actual_vector: torch.tanh(actual_vector),
                lambda actual_vector=actual_vector: torch.tanh(
                    actual_vector, out=None
                ),
            ):
                with self.subTest(nonfinite_vector=f"0x{bits:08x}"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual_vector.grad)
            actual_vector.sum().backward()
            self.assertEqual(actual_vector.grad.tolist(), [1.0, 1.0])

            actual_matrix = torch.tensor([[0.5, value]], requires_grad=True)
            for call in (
                actual_matrix.tanh,
                lambda actual_matrix=actual_matrix: torch.tanh(actual_matrix),
                lambda actual_matrix=actual_matrix: torch.tanh(
                    actual_matrix, out=None
                ),
            ):
                with self.subTest(nonfinite_matrix=f"0x{bits:08x}"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual_matrix.grad)
            actual_matrix.sum().backward()
            self.assertEqual(actual_matrix.grad.tolist(), [[1.0, 1.0]])

            actual_rank_three = torch.tensor(
                [[[0.5, value]]], requires_grad=True
            )
            for call in (
                actual_rank_three.tanh,
                lambda actual_rank_three=actual_rank_three: torch.tanh(
                    actual_rank_three
                ),
                lambda actual_rank_three=actual_rank_three: torch.tanh(
                    actual_rank_three, out=None
                ),
            ):
                with self.subTest(nonfinite_rank_three=f"0x{bits:08x}"):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual_rank_three.grad)
            actual_rank_three.sum().backward()
            self.assertEqual(actual_rank_three.grad.tolist(), [[[1.0, 1.0]]])

        actual_rank_four = torch.tensor([[[[0.5, -1.0]]]], requires_grad=True)
        for call in (
            actual_rank_four.tanh,
            lambda: torch.tanh(actual_rank_four),
            lambda: torch.tanh(actual_rank_four, out=None),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        self.assertIsNone(actual_rank_four.grad)
        actual_rank_four.sum().backward()
        self.assertEqual(actual_rank_four.grad.tolist(), [[[[1.0, 1.0]]]])

        actual_leaf = torch.tensor([0.5, -1.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [0.5, -1.0], dtype=reference_torch.float32, requires_grad=True
        )
        self.assertTrue(actual_leaf.tanh().requires_grad)
        self.assertTrue(torch.tanh(actual_leaf).requires_grad)
        self.assertTrue(expected_leaf.tanh().requires_grad)
        self.assertTrue(reference_torch.tanh(expected_leaf).requires_grad)

        actual_view_base = torch.tensor([0.5], requires_grad=True)
        actual_view = actual_view_base[0]
        for call in (actual_view.tanh, lambda: torch.tanh(actual_view, out=None)):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_view.backward()
        self.assertEqual(actual_view_base.grad.tolist(), [1.0])

        actual_vector_view_base = torch.tensor(
            [[0.5, -1.0], [2.0, -3.0]], requires_grad=True
        )
        actual_vector_view = actual_vector_view_base[0]
        for call in (
            actual_vector_view.tanh,
            lambda: torch.tanh(actual_vector_view, out=None),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_vector_view.sum().backward()
        self.assertEqual(
            actual_vector_view_base.grad.tolist(), [[1.0, 1.0], [0.0, 0.0]]
        )

        actual_matrix_view_base = torch.tensor(
            [
                [[0.5, -1.0], [2.0, -3.0]],
                [[4.0, -5.0], [6.0, -7.0]],
            ],
            requires_grad=True,
        )
        actual_matrix_view = actual_matrix_view_base[0]
        for call in (
            actual_matrix_view.tanh,
            lambda: torch.tanh(actual_matrix_view, out=None),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_matrix_view.sum().backward()
        self.assertEqual(
            actual_matrix_view_base.grad.tolist(),
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        )

        actual_rank_three_view_base = torch.tensor(
            [[[[0.5, -1.0]]], [[[2.0, -3.0]]]], requires_grad=True
        )
        actual_rank_three_view = actual_rank_three_view_base[0]
        for call in (
            actual_rank_three_view.tanh,
            lambda: torch.tanh(actual_rank_three_view, out=None),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_rank_three_view.sum().backward()
        self.assertEqual(
            actual_rank_three_view_base.grad.tolist(),
            [[[[1.0, 1.0]]], [[[0.0, 0.0]]]],
        )

        actual_nonleaf_base = torch.tensor([[[0.5, -0.5]]], requires_grad=True)
        actual_nonleaf = actual_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            actual_nonleaf.tanh()
        actual_nonleaf.sum().backward()
        self.assertIsNotNone(actual_nonleaf_base.grad)

        empty_view_base = torch.zeros((1, 0), requires_grad=True)
        with torch.no_grad():
            empty_view = empty_view_base[0]
        self.assertTrue(empty_view.requires_grad)
        self.assertTrue(empty_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            empty_view.tanh()

        with torch.no_grad():
            matrix_view = actual_matrix_view_base[0]
        self.assertTrue(matrix_view.requires_grad)
        self.assertTrue(matrix_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            matrix_view.tanh()

        with torch.no_grad():
            rank_three_view = actual_rank_three_view_base[0]
        self.assertTrue(rank_three_view.requires_grad)
        self.assertTrue(rank_three_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            rank_three_view.tanh()

        with torch.no_grad():
            actual_no_grad = torch.tanh(actual_leaf, out=None)
        with reference_torch.no_grad():
            expected_no_grad = reference_torch.tanh(expected_leaf, out=None)
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        actual_detached = torch.tanh(actual_leaf.detach())
        expected_detached = reference_torch.tanh(expected_leaf.detach())
        self.assert_tensor_matches(actual_detached, expected_detached, case="detached")

        self.assertTrue(hasattr(torch, "tanh"))
        self.assertTrue(hasattr(reference_torch, "tanh"))
        self.assertTrue(hasattr(torch.nn.functional, "tanh"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "tanh"))
        self.assertFalse(hasattr(torch.Tensor, "tanh_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "tanh_"))


if __name__ == "__main__":
    unittest.main()
