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
    from .test_sigmoid import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
        SPECIAL_INPUT_BITS,
        SPECIAL_OUTPUT_BITS,
        rank_preserving_nonleaf_parent_cases,
        scalar_nonleaf_parent_cases,
    )
else:
    from test_sigmoid import (
        AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        AUTOGRAD_GRADIENT_BITS,
        AUTOGRAD_INPUT_BITS,
        AUTOGRAD_OUTPUT_BITS,
        AUTOGRAD_WEIGHTS,
        SPECIAL_INPUT_BITS,
        SPECIAL_OUTPUT_BITS,
        rank_preserving_nonleaf_parent_cases,
        scalar_nonleaf_parent_cases,
    )

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSigmoidReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.sigmoid differentials require pinned PyTorch 2.13.0"
            )

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
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                self.tensor_values(actual),
                self.tensor_values(expected),
                rtol=2.0e-6,
                atol=np.nextafter(np.float32(0), np.float32(1)),
                equal_nan=True,
            )
            if exact_bits:
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                )

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        channels_last = module.tensor(
            np.linspace(-15.0, 15.0, 120, dtype=np.float32)
            .reshape(2, 3, 4, 5)
            .tolist(),
            dtype=module.float32,
        ).contiguous(memory_format=module.channels_last)
        channels_last_3d = module.tensor(
            np.linspace(-90.0, 90.0, 720, dtype=np.float32)
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
            (
                "empty singleton middle",
                module.zeros((0, 1, 2), dtype=module.float32),
            ),
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
                module.tensor(memoryview(SPECIAL_INPUT_BITS.view(np.float32))),
            ),
        )

    @staticmethod
    def call_top_level_sigmoid(module, tensor, form):
        if form == "positional":
            return module.sigmoid(tensor)
        if form == "out none":
            return module.sigmoid(tensor, out=None)
        if form == "alias and out none":
            return module.sigmoid(x=tensor, out=None)
        return module.sigmoid(**{form: tensor})

    def test_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            actual = actual_input.sigmoid()
            expected = expected_input.sigmoid()
            self.assert_tensor_matches(actual, expected, case=case)
            self.assertFalse(actual.is_set_to(actual_input))
            self.assertFalse(expected.is_set_to(expected_input))
            if actual_input.numel():
                self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
            if case == "numerical edges":
                np.testing.assert_array_equal(
                    self.tensor_values(actual).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )
                np.testing.assert_array_equal(
                    self.tensor_values(expected).reshape(-1).view(np.uint32),
                    SPECIAL_OUTPUT_BITS,
                )

    def test_top_level_values_layouts_offsets_empty_tensors_and_storage_match_pytorch(
        self,
    ):
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
                actual = self.call_top_level_sigmoid(torch, actual_input, form)
                expected = self.call_top_level_sigmoid(
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

    def test_seeded_float32_values_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x5160_213)
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))
        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(
                tuple(int(value) for value in rng.integers(0, 9, size=rank))
            )

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-100.0, 100.0, size=elements).astype(np.float32)
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
            actual = actual_input.sigmoid()
            expected = expected_input.sigmoid()

            with self.subTest(case=case, shape=shape):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                np.testing.assert_allclose(
                    self.tensor_values(actual),
                    self.tensor_values(expected),
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )

    def test_finite_owned_scalar_autograd_bits_match_pytorch_2_13(self):
        cases = (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x3F00_0000,
            0xBF00_0000,
            0x4185_1591,
            0x4185_1592,
            0xC2B1_7217,
            0xC2B1_7218,
        )

        for input_bits in cases:
            value = np.asarray(input_bits, dtype=np.uint32).view(np.float32).item()
            actual_leaf = torch.tensor(value, requires_grad=True)
            expected_leaf = reference_torch.tensor(
                value, dtype=reference_torch.float32, requires_grad=True
            )
            actual_output = actual_leaf.sigmoid()
            expected_output = expected_leaf.sigmoid()

            actual_output.backward()
            expected_output.backward()
            self.assert_tensor_matches(
                actual_output,
                expected_output,
                case=(f"0x{input_bits:08x}", "forward"),
                exact_bits=True,
            )
            self.assert_tensor_matches(
                actual_leaf.grad,
                expected_leaf.grad,
                case=(f"0x{input_bits:08x}", "gradient"),
                exact_bits=True,
            )

    def test_scalar_composition_accumulation_and_freed_graph_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            composed = module.tensor(
                0.5, dtype=module.float32, requires_grad=True
            )
            composed.sigmoid().sin().backward()
            composed_gradient = self.tensor_values(composed.grad).copy()

            accumulated = module.tensor(
                -0.5, dtype=module.float32, requires_grad=True
            )
            accumulated.sigmoid().backward()
            first = self.tensor_values(accumulated.grad).copy()
            accumulated.sigmoid().backward()
            second = self.tensor_values(accumulated.grad).copy()

            freed = module.tensor(
                0.25, dtype=module.float32, requires_grad=True
            )
            loss = freed.sigmoid()
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

    def test_owned_scalar_nonleaf_autograd_matches_pytorch_2_13_across_parent_nodes(
        self,
    ):
        actual_cases = scalar_nonleaf_parent_cases(torch)
        expected_cases = scalar_nonleaf_parent_cases(reference_torch)
        for (case, make_actual_parent), (
            expected_case,
            make_expected_parent,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            for form in ("method", "functional"):
                with self.subTest(parent=case, form=form):
                    actual_leaf = torch.tensor(0.5, requires_grad=True)
                    expected_leaf = reference_torch.tensor(
                        0.5,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    actual_parent = make_actual_parent(actual_leaf)
                    expected_parent = make_expected_parent(expected_leaf)
                    self.assert_tensor_matches(
                        actual_parent,
                        expected_parent,
                        case=(case, form, "parent"),
                        exact_bits=True,
                    )
                    self.assertFalse(actual_parent.is_set_to(actual_leaf))
                    self.assertFalse(expected_parent.is_set_to(expected_leaf))

                    if form == "method":
                        actual_output = actual_parent.sigmoid()
                        expected_output = expected_parent.sigmoid()
                    else:
                        actual_output = torch.nn.functional.sigmoid(actual_parent)
                        expected_output = reference_torch.nn.functional.sigmoid(
                            expected_parent
                        )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(case, form, "sigmoid"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual_output
                        ),
                        f", grad_fn=<{type(expected_output.grad_fn).__name__}>",
                    )

                    actual_output.backward()
                    expected_output.backward()
                    self.assert_tensor_matches(
                        actual_leaf.grad,
                        expected_leaf.grad,
                        case=(case, form, "composed gradient"),
                        exact_bits=True,
                    )
                    actual_gradient = self.tensor_values(actual_leaf.grad).copy()
                    expected_gradient = self.tensor_values(expected_leaf.grad).copy()
                    self.assertEqual(
                        self.error(actual_output.backward),
                        self.error(expected_output.backward),
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(actual_leaf.grad), actual_gradient
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(expected_leaf.grad), expected_gradient
                    )

                    actual_accumulated = torch.tensor(0.5, requires_grad=True)
                    expected_accumulated = reference_torch.tensor(
                        0.5,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    for _ in range(2):
                        actual_parent = make_actual_parent(actual_accumulated)
                        expected_parent = make_expected_parent(expected_accumulated)
                        if form == "method":
                            actual_parent.sigmoid().backward()
                            expected_parent.sigmoid().backward()
                        else:
                            torch.nn.functional.sigmoid(actual_parent).backward()
                            reference_torch.nn.functional.sigmoid(
                                expected_parent
                            ).backward()
                    self.assert_tensor_matches(
                        actual_accumulated.grad,
                        expected_accumulated.grad,
                        case=(case, form, "accumulated gradient"),
                        exact_bits=True,
                    )

    def test_owned_rank_one_nonleaf_autograd_matches_pytorch_2_13_across_parent_nodes(
        self,
    ):
        actual_cases = rank_preserving_nonleaf_parent_cases(torch)
        expected_cases = rank_preserving_nonleaf_parent_cases(reference_torch)
        values = [0.25, 0.5]
        actual_weights = torch.tensor([1.25, -0.75])
        expected_weights = reference_torch.tensor(
            [1.25, -0.75], dtype=reference_torch.float32
        )
        for (case, make_actual_parent), (
            expected_case,
            make_expected_parent,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            for form in ("method", "functional"):
                with self.subTest(parent=case, form=form):
                    actual_leaf = torch.tensor(values, requires_grad=True)
                    expected_leaf = reference_torch.tensor(
                        values,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    actual_parent = make_actual_parent(actual_leaf)
                    expected_parent = make_expected_parent(expected_leaf)
                    self.assert_tensor_matches(
                        actual_parent,
                        expected_parent,
                        case=(case, form, "parent"),
                        exact_bits=True,
                    )
                    self.assertFalse(actual_parent.is_set_to(actual_leaf))
                    self.assertFalse(expected_parent.is_set_to(expected_leaf))
                    self.assertNotEqual(actual_parent.data_ptr(), actual_leaf.data_ptr())
                    self.assertNotEqual(
                        expected_parent.data_ptr(), expected_leaf.data_ptr()
                    )

                    if form == "method":
                        actual_output = actual_parent.sigmoid()
                        expected_output = expected_parent.sigmoid()
                    else:
                        actual_output = torch.nn.functional.sigmoid(actual_parent)
                        expected_output = reference_torch.nn.functional.sigmoid(
                            expected_parent
                        )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(case, form, "sigmoid"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual_output
                        ),
                        f", grad_fn=<{type(expected_output.grad_fn).__name__}>",
                    )

                    actual_loss = (actual_output * actual_weights).sum()
                    expected_loss = (expected_output * expected_weights).sum()
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assert_tensor_matches(
                        actual_leaf.grad,
                        expected_leaf.grad,
                        case=(case, form, "weighted VJP"),
                        exact_bits=True,
                    )
                    actual_gradient = self.tensor_values(actual_leaf.grad).copy()
                    expected_gradient = self.tensor_values(expected_leaf.grad).copy()
                    self.assertEqual(
                        self.error(actual_loss.backward),
                        self.error(expected_loss.backward),
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(actual_leaf.grad), actual_gradient
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(expected_leaf.grad), expected_gradient
                    )

                    actual_accumulated = torch.tensor(values, requires_grad=True)
                    expected_accumulated = reference_torch.tensor(
                        values,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    for _ in range(2):
                        actual_parent = make_actual_parent(actual_accumulated)
                        expected_parent = make_expected_parent(expected_accumulated)
                        if form == "method":
                            actual_output = actual_parent.sigmoid()
                            expected_output = expected_parent.sigmoid()
                        else:
                            actual_output = torch.nn.functional.sigmoid(actual_parent)
                            expected_output = reference_torch.nn.functional.sigmoid(
                                expected_parent
                            )
                        (actual_output * actual_weights).sum().backward()
                        (expected_output * expected_weights).sum().backward()
                    self.assert_tensor_matches(
                        actual_accumulated.grad,
                        expected_accumulated.grad,
                        case=(case, form, "accumulated VJP"),
                        exact_bits=True,
                    )

                    actual_empty = torch.tensor([], requires_grad=True)
                    expected_empty = reference_torch.tensor(
                        [], dtype=reference_torch.float32, requires_grad=True
                    )
                    actual_empty_parent = make_actual_parent(actual_empty)
                    expected_empty_parent = make_expected_parent(expected_empty)
                    self.assert_tensor_matches(
                        actual_empty_parent,
                        expected_empty_parent,
                        case=(case, form, "empty parent"),
                        exact_bits=True,
                    )
                    self.assertFalse(actual_empty_parent.is_set_to(actual_empty))
                    self.assertFalse(expected_empty_parent.is_set_to(expected_empty))
                    if form == "method":
                        actual_empty_output = actual_empty_parent.sigmoid()
                        expected_empty_output = expected_empty_parent.sigmoid()
                    else:
                        actual_empty_output = torch.nn.functional.sigmoid(
                            actual_empty_parent
                        )
                        expected_empty_output = (
                            reference_torch.nn.functional.sigmoid(
                                expected_empty_parent
                            )
                        )
                    self.assert_tensor_matches(
                        actual_empty_output,
                        expected_empty_output,
                        case=(case, form, "empty sigmoid"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual_empty_output
                        ),
                        f", grad_fn=<{type(expected_empty_output.grad_fn).__name__}>",
                    )
                    actual_empty_loss = actual_empty_output.sum()
                    expected_empty_loss = expected_empty_output.sum()
                    actual_empty_loss.backward()
                    expected_empty_loss.backward()
                    self.assert_tensor_matches(
                        actual_empty.grad,
                        expected_empty.grad,
                        case=(case, form, "empty VJP"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        self.error(actual_empty_loss.backward),
                        self.error(expected_empty_loss.backward),
                    )

    def test_owned_rank_two_nonleaf_autograd_matches_pytorch_2_13_across_parent_nodes(
        self,
    ):
        actual_cases = rank_preserving_nonleaf_parent_cases(torch)
        expected_cases = rank_preserving_nonleaf_parent_cases(reference_torch)
        values = [[0.25, 0.5], [0.5, 0.25]]
        actual_weights = torch.tensor([[1.25, -0.75], [0.5, -1.5]])
        expected_weights = reference_torch.tensor(
            [[1.25, -0.75], [0.5, -1.5]], dtype=reference_torch.float32
        )
        for (case, make_actual_parent), (
            expected_case,
            make_expected_parent,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            for form in ("method", "functional"):
                with self.subTest(parent=case, form=form):
                    actual_leaf = torch.tensor(values, requires_grad=True)
                    expected_leaf = reference_torch.tensor(
                        values,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    actual_parent = make_actual_parent(actual_leaf)
                    expected_parent = make_expected_parent(expected_leaf)
                    self.assert_tensor_matches(
                        actual_parent,
                        expected_parent,
                        case=(case, form, "parent"),
                        exact_bits=True,
                    )
                    self.assertFalse(actual_parent.is_set_to(actual_leaf))
                    self.assertFalse(expected_parent.is_set_to(expected_leaf))
                    self.assertNotEqual(actual_parent.data_ptr(), actual_leaf.data_ptr())
                    self.assertNotEqual(
                        expected_parent.data_ptr(), expected_leaf.data_ptr()
                    )

                    if form == "method":
                        actual_output = actual_parent.sigmoid()
                        expected_output = expected_parent.sigmoid()
                    else:
                        actual_output = torch.nn.functional.sigmoid(actual_parent)
                        expected_output = reference_torch.nn.functional.sigmoid(
                            expected_parent
                        )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(case, form, "sigmoid"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual_output
                        ),
                        f", grad_fn=<{type(expected_output.grad_fn).__name__}>",
                    )

                    actual_loss = (actual_output * actual_weights).sum()
                    expected_loss = (expected_output * expected_weights).sum()
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assert_tensor_matches(
                        actual_leaf.grad,
                        expected_leaf.grad,
                        case=(case, form, "weighted VJP"),
                        exact_bits=True,
                    )
                    actual_gradient = self.tensor_values(actual_leaf.grad).copy()
                    expected_gradient = self.tensor_values(expected_leaf.grad).copy()
                    self.assertEqual(
                        self.error(actual_loss.backward),
                        self.error(expected_loss.backward),
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(actual_leaf.grad), actual_gradient
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(expected_leaf.grad), expected_gradient
                    )

                    actual_accumulated = torch.tensor(values, requires_grad=True)
                    expected_accumulated = reference_torch.tensor(
                        values,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    for _ in range(2):
                        actual_parent = make_actual_parent(actual_accumulated)
                        expected_parent = make_expected_parent(expected_accumulated)
                        if form == "method":
                            actual_output = actual_parent.sigmoid()
                            expected_output = expected_parent.sigmoid()
                        else:
                            actual_output = torch.nn.functional.sigmoid(actual_parent)
                            expected_output = reference_torch.nn.functional.sigmoid(
                                expected_parent
                            )
                        (actual_output * actual_weights).sum().backward()
                        (expected_output * expected_weights).sum().backward()
                    self.assert_tensor_matches(
                        actual_accumulated.grad,
                        expected_accumulated.grad,
                        case=(case, form, "accumulated VJP"),
                        exact_bits=True,
                    )

                    for empty_shape in ((0, 0), (0, 3), (2, 0)):
                        actual_empty = torch.zeros(
                            empty_shape, requires_grad=True
                        )
                        expected_empty = reference_torch.zeros(
                            empty_shape,
                            dtype=reference_torch.float32,
                            requires_grad=True,
                        )
                        actual_empty_parent = make_actual_parent(actual_empty)
                        expected_empty_parent = make_expected_parent(expected_empty)
                        self.assert_tensor_matches(
                            actual_empty_parent,
                            expected_empty_parent,
                            case=(case, form, "empty parent", empty_shape),
                            exact_bits=True,
                        )
                        self.assertFalse(actual_empty_parent.is_set_to(actual_empty))
                        self.assertFalse(
                            expected_empty_parent.is_set_to(expected_empty)
                        )
                        if form == "method":
                            actual_empty_output = actual_empty_parent.sigmoid()
                            expected_empty_output = expected_empty_parent.sigmoid()
                        else:
                            actual_empty_output = torch.nn.functional.sigmoid(
                                actual_empty_parent
                            )
                            expected_empty_output = (
                                reference_torch.nn.functional.sigmoid(
                                    expected_empty_parent
                                )
                            )
                        self.assert_tensor_matches(
                            actual_empty_output,
                            expected_empty_output,
                            case=(case, form, "empty sigmoid", empty_shape),
                            exact_bits=True,
                        )
                        self.assertEqual(
                            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                                actual_empty_output
                            ),
                            f", grad_fn=<{type(expected_empty_output.grad_fn).__name__}>",
                        )
                        actual_empty_loss = actual_empty_output.sum()
                        expected_empty_loss = expected_empty_output.sum()
                        actual_empty_loss.backward()
                        expected_empty_loss.backward()
                        self.assert_tensor_matches(
                            actual_empty.grad,
                            expected_empty.grad,
                            case=(case, form, "empty VJP", empty_shape),
                            exact_bits=True,
                        )
                        self.assertEqual(
                            self.error(actual_empty_loss.backward),
                            self.error(expected_empty_loss.backward),
                        )

    def test_owned_rank_three_nonleaf_autograd_matches_pytorch_2_13_across_parent_nodes(
        self,
    ):
        actual_cases = rank_preserving_nonleaf_parent_cases(torch)
        expected_cases = rank_preserving_nonleaf_parent_cases(reference_torch)
        values = [[[0.25, 0.5]], [[0.5, 0.25]]]
        actual_weights = torch.tensor([[[1.25, -0.75]], [[0.5, -1.5]]])
        expected_weights = reference_torch.tensor(
            [[[1.25, -0.75]], [[0.5, -1.5]]], dtype=reference_torch.float32
        )
        for (case, make_actual_parent), (
            expected_case,
            make_expected_parent,
        ) in zip(actual_cases, expected_cases, strict=True):
            self.assertEqual(case, expected_case)
            for form in ("method", "functional"):
                with self.subTest(parent=case, form=form):
                    actual_leaf = torch.tensor(values, requires_grad=True)
                    expected_leaf = reference_torch.tensor(
                        values,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    actual_parent = make_actual_parent(actual_leaf)
                    expected_parent = make_expected_parent(expected_leaf)
                    self.assert_tensor_matches(
                        actual_parent,
                        expected_parent,
                        case=(case, form, "parent"),
                        exact_bits=True,
                    )
                    self.assertFalse(actual_parent.is_set_to(actual_leaf))
                    self.assertFalse(expected_parent.is_set_to(expected_leaf))
                    self.assertNotEqual(actual_parent.data_ptr(), actual_leaf.data_ptr())
                    self.assertNotEqual(
                        expected_parent.data_ptr(), expected_leaf.data_ptr()
                    )

                    if form == "method":
                        actual_output = actual_parent.sigmoid()
                        expected_output = expected_parent.sigmoid()
                    else:
                        actual_output = torch.nn.functional.sigmoid(actual_parent)
                        expected_output = reference_torch.nn.functional.sigmoid(
                            expected_parent
                        )
                    self.assert_tensor_matches(
                        actual_output,
                        expected_output,
                        case=(case, form, "sigmoid"),
                        exact_bits=True,
                    )
                    self.assertEqual(
                        torch._C._nn_functional_dropout_tensor_autograd_suffix(
                            actual_output
                        ),
                        f", grad_fn=<{type(expected_output.grad_fn).__name__}>",
                    )

                    actual_loss = (actual_output * actual_weights).sum()
                    expected_loss = (expected_output * expected_weights).sum()
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assert_tensor_matches(
                        actual_leaf.grad,
                        expected_leaf.grad,
                        case=(case, form, "weighted VJP"),
                        exact_bits=True,
                    )
                    actual_gradient = self.tensor_values(actual_leaf.grad).copy()
                    expected_gradient = self.tensor_values(expected_leaf.grad).copy()
                    self.assertEqual(
                        self.error(actual_loss.backward),
                        self.error(expected_loss.backward),
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(actual_leaf.grad), actual_gradient
                    )
                    np.testing.assert_array_equal(
                        self.tensor_values(expected_leaf.grad), expected_gradient
                    )

                    actual_accumulated = torch.tensor(values, requires_grad=True)
                    expected_accumulated = reference_torch.tensor(
                        values,
                        dtype=reference_torch.float32,
                        requires_grad=True,
                    )
                    for _ in range(2):
                        actual_parent = make_actual_parent(actual_accumulated)
                        expected_parent = make_expected_parent(expected_accumulated)
                        if form == "method":
                            actual_output = actual_parent.sigmoid()
                            expected_output = expected_parent.sigmoid()
                        else:
                            actual_output = torch.nn.functional.sigmoid(actual_parent)
                            expected_output = reference_torch.nn.functional.sigmoid(
                                expected_parent
                            )
                        (actual_output * actual_weights).sum().backward()
                        (expected_output * expected_weights).sum().backward()
                    self.assert_tensor_matches(
                        actual_accumulated.grad,
                        expected_accumulated.grad,
                        case=(case, form, "accumulated VJP"),
                        exact_bits=True,
                    )

                    for empty_shape in (
                        (0, 1, 3),
                        (1, 0, 3),
                        (2, 3, 0),
                        (0, 0, 0),
                    ):
                        actual_empty = torch.zeros(
                            empty_shape, requires_grad=True
                        )
                        expected_empty = reference_torch.zeros(
                            empty_shape,
                            dtype=reference_torch.float32,
                            requires_grad=True,
                        )
                        actual_empty_parent = make_actual_parent(actual_empty)
                        expected_empty_parent = make_expected_parent(expected_empty)
                        self.assert_tensor_matches(
                            actual_empty_parent,
                            expected_empty_parent,
                            case=(case, form, "empty parent", empty_shape),
                            exact_bits=True,
                        )
                        self.assertFalse(actual_empty_parent.is_set_to(actual_empty))
                        self.assertFalse(
                            expected_empty_parent.is_set_to(expected_empty)
                        )
                        if form == "method":
                            actual_empty_output = actual_empty_parent.sigmoid()
                            expected_empty_output = expected_empty_parent.sigmoid()
                        else:
                            actual_empty_output = torch.nn.functional.sigmoid(
                                actual_empty_parent
                            )
                            expected_empty_output = (
                                reference_torch.nn.functional.sigmoid(
                                    expected_empty_parent
                                )
                            )
                        self.assert_tensor_matches(
                            actual_empty_output,
                            expected_empty_output,
                            case=(case, form, "empty sigmoid", empty_shape),
                            exact_bits=True,
                        )
                        self.assertEqual(
                            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                                actual_empty_output
                            ),
                            f", grad_fn=<{type(expected_empty_output.grad_fn).__name__}>",
                        )
                        actual_empty_loss = actual_empty_output.sum()
                        expected_empty_loss = expected_empty_output.sum()
                        actual_empty_loss.backward()
                        expected_empty_loss.backward()
                        self.assert_tensor_matches(
                            actual_empty.grad,
                            expected_empty.grad,
                            case=(case, form, "empty VJP", empty_shape),
                            exact_bits=True,
                        )
                        self.assertEqual(
                            self.error(actual_empty_loss.backward),
                            self.error(expected_empty_loss.backward),
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
        actual_output = actual_leaf.sigmoid()
        expected_output = expected_leaf.sigmoid()

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-one forward",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).view(np.uint32), AUTOGRAD_OUTPUT_BITS
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
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
            (actual_accumulated.sigmoid() * actual_weights).sum().backward()
            (expected_accumulated.sigmoid() * expected_weights).sum().backward()
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

        actual_empty = torch.tensor([], requires_grad=True)
        expected_empty = reference_torch.tensor(
            [], dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_output = actual_empty.sigmoid()
        expected_empty_output = expected_empty.sigmoid()
        self.assert_tensor_matches(
            actual_empty_output,
            expected_empty_output,
            case="empty forward",
            exact_bits=True,
        )
        self.assertEqual(type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_empty_output),
            ", grad_fn=<SigmoidBackward0>",
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

    def test_rank_two_weighted_autograd_empty_shapes_and_graph_lifetime_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 4).tolist()
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 4).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(2, 4), dtype=reference_torch.float32
        )
        actual_output = actual_leaf.sigmoid()
        expected_output = expected_leaf.sigmoid()

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-two forward",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-two weighted gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
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
        for _ in range(2):
            (actual_accumulated.sigmoid() * actual_weights).sum().backward()
            (expected_accumulated.sigmoid() * expected_weights).sum().backward()
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
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = actual_empty.sigmoid()
            expected_empty_output = expected_empty.sigmoid()
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-two forward", shape),
                exact_bits=True,
            )
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            self.assertEqual(
                torch._C._nn_functional_dropout_tensor_autograd_suffix(
                    actual_empty_output
                ),
                ", grad_fn=<SigmoidBackward0>",
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-two gradient", shape),
                exact_bits=True,
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

    def test_rank_three_singletons_empty_shapes_and_autograd_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(2, 1, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(2, 1, 4).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(2, 1, 4), dtype=reference_torch.float32
        )
        actual_output = actual_leaf.sigmoid()
        expected_output = expected_leaf.sigmoid()

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-three singleton forward",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-three weighted gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (actual_accumulated.sigmoid() * actual_weights).sum().backward()
            (expected_accumulated.sigmoid() * expected_weights).sum().backward()
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

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_composed.sigmoid().sin().sum().backward()
        expected_composed.sigmoid().sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-three composition gradient",
        )

        for shape in ((0, 1, 3), (1, 0, 3), (2, 3, 0), (0, 0, 0)):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = actual_empty.sigmoid()
            expected_empty_output = expected_empty.sigmoid()
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-three forward", shape),
                exact_bits=True,
            )
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-three gradient", shape),
                exact_bits=True,
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

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

    def test_rank_four_singletons_empty_shapes_and_autograd_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 4), dtype=reference_torch.float32
        )
        actual_output = actual_leaf.sigmoid()
        expected_output = expected_leaf.sigmoid()

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-four singleton forward",
            exact_bits=True,
        )
        self.assertFalse(actual_output.is_set_to(actual_leaf))
        self.assertFalse(expected_output.is_set_to(expected_leaf))
        self.assertNotEqual(actual_output.data_ptr(), actual_leaf.data_ptr())
        self.assertNotEqual(expected_output.data_ptr(), expected_leaf.data_ptr())
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-four weighted gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (actual_accumulated.sigmoid() * actual_weights).sum().backward()
            (expected_accumulated.sigmoid() * expected_weights).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-four accumulated gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_composed.sigmoid().sin().sum().backward()
        expected_composed.sigmoid().sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-four composition gradient",
        )

        for shape in (
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (1, 2, 0, 3),
            (1, 2, 3, 0),
            (0, 0, 0, 0),
        ):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = actual_empty.sigmoid()
            expected_empty_output = expected_empty.sigmoid()
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-four forward", shape),
                exact_bits=True,
            )
            self.assertFalse(actual_empty_output.is_set_to(actual_empty))
            self.assertFalse(expected_empty_output.is_set_to(expected_empty))
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            self.assertEqual(
                torch._C._nn_functional_dropout_tensor_autograd_suffix(
                    actual_empty_output
                ),
                ", grad_fn=<SigmoidBackward0>",
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-four gradient", shape),
                exact_bits=True,
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

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

    def test_rank_five_ncdhw_singletons_empty_shapes_and_autograd_match_pytorch_2_13(
        self,
    ):
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(1, 2, 1, 1, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 1, 4).tolist()
        )
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(1, 2, 1, 1, 4),
            dtype=reference_torch.float32,
        )
        actual_output = actual_leaf.sigmoid()
        expected_output = expected_leaf.sigmoid()

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-five NCDHW singleton forward",
            exact_bits=True,
        )
        self.assertFalse(actual_output.is_set_to(actual_leaf))
        self.assertFalse(expected_output.is_set_to(expected_leaf))
        self.assertNotEqual(actual_output.data_ptr(), actual_leaf.data_ptr())
        self.assertNotEqual(expected_output.data_ptr(), expected_leaf.data_ptr())
        np.testing.assert_array_equal(
            self.tensor_values(actual_output).reshape(-1).view(np.uint32),
            AUTOGRAD_OUTPUT_BITS,
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-five weighted gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad).reshape(-1).view(np.uint32),
            AUTOGRAD_GRADIENT_BITS,
        )
        actual_gradient_before = self.tensor_values(actual_leaf.grad).copy()
        expected_gradient_before = self.tensor_values(expected_leaf.grad).copy()
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_leaf.grad), actual_gradient_before
        )
        np.testing.assert_array_equal(
            self.tensor_values(expected_leaf.grad), expected_gradient_before
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (actual_accumulated.sigmoid() * actual_weights).sum().backward()
            (expected_accumulated.sigmoid() * expected_weights).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-five accumulated gradient",
            exact_bits=True,
        )
        np.testing.assert_array_equal(
            self.tensor_values(actual_accumulated.grad)
            .reshape(-1)
            .view(np.uint32),
            AUTOGRAD_ACCUMULATED_GRADIENT_BITS,
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_composed.sigmoid().sin().sum().backward()
        expected_composed.sigmoid().sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-five composition gradient",
        )

        for shape in (
            (0, 1, 2, 3, 4),
            (1, 0, 2, 3, 4),
            (1, 2, 0, 3, 4),
            (1, 2, 3, 0, 4),
            (1, 2, 3, 4, 0),
            (0, 0, 0, 0, 0),
        ):
            actual_empty = torch.zeros(shape, requires_grad=True)
            expected_empty = reference_torch.zeros(
                shape, dtype=reference_torch.float32, requires_grad=True
            )
            actual_empty_output = actual_empty.sigmoid()
            expected_empty_output = expected_empty.sigmoid()
            self.assert_tensor_matches(
                actual_empty_output,
                expected_empty_output,
                case=("empty rank-five forward", shape),
                exact_bits=True,
            )
            self.assertFalse(actual_empty_output.is_set_to(actual_empty))
            self.assertFalse(expected_empty_output.is_set_to(expected_empty))
            self.assertEqual(
                type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
            )
            self.assertEqual(
                torch._C._nn_functional_dropout_tensor_autograd_suffix(
                    actual_empty_output
                ),
                ", grad_fn=<SigmoidBackward0>",
            )
            actual_empty_loss = actual_empty_output.sum()
            expected_empty_loss = expected_empty_output.sum()
            actual_empty_loss.backward()
            expected_empty_loss.backward()
            self.assert_tensor_matches(
                actual_empty.grad,
                expected_empty.grad,
                case=("empty rank-five gradient", shape),
                exact_bits=True,
            )
            self.assertEqual(
                self.error(actual_empty_loss.backward),
                self.error(expected_empty_loss.backward),
            )

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

    def test_rank_six_and_high_rank_autograd_match_pytorch_2_13(self):
        shape = (1, 2, 1, 1, 1, 4)
        values = AUTOGRAD_INPUT_BITS.view(np.float32).reshape(shape)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_weights = torch.tensor(AUTOGRAD_WEIGHTS.reshape(shape).tolist())
        expected_weights = reference_torch.tensor(
            AUTOGRAD_WEIGHTS.reshape(shape), dtype=reference_torch.float32
        )
        actual_output = actual_leaf.sigmoid()
        expected_output = expected_leaf.sigmoid()

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="rank-six singleton forward",
            exact_bits=True,
        )
        self.assertEqual(type(expected_output.grad_fn).__name__, "SigmoidBackward0")
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(actual_output),
            ", grad_fn=<SigmoidBackward0>",
        )

        actual_loss = (actual_output * actual_weights).sum()
        expected_loss = (expected_output * expected_weights).sum()
        actual_loss.backward()
        expected_loss.backward()
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="rank-six weighted gradient",
            exact_bits=True,
        )
        self.assertEqual(
            self.error(actual_loss.backward), self.error(expected_loss.backward)
        )

        actual_accumulated = torch.tensor(values.tolist(), requires_grad=True)
        expected_accumulated = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        for _ in range(2):
            (actual_accumulated.sigmoid() * actual_weights).sum().backward()
            (expected_accumulated.sigmoid() * expected_weights).sum().backward()
        self.assert_tensor_matches(
            actual_accumulated.grad,
            expected_accumulated.grad,
            case="rank-six accumulated gradient",
            exact_bits=True,
        )

        actual_composed = torch.tensor(values.tolist(), requires_grad=True)
        expected_composed = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_composed.sigmoid().sin().sum().backward()
        expected_composed.sigmoid().sin().sum().backward()
        self.assert_tensor_matches(
            actual_composed.grad,
            expected_composed.grad,
            case="rank-six composition gradient",
        )

        actual_empty = torch.zeros((1, 2, 0, 1, 1, 4), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (1, 2, 0, 1, 1, 4),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_empty_output = actual_empty.sigmoid()
        expected_empty_output = expected_empty.sigmoid()
        self.assert_tensor_matches(
            actual_empty_output,
            expected_empty_output,
            case="empty rank-six forward",
            exact_bits=True,
        )
        self.assertEqual(
            type(expected_empty_output.grad_fn).__name__, "SigmoidBackward0"
        )
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                actual_empty_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        actual_empty_loss = actual_empty_output.sum()
        expected_empty_loss = expected_empty_output.sum()
        actual_empty_loss.backward()
        expected_empty_loss.backward()
        self.assert_tensor_matches(
            actual_empty.grad,
            expected_empty.grad,
            case="empty rank-six gradient",
            exact_bits=True,
        )
        self.assertEqual(
            self.error(actual_empty_loss.backward),
            self.error(expected_empty_loss.backward),
        )

        high_rank_shape = (1,) * 65
        actual_high_rank = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        expected_high_rank = reference_torch.full(
            high_rank_shape,
            0.5,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_high_rank_output = actual_high_rank.sigmoid()
        expected_high_rank_output = expected_high_rank.sigmoid()
        self.assertEqual(actual_high_rank_output.shape, expected_high_rank_output.shape)
        self.assertEqual(
            actual_high_rank_output.stride(), expected_high_rank_output.stride()
        )
        self.assertEqual(
            np.float32(actual_high_rank_output.item()).view(np.uint32).item(),
            np.float32(expected_high_rank_output.item()).view(np.uint32).item(),
        )
        self.assertEqual(
            type(expected_high_rank_output.grad_fn).__name__, "SigmoidBackward0"
        )
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                actual_high_rank_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        actual_high_rank_output.backward()
        expected_high_rank_output.backward()
        self.assertEqual(actual_high_rank.grad.shape, expected_high_rank.grad.shape)
        self.assertEqual(actual_high_rank.grad.stride(), expected_high_rank.grad.stride())
        self.assertEqual(
            np.float32(actual_high_rank.grad.item()).view(np.uint32).item(),
            np.float32(expected_high_rank.grad.item()).view(np.uint32).item(),
        )
        self.assertEqual(
            self.error(actual_high_rank_output.backward),
            self.error(expected_high_rank_output.backward),
        )

        actual_high_rank_accumulated = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        expected_high_rank_accumulated = reference_torch.full(
            high_rank_shape,
            0.5,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        for _ in range(2):
            actual_high_rank_accumulated.sigmoid().backward()
            expected_high_rank_accumulated.sigmoid().backward()
        self.assertEqual(
            np.float32(actual_high_rank_accumulated.grad.item())
            .view(np.uint32)
            .item(),
            np.float32(expected_high_rank_accumulated.grad.item())
            .view(np.uint32)
            .item(),
        )

        actual_high_rank_composed = torch.full(
            high_rank_shape, 0.5, requires_grad=True
        )
        expected_high_rank_composed = reference_torch.full(
            high_rank_shape,
            0.5,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_high_rank_composed.sigmoid().sin().backward()
        expected_high_rank_composed.sigmoid().sin().backward()
        np.testing.assert_allclose(
            np.float32(actual_high_rank_composed.grad.item()),
            np.float32(expected_high_rank_composed.grad.item()),
            rtol=2.0e-6,
            atol=0.0,
        )

        high_rank_empty_shape = (1,) * 32 + (0,) + (1,) * 32
        actual_high_rank_empty = torch.zeros(
            high_rank_empty_shape, requires_grad=True
        )
        expected_high_rank_empty = reference_torch.zeros(
            high_rank_empty_shape,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_high_rank_empty_output = actual_high_rank_empty.sigmoid()
        expected_high_rank_empty_output = expected_high_rank_empty.sigmoid()
        self.assertEqual(
            actual_high_rank_empty_output.shape,
            expected_high_rank_empty_output.shape,
        )
        self.assertEqual(
            actual_high_rank_empty_output.stride(),
            expected_high_rank_empty_output.stride(),
        )
        self.assertEqual(actual_high_rank_empty_output.numel(), 0)
        self.assertEqual(expected_high_rank_empty_output.numel(), 0)
        self.assertEqual(
            type(expected_high_rank_empty_output.grad_fn).__name__,
            "SigmoidBackward0",
        )
        self.assertEqual(
            torch._C._nn_functional_dropout_tensor_autograd_suffix(
                actual_high_rank_empty_output
            ),
            ", grad_fn=<SigmoidBackward0>",
        )
        actual_high_rank_empty_loss = actual_high_rank_empty_output.sum()
        expected_high_rank_empty_loss = expected_high_rank_empty_output.sum()
        actual_high_rank_empty_loss.backward()
        expected_high_rank_empty_loss.backward()
        self.assertEqual(
            actual_high_rank_empty.grad.shape, expected_high_rank_empty.grad.shape
        )
        self.assertEqual(
            actual_high_rank_empty.grad.stride(),
            expected_high_rank_empty.grad.stride(),
        )
        self.assertEqual(actual_high_rank_empty.grad.numel(), 0)
        self.assertEqual(expected_high_rank_empty.grad.numel(), 0)
        self.assertEqual(
            self.error(actual_high_rank_empty_loss.backward),
            self.error(expected_high_rank_empty_loss.backward),
        )

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
        input_value = np.asarray(0x40E6_E0D6, dtype=np.uint32).view(np.float32).item()
        try:
            if fesetround(0) != 0 or fegetround() != 0:
                self.skipTest("the platform does not expose round-to-nearest as zero")

            actual_leaf = torch.tensor(input_value, requires_grad=True)
            expected_leaf = reference_torch.tensor(
                input_value,
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            actual_output = actual_leaf.sigmoid()
            expected_output = expected_leaf.sigmoid()
            saved_bits = self.tensor_values(actual_output).view(np.uint32).item()

            alternate_rounding = None
            for candidate in (0x400, 0x80_0000):
                if fesetround(candidate) != 0 or fegetround() != candidate:
                    continue
                actual_probe = torch.tensor(input_value).sigmoid()
                expected_probe = reference_torch.tensor(
                    input_value, dtype=reference_torch.float32
                ).sigmoid()
                actual_probe_bits = self.tensor_values(actual_probe).view(np.uint32).item()
                expected_probe_bits = (
                    self.tensor_values(expected_probe).view(np.uint32).item()
                )
                if actual_probe_bits == expected_probe_bits != saved_bits:
                    alternate_rounding = candidate
                    break
            if alternate_rounding is None:
                self.skipTest("native sigmoid is not sensitive to available fenv modes")

            actual_output.backward()
            expected_output.backward()
        finally:
            self.assertEqual(fesetround(original_rounding), 0)

        self.assert_tensor_matches(
            actual_output,
            expected_output,
            case="saved output forward",
            exact_bits=True,
        )
        self.assert_tensor_matches(
            actual_leaf.grad,
            expected_leaf.grad,
            case="saved output gradient",
            exact_bits=True,
        )

    def test_sigmoid_backward_node_identity_matches_pytorch_2_13(self):
        actual = torch.tensor(0.5, requires_grad=True).sigmoid()
        expected = reference_torch.tensor(
            0.5, dtype=reference_torch.float32, requires_grad=True
        ).sigmoid()
        self.assertEqual(type(expected.grad_fn).__name__, "SigmoidBackward0")
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
        raise AssertionError("Tensor.sigmoid unexpectedly accepted an invalid call")

    @staticmethod
    def signature_outcome(callable_object):
        try:
            return "signature", str(inspect.signature(callable_object))
        except Exception as error:
            return "error", type(error).__name__

    def callable_contract(self, module):
        tensor = module.tensor([1.25], dtype=module.float32)
        descriptor = inspect.getattr_static(module.Tensor, "sigmoid")
        bound = tensor.sigmoid
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
                    lambda: tensor.sigmoid(1),
                    lambda: bound(1),
                    lambda: descriptor(tensor, 1),
                    lambda: tensor.sigmoid(1, 2),
                    lambda: tensor.sigmoid(input=tensor),
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
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def top_level_callable_contract(self, module):
        function = module.sigmoid
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
            "owner_name": owner.__name__,
            "owner_qualname": owner.__qualname__,
            "owner_module": owner.__module__.replace("torch_rs._C", "torch._C"),
            "owner_path_identity": owner is module._C._VariableFunctionsClass,
            "owner_callable_identity": owner.sigmoid is function,
            "doc": function.__doc__,
            "text_signature": function.__text_signature__,
            "repr": re.sub(r"0x[0-9a-f]+", "0x...", repr(function)),
            "signature_error": signature_error,
            "all_count": module.__all__.count("sigmoid"),
            "owner_not_in_all": "_VariableFunctionsClass" not in module.__all__,
            "owner_not_top_level": not hasattr(module, "_VariableFunctionsClass"),
            "wildcard_identity": wildcard_namespace["sigmoid"] is function,
            "copy_identity": copy.copy(function) is function,
            "deepcopy_identity": copy.deepcopy(function) is function,
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

    def top_level_dispatch_observation(self, module):
        tensor = module.tensor([0.5], dtype=module.float32)
        tracked = module.tensor([0.5], dtype=module.float32, requires_grad=True)
        destination = module.tensor([0.0], dtype=module.float32)
        function = module.sigmoid
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        mode_calls = (
            (lambda: function(tensor), None),
            (lambda: function(input=tensor), ("input",)),
            (lambda: function(x=tensor), ("x",)),
            (lambda: function(tensor, out=None), ("out",)),
            (lambda: function(input=tracked, out=destination), ("input", "out")),
        )
        for call, keyword_names in mode_calls:
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is function,
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
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is function,
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

        forward_order = []

        class ForwardingMode(module.overrides.TorchFunctionMode):
            def __init__(self, label):
                self.label = label

            def __torch_function__(self, func, types, args=(), kwargs=None):
                forward_order.append(self.label)
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
            forward_order,
            tuple(self.tensor_values(forwarded).reshape(-1)),
            fallback_result is marker,
            len(declining_mode.calls),
            fallback_events,
            invalid_observations,
        )

    def test_top_level_torch_function_dispatch_matches_pytorch_2_13(self):
        self.assertEqual(
            self.top_level_dispatch_observation(torch),
            self.top_level_dispatch_observation(reference_torch),
        )

    @staticmethod
    def mode_dispatch_observation(module_name):
        source = r'''
import importlib
import inspect
import json
import sys

module = importlib.import_module(MODULE)
tensor = module.tensor([1.25], dtype=module.float32)
descriptor = inspect.getattr_static(module.Tensor, "sigmoid")
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
    intercepted = tensor.sigmoid()
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
        forwarded = tensor.sigmoid()

sys.setrecursionlimit(80)
declining = RecordingMode(NotImplemented)
try:
    with declining:
        tensor.sigmoid()
except Exception as error:
    declining_error = [type(error).__name__, str(error)]
else:
    declining_error = None

invalid = RecordingMode(marker)
try:
    with invalid:
        tensor.sigmoid(1)
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

    def test_scalar_autograd_modes_and_unsupported_boundaries_remain_explicit(self):
        actual_scalar = torch.tensor(0.5, requires_grad=True)
        expected_scalar = reference_torch.tensor(
            0.5, dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_no_grad_scalar = actual_scalar.sigmoid()
        with reference_torch.no_grad():
            expected_no_grad_scalar = expected_scalar.sigmoid()
        self.assert_tensor_matches(
            actual_no_grad_scalar,
            expected_no_grad_scalar,
            case="scalar no_grad",
            exact_bits=True,
        )
        self.assert_tensor_matches(
            actual_scalar.detach().sigmoid(),
            expected_scalar.detach().sigmoid(),
            case="scalar detached",
            exact_bits=True,
        )

        higher_order = torch.tensor([[0.25, -0.25]], requires_grad=True)
        loss = higher_order.sigmoid().sum()
        with self.assertRaisesRegex(
            NotImplementedError,
            r"^torch_rs\.Tensor\.backward does not support create_graph=True$",
        ):
            loss.backward(create_graph=True)
        self.assertIsNone(higher_order.grad)
        loss.backward()
        self.assertIsNotNone(higher_order.grad)

        message = r"^sigmoid\(\): autograd recording is not supported$"
        for bits in (0x7F80_0000, 0xFF80_0000, 0x7FC1_2345, 0xFFC5_4321):
            value = np.asarray(bits, dtype=np.uint32).view(np.float32).item()
            actual = torch.tensor(value, requires_grad=True)
            with self.subTest(nonfinite=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual.sigmoid()
            self.assertIsNone(actual.grad)
            actual.sum().backward()
            self.assertEqual(actual.grad.item(), 1.0)

            actual_nonleaf_base = torch.tensor(
                [0.5, 0.25], requires_grad=True
            )
            actual_nonleaf = actual_nonleaf_base + value
            for call in (
                actual_nonleaf.sigmoid,
                lambda: torch.nn.functional.sigmoid(actual_nonleaf),
            ):
                with self.subTest(
                    nonfinite_rank_one_nonleaf=f"0x{bits:08x}", call=call
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual_nonleaf_base.grad)
            actual_nonleaf.sum().backward()
            self.assertEqual(actual_nonleaf_base.grad.tolist(), [1.0, 1.0])

            actual_matrix_nonleaf_base = torch.tensor(
                [[0.5, 0.25]], requires_grad=True
            )
            actual_matrix_nonleaf = actual_matrix_nonleaf_base + value
            for call in (
                actual_matrix_nonleaf.sigmoid,
                lambda: torch.nn.functional.sigmoid(actual_matrix_nonleaf),
            ):
                with self.subTest(
                    nonfinite_rank_two_nonleaf=f"0x{bits:08x}", call=call
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual_matrix_nonleaf_base.grad)
            actual_matrix_nonleaf.sum().backward()
            self.assertEqual(
                actual_matrix_nonleaf_base.grad.tolist(), [[1.0, 1.0]]
            )

            actual_rank_three_nonleaf_base = torch.tensor(
                [[[0.5, 0.25]]], requires_grad=True
            )
            actual_rank_three_nonleaf = actual_rank_three_nonleaf_base + value
            for call in (
                actual_rank_three_nonleaf.sigmoid,
                lambda: torch.nn.functional.sigmoid(actual_rank_three_nonleaf),
            ):
                with self.subTest(
                    nonfinite_rank_three_nonleaf=f"0x{bits:08x}", call=call
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        call()
            self.assertIsNone(actual_rank_three_nonleaf_base.grad)
            actual_rank_three_nonleaf.sum().backward()
            self.assertEqual(
                actual_rank_three_nonleaf_base.grad.tolist(), [[[1.0, 1.0]]]
            )

            actual_vector = torch.tensor([0.5, value], requires_grad=True)
            with self.subTest(nonfinite_vector=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_vector.sigmoid()
            self.assertIsNone(actual_vector.grad)
            actual_vector.sum().backward()
            self.assertEqual(actual_vector.grad.tolist(), [1.0, 1.0])

            actual_matrix = torch.tensor([[0.5, value]], requires_grad=True)
            with self.subTest(nonfinite_matrix=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_matrix.sigmoid()
            self.assertIsNone(actual_matrix.grad)
            actual_matrix.sum().backward()
            self.assertEqual(actual_matrix.grad.tolist(), [[1.0, 1.0]])

            actual_rank_three = torch.tensor([[[0.5, value]]], requires_grad=True)
            with self.subTest(nonfinite_rank_three=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_rank_three.sigmoid()
            self.assertIsNone(actual_rank_three.grad)
            actual_rank_three.sum().backward()
            self.assertEqual(actual_rank_three.grad.tolist(), [[[1.0, 1.0]]])

            actual_rank_four = torch.tensor([[[[0.5, value]]]], requires_grad=True)
            with self.subTest(nonfinite_rank_four=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_rank_four.sigmoid()
            self.assertIsNone(actual_rank_four.grad)
            actual_rank_four.sum().backward()
            self.assertEqual(actual_rank_four.grad.tolist(), [[[[1.0, 1.0]]]])

            actual_rank_five = torch.tensor(
                [[[[[0.5, value]]]]], requires_grad=True
            )
            with self.subTest(nonfinite_rank_five=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_rank_five.sigmoid()
            self.assertIsNone(actual_rank_five.grad)
            actual_rank_five.sum().backward()
            self.assertEqual(actual_rank_five.grad.tolist(), [[[[[1.0, 1.0]]]]])

            actual_rank_six = torch.tensor(
                [[[[[[0.5, value]]]]]], requires_grad=True
            )
            with self.subTest(nonfinite_rank_six=f"0x{bits:08x}"):
                with self.assertRaisesRegex(RuntimeError, message):
                    actual_rank_six.sigmoid()
            self.assertIsNone(actual_rank_six.grad)
            actual_rank_six.sum().backward()
            self.assertEqual(actual_rank_six.grad.tolist(), [[[[[[1.0, 1.0]]]]]])

        actual_high_rank_nonfinite = torch.full(
            (1,) * 65, float("inf"), requires_grad=True
        )
        with self.assertRaisesRegex(RuntimeError, message):
            actual_high_rank_nonfinite.sigmoid()
        self.assertIsNone(actual_high_rank_nonfinite.grad)
        actual_high_rank_nonfinite.sum().backward()
        self.assertEqual(actual_high_rank_nonfinite.grad.item(), 1.0)

        actual_leaf = torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            requires_grad=True,
        )
        expected_leaf = reference_torch.tensor(
            np.linspace(-3.75, 3.75, 24, dtype=np.float32).reshape(2, 3, 4),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_input = actual_leaf.transpose(0, 2)[1]
        expected_input = expected_leaf.transpose(0, 2)[1]

        with self.assertRaisesRegex(RuntimeError, message):
            actual_input.sigmoid()
        self.assertTrue(expected_input.sigmoid().requires_grad)

        actual_view_base = torch.tensor([0.5], requires_grad=True)
        actual_view = actual_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            actual_view.sigmoid()
        actual_view.backward()
        self.assertEqual(actual_view_base.grad.tolist(), [1.0])

        actual_full_vector_view_base = torch.tensor(
            [0.5, -0.5], requires_grad=True
        )
        actual_full_vector_view = actual_full_vector_view_base.view((2,))
        self.assertTrue(
            actual_full_vector_view.is_set_to(actual_full_vector_view_base)
        )
        for call in (
            actual_full_vector_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(actual_full_vector_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_full_vector_view.sum().backward()
        self.assertEqual(actual_full_vector_view_base.grad.tolist(), [1.0, 1.0])

        actual_full_matrix_view_base = torch.tensor(
            [[0.5, -0.5], [1.0, -1.0]], requires_grad=True
        )
        actual_full_matrix_view = actual_full_matrix_view_base.view((2, 2))
        self.assertTrue(
            actual_full_matrix_view.is_set_to(actual_full_matrix_view_base)
        )
        for call in (
            actual_full_matrix_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(actual_full_matrix_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_full_matrix_view.sum().backward()
        self.assertEqual(
            actual_full_matrix_view_base.grad.tolist(),
            [[1.0, 1.0], [1.0, 1.0]],
        )

        actual_full_rank_three_view_base = torch.tensor(
            [[[0.5, -0.5]], [[1.0, -1.0]]], requires_grad=True
        )
        actual_full_rank_three_view = actual_full_rank_three_view_base.view(
            (2, 1, 2)
        )
        self.assertTrue(
            actual_full_rank_three_view.is_set_to(
                actual_full_rank_three_view_base
            )
        )
        for call in (
            actual_full_rank_three_view.sigmoid,
            lambda: torch.nn.functional.sigmoid(actual_full_rank_three_view),
        ):
            with self.assertRaisesRegex(RuntimeError, message):
                call()
        actual_full_rank_three_view.sum().backward()
        self.assertEqual(
            actual_full_rank_three_view_base.grad.tolist(),
            [[[1.0, 1.0]], [[1.0, 1.0]]],
        )

        actual_vector_view_base = torch.tensor(
            [[0.5, -1.0], [2.0, -3.0]], requires_grad=True
        )
        actual_vector_view = actual_vector_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            actual_vector_view.sigmoid()
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
        with self.assertRaisesRegex(RuntimeError, message):
            actual_matrix_view.sigmoid()
        actual_matrix_view.sum().backward()
        self.assertEqual(
            actual_matrix_view_base.grad.tolist(),
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
        )

        actual_rank_four_view_base = torch.tensor(
            [[[[[0.5, -1.0]]]], [[[[2.0, -3.0]]]]], requires_grad=True
        )
        actual_rank_four_view = actual_rank_four_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            actual_rank_four_view.sigmoid()
        actual_rank_four_view.sum().backward()
        self.assertEqual(
            actual_rank_four_view_base.grad.tolist(),
            [[[[[1.0, 1.0]]]], [[[[0.0, 0.0]]]]],
        )

        actual_rank_five_view_base = torch.tensor(
            [[[[[[0.5, -1.0]]]]], [[[[[2.0, -3.0]]]]]], requires_grad=True
        )
        actual_rank_five_view = actual_rank_five_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            actual_rank_five_view.sigmoid()
        actual_rank_five_view.sum().backward()
        self.assertEqual(
            actual_rank_five_view_base.grad.tolist(),
            [[[[[[1.0, 1.0]]]]], [[[[[0.0, 0.0]]]]]],
        )

        actual_rank_six_view_base = torch.full(
            (2,) + (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        actual_rank_six_view = actual_rank_six_view_base[0]
        with self.assertRaisesRegex(RuntimeError, message):
            actual_rank_six_view.sigmoid()
        actual_rank_six_view.sum().backward()
        self.assertEqual(actual_rank_six_view_base.grad.sum().item(), 2.0)

        actual_high_rank_view_base = torch.full(
            (2,) + (1,) * 65, 0.5, requires_grad=True
        )
        actual_high_rank_view = actual_high_rank_view_base[0]
        self.assertEqual(actual_high_rank_view.shape, (1,) * 65)
        with self.assertRaisesRegex(RuntimeError, message):
            actual_high_rank_view.sigmoid()
        actual_high_rank_view.backward()
        self.assertEqual(actual_high_rank_view_base.grad.sum().item(), 1.0)

        actual_rank_four_nonleaf_base = torch.tensor(
            [[[[0.5, -0.5]]]], requires_grad=True
        )
        actual_rank_four_nonleaf = actual_rank_four_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            actual_rank_four_nonleaf.sigmoid()
        actual_rank_four_nonleaf.sum().backward()
        self.assertIsNotNone(actual_rank_four_nonleaf_base.grad)

        actual_rank_five_nonleaf_base = torch.tensor(
            [[[[[0.5, -0.5]]]]], requires_grad=True
        )
        actual_rank_five_nonleaf = actual_rank_five_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            actual_rank_five_nonleaf.sigmoid()
        actual_rank_five_nonleaf.sum().backward()
        self.assertIsNotNone(actual_rank_five_nonleaf_base.grad)

        actual_rank_six_nonleaf_base = torch.full(
            (1,) * 5 + (2,), 0.5, requires_grad=True
        )
        actual_rank_six_nonleaf = actual_rank_six_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            actual_rank_six_nonleaf.sigmoid()
        actual_rank_six_nonleaf.sum().backward()
        self.assertIsNotNone(actual_rank_six_nonleaf_base.grad)

        actual_high_rank_nonleaf_base = torch.full(
            (1,) * 65, 0.5, requires_grad=True
        )
        actual_high_rank_nonleaf = actual_high_rank_nonleaf_base.sin()
        with self.assertRaisesRegex(RuntimeError, message):
            actual_high_rank_nonleaf.sigmoid()
        actual_high_rank_nonleaf.backward()
        self.assertIsNotNone(actual_high_rank_nonleaf_base.grad)

        empty_view_base = torch.zeros((1, 0), requires_grad=True)
        with torch.no_grad():
            empty_view = empty_view_base[0]
        self.assertTrue(empty_view.requires_grad)
        self.assertTrue(empty_view.is_leaf)
        with self.assertRaisesRegex(RuntimeError, message):
            empty_view.sigmoid()

        with torch.no_grad():
            actual_no_grad = actual_input.sigmoid()
        with reference_torch.no_grad():
            expected_no_grad = expected_input.sigmoid()
        self.assert_tensor_matches(actual_no_grad, expected_no_grad, case="no_grad")

        actual_detached = actual_input.detach().sigmoid()
        expected_detached = expected_input.detach().sigmoid()
        self.assert_tensor_matches(actual_detached, expected_detached, case="detached")

    def test_top_level_is_supported_and_inplace_boundary_remains_unsupported(self):
        self.assertTrue(hasattr(torch, "sigmoid"))
        self.assertTrue(hasattr(reference_torch, "sigmoid"))
        self.assertEqual(
            torch.__all__.count("sigmoid"),
            reference_torch.__all__.count("sigmoid"),
        )
        self.assertTrue(hasattr(torch.nn.functional, "sigmoid"))
        self.assertTrue(hasattr(reference_torch.nn.functional, "sigmoid"))
        self.assertFalse(hasattr(torch.Tensor, "sigmoid_"))
        self.assertTrue(hasattr(reference_torch.Tensor, "sigmoid_"))
        self.assertFalse(hasattr(torch, "sigmoid_"))
        self.assertNotIn("sigmoid_", torch.__all__)

        actual = torch.tensor([0.5, -1.0], requires_grad=True)
        expected = reference_torch.tensor(
            [0.5, -1.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_out = torch.tensor([17.0, 19.0])
        expected_out = reference_torch.tensor([17.0, 19.0])
        expected_plain = reference_torch.tensor(
            [0.5, -1.0], dtype=reference_torch.float32
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^sigmoid\(\): the 'out' argument is not supported$",
        ):
            torch.sigmoid(actual, out=actual_out)
        expected_result = reference_torch.sigmoid(expected_plain, out=expected_out)
        self.assertEqual(actual_out.tolist(), [17.0, 19.0])
        self.assertIs(expected_result, expected_out)
        self.assert_tensor_matches(
            actual.detach().sigmoid(),
            expected_result,
            case="unsupported concrete out still computes same values",
        )

        actual_before = self.tensor_values(actual).copy()
        expected_before = self.tensor_values(expected).copy()
        with self.assertRaises(AttributeError):
            actual.sigmoid_()
        with self.assertRaises(RuntimeError):
            expected.sigmoid_()
        np.testing.assert_array_equal(self.tensor_values(actual), actual_before)
        np.testing.assert_array_equal(self.tensor_values(expected), expected_before)
        self.assertIsNone(actual.grad)
        actual.sigmoid().sum().backward()
        self.assertIsNotNone(actual.grad)


if __name__ == "__main__":
    unittest.main()
