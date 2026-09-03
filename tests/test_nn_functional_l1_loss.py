import importlib
import inspect
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalL1LossTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    @classmethod
    def tensor_state(cls, tensor):
        return (
            tensor.shape,
            tensor.stride(),
            tensor.storage_offset(),
            tensor.data_ptr(),
            cls.tensor_bits(tensor).copy(),
        )

    @classmethod
    def expected_l1_bits(cls, input, target):
        expected = (input - target).abs()
        expected_bits = cls.tensor_bits(expected).copy()
        target_bits = cls.tensor_bits(target)
        target_nan = (target_bits & 0x7FFF_FFFF) > 0x7F80_0000
        expected_bits[target_nan] = (
            target_bits[target_nan] | 0x0040_0000
        ) & 0x7FFF_FFFF
        return expected, expected_bits

    def assert_matches_composition(
        self,
        actual,
        expected,
        *,
        case,
        expected_stride=None,
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(
                actual.stride(),
                expected.stride() if expected_stride is None else expected_stride,
            )
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                self.tensor_bits(actual),
                self.tensor_bits(expected),
            )

    def layout_cases(self):
        offset_input_base = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist()
        )
        offset_target_base = torch.tensor(
            np.linspace(-3.0, 4.0, 48, dtype=np.float32)
            .reshape(2, 2, 3, 4)
            .tolist()
        )
        noncontiguous_input = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
        ).transpose(1, 2)
        noncontiguous_target = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32)
            .reshape(2, 4, 3)
            .tolist()
        ).transpose(1, 2)
        mixed_layout_target = torch.tensor(
            np.linspace(3.0, -3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        offset_strided_input = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist()
        )[1].transpose(1, 2)
        offset_strided_target = torch.tensor(
            np.linspace(5.0, -5.0, 48, dtype=np.float32)
            .reshape(2, 2, 4, 3)
            .tolist()
        )[1].transpose(1, 2)
        channels_last_input = offset_input_base.contiguous(
            memory_format=torch.channels_last
        )
        channels_last_target = offset_target_base.contiguous(
            memory_format=torch.channels_last
        )
        empty_input = torch.zeros((2, 0, 3)).transpose(0, 2)
        empty_target = torch.ones((2, 0, 3)).transpose(0, 2)
        mixed_singleton_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist()
        )
        mixed_singleton_target = torch.tensor(
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist()
        ).permute(2, 1, 0)
        same = torch.tensor([[1.0, -2.0], [3.0, -4.0]])

        return (
            ("scalar", torch.tensor(-0.0), torch.tensor(2.5)),
            ("empty", empty_input, empty_target),
            ("offset", offset_input_base[1], offset_target_base[0]),
            (
                "matching noncontiguous",
                noncontiguous_input,
                noncontiguous_target,
            ),
            ("mixed noncontiguous", noncontiguous_input, mixed_layout_target),
            (
                "offset noncontiguous",
                offset_strided_input,
                offset_strided_target,
            ),
            ("channels last", channels_last_input, channels_last_target),
            (
                "mixed singleton strides",
                mixed_singleton_input,
                mixed_singleton_target,
            ),
            ("same operand", same, same),
        )

    def matching_dense_cases(self):
        input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
            ],
            dtype=np.uint32,
        )
        target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0x8000_0001,
                0x0000_0001,
                0x807F_FFFF,
                0x007F_FFFF,
                0x8080_0000,
                0x0080_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF80_0000,
                0x7F80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
                0x0000_0000,
                0x8000_0000,
            ],
            dtype=np.uint32,
        )
        input_values = input_bits.view(np.float32)
        target_values = target_bits.view(np.float32)
        transposed_input = torch.tensor(memoryview(input_values)).view(3, 6)
        transposed_target = torch.tensor(memoryview(target_values)).view(3, 6)

        input_padding = np.linspace(
            -11.0, 11.0, input_values.size, dtype=np.float32
        )
        target_padding = np.linspace(
            13.0, -13.0, target_values.size, dtype=np.float32
        )
        offset_input_base = torch.tensor(
            memoryview(np.concatenate([input_padding, input_values]))
        ).view(2, 3, 6)
        offset_target_base = torch.tensor(
            memoryview(np.concatenate([target_padding, target_values]))
        ).view(2, 3, 6)
        empty_input = torch.zeros((2, 0, 3)).transpose(0, 2)
        empty_target = torch.ones((2, 0, 3)).transpose(0, 2)

        return (
            (
                "transposed edge bits",
                transposed_input.transpose(0, 1),
                transposed_target.transpose(0, 1),
            ),
            (
                "offset transposed edge bits",
                offset_input_base[1].transpose(0, 1),
                offset_target_base[1].transpose(0, 1),
            ),
            ("empty transposed", empty_input, empty_target),
        )

    def channels_last_cases(self):
        edge_input_patterns = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        edge_target_patterns = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0x8000_0001,
                0x0000_0001,
                0x807F_FFFF,
                0x007F_FFFF,
                0x8080_0000,
                0x0080_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF7F_FFFF,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
            ],
            dtype=np.uint32,
        )
        edge_input = torch.tensor(
            memoryview(np.resize(edge_input_patterns, 24).view(np.float32))
        ).view(2, 3, 2, 2)
        edge_target = torch.tensor(
            memoryview(np.resize(edge_target_patterns, 24).view(np.float32))
        ).view(2, 3, 2, 2)
        singleton_input = torch.tensor(
            np.linspace(-3.0, 4.0, 2 * 3 * 1 * 4, dtype=np.float32)
            .reshape(2, 3, 1, 4)
            .tolist()
        )
        singleton_target = torch.tensor(
            np.linspace(5.0, -7.0, 2 * 3 * 1 * 4, dtype=np.float32)
            .reshape(2, 3, 1, 4)
            .tolist()
        )
        empty_input = torch.zeros((2, 3, 0, 5))
        empty_target = torch.ones((2, 3, 0, 5))

        return (
            (
                "edge bits",
                edge_input.contiguous(memory_format=torch.channels_last),
                edge_target.contiguous(memory_format=torch.channels_last),
            ),
            (
                "singleton",
                singleton_input.contiguous(memory_format=torch.channels_last),
                singleton_target.contiguous(memory_format=torch.channels_last),
            ),
            (
                "empty",
                empty_input.contiguous(memory_format=torch.channels_last),
                empty_target.contiguous(memory_format=torch.channels_last),
            ),
        )

    def broadcast_cases(self):
        scalar = torch.tensor(-0.0)
        offset_scalar = torch.tensor([17.0, 0.5])[1]
        matrix = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
        )
        offset_matrix = torch.tensor(
            np.arange(12, dtype=np.float32).reshape(2, 2, 3).tolist()
        )[1]
        noncontiguous_matrix = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(3, 2).tolist()
        ).transpose(0, 1)
        empty_contiguous = torch.zeros((0, 4))
        empty_strided = torch.zeros((2, 0, 3)).transpose(0, 2)

        return (
            ("scalar target", matrix, scalar),
            ("vector target", matrix, torch.tensor([1.0, 2.0, 3.0])),
            ("column target", matrix, torch.tensor([[1.0], [2.0]])),
            ("scalar input", scalar, offset_matrix),
            ("empty scalar input", scalar, empty_contiguous),
            ("empty scalar target", empty_contiguous, scalar),
            ("noncontiguous scalar input", offset_scalar, noncontiguous_matrix),
            ("noncontiguous scalar target", noncontiguous_matrix, offset_scalar),
            (
                "empty singleton broadcast",
                empty_strided,
                torch.ones((1, 0, 1)),
            ),
        )

    def sum_reduction_cases(self):
        offset_input_base = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist()
        )
        offset_target_base = torch.tensor(
            np.linspace(-3.0, 4.0, 48, dtype=np.float32)
            .reshape(2, 2, 3, 4)
            .tolist()
        )
        transposed_input = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist()
        ).transpose(1, 2)
        transposed_target = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32)
            .reshape(2, 4, 3)
            .tolist()
        ).transpose(1, 2)
        broadcast_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist()
        )
        broadcast_target = torch.tensor([1.0, -2.0, 0.5])
        empty_input = torch.zeros((2, 0, 3)).transpose(0, 2)
        empty_target = torch.ones((2, 0, 3)).transpose(0, 2)

        return (
            ("scalar", torch.tensor(-0.0), torch.tensor(2.5), False),
            ("empty", empty_input, empty_target, False),
            ("broadcasted", broadcast_input, broadcast_target, True),
            ("offset", offset_input_base[1], offset_target_base[0], False),
            ("noncontiguous", transposed_input, transposed_target, False),
        )

    def rank_two_trailing_vector_sum_cases(self):
        edge_matrix_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        edge_vector_bits = np.asarray(
            [0x8000_0000, 0x0000_0000, 0xFF80_0000, 0x7F82_ABCD],
            dtype=np.uint32,
        )
        matrix = torch.tensor(
            np.linspace(-6.0, 8.0, 12, dtype=np.float32).reshape(3, 4).tolist()
        )
        vector = torch.tensor([1.0, -2.0, 3.5, -4.5])
        edge_matrix = torch.tensor(memoryview(edge_matrix_bits.view(np.float32))).view(2, 4)
        edge_vector = torch.tensor(memoryview(edge_vector_bits.view(np.float32)))
        empty_matrix = torch.zeros((0, 4))
        empty_vector = torch.tensor([1.0, -2.0, 3.5, -4.5])

        return (
            ("target broadcast", matrix, vector),
            ("input broadcast", vector, matrix),
            ("target broadcast signed-zero nan inf", edge_matrix, edge_vector),
            ("input broadcast signed-zero nan inf", edge_vector, edge_matrix),
            (
                "target broadcast empty leading dimension",
                empty_matrix,
                empty_vector,
            ),
            (
                "input broadcast empty leading dimension",
                empty_vector,
                empty_matrix,
            ),
        )

    @staticmethod
    def broadcast_warning(input, target):
        return (
            f"Using a target size (torch.Size({list(target.shape)})) that is "
            f"different to the input size (torch.Size({list(input.shape)})). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size."
        )

    @staticmethod
    def call(input, target, form):
        if form == "reduction keyword":
            return functional.l1_loss(input, target, reduction="none")
        if form == "legacy none keywords":
            return functional.l1_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="none",
                weight=None,
            )
        if form == "five positional":
            return functional.l1_loss(input, target, None, None, "none")
        return functional.l1_loss(input, target, None, None, "none", None)

    def test_import_signature_documentation_and_exports(self):
        imported = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn.functional import l1_loss

        self.assertIs(imported, functional)
        self.assertIs(l1_loss, functional.l1_loss)
        self.assertIs(type(l1_loss), types.FunctionType)
        self.assertEqual(l1_loss.__name__, "l1_loss")
        self.assertEqual(l1_loss.__qualname__, "l1_loss")
        self.assertEqual(l1_loss.__module__, "torch_rs.nn.functional")
        self.assertEqual(l1_loss.__defaults__, (None, None, "mean", None))
        self.assertIsNone(l1_loss.__kwdefaults__)
        self.assertFalse(hasattr(l1_loss, "__text_signature__"))
        self.assertTrue(
            l1_loss.__doc__.startswith(
                "\nl1_loss(input, target, size_average=None, reduce=None, "
                "reduction='mean', weight=None)"
            )
        )
        normalized_doc = " ".join(l1_loss.__doc__.split())
        for documented_limit in (
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "broadcastable shapes",
            "``reduction='none'``",
            "``size_average=None``",
            "``reduce=None``",
            "``weight=None``",
            "fuses same-shape row-major contiguous operands",
            "non-empty same-shape rank-4 channels-last-contiguous operands",
            "same-shape operands with identical strides and non-overlapping dense storage",
            "identical strides and non-overlapping dense storage",
            "rank-0 scalar broadcasts over row-major contiguous tensors",
            "one native absolute-difference pass",
            "subtraction and absolute-value behavior",
            "``reduction='sum'``",
            "direct fused absolute-difference scalar reduction",
            "row-major rank-2 tensor paired with a row-major rank-1 tensor",
            "other supported layouts",
            "supported full-tensor sum",
            "fresh, independent tensor",
            "size-mismatch warning",
            "Unbroadcastable shapes",
            "``reduction='mean'``",
            "legacy ``size_average``/``reduce`` behavior",
            "weights",
            "unsupported dtypes or devices",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording",
            "module loss wrappers",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)

        signature = inspect.signature(l1_loss)
        self.assertEqual(
            tuple(signature.parameters),
            ("input", "target", "size_average", "reduce", "reduction", "weight"),
        )
        self.assertIs(signature.parameters["input"].annotation, torch.Tensor)
        self.assertIs(signature.parameters["target"].annotation, torch.Tensor)
        self.assertEqual(signature.parameters["size_average"].default, None)
        self.assertEqual(signature.parameters["reduce"].default, None)
        self.assertEqual(signature.parameters["reduction"].default, "mean")
        self.assertEqual(signature.parameters["weight"].default, None)
        self.assertIs(signature.return_annotation, torch.Tensor)
        self.assertFalse(hasattr(torch, "_nn_functional_l1_loss"))
        self.assertFalse(hasattr(torch.nn, "L1Loss"))

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["l1_loss"], l1_loss)

    def test_supported_forms_match_subtraction_and_abs_composition(self):
        for case, input, target in self.layout_cases():
            difference = input - target
            expected = difference.abs()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            for form in (
                "reduction keyword",
                "legacy none keywords",
                "five positional",
                "six positional",
            ):
                actual = self.call(input, target, form)
                self.assert_matches_composition(
                    actual,
                    expected,
                    case=(case, form),
                )
                with self.subTest(case=(case, form), nonmutation=True):
                    self.assertEqual(
                        self.tensor_state(input)[:-1], input_state[:-1]
                    )
                    self.assertEqual(
                        self.tensor_state(target)[:-1], target_state[:-1]
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(input)[-1], input_state[-1]
                    )
                    np.testing.assert_array_equal(
                        self.tensor_state(target)[-1], target_state[-1]
                    )

    def test_bandwidth_sized_same_shape_contiguous_matches_composition(self):
        input_values = np.linspace(
            -1024.0,
            1024.0,
            1024 * 1024,
            dtype=np.float32,
        ).reshape(1024, 1024)
        target_values = np.linspace(
            2048.0,
            -2048.0,
            1024 * 1024,
            dtype=np.float32,
        ).reshape(1024, 1024)
        input = torch.tensor(memoryview(input_values.reshape(-1))).view(1024, 1024)
        target = torch.tensor(memoryview(target_values.reshape(-1))).view(1024, 1024)

        self.assertTrue(input.is_contiguous())
        self.assertTrue(target.is_contiguous())
        actual = functional.l1_loss(input, target, reduction="none")
        expected = (input - target).abs()

        self.assert_matches_composition(
            actual,
            expected,
            case="bandwidth-sized contiguous",
        )
        self.assertFalse(actual.is_set_to(input))
        self.assertFalse(actual.is_set_to(target))
        self.assertNotEqual(actual.data_ptr(), input.data_ptr())
        self.assertNotEqual(actual.data_ptr(), target.data_ptr())

    def test_matching_dense_cases_cover_edges_storage_sum_and_nonmutation(self):
        for case, input, target in self.matching_dense_cases():
            self.assertEqual(input.shape, target.shape)
            self.assertEqual(input.stride(), target.stride())
            if input.numel() != 0:
                self.assertFalse(input.is_contiguous())
                self.assertFalse(target.is_contiguous())
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            expected_none, expected_none_bits = self.expected_l1_bits(input, target)
            expected_values = expected_none_bits.view(np.float32).copy()
            expected_sum = torch.tensor(memoryview(expected_values)).view(
                expected_none.shape
            ).sum()

            for reduction, expected in (
                ("none", expected_none),
                ("sum", expected_sum),
            ):
                with self.subTest(case=case, reduction=reduction):
                    actual = functional.l1_loss(input, target, reduction=reduction)
                    if reduction == "none":
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertEqual(
                            actual.storage_offset(),
                            expected.storage_offset(),
                        )
                        self.assertEqual(
                            actual.is_contiguous(),
                            expected.is_contiguous(),
                        )
                        self.assertEqual(actual.requires_grad, expected.requires_grad)
                        self.assertEqual(actual.is_leaf, expected.is_leaf)
                        self.assertIs(actual.dtype, torch.float32)
                        self.assertEqual(actual.device, torch.device("cpu"))
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual),
                            expected_none_bits,
                        )
                    else:
                        self.assert_matches_composition(
                            actual,
                            expected,
                            case=(case, reduction),
                        )

                    repeated = functional.l1_loss(input, target, reduction=reduction)
                    self.assertFalse(actual.is_set_to(repeated))
                    self.assertFalse(actual.is_set_to(input))
                    self.assertFalse(actual.is_set_to(target))
                    if actual.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), target.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1],
                    input_state[-1],
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1],
                    target_state[-1],
                )

    def test_channels_last_cases_match_composition_edges_and_storage(self):
        for case, input, target in self.channels_last_cases():
            self.assertEqual(input.shape, target.shape)
            self.assertEqual(input.stride(), target.stride())
            self.assertTrue(input.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(target.is_contiguous(memory_format=torch.channels_last))
            if input.numel() != 0:
                self.assertFalse(input.is_contiguous())
                self.assertFalse(target.is_contiguous())
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            expected, expected_bits = self.expected_l1_bits(input, target)

            actual = functional.l1_loss(input, target, reduction="none")

            with self.subTest(case=case, metadata=True):
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
                self.assertEqual(
                    actual.is_contiguous(memory_format=torch.channels_last),
                    expected.is_contiguous(memory_format=torch.channels_last),
                )
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIs(actual.dtype, torch.float32)
                self.assertEqual(actual.device, torch.device("cpu"))
            with self.subTest(case=case, values=True):
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    expected_bits,
                )
            with self.subTest(case=case, storage=True):
                repeated = functional.l1_loss(input, target, reduction="none")
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                if actual.numel() != 0:
                    self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), target.data_ptr())
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1],
                    input_state[-1],
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1],
                    target_state[-1],
                )

    def test_broadcasted_inputs_match_composition_warning_and_storage(self):
        for case, input, target in self.broadcast_cases():
            difference = input - target
            expected = difference.abs()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warning_line = inspect.currentframe().f_lineno + 1
                actual = functional.l1_loss(input, target, reduction="none")

            with self.subTest(case=case, warning=True):
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    str(caught[0].message),
                    self.broadcast_warning(input, target),
                )
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, warning_line)

            self.assert_matches_composition(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    repeated = functional.l1_loss(input, target, reduction="none")
                self.assertIsNot(actual, repeated)
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                if actual.numel() != 0:
                    self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), input.data_ptr())
                    self.assertNotEqual(actual.data_ptr(), target.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1], input_state[-1]
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1], target_state[-1]
                )

    def test_sum_reduction_cases_match_composition_warning_and_storage(self):
        for case, input, target, warns in self.sum_reduction_cases():
            expected = (input - target).abs().sum()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                actual = functional.l1_loss(input, target, reduction="sum")

            self.assert_matches_composition(actual, expected, case=case)
            with self.subTest(case=case, warning=True):
                self.assertEqual(len(caught), int(warns))
                if warns:
                    self.assertIs(caught[0].category, UserWarning)
                    self.assertEqual(
                        str(caught[0].message),
                        self.broadcast_warning(input, target),
                    )
            with self.subTest(case=case, storage=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    repeated = functional.l1_loss(input, target, reduction="sum")
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1], input_state[-1]
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1], target_state[-1]
                )

    def test_rank_two_trailing_vector_sum_broadcast_matches_composition_warning_metadata_and_nonmutation(
        self,
    ):
        for case, input, target in self.rank_two_trailing_vector_sum_cases():
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected_none = functional.l1_loss(input, target, reduction="none")
            expected = expected_none.sum()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warning_line = inspect.currentframe().f_lineno + 1
                actual = functional.l1_loss(input, target, reduction="sum")

            self.assert_matches_composition(actual, expected, case=case)
            with self.subTest(case=case, warning=True):
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    str(caught[0].message),
                    self.broadcast_warning(input, target),
                )
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, warning_line)
            with self.subTest(case=case, metadata=True):
                self.assertEqual(actual.shape, ())
                self.assertEqual(actual.stride(), ())
                self.assertEqual(actual.storage_offset(), 0)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.numel(), 1)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIs(actual.dtype, torch.float32)
                self.assertEqual(actual.device, torch.device("cpu"))
            with self.subTest(case=case, storage=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    repeated = functional.l1_loss(input, target, reduction="sum")
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
            with self.subTest(case=case, none_reduction=True):
                with warnings.catch_warnings(record=True) as none_warnings:
                    warnings.simplefilter("always")
                    none_actual = functional.l1_loss(
                        input,
                        target,
                        reduction="none",
                    )
                self.assertEqual(len(none_warnings), 1)
                self.assert_matches_composition(
                    none_actual,
                    expected_none,
                    case=(case, "none"),
                )
            with self.subTest(case=case, nonmutation=True):
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1],
                    input_state[-1],
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1],
                    target_state[-1],
                )

    def test_same_shape_contiguous_sum_edges_metadata_and_nonmutation(self):
        def expected_edge_sum(input, target):
            expected_none, expected_bits = self.expected_l1_bits(input, target)
            expected_values = expected_bits.view(np.float32).copy()
            return torch.tensor(memoryview(expected_values)).view(
                expected_none.shape
            ).sum()

        edge_input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F81_2345,
            ],
            dtype=np.uint32,
        )
        edge_target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0xFF80_0000,
                0x7F80_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF86_789A,
            ],
            dtype=np.uint32,
        )
        inf_input_bits = np.asarray(
            [0x7F80_0000, 0xFF80_0000, 0x3F80_0000, 0xBF80_0000],
            dtype=np.uint32,
        )
        inf_target_bits = np.asarray(
            [0xFF80_0000, 0x7F80_0000, 0xBF80_0000, 0x3F80_0000],
            dtype=np.uint32,
        )
        edge_input = torch.tensor(memoryview(edge_input_bits.view(np.float32)))
        edge_target = torch.tensor(memoryview(edge_target_bits.view(np.float32)))
        inf_input = torch.tensor(memoryview(inf_input_bits.view(np.float32)))
        inf_target = torch.tensor(memoryview(inf_target_bits.view(np.float32)))
        large_values = np.ones(1024 * 1024, dtype=np.float32)
        large_input = torch.tensor(memoryview(large_values)).view(1024, 1024)
        large_target = torch.zeros((1024, 1024))
        mixed_values = np.full(1024, 100_000.0, dtype=np.float32)
        mixed_values[:31] = 100.0
        mixed_input = torch.tensor(memoryview(mixed_values))
        mixed_target = torch.zeros((1024,))

        cases = (
            ("scalar signed zero", torch.tensor(-0.0), torch.tensor(0.0), None, None),
            ("empty", torch.zeros((5, 0, 7)), torch.ones((5, 0, 7)), None, None),
            (
                "large contiguous",
                large_input,
                large_target,
                None,
                None,
            ),
            (
                "mixed magnitudes",
                mixed_input,
                mixed_target,
                None,
                0x4CBD_67D8,
            ),
            ("inf edge bits", inf_input, inf_target, expected_edge_sum, None),
            ("nan edge bits", edge_input, edge_target, expected_edge_sum, None),
        )
        for case, input, target, expected_factory, expected_sum_bits in cases:
            with self.subTest(case=case):
                self.assertEqual(input.shape, target.shape)
                self.assertTrue(input.is_contiguous())
                self.assertTrue(target.is_contiguous())
                input_state = self.tensor_state(input)
                target_state = self.tensor_state(target)
                expected = (
                    (input - target).abs().sum()
                    if expected_factory is None
                    else expected_factory(input, target)
                )

                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual = functional.l1_loss(input, target, reduction="sum")

                self.assertEqual(caught, [])
                self.assert_matches_composition(actual, expected, case=case)
                self.assertEqual(actual.shape, ())
                self.assertEqual(actual.stride(), ())
                self.assertEqual(actual.storage_offset(), 0)
                self.assertTrue(actual.is_contiguous())
                self.assertEqual(actual.numel(), 1)
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                if expected_sum_bits is not None:
                    self.assertEqual(
                        int(self.tensor_bits(actual)[0]),
                        expected_sum_bits,
                    )

                repeated = functional.l1_loss(input, target, reduction="sum")
                self.assertFalse(actual.is_set_to(repeated))
                self.assertFalse(actual.is_set_to(input))
                self.assertFalse(actual.is_set_to(target))
                self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
                np.testing.assert_array_equal(
                    self.tensor_state(input)[-1],
                    input_state[-1],
                )
                np.testing.assert_array_equal(
                    self.tensor_state(target)[-1],
                    target_state[-1],
                )

    def test_mixed_layout_singleton_keeps_binary_tensoriterator_stride(self):
        input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist()
        )
        target = torch.tensor(
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist()
        ).permute(2, 1, 0)

        self.assertEqual(input.stride(), (3, 3, 1))
        self.assertEqual(target.stride(), (1, 2, 2))
        difference = input - target
        expected = difference.abs()
        self.assertEqual(difference.stride(), (3, 6, 1))
        self.assertEqual(expected.stride(), (3, 3, 1))

        actual = functional.l1_loss(input, target, reduction="none")
        self.assertEqual(actual.stride(), (3, 3, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(expected),
        )

    def test_every_call_returns_fresh_independent_storage(self):
        for case, input, target in self.layout_cases():
            first = functional.l1_loss(input, target, reduction="none")
            second = functional.l1_loss(input, target, reduction="none")
            with self.subTest(case=case):
                self.assertIsNot(first, second)
                self.assertFalse(first.is_set_to(second))
                self.assertFalse(first.is_set_to(input))
                self.assertFalse(first.is_set_to(target))
                if first.numel() != 0:
                    self.assertNotEqual(first.data_ptr(), second.data_ptr())
                    self.assertNotEqual(first.data_ptr(), input.data_ptr())
                    self.assertNotEqual(first.data_ptr(), target.data_ptr())

    def test_float32_edge_values_match_kernel_composition_bits(self):
        input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0x8000_0001,
                0x0000_0001,
                0x807F_FFFF,
                0x007F_FFFF,
                0x8080_0000,
                0x0080_0000,
                0xBF80_0000,
                0x3F80_0000,
                0xFF7F_FFFF,
                0x7F7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
            ],
            dtype=np.uint32,
        )
        input = torch.tensor(memoryview(input_bits.view(np.float32))).view(3, 6)
        target = torch.tensor(memoryview(target_bits.view(np.float32))).view(3, 6)

        for case, actual_input, actual_target in (
            ("contiguous", input, target),
            ("transposed", input.transpose(0, 1), target.transpose(0, 1)),
        ):
            difference = actual_input - actual_target
            expected = difference.abs()
            expected_bits = self.tensor_bits(expected).copy()
            target_bits_for_case = self.tensor_bits(actual_target)
            target_nan = (target_bits_for_case & 0x7FFF_FFFF) > 0x7F80_0000
            expected_bits[target_nan] = (
                target_bits_for_case[target_nan] | 0x0040_0000
            ) & 0x7FFF_FFFF
            actual = functional.l1_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            with self.subTest(case=case):
                self.assertEqual(actual.stride(), expected.stride())
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    expected_bits,
                )

    def test_scalar_broadcast_float32_edges_match_kernel_composition_bits(self):
        tensor_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        contiguous_tensor = torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(2, 7)
        empty_tensor = torch.zeros((0, 7))

        for scalar_bits in (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC6_789A,
            0x7F86_789A,
        ):
            scalar_values = np.asarray([scalar_bits], dtype=np.uint32).view(np.float32)
            scalar = torch.tensor(memoryview(scalar_values))[0]
            for layout, tensor in (
                ("contiguous", contiguous_tensor),
                ("empty", empty_tensor),
                ("noncontiguous fallback", contiguous_tensor.transpose(0, 1)),
            ):
                for scalar_on_left in (True, False):
                    input, target = (
                        (scalar, tensor) if scalar_on_left else (tensor, scalar)
                    )
                    difference = input - target
                    expected = difference.abs()
                    expected_bits = self.tensor_bits(expected).copy()
                    if target.shape == ():
                        target_bits_for_case = np.full(
                            expected_bits.shape,
                            self.tensor_bits(target)[0],
                            dtype=np.uint32,
                        )
                    else:
                        target_bits_for_case = self.tensor_bits(target)
                    target_nan = (target_bits_for_case & 0x7FFF_FFFF) > 0x7F80_0000
                    expected_bits[target_nan] = (
                        target_bits_for_case[target_nan] | 0x0040_0000
                    ) & 0x7FFF_FFFF
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = functional.l1_loss(input, target, reduction="none")

                    with self.subTest(
                        layout=layout,
                        scalar_bits=hex(scalar_bits),
                        scalar_on_left=scalar_on_left,
                        warning=True,
                    ):
                        self.assertEqual(len(caught), 1)
                        self.assertIs(caught[0].category, UserWarning)
                        self.assertEqual(
                            str(caught[0].message),
                            self.broadcast_warning(input, target),
                        )

                    with self.subTest(
                        layout=layout,
                        scalar_bits=hex(scalar_bits),
                        scalar_on_left=scalar_on_left,
                    ):
                        self.assertEqual(actual.shape, expected.shape)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertEqual(
                            actual.storage_offset(),
                            expected.storage_offset(),
                        )
                        self.assertEqual(
                            actual.is_contiguous(),
                            expected.is_contiguous(),
                        )
                        self.assertEqual(actual.requires_grad, expected.requires_grad)
                        self.assertEqual(actual.is_leaf, expected.is_leaf)
                        self.assertIs(actual.dtype, torch.float32)
                        self.assertEqual(actual.device, torch.device("cpu"))
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual),
                            expected_bits,
                        )
                        if layout == "noncontiguous fallback":
                            self.assertFalse(tensor.is_contiguous())
                            self.assertEqual(actual.stride(), expected.stride())

    def test_scalar_broadcast_sum_float32_edges_match_composition_warning_metadata_and_nonmutation(
        self,
    ):
        tensor_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x007F_FFFF,
                0x807F_FFFF,
                0x0080_0000,
                0x8080_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        contiguous_tensor = torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(2, 7)
        empty_tensor = torch.zeros((0, 7))

        for scalar_bits in (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x8000_0001,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC6_789A,
            0x7F86_789A,
        ):
            scalar_values = np.asarray([scalar_bits], dtype=np.uint32).view(np.float32)
            scalar = torch.tensor(memoryview(scalar_values))[0]
            for layout, tensor in (
                ("contiguous", contiguous_tensor),
                ("empty", empty_tensor),
                ("noncontiguous fallback", contiguous_tensor.transpose(0, 1)),
            ):
                for scalar_on_left in (True, False):
                    input, target = (
                        (scalar, tensor) if scalar_on_left else (tensor, scalar)
                    )
                    input_state = self.tensor_state(input)
                    target_state = self.tensor_state(target)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        expected = functional.l1_loss(
                            input,
                            target,
                            reduction="none",
                        ).sum()

                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        actual = functional.l1_loss(input, target, reduction="sum")

                    case = (layout, hex(scalar_bits), scalar_on_left)
                    self.assert_matches_composition(actual, expected, case=case)
                    with self.subTest(case=case, warning=True):
                        self.assertEqual(len(caught), 1)
                        self.assertIs(caught[0].category, UserWarning)
                        self.assertEqual(
                            str(caught[0].message),
                            self.broadcast_warning(input, target),
                        )
                    with self.subTest(case=case, metadata=True):
                        self.assertEqual(actual.shape, ())
                        self.assertEqual(actual.stride(), ())
                        self.assertEqual(actual.storage_offset(), 0)
                        self.assertTrue(actual.is_contiguous())
                        self.assertEqual(actual.numel(), 1)
                        self.assertFalse(actual.requires_grad)
                        self.assertTrue(actual.is_leaf)
                    with self.subTest(case=case, storage=True):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            repeated = functional.l1_loss(
                                input,
                                target,
                                reduction="sum",
                            )
                        self.assertFalse(actual.is_set_to(repeated))
                        self.assertFalse(actual.is_set_to(input))
                        self.assertFalse(actual.is_set_to(target))
                        self.assertNotEqual(actual.data_ptr(), repeated.data_ptr())
                    with self.subTest(case=case, nonmutation=True):
                        self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
                        self.assertEqual(
                            self.tensor_state(target)[:-1],
                            target_state[:-1],
                        )
                        np.testing.assert_array_equal(
                            self.tensor_state(input)[-1],
                            input_state[-1],
                        )
                        np.testing.assert_array_equal(
                            self.tensor_state(target)[-1],
                            target_state[-1],
                        )
                    if layout == "noncontiguous fallback":
                        with self.subTest(case=case, fallback_layout=True):
                            self.assertFalse(tensor.is_contiguous())

    def test_requires_grad_operands_need_no_grad(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            input = torch.tensor(
                [[1.0, -2.0], [3.0, -4.0]],
                requires_grad=input_requires_grad,
            )
            target = torch.tensor(
                [[0.5, 2.0], [-3.0, 4.5]],
                requires_grad=target_requires_grad,
            )
            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                for reduction in ("none", "sum"):
                    with self.subTest(reduction=reduction):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"^l1_loss\(\): autograd recording is not supported$",
                        ):
                            functional.l1_loss(input, target, reduction=reduction)

                        with torch.no_grad():
                            expected = (input - target).abs()
                            if reduction == "sum":
                                expected = expected.sum()
                            actual = functional.l1_loss(input, target, reduction=reduction)
                        self.assert_matches_composition(
                            actual,
                            expected,
                            case=("no_grad", reduction),
                        )
                        self.assertFalse(actual.requires_grad)
                        self.assertTrue(actual.is_leaf)
                        self.assertIsNone(input.grad)
                        self.assertIsNone(target.grad)

    def test_rank_two_trailing_vector_sum_requires_grad_operands_need_no_grad(self):
        def target_broadcast(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor(
                    [0.5, -1.5, 2.5],
                    requires_grad=target_requires_grad,
                ),
            )

        def input_broadcast(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [0.5, -1.5, 2.5],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor(
                    [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                    requires_grad=target_requires_grad,
                ),
            )

        for case, factory in (
            ("target broadcast", target_broadcast),
            ("input broadcast", input_broadcast),
        ):
            for input_requires_grad, target_requires_grad in (
                (True, False),
                (False, True),
                (True, True),
            ):
                input, target = factory(input_requires_grad, target_requires_grad)
                with self.subTest(
                    case=case,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                ):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"^l1_loss\(\): autograd recording is not supported$",
                        ):
                            functional.l1_loss(input, target, reduction="sum")

                    with warnings.catch_warnings(), torch.no_grad():
                        warnings.simplefilter("ignore")
                        expected = (input - target).abs().sum()
                        actual = functional.l1_loss(input, target, reduction="sum")
                    self.assert_matches_composition(
                        actual,
                        expected,
                        case=(case, "no_grad"),
                    )
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertIsNone(input.grad)
                    self.assertIsNone(target.grad)

    def test_channels_last_requires_grad_operands_need_no_grad(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            input_base = torch.tensor(
                np.linspace(-2.0, 3.0, 2 * 3 * 2 * 4, dtype=np.float32)
                .reshape(2, 3, 2, 4)
                .tolist(),
                requires_grad=input_requires_grad,
            )
            target_base = torch.tensor(
                np.linspace(5.0, -7.0, 2 * 3 * 2 * 4, dtype=np.float32)
                .reshape(2, 3, 2, 4)
                .tolist(),
                requires_grad=target_requires_grad,
            )
            input = input_base.contiguous(memory_format=torch.channels_last)
            target = target_base.contiguous(memory_format=torch.channels_last)
            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                self.assertEqual(input.stride(), target.stride())
                self.assertTrue(input.is_contiguous(memory_format=torch.channels_last))
                self.assertTrue(target.is_contiguous(memory_format=torch.channels_last))
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^l1_loss\(\): autograd recording is not supported$",
                ):
                    functional.l1_loss(input, target, reduction="none")

                with torch.no_grad():
                    actual = functional.l1_loss(input, target, reduction="none")
                    difference = input - target
                    expected = difference.abs()
                self.assert_matches_composition(actual, expected, case="no_grad")
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIsNone(input_base.grad)
                self.assertIsNone(target_base.grad)

    def test_scalar_broadcast_requires_grad_operands_need_no_grad(self):
        def scalar_input(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(-0.5, requires_grad=input_requires_grad),
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=target_requires_grad,
                ),
            )

        def scalar_target(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor(-0.5, requires_grad=target_requires_grad),
            )

        for case, factory in (
            ("scalar input", scalar_input),
            ("scalar target", scalar_target),
        ):
            for input_requires_grad, target_requires_grad in (
                (True, False),
                (False, True),
                (True, True),
            ):
                input, target = factory(input_requires_grad, target_requires_grad)
                with self.subTest(
                    case=case,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                ):
                    for reduction in ("none", "sum"):
                        with self.subTest(reduction=reduction):
                            with self.assertWarnsRegex(
                                UserWarning,
                                "Using a target size",
                            ):
                                with self.assertRaisesRegex(
                                    RuntimeError,
                                    r"^l1_loss\(\): autograd recording is not supported$",
                                ):
                                    functional.l1_loss(
                                        input,
                                        target,
                                        reduction=reduction,
                                    )

                            with warnings.catch_warnings(), torch.no_grad():
                                warnings.simplefilter("ignore")
                                expected = (input - target).abs()
                                if reduction == "sum":
                                    expected = expected.sum()
                                actual = functional.l1_loss(
                                    input,
                                    target,
                                    reduction=reduction,
                                )
                            self.assert_matches_composition(
                                actual,
                                expected,
                                case=("no_grad", reduction),
                            )
                            self.assertFalse(actual.requires_grad)
                            self.assertTrue(actual.is_leaf)
                            self.assertIsNone(input.grad)
                            self.assertIsNone(target.grad)

    def test_unsupported_options_shapes_and_operands_are_rejected(self):
        input = torch.ones((2, 3))
        target = torch.zeros((2, 3))

        reduction_error = (
            "torch_rs.nn.functional.l1_loss only supports "
            "reduction='none' or reduction='sum'"
        )
        for reduction in ("mean", "batchmean", None, 1, object()):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(reduction_error)}$",
                ):
                    functional.l1_loss(input, target, reduction=reduction)

        legacy_error = (
            "torch_rs.nn.functional.l1_loss only supports "
            "size_average=None and reduce=None"
        )
        for legacy_arguments in (
            {"size_average": False},
            {"size_average": True},
            {"reduce": False},
            {"reduce": True},
            {"size_average": False, "reduce": False},
        ):
            for reduction in ("none", "sum"):
                with self.subTest(
                    legacy_arguments=legacy_arguments,
                    reduction=reduction,
                ):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"^{re.escape(legacy_error)}$",
                    ):
                        functional.l1_loss(
                            input,
                            target,
                            reduction=reduction,
                            **legacy_arguments,
                        )

        weight_error = "torch_rs.nn.functional.l1_loss only supports weight=None"
        for weight in (torch.ones((2, 3)), 1.0, [1.0, 1.0]):
            for reduction in ("none", "sum"):
                with self.subTest(weight=type(weight), reduction=reduction):
                    with self.assertRaisesRegex(
                        NotImplementedError,
                        f"^{re.escape(weight_error)}$",
                    ):
                        functional.l1_loss(
                            input,
                            target,
                            reduction=reduction,
                            weight=weight,
                        )

        unbroadcastable_target = torch.zeros((2, 2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaisesRegex(
                RuntimeError,
                r"^The size of tensor a \(3\) must match the size of tensor b "
                r"\(2\) at non-singleton dimension 1$",
            ):
                functional.l1_loss(
                    input,
                    unbroadcastable_target,
                    reduction="none",
                )
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(
            str(caught[0].message),
            self.broadcast_warning(input, unbroadcastable_target),
        )

        class Override:
            calls = 0

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls += 1
                return object()

        exact_tensor_error = (
            "l1_loss() only supports exact native Tensor input and target operands"
        )
        for actual_input, actual_target in (
            (Override(), target),
            (input, Override()),
            (1.0, target),
            (input, [0.0]),
        ):
            with self.subTest(
                input_type=type(actual_input),
                target_type=type(actual_target),
            ):
                with self.assertRaisesRegex(
                    TypeError,
                    f"^{re.escape(exact_tensor_error)}$",
                ):
                    functional.l1_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
                    )
        self.assertEqual(Override.calls, 0)

    def test_active_torch_function_mode_is_rejected_without_dispatch(self):
        class RecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self):
                self.calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls += 1
                return object()

        mode = RecordingMode()
        with self.assertRaisesRegex(
            TypeError,
            r"^l1_loss\(\) does not support an active TorchFunctionMode$",
        ):
            with mode:
                functional.l1_loss(
                    torch.ones((2, 3), requires_grad=True),
                    torch.zeros((3,)),
                    reduction="mean",
                )
        self.assertEqual(mode.calls, 0)

    def test_python_argument_binding_matches_the_canonical_signature(self):
        input = torch.ones((1,))
        target = torch.zeros((1,))
        cases = (
            (
                lambda: functional.l1_loss(),
                "l1_loss() missing 2 required positional arguments: 'input' and 'target'",
            ),
            (
                lambda: functional.l1_loss(input),
                "l1_loss() missing 1 required positional argument: 'target'",
            ),
            (
                lambda: functional.l1_loss(input, target, input=input),
                "l1_loss() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.l1_loss(
                    input, target, None, None, "none", None, None
                ),
                "l1_loss() takes from 2 to 6 positional arguments but 7 were given",
            ),
            (
                lambda: functional.l1_loss(
                    input, target, reduction="none", unexpected=True
                ),
                "l1_loss() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
