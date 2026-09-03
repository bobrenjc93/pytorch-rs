import importlib
import inspect
import re
import subprocess
import sys
import textwrap
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional


class FunctionalMseLossTests(unittest.TestCase):
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

    def assert_mean_scalar_matches_composition(self, actual, expected, *, case):
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
            actual_value = np.asarray(actual)
            expected_value = np.asarray(expected)
            if np.isnan(expected_value).all():
                self.assertTrue(np.isnan(actual_value).all())
            elif case[0] == "signed zero":
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    self.tensor_bits(expected),
                )
            elif case[0] == "large contiguous":
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=1e-3,
                    atol=1e-5,
                )
            else:
                np.testing.assert_array_max_ulp(
                    actual_value,
                    expected_value,
                    maxulp=2,
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
        vector = torch.tensor([1.0, 2.0, 3.0])
        column = torch.tensor([[1.0], [2.0]])
        contiguous = torch.tensor(
            np.linspace(-3.0, 4.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist()
        )
        empty_contiguous = torch.zeros((0, 4))
        offset_strided = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist()
        )[1].transpose(1, 2)
        channels_last = torch.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist()
        ).contiguous(memory_format=torch.channels_last)
        singleton_strided = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(3, 1, 2).tolist()
        ).permute(2, 1, 0)
        empty_strided = torch.zeros((2, 0, 3)).transpose(0, 2)

        return (
            ("vector target", matrix, vector),
            ("column target", matrix, column),
            ("offset vector target", offset_matrix, vector),
            ("noncontiguous vector target", noncontiguous_matrix, vector),
            ("contiguous scalar input", scalar, contiguous),
            ("contiguous scalar target", contiguous, scalar),
            ("contiguous empty scalar input", scalar, empty_contiguous),
            ("contiguous empty scalar target", empty_contiguous, scalar),
            ("offset strided scalar input", offset_scalar, offset_strided),
            ("offset strided scalar target", offset_strided, offset_scalar),
            ("channels last scalar input", scalar, channels_last),
            ("channels last scalar target", channels_last, scalar),
            ("singleton strided scalar input", scalar, singleton_strided),
            ("singleton strided scalar target", singleton_strided, scalar),
            ("empty strided scalar input", scalar, empty_strided),
            ("empty strided scalar target", empty_strided, scalar),
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

    def same_shape_contiguous_mean_cases(self):
        signed_zero_input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0x8000_0000,
            ],
            dtype=np.uint32,
        )
        signed_zero_target_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x8000_0000,
                0x0000_0000,
            ],
            dtype=np.uint32,
        )
        edge_input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        edge_target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0xFF80_0000,
                0x7F80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
            ],
            dtype=np.uint32,
        )
        large_input = np.linspace(-3.0, 5.0, 1 << 20, dtype=np.float32)
        large_target = np.linspace(4.0, -2.0, 1 << 20, dtype=np.float32)

        return (
            ("scalar", torch.tensor(-0.0), torch.tensor(2.5)),
            ("empty", torch.zeros((0, 1024)), torch.ones((0, 1024))),
            (
                "signed zero",
                torch.tensor(memoryview(signed_zero_input_bits.view(np.float32))).view(2, 2),
                torch.tensor(memoryview(signed_zero_target_bits.view(np.float32))).view(2, 2),
            ),
            (
                "nan and infinity",
                torch.tensor(memoryview(edge_input_bits.view(np.float32))).view(2, 4),
                torch.tensor(memoryview(edge_target_bits.view(np.float32))).view(2, 4),
            ),
            (
                "large contiguous",
                torch.tensor(memoryview(large_input)).view(1024, 1024),
                torch.tensor(memoryview(large_target)).view(1024, 1024),
            ),
        )

    def same_stride_noncontiguous_cases(self):
        edge_input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
                0x3F80_0000,
                0xBF80_0000,
            ],
            dtype=np.uint32,
        )
        edge_target_bits = np.asarray(
            [
                0x8000_0000,
                0x0000_0000,
                0x8000_0001,
                0x0000_0001,
                0xFF80_0000,
                0x7F80_0000,
                0xFFC6_789A,
                0x7FC2_ABCD,
                0xFF86_789A,
                0x7F82_ABCD,
                0xBF80_0000,
                0x3F80_0000,
            ],
            dtype=np.uint32,
        )
        edge_input = torch.tensor(memoryview(edge_input_bits.view(np.float32))).view(
            3, 4
        )
        edge_target = torch.tensor(memoryview(edge_target_bits.view(np.float32))).view(
            3, 4
        )
        offset_input = torch.tensor(
            np.linspace(-5.0, 5.0, 60, dtype=np.float32).reshape(3, 4, 5).tolist()
        )[1].transpose(0, 1)
        offset_target = torch.tensor(
            np.linspace(7.0, -3.0, 60, dtype=np.float32).reshape(3, 4, 5).tolist()
        )[2].transpose(0, 1)
        channels_last_input = torch.tensor(
            np.linspace(-3.0, 4.0, 2 * 3 * 5 * 7, dtype=np.float32)
            .reshape(2, 3, 5, 7)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        channels_last_target = torch.tensor(
            np.linspace(11.0, -13.0, 2 * 3 * 5 * 7, dtype=np.float32)
            .reshape(2, 3, 5, 7)
            .tolist()
        ).contiguous(memory_format=torch.channels_last)
        singleton_input = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(3, 1, 2).tolist()
        ).permute(2, 1, 0)
        singleton_target = torch.tensor(
            np.linspace(3.5, -2.5, 6, dtype=np.float32).reshape(3, 1, 2).tolist()
        ).permute(2, 1, 0)
        empty_input = torch.zeros((2, 0, 3)).transpose(0, 2)
        empty_target = torch.ones((2, 0, 3)).transpose(0, 2)

        return (
            ("transposed edge bits", edge_input.transpose(0, 1), edge_target.transpose(0, 1)),
            ("offset transposed", offset_input, offset_target),
            ("channels-last-like", channels_last_input, channels_last_target),
            ("singleton strided", singleton_input, singleton_target),
            ("empty transposed", empty_input, empty_target),
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
            return functional.mse_loss(input, target, reduction="none")
        if form == "legacy none keywords":
            return functional.mse_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="none",
                weight=None,
            )
        if form == "five positional":
            return functional.mse_loss(input, target, None, None, "none")
        return functional.mse_loss(input, target, None, None, "none", None)

    def test_import_signature_documentation_and_exports(self):
        imported = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn.functional import mse_loss

        self.assertIs(imported, functional)
        self.assertIs(mse_loss, functional.mse_loss)
        self.assertIs(type(mse_loss), types.FunctionType)
        self.assertEqual(mse_loss.__name__, "mse_loss")
        self.assertEqual(mse_loss.__qualname__, "mse_loss")
        self.assertEqual(mse_loss.__module__, "torch_rs.nn.functional")
        self.assertEqual(mse_loss.__defaults__, (None, None, "mean", None))
        self.assertIsNone(mse_loss.__kwdefaults__)
        self.assertFalse(hasattr(mse_loss, "__text_signature__"))
        self.assertTrue(
            mse_loss.__doc__.startswith(
                "\nmse_loss(input, target, size_average=None, reduce=None, "
                "reduction='mean', weight=None)"
            )
        )
        normalized_doc = " ".join(mse_loss.__doc__.split())
        for documented_limit in (
            "exact ``torch_rs.Tensor`` operands",
            "CPU ``float32`` storage",
            "broadcastable shapes",
            "``reduction='none'``",
            "``reduction='mean'``",
            "``reduction='sum'``",
            "``size_average=None``",
            "``reduce=None``",
            "``weight=None``",
            "fuses subtraction and square into one native pass",
            "direct scalar reduction",
            "supported full-tensor reduction",
            "fresh, independent tensor",
            "size-mismatch warning",
            "Unbroadcastable shapes",
            "legacy ``size_average``/``reduce`` behavior",
            "weights",
            "unsupported dtypes or devices",
            "Tensor subclasses",
            "active ``TorchFunctionMode`` contexts",
            "active autograd recording for ``reduction='none'``",
            "module loss wrappers",
            "``reduction='mean'`` or ``reduction='sum'``",
            "inside ``torch.no_grad()``",
        ):
            self.assertIn(documented_limit, normalized_doc)

        signature = inspect.signature(mse_loss)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "input",
                "target",
                "size_average",
                "reduce",
                "reduction",
                "weight",
            ),
        )
        self.assertIs(signature.parameters["input"].annotation, torch.Tensor)
        self.assertIs(signature.parameters["target"].annotation, torch.Tensor)
        self.assertEqual(signature.parameters["size_average"].default, None)
        self.assertEqual(signature.parameters["reduce"].default, None)
        self.assertEqual(signature.parameters["reduction"].default, "mean")
        self.assertEqual(signature.parameters["weight"].default, None)
        self.assertIs(signature.return_annotation, torch.Tensor)
        self.assertFalse(hasattr(torch, "_nn_functional_mse_loss"))
        self.assertFalse(hasattr(torch.nn, "MSELoss"))

        wildcard = {}
        exec("from torch_rs.nn.functional import *", wildcard)
        self.assertIs(wildcard["mse_loss"], mse_loss)

    def test_supported_forms_match_subtraction_and_square_composition(self):
        for case, input, target in self.layout_cases():
            difference = input - target
            expected = difference.square()
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
                    expected_stride=difference.stride(),
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

    def test_same_stride_noncontiguous_cases_match_composition(self):
        for case, input, target in self.same_stride_noncontiguous_cases():
            self.assertEqual(input.shape, target.shape)
            self.assertEqual(input.stride(), target.stride())
            if input.numel() != 0:
                self.assertFalse(input.is_contiguous())
                self.assertFalse(target.is_contiguous())
            difference = input - target
            expected = difference.square()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            actual = functional.mse_loss(input, target, reduction="none")
            self.assert_matches_composition(
                actual,
                expected,
                case=case,
                expected_stride=difference.stride(),
            )
            with self.subTest(case=case, storage=True):
                repeated = functional.mse_loss(input, target, reduction="none")
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

    def test_broadcasted_inputs_match_composition_warning_and_storage(self):
        for case, input, target in self.broadcast_cases():
            difference = input - target
            expected = difference.square()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warning_line = inspect.currentframe().f_lineno + 1
                actual = functional.mse_loss(input, target, reduction="none")

            with self.subTest(case=case, warning=True):
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(str(caught[0].message), self.broadcast_warning(input, target))
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, warning_line)

            self.assert_matches_composition(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    repeated = functional.mse_loss(input, target, reduction="none")
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
            expected = (input - target).square().sum()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                actual = functional.mse_loss(input, target, reduction="sum")

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
                    repeated = functional.mse_loss(input, target, reduction="sum")
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

    def test_mean_reduction_cases_match_composition_warning_storage_and_default(self):
        for case, input, target, warns in self.sum_reduction_cases():
            expected = (input - target).square().mean()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            for form, call in (
                (
                    "explicit mean",
                    lambda: functional.mse_loss(input, target, reduction="mean"),
                ),
                ("default mean", lambda: functional.mse_loss(input, target)),
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    actual = call()

                self.assert_mean_scalar_matches_composition(
                    actual, expected, case=(case, form)
                )
                with self.subTest(case=(case, form), warning=True):
                    self.assertEqual(len(caught), int(warns))
                    if warns:
                        self.assertIs(caught[0].category, UserWarning)
                        self.assertEqual(
                            str(caught[0].message),
                            self.broadcast_warning(input, target),
                        )
                with self.subTest(case=(case, form), storage=True):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        repeated = call()
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

    def test_mean_reduction_same_shape_contiguous_fast_path_cases(self):
        for case, input, target in self.same_shape_contiguous_mean_cases():
            self.assertEqual(input.shape, target.shape)
            self.assertTrue(input.is_contiguous())
            self.assertTrue(target.is_contiguous())
            expected = (input - target).square().mean()
            input_state = self.tensor_state(input)
            target_state = self.tensor_state(target)

            for form, call in (
                (
                    "explicit mean",
                    lambda: functional.mse_loss(input, target, reduction="mean"),
                ),
                ("default mean", lambda: functional.mse_loss(input, target)),
            ):
                actual = call()
                self.assert_mean_scalar_matches_composition(
                    actual, expected, case=(case, form)
                )
                with self.subTest(case=(case, form), scalar_metadata=True):
                    self.assertEqual(actual.shape, ())
                    self.assertEqual(actual.stride(), ())
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertTrue(actual.is_contiguous())
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                with self.subTest(case=(case, form), storage=True):
                    repeated = call()
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

    def test_mean_reduction_same_shape_contiguous_no_grad_leaves(self):
        input = torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]],
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.5, 2.0], [-3.0, 4.5]],
            requires_grad=True,
        )
        input_state = self.tensor_state(input)
        target_state = self.tensor_state(target)

        with torch.no_grad():
            actual = functional.mse_loss(input, target, reduction="mean")
            expected = (input - target).square().mean()

        self.assert_matches_composition(actual, expected, case="no_grad leaves")
        self.assertFalse(actual.requires_grad)
        self.assertTrue(actual.is_leaf)
        self.assertIsNone(input.grad)
        self.assertIsNone(target.grad)
        self.assertEqual(self.tensor_state(input)[:-1], input_state[:-1])
        self.assertEqual(self.tensor_state(target)[:-1], target_state[:-1])
        np.testing.assert_array_equal(self.tensor_state(input)[-1], input_state[-1])
        np.testing.assert_array_equal(self.tensor_state(target)[-1], target_state[-1])

    def test_sum_reduction_no_grad_view_repeated_backward_reports_freed_graph(self):
        leaf = torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]],
            requires_grad=True,
        )
        with torch.no_grad():
            view = leaf[0]

        self.assertTrue(view.requires_grad)
        self.assertTrue(view.is_leaf)
        loss = functional.mse_loss(view, view, reduction="sum")
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)

        loss.backward()
        self.assertIsNone(leaf.grad)
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_high_rank_scalar_broadcast_warning_is_fallible(self):
        script = textwrap.dedent(
            """\
            import os
            import resource
            import sys
            import warnings

            import torch_rs as torch
            import torch_rs.nn.functional as functional

            mode = sys.argv[1]
            rank = 750_000
            scalar = torch.tensor(0.0)
            empty = torch.zeros((0,) * rank)
            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            allowance = {"memory_error": 1, "warning": 64}[mode]
            limit = current_virtual_size + allowance * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            warnings.simplefilter("error")
            try:
                functional.mse_loss(scalar, empty, reduction="none")
            except MemoryError as error:
                if mode != "memory_error":
                    raise
                assert str(error) == "unable to allocate mse_loss broadcast warning"
            except UserWarning as warning:
                if mode != "warning":
                    raise
                message = str(warning)
                assert message.startswith("Using a target size (torch.Size([")
                assert message.endswith("Please ensure they have the same size.")
            else:
                raise AssertionError("the scalar-broadcast warning was not raised")
            """
        )
        for mode in ("memory_error", "warning"):
            with self.subTest(mode=mode):
                completed = subprocess.run(
                    [sys.executable, "-c", script, mode],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=60,
                )
                if completed.returncode == 77:
                    self.skipTest("process hard address-space limit is too low")
                self.assertEqual(
                    completed.returncode,
                    0,
                    msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
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
        self.assertEqual(difference.stride(), (3, 6, 1))
        self.assertEqual(difference.square().stride(), (3, 3, 1))

        actual = functional.mse_loss(input, target, reduction="none")
        self.assertEqual(actual.stride(), (3, 6, 1))
        np.testing.assert_array_equal(
            self.tensor_bits(actual),
            self.tensor_bits(difference.square()),
        )

    def test_broadcasted_leading_singleton_stride_matches_pytorch_mse_loss(self):
        input = torch.tensor([[0.0, 1.0, 2.0]]).transpose(0, 1)
        target = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3, 1).tolist()
        ).permute(2, 1, 0)
        difference = input - target
        expected = difference.square()

        self.assertEqual(input.shape, (3, 1))
        self.assertEqual(input.stride(), (1, 3))
        self.assertEqual(target.shape, (1, 3, 2))
        self.assertEqual(target.stride(), (1, 1, 3))
        self.assertEqual(difference.stride(), (1, 1, 3))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = functional.mse_loss(input, target, reduction="none")

        with self.subTest(warning=True):
            self.assertEqual(len(caught), 1)
            self.assertIs(caught[0].category, UserWarning)
            self.assertEqual(str(caught[0].message), self.broadcast_warning(input, target))

        self.assert_matches_composition(
            actual,
            expected,
            case="leading singleton broadcast",
            expected_stride=(3, 1, 3),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            repeated = functional.mse_loss(input, target, reduction="none")
        self.assertFalse(actual.is_set_to(repeated))
        self.assertFalse(actual.is_set_to(input))
        self.assertFalse(actual.is_set_to(target))

    def test_broadcasted_singleton_output_stride_matches_pytorch_mse_loss(self):
        input = torch.tensor([[0.0], [1.0]]).transpose(0, 1)
        target = torch.tensor([0.0, 1.0])
        expected = (input - target).square()

        self.assertEqual(input.shape, (1, 2))
        self.assertEqual(input.stride(), (1, 1))
        self.assertEqual(target.shape, (2,))
        self.assertEqual(target.stride(), (1,))
        self.assertEqual((input - target).stride(), (1, 1))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = functional.mse_loss(input, target, reduction="none")

        with self.subTest(warning=True):
            self.assertEqual(len(caught), 1)
            self.assertIs(caught[0].category, UserWarning)
            self.assertEqual(str(caught[0].message), self.broadcast_warning(input, target))

        self.assert_matches_composition(
            actual,
            expected,
            case="singleton output broadcast",
            expected_stride=(2, 1),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            repeated = functional.mse_loss(input, target, reduction="none")
        self.assertFalse(actual.is_set_to(repeated))
        self.assertFalse(actual.is_set_to(input))
        self.assertFalse(actual.is_set_to(target))

    def test_every_call_returns_fresh_independent_storage(self):
        for case, input, target in self.layout_cases():
            first = functional.mse_loss(input, target, reduction="none")
            second = functional.mse_loss(input, target, reduction="none")
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
            actual = functional.mse_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            expected = difference.square()
            with self.subTest(case=case):
                self.assertEqual(actual.stride(), difference.stride())
                np.testing.assert_array_equal(
                    self.tensor_bits(actual),
                    self.tensor_bits(expected),
                )

    def test_scalar_broadcast_float32_edges_match_kernel_composition_bits(self):
        tensor_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
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
        contiguous_tensor = torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(3, 4)

        for scalar_bits in (
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC6_789A,
            0x7F86_789A,
        ):
            scalar_values = np.asarray([scalar_bits], dtype=np.uint32).view(np.float32)
            scalar = torch.tensor(memoryview(scalar_values))[0]
            for layout, tensor in (
                ("contiguous", contiguous_tensor),
                ("noncontiguous", contiguous_tensor.transpose(0, 1)),
            ):
                for scalar_on_left in (True, False):
                    input, target = (
                        (scalar, tensor) if scalar_on_left else (tensor, scalar)
                    )
                    difference = input - target
                    expected = difference.square()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        actual = functional.mse_loss(input, target, reduction="none")
                    with self.subTest(
                        layout=layout,
                        scalar_bits=hex(scalar_bits),
                        scalar_on_left=scalar_on_left,
                    ):
                        self.assertEqual(actual.stride(), expected.stride())
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual),
                            self.tensor_bits(expected),
                        )

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
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^mse_loss\(\): autograd recording is not supported$",
                ):
                    functional.mse_loss(input, target, reduction="none")

                with torch.no_grad():
                    actual = functional.mse_loss(input, target, reduction="none")
                    difference = input - target
                    expected = difference.square()
                self.assert_matches_composition(
                    actual,
                    expected,
                    case="no_grad",
                    expected_stride=difference.stride(),
                )
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(target.grad)

    def test_sum_reduction_requires_grad_operands_match_inside_no_grad(self):
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
                with torch.no_grad():
                    actual = functional.mse_loss(input, target, reduction="sum")
                    expected = (input - target).square().sum()
                self.assert_matches_composition(actual, expected, case="sum no_grad")
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIsNone(input.grad)
                self.assertIsNone(target.grad)

    def test_mean_reduction_requires_grad_operands_match_inside_no_grad(self):
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
            for form, call in (
                (
                    "explicit mean",
                    lambda: functional.mse_loss(input, target, reduction="mean"),
                ),
                ("default mean", lambda: functional.mse_loss(input, target)),
            ):
                with self.subTest(
                    form=form,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                ):
                    with torch.no_grad():
                        actual = call()
                        expected = (input - target).square().mean()
                    self.assert_matches_composition(actual, expected, case=form)
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertIsNone(input.grad)
                    self.assertIsNone(target.grad)

    def test_sum_reduction_first_order_backward_matches_composition(self):
        def assert_gradients(actual_sources, expected_sources, *, case):
            for actual, expected in zip(actual_sources, expected_sources, strict=True):
                with self.subTest(case=case, source=actual.shape):
                    if expected.grad is None:
                        self.assertIsNone(actual.grad)
                    else:
                        self.assertEqual(actual.grad.shape, expected.grad.shape)
                        self.assertEqual(actual.grad.stride(), expected.grad.stride())
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual.grad),
                            self.tensor_bits(expected.grad),
                        )

        actual_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        actual_target = torch.tensor(
            [[0.5, 2.0, -3.0], [1.0, -1.0, 0.0]], requires_grad=True
        )
        expected_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        expected_target = torch.tensor(
            [[0.5, 2.0, -3.0], [1.0, -1.0, 0.0]], requires_grad=True
        )
        actual_loss = functional.mse_loss(
            actual_input, actual_target, reduction="sum"
        )
        expected_loss = (expected_input - expected_target).square().sum()
        self.assert_matches_composition(actual_loss, expected_loss, case="same shape")
        self.assertTrue(actual_loss.requires_grad)
        self.assertFalse(actual_loss.is_leaf)
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="same shape",
        )

        actual_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        actual_target = torch.tensor([0.5, -1.5, 2.0], requires_grad=True)
        expected_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        expected_target = torch.tensor([0.5, -1.5, 2.0], requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_loss = functional.mse_loss(
                actual_input, actual_target, reduction="sum"
            )
            expected_loss = (expected_input - expected_target).square().sum()
        self.assert_matches_composition(actual_loss, expected_loss, case="broadcast")
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="broadcast",
        )

        actual_input_base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        actual_target_base = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        expected_input_base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        expected_target_base = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        actual_input = actual_input_base[1].transpose(0, 1)
        actual_target = actual_target_base[0].transpose(0, 1)
        expected_input = expected_input_base[1].transpose(0, 1)
        expected_target = expected_target_base[0].transpose(0, 1)
        actual_loss = functional.mse_loss(
            actual_input, actual_target, reduction="sum"
        )
        expected_loss = (expected_input - expected_target).square().sum()
        self.assert_matches_composition(
            actual_loss,
            expected_loss,
            case="offset noncontiguous",
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input_base, actual_target_base),
            (expected_input_base, expected_target_base),
            case="offset noncontiguous",
        )

        actual_input = torch.zeros((2, 0, 3), requires_grad=True)
        actual_target = torch.ones((2, 0, 3), requires_grad=True)
        expected_input = torch.zeros((2, 0, 3), requires_grad=True)
        expected_target = torch.ones((2, 0, 3), requires_grad=True)
        actual_loss = functional.mse_loss(
            actual_input.transpose(0, 2),
            actual_target.transpose(0, 2),
            reduction="sum",
        )
        expected_loss = (
            expected_input.transpose(0, 2) - expected_target.transpose(0, 2)
        ).square().sum()
        self.assert_matches_composition(actual_loss, expected_loss, case="empty")
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="empty",
        )

    def test_sum_reduction_backward_edge_bits_match_pytorch_2_13_contract(self):
        input_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0x8000_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7F81_2345,
                0xFF85_4321,
            ],
            dtype=np.uint32,
        )
        target_bits = np.asarray(
            [
                0x0000_0000,
                0x8000_0000,
                0x8000_0000,
                0x0000_0000,
                0xFFC5_4321,
                0x7FC1_2345,
                0xFF85_4321,
                0x7F81_2345,
            ],
            dtype=np.uint32,
        )
        expected_input_grad_bits = np.asarray(
            [
                0x0000_0000,
                0x0000_0000,
                0x0000_0000,
                0x8000_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7FC1_2345,
                0xFFC5_4321,
            ],
            dtype=np.uint32,
        )
        expected_target_grad_bits = np.asarray(
            [
                0x0000_0000,
                0x0000_0000,
                0x8000_0000,
                0x0000_0000,
                0xFFC5_4321,
                0x7FC1_2345,
                0xFFC5_4321,
                0x7FC1_2345,
            ],
            dtype=np.uint32,
        )
        input = torch.tensor(
            memoryview(input_bits.view(np.float32)),
            requires_grad=True,
        )
        target = torch.tensor(
            memoryview(target_bits.view(np.float32)),
            requires_grad=True,
        )

        functional.mse_loss(input, target, reduction="sum").backward()

        np.testing.assert_array_equal(
            self.tensor_bits(input.grad),
            expected_input_grad_bits,
        )
        np.testing.assert_array_equal(
            self.tensor_bits(target.grad),
            expected_target_grad_bits,
        )

    def test_mean_reduction_first_order_backward_matches_composition(self):
        def assert_gradients(actual_sources, expected_sources, *, case):
            for actual, expected in zip(actual_sources, expected_sources, strict=True):
                with self.subTest(case=case, source=actual.shape):
                    if expected.grad is None:
                        self.assertIsNone(actual.grad)
                    else:
                        self.assertEqual(actual.grad.shape, expected.grad.shape)
                        self.assertEqual(actual.grad.stride(), expected.grad.stride())
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual.grad),
                            self.tensor_bits(expected.grad),
                        )

        actual_input = torch.tensor(1.0, requires_grad=True)
        actual_target = torch.tensor(-2.0, requires_grad=True)
        expected_input = torch.tensor(1.0, requires_grad=True)
        expected_target = torch.tensor(-2.0, requires_grad=True)
        actual_loss = functional.mse_loss(actual_input, actual_target)
        expected_loss = (expected_input - expected_target).square().mean()
        self.assert_matches_composition(actual_loss, expected_loss, case="scalar")
        self.assertTrue(actual_loss.requires_grad)
        self.assertFalse(actual_loss.is_leaf)
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="scalar",
        )

        actual_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        actual_target = torch.tensor([0.5, -1.5, 2.0], requires_grad=True)
        expected_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        expected_target = torch.tensor([0.5, -1.5, 2.0], requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_loss = functional.mse_loss(
                actual_input, actual_target, reduction="mean"
            )
            expected_loss = (expected_input - expected_target).square().mean()
        self.assert_matches_composition(actual_loss, expected_loss, case="broadcast")
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="broadcast",
        )

        actual_input_base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        actual_target_base = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        expected_input_base = torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        expected_target_base = torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            requires_grad=True,
        )
        actual_input = actual_input_base[1].transpose(0, 1)
        actual_target = actual_target_base[0].transpose(0, 1)
        expected_input = expected_input_base[1].transpose(0, 1)
        expected_target = expected_target_base[0].transpose(0, 1)
        actual_loss = functional.mse_loss(
            actual_input, actual_target, reduction="mean"
        )
        expected_loss = (expected_input - expected_target).square().mean()
        self.assert_matches_composition(
            actual_loss,
            expected_loss,
            case="offset noncontiguous",
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input_base, actual_target_base),
            (expected_input_base, expected_target_base),
            case="offset noncontiguous",
        )

        actual_input = torch.zeros((2, 0, 3), requires_grad=True)
        actual_target = torch.ones((2, 0, 3), requires_grad=True)
        expected_input = torch.zeros((2, 0, 3), requires_grad=True)
        expected_target = torch.ones((2, 0, 3), requires_grad=True)
        actual_loss = functional.mse_loss(
            actual_input.transpose(0, 2),
            actual_target.transpose(0, 2),
            reduction="mean",
        )
        expected_loss = (
            expected_input.transpose(0, 2) - expected_target.transpose(0, 2)
        ).square().mean()
        self.assert_matches_composition(actual_loss, expected_loss, case="empty")
        actual_loss.backward()
        expected_loss.backward()
        assert_gradients(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="empty",
        )

    def test_same_stride_noncontiguous_requires_grad_operands_need_no_grad(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            input_base = torch.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
                requires_grad=input_requires_grad,
            )
            target_base = torch.tensor(
                np.linspace(-2.0, 3.0, 12, dtype=np.float32).reshape(3, 4).tolist(),
                requires_grad=target_requires_grad,
            )
            input = input_base.transpose(0, 1)
            target = target_base.transpose(0, 1)
            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                self.assertEqual(input.stride(), target.stride())
                self.assertFalse(input.is_contiguous())
                self.assertFalse(target.is_contiguous())
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^mse_loss\(\): autograd recording is not supported$",
                ):
                    functional.mse_loss(input, target, reduction="none")

                with torch.no_grad():
                    actual = functional.mse_loss(input, target, reduction="none")
                    difference = input - target
                    expected = difference.square()
                self.assert_matches_composition(
                    actual,
                    expected,
                    case="same-stride no_grad",
                    expected_stride=difference.stride(),
                )
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                self.assertIsNone(input_base.grad)
                self.assertIsNone(target_base.grad)

    def test_broadcast_requires_grad_operands_need_no_grad(self):
        def scalar_input(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(0.5, requires_grad=input_requires_grad),
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
                torch.tensor(0.5, requires_grad=target_requires_grad),
            )

        def vector_target(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor([0.5, -1.5], requires_grad=target_requires_grad),
            )

        for case, factory in (
            ("scalar input", scalar_input),
            ("scalar target", scalar_target),
            ("vector target", vector_target),
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
                    with self.assertWarnsRegex(UserWarning, "Using a target size"):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"^mse_loss\(\): autograd recording is not supported$",
                        ):
                            functional.mse_loss(input, target, reduction="none")

                    with warnings.catch_warnings(), torch.no_grad():
                        warnings.simplefilter("ignore")
                        actual = functional.mse_loss(input, target, reduction="none")
                        expected = (input - target).square()
                    self.assert_matches_composition(actual, expected, case="no_grad")
                    self.assertFalse(actual.requires_grad)
                    self.assertTrue(actual.is_leaf)
                    self.assertIsNone(input.grad)
                    self.assertIsNone(target.grad)

    def test_unsupported_options_shapes_and_operands_are_rejected(self):
        input = torch.ones((2, 3))
        target = torch.zeros((2, 3))

        reduction_error = (
            "torch_rs.nn.functional.mse_loss only supports "
            "reduction='none', reduction='mean', or reduction='sum'"
        )
        for reduction in ("batchmean", None, 1, object()):
            with self.subTest(reduction=reduction):
                with self.assertRaisesRegex(
                    NotImplementedError,
                    f"^{re.escape(reduction_error)}$",
                ):
                    functional.mse_loss(input, target, reduction=reduction)

        legacy_error = (
            "torch_rs.nn.functional.mse_loss only supports "
            "size_average=None and reduce=None"
        )
        for legacy_arguments in (
            {"size_average": False},
            {"size_average": True},
            {"reduce": False},
            {"reduce": True},
            {"size_average": False, "reduce": False},
        ):
            with self.subTest(legacy_arguments=legacy_arguments):
                for reduction in ("none", "mean", "sum"):
                    with self.subTest(reduction=reduction):
                        with self.assertRaisesRegex(
                            NotImplementedError,
                            f"^{re.escape(legacy_error)}$",
                        ):
                            functional.mse_loss(
                                input,
                                target,
                                reduction=reduction,
                                **legacy_arguments,
                            )

        weight_error = "torch_rs.nn.functional.mse_loss only supports weight=None"
        for weight in (torch.ones((2, 3)), 1.0, [1.0, 1.0]):
            with self.subTest(weight=type(weight)):
                for reduction in ("none", "mean", "sum"):
                    with self.subTest(reduction=reduction):
                        with self.assertRaisesRegex(
                            NotImplementedError,
                            f"^{re.escape(weight_error)}$",
                        ):
                            functional.mse_loss(
                                input,
                                target,
                                reduction=reduction,
                                weight=weight,
                            )

        unbroadcastable_target = torch.zeros((2, 2))
        for reduction in ("none", "mean", "sum"):
            with self.subTest(reduction=reduction, unbroadcastable=True):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^The size of tensor a \(3\) must match the size of tensor b "
                        r"\(2\) at non-singleton dimension 1$",
                    ):
                        functional.mse_loss(
                            input,
                            unbroadcastable_target,
                            reduction=reduction,
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
            "mse_loss() only supports exact native Tensor input and target operands"
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
                for reduction in ("none", "mean", "sum"):
                    with self.subTest(reduction=reduction):
                        with self.assertRaisesRegex(
                            TypeError,
                            f"^{re.escape(exact_tensor_error)}$",
                        ):
                            functional.mse_loss(
                                actual_input,
                                actual_target,
                                reduction=reduction,
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
            r"^mse_loss\(\) does not support an active TorchFunctionMode$",
        ):
            with mode:
                functional.mse_loss(
                    torch.ones((2, 3), requires_grad=True),
                    torch.zeros((3,)),
                    reduction="mean",
                    weight=object(),
                )
        self.assertEqual(mode.calls, 0)

    def test_python_argument_binding_matches_the_canonical_signature(self):
        input = torch.ones((1,))
        target = torch.zeros((1,))
        cases = (
            (
                lambda: functional.mse_loss(),
                "mse_loss() missing 2 required positional arguments: 'input' and 'target'",
            ),
            (
                lambda: functional.mse_loss(input),
                "mse_loss() missing 1 required positional argument: 'target'",
            ),
            (
                lambda: functional.mse_loss(input, target, input=input),
                "mse_loss() got multiple values for argument 'input'",
            ),
            (
                lambda: functional.mse_loss(
                    input, target, None, None, "none", None, None
                ),
                "mse_loss() takes from 2 to 6 positional arguments but 7 were given",
            ),
            (
                lambda: functional.mse_loss(
                    input, target, reduction="none", unexpected=True
                ),
                "mse_loss() got an unexpected keyword argument 'unexpected'",
            ),
        )
        for case, (call, message) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)


if __name__ == "__main__":
    unittest.main()
