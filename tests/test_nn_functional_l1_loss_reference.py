import inspect
import unittest
import warnings

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional

try:
    import torch as reference_torch
    import torch.nn.functional as reference_functional
except ImportError:
    reference_torch = None
    reference_functional = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class FunctionalL1LossReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.l1_loss differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor(module, values):
        return module.tensor(values, dtype=module.float32)

    def make_cases(self, module):
        offset_input_base = self.tensor(
            module,
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist(),
        )
        offset_target_base = self.tensor(
            module,
            np.linspace(-3.0, 4.0, 48, dtype=np.float32)
            .reshape(2, 2, 3, 4)
            .tolist(),
        )
        noncontiguous_input = self.tensor(
            module,
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
        ).transpose(1, 2)
        noncontiguous_target = self.tensor(
            module,
            np.linspace(-2.0, 2.0, 24, dtype=np.float32)
            .reshape(2, 4, 3)
            .tolist(),
        ).transpose(1, 2)
        mixed_layout_target = self.tensor(
            module,
            np.linspace(3.0, -3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
        )
        offset_strided_input = self.tensor(
            module,
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist(),
        )[1].transpose(1, 2)
        offset_strided_target = self.tensor(
            module,
            np.linspace(5.0, -5.0, 48, dtype=np.float32)
            .reshape(2, 2, 4, 3)
            .tolist(),
        )[1].transpose(1, 2)
        channels_last_input = offset_input_base.contiguous(
            memory_format=module.channels_last
        )
        channels_last_target = offset_target_base.contiguous(
            memory_format=module.channels_last
        )
        empty_input = module.zeros(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)
        empty_target = module.ones(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)
        mixed_singleton_input = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist(),
        )
        mixed_singleton_target = self.tensor(
            module,
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist(),
        ).permute(2, 1, 0)
        same = self.tensor(module, [[1.0, -2.0], [3.0, -4.0]])

        return (
            (
                "scalar",
                self.tensor(module, -0.0),
                self.tensor(module, 2.5),
            ),
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

    def make_matching_dense_cases(self, module):
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
        transposed_input = module.tensor(
            memoryview(input_values),
            dtype=module.float32,
        ).view(3, 6)
        transposed_target = module.tensor(
            memoryview(target_values),
            dtype=module.float32,
        ).view(3, 6)

        input_padding = np.linspace(
            -11.0, 11.0, input_values.size, dtype=np.float32
        )
        target_padding = np.linspace(
            13.0, -13.0, target_values.size, dtype=np.float32
        )
        offset_input_base = module.tensor(
            memoryview(np.concatenate([input_padding, input_values])),
            dtype=module.float32,
        ).view(2, 3, 6)
        offset_target_base = module.tensor(
            memoryview(np.concatenate([target_padding, target_values])),
            dtype=module.float32,
        ).view(2, 3, 6)
        empty_input = module.zeros(
            (2, 0, 3),
            dtype=module.float32,
        ).transpose(0, 2)
        empty_target = module.ones(
            (2, 0, 3),
            dtype=module.float32,
        ).transpose(0, 2)

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

    def make_broadcast_cases(self, module):
        scalar = self.tensor(module, -0.0)
        offset_scalar = self.tensor(module, [17.0, 0.5])[1]
        matrix = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
        )
        offset_matrix = self.tensor(
            module,
            np.arange(12, dtype=np.float32).reshape(2, 2, 3).tolist(),
        )[1]
        noncontiguous_matrix = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(3, 2).tolist(),
        ).transpose(0, 1)
        empty_contiguous = module.zeros((0, 4), dtype=module.float32)
        empty_strided = module.zeros(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)

        return (
            ("scalar target", matrix, scalar),
            (
                "vector target",
                matrix,
                self.tensor(module, [1.0, 2.0, 3.0]),
            ),
            (
                "column target",
                matrix,
                self.tensor(module, [[1.0], [2.0]]),
            ),
            ("scalar input", scalar, offset_matrix),
            ("empty scalar input", scalar, empty_contiguous),
            ("empty scalar target", empty_contiguous, scalar),
            ("noncontiguous scalar input", offset_scalar, noncontiguous_matrix),
            ("noncontiguous scalar target", noncontiguous_matrix, offset_scalar),
            (
                "empty singleton broadcast",
                empty_strided,
                module.ones((1, 0, 1), dtype=module.float32),
            ),
        )

    def make_sum_reduction_cases(self, module):
        offset_input_base = self.tensor(
            module,
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist(),
        )
        offset_target_base = self.tensor(
            module,
            np.linspace(-3.0, 4.0, 48, dtype=np.float32)
            .reshape(2, 2, 3, 4)
            .tolist(),
        )
        transposed_input = self.tensor(
            module,
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
        ).transpose(1, 2)
        transposed_target = self.tensor(
            module,
            np.linspace(-2.0, 2.0, 24, dtype=np.float32)
            .reshape(2, 4, 3)
            .tolist(),
        ).transpose(1, 2)
        broadcast_input = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
        )
        broadcast_target = self.tensor(module, [1.0, -2.0, 0.5])
        empty_input = module.zeros(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)
        empty_target = module.ones(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)

        return (
            ("scalar", self.tensor(module, -0.0), self.tensor(module, 2.5), False),
            ("empty", empty_input, empty_target, False),
            ("broadcasted", broadcast_input, broadcast_target, True),
            ("offset", offset_input_base[1], offset_target_base[0], False),
            ("noncontiguous", transposed_input, transposed_target, False),
        )

    def make_rank2_trailing_vector_sum_cases(self, module):
        signed_zero_matrix_bits = np.asarray(
            [0x8000_0000, 0x0000_0000, 0x8000_0000, 0x0000_0000],
            dtype=np.uint32,
        )
        signed_zero_vector_bits = np.asarray(
            [0x0000_0000, 0x8000_0000],
            dtype=np.uint32,
        )
        edge_matrix_bits = np.asarray(
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
                0x7F7F_FFFF,
                0xFF7F_FFFF,
            ],
            dtype=np.uint32,
        )
        edge_vector_bits = np.asarray(
            [0x8000_0000, 0x7F80_0000, 0xFF86_789A, 0x7F82_ABCD],
            dtype=np.uint32,
        )
        inf_matrix_bits = np.asarray(
            [0x7F80_0000, 0xFF80_0000, 0x3F80_0000, 0xBF80_0000],
            dtype=np.uint32,
        )
        inf_vector_bits = np.asarray(
            [0xFF80_0000, 0x3F80_0000],
            dtype=np.uint32,
        )

        signed_zero_matrix = module.tensor(
            memoryview(signed_zero_matrix_bits.view(np.float32)),
            dtype=module.float32,
        ).view(2, 2)
        signed_zero_vector = module.tensor(
            memoryview(signed_zero_vector_bits.view(np.float32)),
            dtype=module.float32,
        )
        edge_matrix = module.tensor(
            memoryview(edge_matrix_bits.view(np.float32)),
            dtype=module.float32,
        ).view(3, 4)
        edge_vector = module.tensor(
            memoryview(edge_vector_bits.view(np.float32)),
            dtype=module.float32,
        )
        inf_matrix = module.tensor(
            memoryview(inf_matrix_bits.view(np.float32)),
            dtype=module.float32,
        ).view(2, 2)
        inf_vector = module.tensor(
            memoryview(inf_vector_bits.view(np.float32)),
            dtype=module.float32,
        )
        empty_rows = module.zeros((0, 4), dtype=module.float32)

        return (
            (
                "target broadcast signed zero",
                signed_zero_matrix,
                signed_zero_vector,
                0x0000_0000,
            ),
            (
                "input broadcast signed zero",
                signed_zero_vector,
                signed_zero_matrix,
                0x0000_0000,
            ),
            ("target broadcast nan inf", edge_matrix, edge_vector, None),
            ("input broadcast nan inf", edge_vector, edge_matrix, None),
            ("target broadcast inf", inf_matrix, inf_vector, 0x7F80_0000),
            ("input broadcast inf", inf_vector, inf_matrix, 0x7F80_0000),
            ("empty leading target broadcast", empty_rows, edge_vector, 0x0000_0000),
            ("empty leading input broadcast", edge_vector, empty_rows, 0x0000_0000),
        )

    def make_channels_last_cases(self, module):
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
        edge_input = module.tensor(
            memoryview(np.resize(edge_input_patterns, 24).view(np.float32)),
            dtype=module.float32,
        ).view(2, 3, 2, 2)
        edge_target = module.tensor(
            memoryview(np.resize(edge_target_patterns, 24).view(np.float32)),
            dtype=module.float32,
        ).view(2, 3, 2, 2)
        singleton_input = self.tensor(
            module,
            np.linspace(-3.0, 4.0, 2 * 3 * 1 * 4, dtype=np.float32)
            .reshape(2, 3, 1, 4)
            .tolist(),
        )
        singleton_target = self.tensor(
            module,
            np.linspace(5.0, -7.0, 2 * 3 * 1 * 4, dtype=np.float32)
            .reshape(2, 3, 1, 4)
            .tolist(),
        )
        empty_input = module.zeros((2, 3, 0, 5), dtype=module.float32)
        empty_target = module.ones((2, 3, 0, 5), dtype=module.float32)

        return (
            (
                "edge bits",
                edge_input.contiguous(memory_format=module.channels_last),
                edge_target.contiguous(memory_format=module.channels_last),
            ),
            (
                "singleton",
                singleton_input.contiguous(memory_format=module.channels_last),
                singleton_target.contiguous(memory_format=module.channels_last),
            ),
            (
                "empty",
                empty_input.contiguous(memory_format=module.channels_last),
                empty_target.contiguous(memory_format=module.channels_last),
            ),
        )

    @staticmethod
    def call(module_functional, input, target, form):
        if form == "reduction keyword":
            return module_functional.l1_loss(input, target, reduction="none")
        if form == "legacy none keywords":
            return module_functional.l1_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="none",
                weight=None,
            )
        if form == "five positional":
            return module_functional.l1_loss(input, target, None, None, "none")
        return module_functional.l1_loss(input, target, None, None, "none", None)

    @staticmethod
    def call_with_warnings(module_functional, input, target, reduction="none"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = module_functional.l1_loss(input, target, reduction=reduction)
        warning_state = [
            (
                warning.category.__name__,
                str(warning.message),
                warning.filename,
                warning.lineno,
            )
            for warning in caught
        ]
        return output, warning_state

    def assert_matches(self, actual, expected, *, case, max_value_ulp=0):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            if expected.dim() == 4:
                self.assertEqual(
                    actual.is_contiguous(memory_format=torch.channels_last),
                    expected.is_contiguous(
                        memory_format=reference_torch.channels_last
                    ),
                )
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_values = np.asarray(actual)
            expected_values = expected.detach().cpu().numpy()
            if max_value_ulp:
                np.testing.assert_array_max_ulp(
                    actual_values,
                    expected_values,
                    maxulp=max_value_ulp,
                )
            else:
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32),
                    expected_values.reshape(-1).view(np.uint32),
                )

    def test_name_signature_defaults_and_annotations_match_pytorch_2_13(self):
        self.assertEqual(
            functional.l1_loss.__name__,
            reference_functional.l1_loss.__name__,
        )
        actual = inspect.signature(functional.l1_loss)
        expected = inspect.signature(reference_functional.l1_loss)
        self.assertEqual(tuple(actual.parameters), tuple(expected.parameters))
        for name in actual.parameters:
            with self.subTest(parameter=name):
                self.assertEqual(
                    actual.parameters[name].kind,
                    expected.parameters[name].kind,
                )
                self.assertEqual(
                    actual.parameters[name].default,
                    expected.parameters[name].default,
                )
        self.assertEqual(
            functional.l1_loss.__defaults__,
            reference_functional.l1_loss.__defaults__,
        )
        self.assertEqual(
            tuple(functional.l1_loss.__annotations__),
            tuple(reference_functional.l1_loss.__annotations__),
        )

    def test_supported_layouts_values_storage_and_nonmutation_match(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_target = actual_case
            expected_name, expected_input, expected_target = expected_case
            self.assertEqual(case, expected_name)
            actual_input_before = np.asarray(actual_input).copy()
            actual_target_before = np.asarray(actual_target).copy()
            expected_input_before = expected_input.clone()
            expected_target_before = expected_target.clone()
            for form in (
                "reduction keyword",
                "legacy none keywords",
                "five positional",
                "six positional",
            ):
                actual = self.call(
                    functional,
                    actual_input,
                    actual_target,
                    form,
                )
                expected = self.call(
                    reference_functional,
                    expected_input,
                    expected_target,
                    form,
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call(
                    functional,
                    actual_input,
                    actual_target,
                    form,
                )
                expected_repeat = self.call(
                    reference_functional,
                    expected_input,
                    expected_target,
                    form,
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_target))
                    self.assertFalse(expected.is_set_to(expected_target))

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(
                    np.asarray(actual_input), actual_input_before
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_target), actual_target_before
                )
                self.assertTrue(
                    reference_torch.equal(expected_input, expected_input_before)
                )
                self.assertTrue(
                    reference_torch.equal(expected_target, expected_target_before)
                )

    def test_matching_dense_transposed_offset_empty_none_sum_match_pytorch_2_13(self):
        actual_cases = self.make_matching_dense_cases(torch)
        expected_cases = self.make_matching_dense_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_target = actual_case
            expected_name, expected_input, expected_target = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_target.shape, tuple(expected_target.shape))
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(actual_target.stride(), expected_target.stride())
            self.assertEqual(actual_input.stride(), actual_target.stride())
            if actual_input.numel() != 0:
                self.assertFalse(actual_input.is_contiguous())
                self.assertFalse(actual_target.is_contiguous())
                self.assertFalse(expected_input.is_contiguous())
                self.assertFalse(expected_target.is_contiguous())

            actual_input_bits_before = (
                np.asarray(actual_input).reshape(-1).view(np.uint32).copy()
            )
            actual_target_bits_before = (
                np.asarray(actual_target).reshape(-1).view(np.uint32).copy()
            )
            expected_input_bits_before = (
                expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )
            expected_target_bits_before = (
                expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )

            for reduction in ("none", "sum"):
                with warnings.catch_warnings(record=True) as actual_warnings:
                    warnings.simplefilter("always")
                    actual = functional.l1_loss(
                        actual_input,
                        actual_target,
                        reduction=reduction,
                    )
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter("always")
                    expected = reference_functional.l1_loss(
                        expected_input,
                        expected_target,
                        reduction=reduction,
                    )

                with self.subTest(case=case, reduction=reduction, warnings=True):
                    self.assertEqual(actual_warnings, [])
                    self.assertEqual(expected_warnings, [])
                self.assert_matches(
                    actual,
                    expected,
                    case=(case, reduction),
                    max_value_ulp=int(reduction == "sum"),
                )

                actual_repeat = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction=reduction,
                )
                expected_repeat = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction=reduction,
                )
                with self.subTest(case=case, reduction=reduction, storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_target))
                    self.assertFalse(expected.is_set_to(expected_target))
                    if actual.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), actual_repeat.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                        self.assertNotEqual(actual.data_ptr(), actual_target.data_ptr())
                        self.assertNotEqual(expected.data_ptr(), expected_repeat.data_ptr())
                        self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())
                        self.assertNotEqual(expected.data_ptr(), expected_target.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(
                    np.asarray(actual_input).reshape(-1).view(np.uint32),
                    actual_input_bits_before,
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_target).reshape(-1).view(np.uint32),
                    actual_target_bits_before,
                )
                np.testing.assert_array_equal(
                    expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32),
                    expected_input_bits_before,
                )
                np.testing.assert_array_equal(
                    expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32),
                    expected_target_bits_before,
                )

    def test_channels_last_edge_singleton_empty_and_nonmutation_match_pytorch_2_13(self):
        actual_cases = self.make_channels_last_cases(torch)
        expected_cases = self.make_channels_last_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_target = actual_case
            expected_name, expected_input, expected_target = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_target.shape, tuple(expected_target.shape))
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(actual_target.stride(), expected_target.stride())
            self.assertEqual(actual_input.stride(), actual_target.stride())
            self.assertTrue(
                actual_input.is_contiguous(memory_format=torch.channels_last)
            )
            self.assertTrue(
                expected_input.is_contiguous(
                    memory_format=reference_torch.channels_last
                )
            )

            actual_input_bits_before = (
                np.asarray(actual_input).reshape(-1).view(np.uint32).copy()
            )
            actual_target_bits_before = (
                np.asarray(actual_target).reshape(-1).view(np.uint32).copy()
            )
            expected_input_bits_before = (
                expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )
            expected_target_bits_before = (
                expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )

            actual = functional.l1_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            expected = reference_functional.l1_loss(
                expected_input,
                expected_target,
                reduction="none",
            )

            self.assert_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                actual_repeat = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
                expected_repeat = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="none",
                )
                self.assertFalse(actual.is_set_to(actual_repeat))
                self.assertFalse(expected.is_set_to(expected_repeat))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assertFalse(actual.is_set_to(actual_target))
                self.assertFalse(expected.is_set_to(expected_target))

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(
                    np.asarray(actual_input).reshape(-1).view(np.uint32),
                    actual_input_bits_before,
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_target).reshape(-1).view(np.uint32),
                    actual_target_bits_before,
                )
                np.testing.assert_array_equal(
                    expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32),
                    expected_input_bits_before,
                )
                np.testing.assert_array_equal(
                    expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32),
                    expected_target_bits_before,
                )

    def test_broadcasted_outputs_strides_warnings_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_broadcast_cases(torch)
        expected_cases = self.make_broadcast_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_target = actual_case
            expected_name, expected_input, expected_target = expected_case
            self.assertEqual(case, expected_name)

            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always")
                actual = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter("always")
                expected = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="none",
                )

            with self.subTest(case=case, warning=True):
                self.assertEqual(len(actual_warnings), len(expected_warnings))
                self.assertEqual(len(actual_warnings), 1)
                self.assertIs(actual_warnings[0].category, UserWarning)
                self.assertIs(expected_warnings[0].category, UserWarning)
                self.assertEqual(
                    str(actual_warnings[0].message),
                    str(expected_warnings[0].message),
                )

            self.assert_matches(actual, expected, case=case)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual_repeat = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
                expected_repeat = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="none",
                )
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_repeat))
                self.assertFalse(expected.is_set_to(expected_repeat))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assertFalse(actual.is_set_to(actual_target))
                self.assertFalse(expected.is_set_to(expected_target))

    def test_sum_reduction_cases_match_pytorch_2_13(self):
        actual_cases = self.make_sum_reduction_cases(torch)
        expected_cases = self.make_sum_reduction_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_target, warns = actual_case
            expected_name, expected_input, expected_target, expected_warns = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(warns, expected_warns)
            actual_input_before = np.asarray(actual_input).copy()
            actual_target_before = np.asarray(actual_target).copy()
            expected_input_before = expected_input.clone()
            expected_target_before = expected_target.clone()

            actual, actual_warnings = self.call_with_warnings(
                functional,
                actual_input,
                actual_target,
                reduction="sum",
            )
            expected, expected_warnings = self.call_with_warnings(
                reference_functional,
                expected_input,
                expected_target,
                reduction="sum",
            )

            self.assert_matches(actual, expected, case=case, max_value_ulp=1)
            with self.subTest(case=case, warnings=True):
                self.assertEqual(actual_warnings, expected_warnings)
                self.assertEqual(len(actual_warnings), int(warns))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual_repeat = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="sum",
                )
                expected_repeat = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="sum",
                )
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_repeat))
                self.assertFalse(expected.is_set_to(expected_repeat))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assertFalse(actual.is_set_to(actual_target))
                self.assertFalse(expected.is_set_to(expected_target))

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(
                    np.asarray(actual_input), actual_input_before
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_target), actual_target_before
                )
                self.assertTrue(
                    reference_torch.equal(expected_input, expected_input_before)
                )
                self.assertTrue(
                    reference_torch.equal(expected_target, expected_target_before)
                )

    def test_same_shape_contiguous_sum_edges_metadata_and_nonmutation_match_pytorch_2_13(
        self,
    ):
        def actual_bits(tensor):
            return np.asarray(tensor).reshape(-1).view(np.uint32).copy()

        def expected_bits(tensor):
            return tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()

        def assert_scalar_matches(actual, expected, *, case, allow_nan=False):
            with self.subTest(case=case, metadata=True):
                self.assertEqual(actual.shape, tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
                self.assertFalse(actual.requires_grad)
                self.assertEqual(actual.is_leaf, expected.is_leaf)
                self.assertIs(actual.dtype, torch.float32)
                self.assertEqual(actual.device, torch.device("cpu"))
                self.assertEqual(actual.numel(), expected.numel())

            actual_values = np.asarray(actual).reshape(-1)
            expected_values = expected.detach().cpu().numpy().reshape(-1)
            with self.subTest(case=case, values=True):
                if allow_nan:
                    actual_nan = np.isnan(actual_values)
                    expected_nan = np.isnan(expected_values)
                    np.testing.assert_array_equal(actual_nan, expected_nan)
                    non_nan = ~expected_nan
                    if np.any(non_nan):
                        np.testing.assert_array_max_ulp(
                            actual_values[non_nan],
                            expected_values[non_nan],
                            maxulp=1,
                        )
                else:
                    np.testing.assert_array_max_ulp(
                        actual_values,
                        expected_values,
                        maxulp=1,
                    )

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
        large_values = np.ones(1024 * 1024, dtype=np.float32)
        mixed_values = np.full(1024, 100_000.0, dtype=np.float32)
        mixed_values[:31] = 100.0
        actual_cases = (
            (
                "scalar signed zero",
                torch.tensor(-0.0),
                torch.tensor(0.0),
                False,
            ),
            (
                "empty",
                torch.zeros((5, 0, 7), dtype=torch.float32),
                torch.ones((5, 0, 7), dtype=torch.float32),
                False,
            ),
            (
                "large contiguous",
                torch.tensor(memoryview(large_values)).view(1024, 1024),
                torch.zeros((1024, 1024), dtype=torch.float32),
                False,
            ),
            (
                "mixed magnitudes",
                torch.tensor(memoryview(mixed_values)),
                torch.zeros((1024,), dtype=torch.float32),
                False,
            ),
            (
                "inf edge bits",
                torch.tensor(memoryview(inf_input_bits.view(np.float32))),
                torch.tensor(memoryview(inf_target_bits.view(np.float32))),
                False,
            ),
            (
                "nan edge bits",
                torch.tensor(memoryview(edge_input_bits.view(np.float32))),
                torch.tensor(memoryview(edge_target_bits.view(np.float32))),
                True,
            ),
        )
        expected_cases = (
            (
                "scalar signed zero",
                reference_torch.tensor(-0.0, dtype=reference_torch.float32),
                reference_torch.tensor(0.0, dtype=reference_torch.float32),
                False,
            ),
            (
                "empty",
                reference_torch.zeros((5, 0, 7), dtype=reference_torch.float32),
                reference_torch.ones((5, 0, 7), dtype=reference_torch.float32),
                False,
            ),
            (
                "large contiguous",
                reference_torch.tensor(
                    memoryview(large_values),
                    dtype=reference_torch.float32,
                ).view(1024, 1024),
                reference_torch.zeros(
                    (1024, 1024),
                    dtype=reference_torch.float32,
                ),
                False,
            ),
            (
                "mixed magnitudes",
                reference_torch.tensor(
                    memoryview(mixed_values),
                    dtype=reference_torch.float32,
                ),
                reference_torch.zeros((1024,), dtype=reference_torch.float32),
                False,
            ),
            (
                "inf edge bits",
                reference_torch.tensor(
                    memoryview(inf_input_bits.view(np.float32)),
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor(
                    memoryview(inf_target_bits.view(np.float32)),
                    dtype=reference_torch.float32,
                ),
                False,
            ),
            (
                "nan edge bits",
                reference_torch.tensor(
                    memoryview(edge_input_bits.view(np.float32)),
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor(
                    memoryview(edge_target_bits.view(np.float32)),
                    dtype=reference_torch.float32,
                ),
                True,
            ),
        )

        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_input, actual_target, allow_nan = actual_case
            expected_name, expected_input, expected_target, expected_allow_nan = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(allow_nan, expected_allow_nan)
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_target.shape, tuple(expected_target.shape))
            self.assertTrue(actual_input.is_contiguous())
            self.assertTrue(actual_target.is_contiguous())
            self.assertTrue(expected_input.is_contiguous())
            self.assertTrue(expected_target.is_contiguous())

            actual_input_before = actual_bits(actual_input)
            actual_target_before = actual_bits(actual_target)
            expected_input_before = expected_bits(expected_input)
            expected_target_before = expected_bits(expected_target)

            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always")
                actual = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="sum",
                )
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter("always")
                expected = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="sum",
                )

            self.assertEqual(actual_warnings, [])
            self.assertEqual(expected_warnings, [])
            assert_scalar_matches(
                actual,
                expected,
                case=case,
                allow_nan=allow_nan,
            )
            if case == "mixed magnitudes":
                self.assertEqual(int(actual_bits(actual)[0]), 0x4CBD_67D8)
                self.assertEqual(int(expected_bits(expected)[0]), 0x4CBD_67D8)

            actual_repeat = functional.l1_loss(
                actual_input,
                actual_target,
                reduction="sum",
            )
            expected_repeat = reference_functional.l1_loss(
                expected_input,
                expected_target,
                reduction="sum",
            )
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_repeat))
                self.assertFalse(expected.is_set_to(expected_repeat))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assertFalse(actual.is_set_to(actual_target))
                self.assertFalse(expected.is_set_to(expected_target))
                self.assertNotEqual(actual.data_ptr(), actual_repeat.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_repeat.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(actual_bits(actual_input), actual_input_before)
                np.testing.assert_array_equal(actual_bits(actual_target), actual_target_before)
                np.testing.assert_array_equal(expected_bits(expected_input), expected_input_before)
                np.testing.assert_array_equal(
                    expected_bits(expected_target),
                    expected_target_before,
                )

    def test_rank2_trailing_vector_sum_broadcast_edges_match_pytorch_2_13(self):
        def actual_bits(tensor):
            return np.asarray(tensor).reshape(-1).view(np.uint32).copy()

        def expected_bits(tensor):
            return tensor.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()

        def assert_scalar_matches(actual, expected, *, case, allow_nan=False):
            with self.subTest(case=case, metadata=True):
                self.assertEqual(actual.shape, tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertEqual(actual.storage_offset(), expected.storage_offset())
                self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
                self.assertFalse(actual.requires_grad)
                self.assertEqual(actual.is_leaf, expected.is_leaf)
                self.assertIs(actual.dtype, torch.float32)
                self.assertEqual(actual.device, torch.device("cpu"))
                self.assertEqual(actual.numel(), expected.numel())

            actual_values = np.asarray(actual).reshape(-1)
            expected_values = expected.detach().cpu().numpy().reshape(-1)
            with self.subTest(case=case, values=True):
                if allow_nan:
                    actual_nan = np.isnan(actual_values)
                    expected_nan = np.isnan(expected_values)
                    np.testing.assert_array_equal(actual_nan, expected_nan)
                    non_nan = ~expected_nan
                    if np.any(non_nan):
                        np.testing.assert_array_max_ulp(
                            actual_values[non_nan],
                            expected_values[non_nan],
                            maxulp=1,
                        )
                else:
                    np.testing.assert_array_max_ulp(
                        actual_values,
                        expected_values,
                        maxulp=1,
                    )

        actual_cases = self.make_rank2_trailing_vector_sum_cases(torch)
        expected_cases = self.make_rank2_trailing_vector_sum_cases(reference_torch)
        for actual_case, expected_case in zip(actual_cases, expected_cases, strict=True):
            case, actual_input, actual_target, expected_sum_bits = actual_case
            (
                expected_name,
                expected_input,
                expected_target,
                reference_sum_bits,
            ) = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(expected_sum_bits, reference_sum_bits)
            self.assertNotEqual(actual_input.shape, actual_target.shape)
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_target.shape, tuple(expected_target.shape))
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(actual_target.stride(), expected_target.stride())
            self.assertTrue(actual_input.is_contiguous())
            self.assertTrue(actual_target.is_contiguous())
            self.assertTrue(expected_input.is_contiguous())
            self.assertTrue(expected_target.is_contiguous())

            actual_input_before = actual_bits(actual_input)
            actual_target_before = actual_bits(actual_target)
            expected_input_before = expected_bits(expected_input)
            expected_target_before = expected_bits(expected_target)

            actual, actual_warnings = self.call_with_warnings(
                functional,
                actual_input,
                actual_target,
                reduction="sum",
            )
            expected, expected_warnings = self.call_with_warnings(
                reference_functional,
                expected_input,
                expected_target,
                reduction="sum",
            )

            assert_scalar_matches(
                actual,
                expected,
                case=case,
                allow_nan="nan" in case,
            )
            if expected_sum_bits is not None:
                self.assertEqual(int(actual_bits(actual)[0]), expected_sum_bits)
                self.assertEqual(int(expected_bits(expected)[0]), expected_sum_bits)
            with self.subTest(case=case, warnings=True):
                self.assertEqual(actual_warnings, expected_warnings)
                self.assertEqual(len(actual_warnings), 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual_repeat = functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="sum",
                )
                expected_repeat = reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="sum",
                )
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_repeat))
                self.assertFalse(expected.is_set_to(expected_repeat))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assertFalse(actual.is_set_to(actual_target))
                self.assertFalse(expected.is_set_to(expected_target))
                self.assertNotEqual(actual.data_ptr(), actual_repeat.data_ptr())
                self.assertNotEqual(expected.data_ptr(), expected_repeat.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(actual_bits(actual_input), actual_input_before)
                np.testing.assert_array_equal(actual_bits(actual_target), actual_target_before)
                np.testing.assert_array_equal(expected_bits(expected_input), expected_input_before)
                np.testing.assert_array_equal(
                    expected_bits(expected_target),
                    expected_target_before,
                )

    def test_rank2_trailing_vector_sum_broadcast_requires_grad_matches_no_grad_and_rejects_active_autograd(
        self,
    ):
        def actual_target_broadcast(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor([0.5, -1.0, 7.0], requires_grad=target_requires_grad),
            )

        def expected_target_broadcast(input_requires_grad, target_requires_grad):
            return (
                reference_torch.tensor(
                    [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                    dtype=reference_torch.float32,
                    requires_grad=input_requires_grad,
                ),
                reference_torch.tensor(
                    [0.5, -1.0, 7.0],
                    dtype=reference_torch.float32,
                    requires_grad=target_requires_grad,
                ),
            )

        def actual_input_broadcast(input_requires_grad, target_requires_grad):
            return (
                torch.tensor([0.5, -1.0, 7.0], requires_grad=input_requires_grad),
                torch.tensor(
                    [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                    requires_grad=target_requires_grad,
                ),
            )

        def expected_input_broadcast(input_requires_grad, target_requires_grad):
            return (
                reference_torch.tensor(
                    [0.5, -1.0, 7.0],
                    dtype=reference_torch.float32,
                    requires_grad=input_requires_grad,
                ),
                reference_torch.tensor(
                    [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]],
                    dtype=reference_torch.float32,
                    requires_grad=target_requires_grad,
                ),
            )

        for case, actual_factory, expected_factory in (
            (
                "target broadcast",
                actual_target_broadcast,
                expected_target_broadcast,
            ),
            (
                "input broadcast",
                actual_input_broadcast,
                expected_input_broadcast,
            ),
        ):
            for input_requires_grad, target_requires_grad in (
                (True, False),
                (False, True),
                (True, True),
            ):
                actual_input, actual_target = actual_factory(
                    input_requires_grad,
                    target_requires_grad,
                )
                expected_input, expected_target = expected_factory(
                    input_requires_grad,
                    target_requires_grad,
                )
                with self.subTest(
                    case=case,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                    active_autograd=True,
                ):
                    with self.assertWarnsRegex(UserWarning, "Using a target size"):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            r"^l1_loss\(\): autograd recording is not supported$",
                        ):
                            functional.l1_loss(
                                actual_input,
                                actual_target,
                                reduction="sum",
                            )

                with torch.no_grad():
                    actual, actual_warnings = self.call_with_warnings(
                        functional,
                        actual_input,
                        actual_target,
                        reduction="sum",
                    )
                with reference_torch.no_grad():
                    expected, expected_warnings = self.call_with_warnings(
                        reference_functional,
                        expected_input,
                        expected_target,
                        reduction="sum",
                    )
                self.assert_matches(
                    actual,
                    expected,
                    case=(case, input_requires_grad, target_requires_grad),
                    max_value_ulp=1,
                )
                with self.subTest(
                    case=case,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                    warnings=True,
                ):
                    self.assertEqual(actual_warnings, expected_warnings)
                    self.assertEqual(len(actual_warnings), 1)
                self.assertIsNone(actual_input.grad)
                self.assertIsNone(actual_target.grad)
                self.assertIsNone(expected_input.grad)
                self.assertIsNone(expected_target.grad)

    def test_unbroadcastable_shape_warning_and_error_match_pytorch_2_13(self):
        actual_input = torch.ones((2, 3))
        actual_target = torch.zeros((2, 2))
        expected_input = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        expected_target = reference_torch.zeros((2, 2), dtype=reference_torch.float32)

        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as actual_error:
                functional.l1_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as expected_error:
                reference_functional.l1_loss(
                    expected_input,
                    expected_target,
                    reduction="none",
                )

        self.assertEqual(str(actual_error.exception), str(expected_error.exception))
        self.assertEqual(len(actual_warnings), len(expected_warnings))
        self.assertEqual(len(actual_warnings), 1)
        self.assertEqual(
            str(actual_warnings[0].message),
            str(expected_warnings[0].message),
        )

    def test_mixed_layout_singleton_stride_matches_pytorch_2_13(self):
        actual_input = self.tensor(
            torch,
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist(),
        )
        actual_target = self.tensor(
            torch,
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist(),
        ).permute(2, 1, 0)
        expected_input = self.tensor(
            reference_torch,
            np.arange(6, dtype=np.float32).reshape(2, 1, 3).tolist(),
        )
        expected_target = self.tensor(
            reference_torch,
            np.linspace(-1.0, 1.0, 6, dtype=np.float32)
            .reshape(3, 1, 2)
            .tolist(),
        ).permute(2, 1, 0)

        self.assertEqual(actual_input.stride(), (3, 3, 1))
        self.assertEqual(actual_target.stride(), (1, 2, 2))
        actual = functional.l1_loss(
            actual_input,
            actual_target,
            reduction="none",
        )
        expected = reference_functional.l1_loss(
            expected_input,
            expected_target,
            reduction="none",
        )
        self.assertEqual(expected.stride(), (3, 3, 1))
        self.assert_matches(actual, expected, case="mixed singleton strides")

    def test_float32_edge_bits_match_pytorch_2_13(self):
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
        actual_input = torch.tensor(memoryview(input_bits.view(np.float32))).view(3, 6)
        actual_target = torch.tensor(memoryview(target_bits.view(np.float32))).view(3, 6)
        expected_input = reference_torch.tensor(
            memoryview(input_bits.view(np.float32))
        ).view(3, 6)
        expected_target = reference_torch.tensor(
            memoryview(target_bits.view(np.float32))
        ).view(3, 6)

        for case, transpose in (("contiguous", False), ("transposed", True)):
            if transpose:
                actual_operands = (
                    actual_input.transpose(0, 1),
                    actual_target.transpose(0, 1),
                )
                expected_operands = (
                    expected_input.transpose(0, 1),
                    expected_target.transpose(0, 1),
                )
            else:
                actual_operands = (actual_input, actual_target)
                expected_operands = (expected_input, expected_target)
            actual = functional.l1_loss(*actual_operands, reduction="none")
            expected = reference_functional.l1_loss(
                *expected_operands,
                reduction="none",
            )
            self.assert_matches(actual, expected, case=("float32 edges", case))

    def test_scalar_broadcast_float32_edges_match_pytorch_2_13(self):
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
        actual_contiguous = torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(2, 7)
        expected_contiguous = reference_torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(2, 7)
        actual_empty = torch.zeros((0, 7), dtype=torch.float32)
        expected_empty = reference_torch.zeros(
            (0, 7),
            dtype=reference_torch.float32,
        )

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
            actual_scalar = torch.tensor(memoryview(scalar_values))[0]
            expected_scalar = reference_torch.tensor(memoryview(scalar_values))[0]
            for layout, actual_tensor, expected_tensor in (
                ("contiguous", actual_contiguous, expected_contiguous),
                ("empty", actual_empty, expected_empty),
                (
                    "noncontiguous fallback",
                    actual_contiguous.transpose(0, 1),
                    expected_contiguous.transpose(0, 1),
                ),
            ):
                for scalar_on_left in (True, False):
                    actual_operands = (
                        (actual_scalar, actual_tensor)
                        if scalar_on_left
                        else (actual_tensor, actual_scalar)
                    )
                    expected_operands = (
                        (expected_scalar, expected_tensor)
                        if scalar_on_left
                        else (expected_tensor, expected_scalar)
                    )
                    with warnings.catch_warnings(record=True) as actual_warnings:
                        warnings.simplefilter("always")
                        actual = functional.l1_loss(
                            *actual_operands,
                            reduction="none",
                        )
                    with warnings.catch_warnings(record=True) as expected_warnings:
                        warnings.simplefilter("always")
                        expected = reference_functional.l1_loss(
                            *expected_operands,
                            reduction="none",
                        )

                    with self.subTest(
                        layout=layout,
                        scalar_bits=hex(scalar_bits),
                        scalar_on_left=scalar_on_left,
                        warning=True,
                    ):
                        self.assertEqual(len(actual_warnings), len(expected_warnings))
                        self.assertEqual(len(actual_warnings), 1)
                        self.assertEqual(
                            str(actual_warnings[0].message),
                            str(expected_warnings[0].message),
                        )

                    self.assert_matches(
                        actual,
                        expected,
                        case=(layout, hex(scalar_bits), scalar_on_left),
                    )

    def test_bandwidth_sized_same_shape_contiguous_matches_pytorch_2_13(self):
        input_values = np.linspace(
            -1024.0,
            1024.0,
            1024 * 1024,
            dtype=np.float32,
        )
        target_values = np.linspace(
            2048.0,
            -2048.0,
            1024 * 1024,
            dtype=np.float32,
        )
        actual_input = torch.tensor(memoryview(input_values)).view(1024, 1024)
        actual_target = torch.tensor(memoryview(target_values)).view(1024, 1024)
        expected_input = reference_torch.tensor(memoryview(input_values)).view(
            1024,
            1024,
        )
        expected_target = reference_torch.tensor(memoryview(target_values)).view(
            1024,
            1024,
        )

        actual = functional.l1_loss(
            actual_input,
            actual_target,
            reduction="none",
        )
        expected = reference_functional.l1_loss(
            expected_input,
            expected_target,
            reduction="none",
        )
        self.assert_matches(actual, expected, case="bandwidth-sized contiguous")

    def test_requires_grad_operands_match_inside_no_grad(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input = torch.tensor(
                [[1.0, -2.0], [3.0, -4.0]],
                requires_grad=input_requires_grad,
            )
            actual_target = torch.tensor(
                [[0.5, 2.0], [-3.0, 4.5]],
                requires_grad=target_requires_grad,
            )
            expected_input = reference_torch.tensor(
                [[1.0, -2.0], [3.0, -4.0]],
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_target = reference_torch.tensor(
                [[0.5, 2.0], [-3.0, 4.5]],
                dtype=reference_torch.float32,
                requires_grad=target_requires_grad,
            )
            for reduction in ("none", "sum"):
                with self.subTest(
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                    reduction=reduction,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        r"^l1_loss\(\): autograd recording is not supported$",
                    ):
                        functional.l1_loss(
                            actual_input,
                            actual_target,
                            reduction=reduction,
                        )

                    with torch.no_grad():
                        actual = functional.l1_loss(
                            actual_input,
                            actual_target,
                            reduction=reduction,
                        )
                    with reference_torch.no_grad():
                        expected = reference_functional.l1_loss(
                            expected_input,
                            expected_target,
                            reduction=reduction,
                        )
                    self.assert_matches(
                        actual,
                        expected,
                        case=(input_requires_grad, target_requires_grad, reduction),
                        max_value_ulp=int(reduction == "sum"),
                    )

    def test_channels_last_requires_grad_matches_no_grad_and_rejects_active_autograd(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input_base = torch.tensor(
                np.linspace(-2.0, 3.0, 2 * 3 * 2 * 4, dtype=np.float32)
                .reshape(2, 3, 2, 4)
                .tolist(),
                requires_grad=input_requires_grad,
            )
            actual_target_base = torch.tensor(
                np.linspace(5.0, -7.0, 2 * 3 * 2 * 4, dtype=np.float32)
                .reshape(2, 3, 2, 4)
                .tolist(),
                requires_grad=target_requires_grad,
            )
            expected_input_base = reference_torch.tensor(
                np.linspace(-2.0, 3.0, 2 * 3 * 2 * 4, dtype=np.float32).reshape(
                    2, 3, 2, 4
                ),
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_target_base = reference_torch.tensor(
                np.linspace(5.0, -7.0, 2 * 3 * 2 * 4, dtype=np.float32).reshape(
                    2, 3, 2, 4
                ),
                dtype=reference_torch.float32,
                requires_grad=target_requires_grad,
            )
            actual_input = actual_input_base.contiguous(
                memory_format=torch.channels_last
            )
            actual_target = actual_target_base.contiguous(
                memory_format=torch.channels_last
            )
            expected_input = expected_input_base.contiguous(
                memory_format=reference_torch.channels_last
            )
            expected_target = expected_target_base.contiguous(
                memory_format=reference_torch.channels_last
            )
            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                self.assertEqual(actual_input.stride(), expected_input.stride())
                self.assertEqual(actual_target.stride(), expected_target.stride())
                self.assertTrue(
                    actual_input.is_contiguous(memory_format=torch.channels_last)
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^l1_loss\(\): autograd recording is not supported$",
                ):
                    functional.l1_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
                    )

                with torch.no_grad():
                    actual = functional.l1_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
                    )
                with reference_torch.no_grad():
                    expected = reference_functional.l1_loss(
                        expected_input,
                        expected_target,
                        reduction="none",
                    )
                self.assert_matches(
                    actual,
                    expected,
                    case=(input_requires_grad, target_requires_grad),
                )
                self.assertIsNone(actual_input_base.grad)
                self.assertIsNone(actual_target_base.grad)
                self.assertIsNone(expected_input_base.grad)
                self.assertIsNone(expected_target_base.grad)

    def test_scalar_broadcast_requires_grad_operands_match_inside_no_grad(self):
        def actual_scalar_input(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(-0.5, requires_grad=input_requires_grad),
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=target_requires_grad,
                ),
            )

        def expected_scalar_input(input_requires_grad, target_requires_grad):
            return (
                reference_torch.tensor(
                    -0.5,
                    dtype=reference_torch.float32,
                    requires_grad=input_requires_grad,
                ),
                reference_torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    dtype=reference_torch.float32,
                    requires_grad=target_requires_grad,
                ),
            )

        def actual_scalar_target(input_requires_grad, target_requires_grad):
            return (
                torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    requires_grad=input_requires_grad,
                ),
                torch.tensor(-0.5, requires_grad=target_requires_grad),
            )

        def expected_scalar_target(input_requires_grad, target_requires_grad):
            return (
                reference_torch.tensor(
                    [[1.0, -2.0], [3.0, -4.0]],
                    dtype=reference_torch.float32,
                    requires_grad=input_requires_grad,
                ),
                reference_torch.tensor(
                    -0.5,
                    dtype=reference_torch.float32,
                    requires_grad=target_requires_grad,
                ),
            )

        for case, actual_factory, expected_factory in (
            ("scalar input", actual_scalar_input, expected_scalar_input),
            ("scalar target", actual_scalar_target, expected_scalar_target),
        ):
            for input_requires_grad, target_requires_grad in (
                (True, False),
                (False, True),
                (True, True),
            ):
                actual_input, actual_target = actual_factory(
                    input_requires_grad,
                    target_requires_grad,
                )
                expected_input, expected_target = expected_factory(
                    input_requires_grad,
                    target_requires_grad,
                )
                with warnings.catch_warnings(), torch.no_grad():
                    warnings.simplefilter("ignore")
                    actual = functional.l1_loss(
                        actual_input,
                        actual_target,
                        reduction="none",
                    )
                with warnings.catch_warnings(), reference_torch.no_grad():
                    warnings.simplefilter("ignore")
                    expected = reference_functional.l1_loss(
                        expected_input,
                        expected_target,
                        reduction="none",
                    )
                self.assert_matches(
                    actual,
                    expected,
                    case=(case, input_requires_grad, target_requires_grad),
                )


if __name__ == "__main__":
    unittest.main()
