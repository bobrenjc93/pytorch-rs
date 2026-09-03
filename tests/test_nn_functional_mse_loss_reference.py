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
class FunctionalMseLossReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.mse_loss differentials require pinned PyTorch 2.13.0"
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

    def make_same_shape_contiguous_cases(self, module):
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
        bandwidth_input = np.linspace(-3.0, 5.0, 1 << 20, dtype=np.float32)
        bandwidth_target = np.linspace(4.0, -2.0, 1 << 20, dtype=np.float32)

        return (
            (
                "scalar",
                self.tensor(module, -0.0),
                self.tensor(module, 2.5),
            ),
            (
                "empty",
                module.zeros((0, 1024), dtype=module.float32),
                module.ones((0, 1024), dtype=module.float32),
            ),
            (
                "small",
                self.tensor(module, [[1.0, -2.0, 3.5], [-0.0, 5.0, -6.5]]),
                self.tensor(module, [[0.25, 4.0, -3.5], [0.0, -7.0, 8.5]]),
            ),
            (
                "signed zero",
                module.tensor(
                    memoryview(
                        np.asarray([0x0000_0000, 0x8000_0000], dtype=np.uint32)
                        .view(np.float32)
                    )
                ),
                module.tensor(
                    memoryview(
                        np.asarray([0x8000_0000, 0x0000_0000], dtype=np.uint32)
                        .view(np.float32)
                    )
                ),
            ),
            (
                "signed-zero-nan-infinity",
                module.tensor(memoryview(edge_input_bits.view(np.float32))).view(2, 4),
                module.tensor(memoryview(edge_target_bits.view(np.float32))).view(2, 4),
            ),
            (
                "bandwidth-sized",
                module.tensor(memoryview(bandwidth_input)),
                module.tensor(memoryview(bandwidth_target)),
            ),
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
        leading_singleton_input = self.tensor(
            module,
            [[0.0, 1.0, 2.0]],
        ).transpose(0, 1)
        leading_singleton_target = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(2, 3, 1).tolist(),
        ).permute(2, 1, 0)
        singleton_output_input = self.tensor(module, [[0.0], [1.0]]).transpose(0, 1)
        singleton_output_target = self.tensor(module, [0.0, 1.0])
        vector = self.tensor(module, [1.0, 2.0, 3.0])
        column = self.tensor(module, [[1.0], [2.0]])
        contiguous = self.tensor(
            module,
            np.linspace(-3.0, 4.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
        )
        empty_contiguous = module.zeros((0, 4), dtype=module.float32)
        offset_strided = self.tensor(
            module,
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist(),
        )[1].transpose(1, 2)
        channels_last = self.tensor(
            module,
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist(),
        ).contiguous(memory_format=module.channels_last)
        singleton_strided = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(3, 1, 2).tolist(),
        ).permute(2, 1, 0)
        empty_strided = module.zeros(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)

        return (
            ("vector target", matrix, vector),
            ("column target", matrix, column),
            ("offset vector target", offset_matrix, vector),
            ("noncontiguous vector target", noncontiguous_matrix, vector),
            (
                "leading singleton broadcast",
                leading_singleton_input,
                leading_singleton_target,
            ),
            (
                "singleton output broadcast",
                singleton_output_input,
                singleton_output_target,
            ),
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

    def make_same_stride_noncontiguous_cases(self, module):
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
        edge_input = module.tensor(memoryview(edge_input_bits.view(np.float32))).view(
            3, 4
        )
        edge_target = module.tensor(memoryview(edge_target_bits.view(np.float32))).view(
            3, 4
        )
        offset_input = self.tensor(
            module,
            np.linspace(-5.0, 5.0, 60, dtype=np.float32).reshape(3, 4, 5).tolist(),
        )[1].transpose(0, 1)
        offset_target = self.tensor(
            module,
            np.linspace(7.0, -3.0, 60, dtype=np.float32).reshape(3, 4, 5).tolist(),
        )[2].transpose(0, 1)
        channels_last_input = self.tensor(
            module,
            np.linspace(-3.0, 4.0, 2 * 3 * 5 * 7, dtype=np.float32)
            .reshape(2, 3, 5, 7)
            .tolist(),
        ).contiguous(memory_format=module.channels_last)
        channels_last_target = self.tensor(
            module,
            np.linspace(11.0, -13.0, 2 * 3 * 5 * 7, dtype=np.float32)
            .reshape(2, 3, 5, 7)
            .tolist(),
        ).contiguous(memory_format=module.channels_last)
        singleton_input = self.tensor(
            module,
            np.arange(6, dtype=np.float32).reshape(3, 1, 2).tolist(),
        ).permute(2, 1, 0)
        singleton_target = self.tensor(
            module,
            np.linspace(3.5, -2.5, 6, dtype=np.float32).reshape(3, 1, 2).tolist(),
        ).permute(2, 1, 0)
        empty_input = module.zeros(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)
        empty_target = module.ones(
            (2, 0, 3), dtype=module.float32
        ).transpose(0, 2)

        return (
            ("transposed edge bits", edge_input.transpose(0, 1), edge_target.transpose(0, 1)),
            ("offset transposed", offset_input, offset_target),
            ("channels-last-like", channels_last_input, channels_last_target),
            ("singleton strided", singleton_input, singleton_target),
            ("empty transposed", empty_input, empty_target),
        )

    @staticmethod
    def call(module_functional, input, target, form):
        if form == "reduction keyword":
            return module_functional.mse_loss(input, target, reduction="none")
        if form == "legacy none keywords":
            return module_functional.mse_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="none",
                weight=None,
            )
        if form == "five positional":
            return module_functional.mse_loss(
                input, target, None, None, "none"
            )
        return module_functional.mse_loss(
            input, target, None, None, "none", None
        )

    @staticmethod
    def call_with_warnings(
        module_functional, input, target, reduction="none", use_default=False
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if use_default:
                output = module_functional.mse_loss(input, target)
            else:
                output = module_functional.mse_loss(input, target, reduction=reduction)
        warning_state = [
            (warning.category.__name__, str(warning.message), warning.filename, warning.lineno)
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
            functional.mse_loss.__name__,
            reference_functional.mse_loss.__name__,
        )
        actual = inspect.signature(functional.mse_loss)
        expected = inspect.signature(reference_functional.mse_loss)
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
            functional.mse_loss.__defaults__,
            reference_functional.mse_loss.__defaults__,
        )
        self.assertEqual(
            tuple(functional.mse_loss.__annotations__),
            tuple(reference_functional.mse_loss.__annotations__),
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

    def test_same_shape_contiguous_cases_match_pytorch_2_13(self):
        actual_cases = self.make_same_shape_contiguous_cases(torch)
        expected_cases = self.make_same_shape_contiguous_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_target = actual_case
            expected_name, expected_input, expected_target = expected_case
            self.assertEqual(case, expected_name)
            self.assertTrue(actual_input.is_contiguous())
            self.assertTrue(actual_target.is_contiguous())
            self.assertTrue(expected_input.is_contiguous())
            self.assertTrue(expected_target.is_contiguous())

            actual = functional.mse_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            expected = reference_functional.mse_loss(
                expected_input,
                expected_target,
                reduction="none",
            )

            self.assert_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(actual.is_set_to(actual_target))
                self.assertFalse(expected.is_set_to(expected_input))
                self.assertFalse(expected.is_set_to(expected_target))

    def test_mean_reduction_same_shape_contiguous_cases_match_pytorch_2_13(self):
        actual_cases = self.make_same_shape_contiguous_cases(torch)
        expected_cases = self.make_same_shape_contiguous_cases(reference_torch)
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
            self.assertTrue(actual_input.is_contiguous())
            self.assertTrue(actual_target.is_contiguous())
            self.assertTrue(expected_input.is_contiguous())
            self.assertTrue(expected_target.is_contiguous())
            actual_input_before = np.asarray(actual_input).reshape(-1).view(np.uint32).copy()
            actual_target_before = np.asarray(actual_target).reshape(-1).view(np.uint32).copy()
            expected_input_before = (
                expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )
            expected_target_before = (
                expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )

            for form, call_kwargs in (
                ("explicit mean", {"reduction": "mean"}),
                ("default mean", {"use_default": True}),
            ):
                actual, actual_warnings = self.call_with_warnings(
                    functional,
                    actual_input,
                    actual_target,
                    **call_kwargs,
                )
                expected, expected_warnings = self.call_with_warnings(
                    reference_functional,
                    expected_input,
                    expected_target,
                    **call_kwargs,
                )

                self.assert_matches(
                    actual,
                    expected,
                    case=(case, form),
                    max_value_ulp=8192 if case == "bandwidth-sized" else 2,
                )
                with self.subTest(case=(case, form), warnings=True):
                    self.assertEqual(actual_warnings, expected_warnings)
                    self.assertEqual(actual_warnings, [])

                actual_repeat, _ = self.call_with_warnings(
                    functional,
                    actual_input,
                    actual_target,
                    **call_kwargs,
                )
                expected_repeat, _ = self.call_with_warnings(
                    reference_functional,
                    expected_input,
                    expected_target,
                    **call_kwargs,
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_target))
                    self.assertFalse(expected.is_set_to(expected_target))
                    self.assertNotEqual(actual.data_ptr(), actual_repeat.data_ptr())

            with self.subTest(case=case, nonmutation=True):
                np.testing.assert_array_equal(
                    np.asarray(actual_input).reshape(-1).view(np.uint32),
                    actual_input_before,
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_target).reshape(-1).view(np.uint32),
                    actual_target_before,
                )
                np.testing.assert_array_equal(
                    expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32),
                    expected_input_before,
                )
                np.testing.assert_array_equal(
                    expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32),
                    expected_target_before,
                )

    def test_same_stride_noncontiguous_cases_match_pytorch_2_13(self):
        actual_cases = self.make_same_stride_noncontiguous_cases(torch)
        expected_cases = self.make_same_stride_noncontiguous_cases(reference_torch)
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
            self.assertEqual(actual_input.shape, actual_target.shape)
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(actual_target.stride(), expected_target.stride())
            self.assertEqual(actual_input.stride(), actual_target.stride())
            if actual_input.numel() != 0:
                self.assertFalse(actual_input.is_contiguous())
                self.assertFalse(actual_target.is_contiguous())
                self.assertFalse(expected_input.is_contiguous())
                self.assertFalse(expected_target.is_contiguous())

            actual_input_bits_before = np.asarray(actual_input).reshape(-1).view(np.uint32).copy()
            actual_target_bits_before = np.asarray(actual_target).reshape(-1).view(np.uint32).copy()
            expected_input_bits_before = (
                expected_input.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )
            expected_target_bits_before = (
                expected_target.detach().cpu().numpy().reshape(-1).view(np.uint32).copy()
            )

            actual = functional.mse_loss(
                actual_input,
                actual_target,
                reduction="none",
            )
            expected = reference_functional.mse_loss(
                expected_input,
                expected_target,
                reduction="none",
            )

            self.assert_matches(actual, expected, case=case)
            with self.subTest(case=case, storage=True):
                actual_repeat = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
                expected_repeat = reference_functional.mse_loss(
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

    def test_broadcasted_outputs_strides_warnings_storage_and_nonmutation_match(self):
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
            actual_input_before = np.asarray(actual_input).copy()
            actual_target_before = np.asarray(actual_target).copy()
            expected_input_before = expected_input.clone()
            expected_target_before = expected_target.clone()

            actual, actual_warnings = self.call_with_warnings(
                functional,
                actual_input,
                actual_target,
            )
            expected, expected_warnings = self.call_with_warnings(
                reference_functional,
                expected_input,
                expected_target,
            )
            self.assert_matches(actual, expected, case=case)
            with self.subTest(case=case, warnings=True):
                self.assertEqual(actual_warnings, expected_warnings)
                self.assertEqual(len(actual_warnings), 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                actual_repeat = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
                expected_repeat = reference_functional.mse_loss(
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
                actual_repeat = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="sum",
                )
                expected_repeat = reference_functional.mse_loss(
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

    def test_mean_reduction_cases_match_pytorch_2_13(self):
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

            for form, call_kwargs in (
                ("explicit mean", {"reduction": "mean"}),
                ("default mean", {"use_default": True}),
            ):
                actual, actual_warnings = self.call_with_warnings(
                    functional,
                    actual_input,
                    actual_target,
                    **call_kwargs,
                )
                expected, expected_warnings = self.call_with_warnings(
                    reference_functional,
                    expected_input,
                    expected_target,
                    **call_kwargs,
                )

                self.assert_matches(
                    actual,
                    expected,
                    case=(case, form),
                    max_value_ulp=2,
                )
                with self.subTest(case=(case, form), warnings=True):
                    self.assertEqual(actual_warnings, expected_warnings)
                    self.assertEqual(len(actual_warnings), int(warns))

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if call_kwargs.get("use_default", False):
                        actual_repeat = functional.mse_loss(
                            actual_input,
                            actual_target,
                        )
                        expected_repeat = reference_functional.mse_loss(
                            expected_input,
                            expected_target,
                        )
                    else:
                        actual_repeat = functional.mse_loss(
                            actual_input,
                            actual_target,
                            reduction="mean",
                        )
                        expected_repeat = reference_functional.mse_loss(
                            expected_input,
                            expected_target,
                            reduction="mean",
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
        actual = functional.mse_loss(
            actual_input,
            actual_target,
            reduction="none",
        )
        expected = reference_functional.mse_loss(
            expected_input,
            expected_target,
            reduction="none",
        )
        self.assertEqual(expected.stride(), (3, 6, 1))
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
            actual = functional.mse_loss(*actual_operands, reduction="none")
            expected = reference_functional.mse_loss(
                *expected_operands,
                reduction="none",
            )
            self.assert_matches(actual, expected, case=("float32 edges", case))

    def test_scalar_broadcast_float32_edge_bits_match_pytorch_2_13(self):
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
        actual_contiguous_tensor = torch.tensor(
            memoryview(tensor_bits.view(np.float32))
        ).view(3, 4)
        expected_contiguous_tensor = reference_torch.tensor(
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
            actual_scalar = torch.tensor(memoryview(scalar_values))[0]
            expected_scalar = reference_torch.tensor(memoryview(scalar_values))[0]
            for layout, actual_tensor, expected_tensor in (
                (
                    "contiguous",
                    actual_contiguous_tensor,
                    expected_contiguous_tensor,
                ),
                (
                    "noncontiguous",
                    actual_contiguous_tensor.transpose(0, 1),
                    expected_contiguous_tensor.transpose(0, 1),
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
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        actual = functional.mse_loss(
                            *actual_operands,
                            reduction="none",
                        )
                        expected = reference_functional.mse_loss(
                            *expected_operands,
                            reduction="none",
                        )
                    self.assert_matches(
                        actual,
                        expected,
                        case=(layout, hex(scalar_bits), scalar_on_left),
                    )

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
            with torch.no_grad():
                actual = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
            with reference_torch.no_grad():
                expected = reference_functional.mse_loss(
                    expected_input,
                    expected_target,
                    reduction="none",
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, target_requires_grad),
            )

    def test_same_stride_noncontiguous_requires_grad_matches_inside_no_grad(self):
        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input_base = torch.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
                requires_grad=input_requires_grad,
            )
            actual_target_base = torch.tensor(
                np.linspace(-2.0, 3.0, 12, dtype=np.float32).reshape(3, 4).tolist(),
                requires_grad=target_requires_grad,
            )
            expected_input_base = reference_torch.tensor(
                np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_target_base = reference_torch.tensor(
                np.linspace(-2.0, 3.0, 12, dtype=np.float32).reshape(3, 4).tolist(),
                dtype=reference_torch.float32,
                requires_grad=target_requires_grad,
            )
            actual_input = actual_input_base.transpose(0, 1)
            actual_target = actual_target_base.transpose(0, 1)
            expected_input = expected_input_base.transpose(0, 1)
            expected_target = expected_target_base.transpose(0, 1)

            with torch.no_grad():
                actual = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
            with reference_torch.no_grad():
                expected = reference_functional.mse_loss(
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

    def test_broadcast_requires_grad_operands_match_inside_no_grad(self):
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
                [0.5, -1.5],
                requires_grad=target_requires_grad,
            )
            expected_input = reference_torch.tensor(
                [[1.0, -2.0], [3.0, -4.0]],
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_target = reference_torch.tensor(
                [0.5, -1.5],
                dtype=reference_torch.float32,
                requires_grad=target_requires_grad,
            )
            with warnings.catch_warnings(), torch.no_grad():
                warnings.simplefilter("ignore")
                actual = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
            with warnings.catch_warnings(), reference_torch.no_grad():
                warnings.simplefilter("ignore")
                expected = reference_functional.mse_loss(
                    expected_input,
                    expected_target,
                    reduction="none",
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, target_requires_grad),
            )

    def test_sum_reduction_requires_grad_operands_match_inside_no_grad(self):
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
            with torch.no_grad():
                actual = functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="sum",
                )
            with reference_torch.no_grad():
                expected = reference_functional.mse_loss(
                    expected_input,
                    expected_target,
                    reduction="sum",
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, target_requires_grad),
                max_value_ulp=1,
            )

    def test_mean_reduction_requires_grad_operands_match_inside_no_grad(self):
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
            for form, actual_call, expected_call in (
                (
                    "explicit mean",
                    lambda: functional.mse_loss(
                        actual_input,
                        actual_target,
                        reduction="mean",
                    ),
                    lambda: reference_functional.mse_loss(
                        expected_input,
                        expected_target,
                        reduction="mean",
                    ),
                ),
                (
                    "default mean",
                    lambda: functional.mse_loss(actual_input, actual_target),
                    lambda: reference_functional.mse_loss(
                        expected_input,
                        expected_target,
                    ),
                ),
            ):
                with self.subTest(
                    form=form,
                    input_requires_grad=input_requires_grad,
                    target_requires_grad=target_requires_grad,
                ):
                    with torch.no_grad():
                        actual = actual_call()
                    with reference_torch.no_grad():
                        expected = expected_call()
                    self.assert_matches(
                        actual,
                        expected,
                        case=form,
                        max_value_ulp=1,
                    )

    def test_sum_reduction_no_grad_view_repeated_backward_matches_pytorch_2_13(self):
        actual_leaf = torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]],
            requires_grad=True,
        )
        expected_leaf = reference_torch.tensor(
            [[1.0, -2.0], [3.0, -4.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            actual_view = actual_leaf[0]
        with reference_torch.no_grad():
            expected_view = expected_leaf[0]

        actual_loss = functional.mse_loss(actual_view, actual_view, reduction="sum")
        expected_loss = reference_functional.mse_loss(
            expected_view,
            expected_view,
            reduction="sum",
        )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="no_grad view",
            max_value_ulp=1,
        )

        actual_loss.backward()
        expected_loss.backward()
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            actual_loss.backward()
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            expected_loss.backward()

    def test_mean_reduction_active_autograd_same_shape_contiguous_matches_pytorch_2_13(self):
        def assert_grads_match(actual_sources, expected_sources, *, case):
            for actual, expected in zip(actual_sources, expected_sources, strict=True):
                with self.subTest(case=case, source=tuple(expected.shape)):
                    if expected.grad is None:
                        self.assertIsNone(actual.grad)
                    else:
                        self.assertEqual(actual.grad.shape, tuple(expected.grad.shape))
                        self.assertEqual(actual.grad.stride(), expected.grad.stride())
                        np.testing.assert_array_max_ulp(
                            np.asarray(actual.grad),
                            expected.grad.detach().cpu().numpy(),
                            maxulp=1,
                        )

        for input_requires_grad, target_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input = torch.tensor(
                [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
                requires_grad=input_requires_grad,
            )
            actual_target = torch.tensor(
                [[0.5, 2.0, -3.0], [1.0, -1.0, 0.0]],
                requires_grad=target_requires_grad,
            )
            expected_input = reference_torch.tensor(
                [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_target = reference_torch.tensor(
                [[0.5, 2.0, -3.0], [1.0, -1.0, 0.0]],
                dtype=reference_torch.float32,
                requires_grad=target_requires_grad,
            )

            with self.subTest(
                input_requires_grad=input_requires_grad,
                target_requires_grad=target_requires_grad,
            ):
                actual_loss = functional.mse_loss(actual_input, actual_target)
                expected_loss = reference_functional.mse_loss(
                    expected_input,
                    expected_target,
                )
                self.assert_matches(
                    actual_loss,
                    expected_loss,
                    case="same-shape contiguous autograd",
                    max_value_ulp=2,
                )
                self.assertTrue(actual_loss.requires_grad)
                self.assertFalse(actual_loss.is_leaf)
                actual_loss.backward()
                expected_loss.backward()
                assert_grads_match(
                    (actual_input, actual_target),
                    (expected_input, expected_target),
                    case=(input_requires_grad, target_requires_grad),
                )

    def test_sum_reduction_first_order_backward_matches_pytorch_2_13(self):
        def assert_grads_match(actual_sources, expected_sources, *, case):
            for actual, expected in zip(actual_sources, expected_sources, strict=True):
                with self.subTest(case=case, source=tuple(expected.shape)):
                    if expected.grad is None:
                        self.assertIsNone(actual.grad)
                    else:
                        self.assertEqual(actual.grad.shape, tuple(expected.grad.shape))
                        self.assertEqual(actual.grad.stride(), expected.grad.stride())
                        np.testing.assert_array_equal(
                            np.asarray(actual.grad).reshape(-1).view(np.uint32),
                            expected.grad.detach().cpu().numpy().reshape(-1).view(np.uint32),
                        )

        actual_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        actual_target = torch.tensor(
            [[0.5, 2.0, -3.0], [1.0, -1.0, 0.0]], requires_grad=True
        )
        expected_input = reference_torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_target = reference_torch.tensor(
            [[0.5, 2.0, -3.0], [1.0, -1.0, 0.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_loss = functional.mse_loss(
            actual_input, actual_target, reduction="sum"
        )
        expected_loss = reference_functional.mse_loss(
            expected_input, expected_target, reduction="sum"
        )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="same shape",
            max_value_ulp=1,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="same shape",
        )

        actual_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        actual_target = torch.tensor([0.5, -1.5, 2.0], requires_grad=True)
        expected_input = reference_torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_target = reference_torch.tensor(
            [0.5, -1.5, 2.0],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_loss = functional.mse_loss(
                actual_input, actual_target, reduction="sum"
            )
            expected_loss = reference_functional.mse_loss(
                expected_input, expected_target, reduction="sum"
            )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="broadcast",
            max_value_ulp=1,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
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
        expected_input_base = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_target_base = reference_torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 4, 3),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_loss = functional.mse_loss(
            actual_input_base[1].transpose(0, 1),
            actual_target_base[0].transpose(0, 1),
            reduction="sum",
        )
        expected_loss = reference_functional.mse_loss(
            expected_input_base[1].transpose(0, 1),
            expected_target_base[0].transpose(0, 1),
            reduction="sum",
        )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="offset noncontiguous",
            max_value_ulp=1,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
            (actual_input_base, actual_target_base),
            (expected_input_base, expected_target_base),
            case="offset noncontiguous",
        )

        actual_input = torch.zeros((2, 0, 3), requires_grad=True)
        actual_target = torch.ones((2, 0, 3), requires_grad=True)
        expected_input = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        expected_target = reference_torch.ones(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = functional.mse_loss(
            actual_input.transpose(0, 2),
            actual_target.transpose(0, 2),
            reduction="sum",
        )
        expected_loss = reference_functional.mse_loss(
            expected_input.transpose(0, 2),
            expected_target.transpose(0, 2),
            reduction="sum",
        )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="empty",
            max_value_ulp=1,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="empty",
        )

    def test_mean_reduction_first_order_backward_matches_pytorch_2_13(self):
        def assert_grads_match(actual_sources, expected_sources, *, case):
            for actual, expected in zip(actual_sources, expected_sources, strict=True):
                with self.subTest(case=case, source=tuple(expected.shape)):
                    if expected.grad is None:
                        self.assertIsNone(actual.grad)
                    else:
                        self.assertEqual(actual.grad.shape, tuple(expected.grad.shape))
                        self.assertEqual(actual.grad.stride(), expected.grad.stride())
                        np.testing.assert_array_max_ulp(
                            np.asarray(actual.grad),
                            expected.grad.detach().cpu().numpy(),
                            maxulp=1,
                        )

        actual_input = torch.tensor(1.0, requires_grad=True)
        actual_target = torch.tensor(-2.0, requires_grad=True)
        expected_input = reference_torch.tensor(
            1.0,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_target = reference_torch.tensor(
            -2.0,
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_loss = functional.mse_loss(actual_input, actual_target)
        expected_loss = reference_functional.mse_loss(expected_input, expected_target)
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="scalar",
            max_value_ulp=2,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="scalar",
        )

        actual_input = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]], requires_grad=True
        )
        actual_target = torch.tensor([0.5, -1.5, 2.0], requires_grad=True)
        expected_input = reference_torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_target = reference_torch.tensor(
            [0.5, -1.5, 2.0],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actual_loss = functional.mse_loss(
                actual_input,
                actual_target,
                reduction="mean",
            )
            expected_loss = reference_functional.mse_loss(
                expected_input,
                expected_target,
                reduction="mean",
            )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="broadcast",
            max_value_ulp=2,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
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
        expected_input_base = reference_torch.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_target_base = reference_torch.tensor(
            np.linspace(-2.0, 2.0, 24, dtype=np.float32).reshape(2, 4, 3),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_loss = functional.mse_loss(
            actual_input_base[1].transpose(0, 1),
            actual_target_base[0].transpose(0, 1),
            reduction="mean",
        )
        expected_loss = reference_functional.mse_loss(
            expected_input_base[1].transpose(0, 1),
            expected_target_base[0].transpose(0, 1),
            reduction="mean",
        )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="offset noncontiguous",
            max_value_ulp=2,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
            (actual_input_base, actual_target_base),
            (expected_input_base, expected_target_base),
            case="offset noncontiguous",
        )

        actual_input = torch.zeros((2, 0, 3), requires_grad=True)
        actual_target = torch.ones((2, 0, 3), requires_grad=True)
        expected_input = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        expected_target = reference_torch.ones(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = functional.mse_loss(
            actual_input.transpose(0, 2),
            actual_target.transpose(0, 2),
            reduction="mean",
        )
        expected_loss = reference_functional.mse_loss(
            expected_input.transpose(0, 2),
            expected_target.transpose(0, 2),
            reduction="mean",
        )
        self.assert_matches(
            actual_loss,
            expected_loss,
            case="empty",
            max_value_ulp=2,
        )
        actual_loss.backward()
        expected_loss.backward()
        assert_grads_match(
            (actual_input, actual_target),
            (expected_input, expected_target),
            case="empty",
        )

    def test_sum_reduction_backward_edge_bits_match_pytorch_2_13(self):
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
        actual_input = torch.tensor(
            memoryview(input_bits.view(np.float32)),
            requires_grad=True,
        )
        actual_target = torch.tensor(
            memoryview(target_bits.view(np.float32)),
            requires_grad=True,
        )
        expected_input = reference_torch.tensor(
            memoryview(input_bits.view(np.float32)),
            requires_grad=True,
        )
        expected_target = reference_torch.tensor(
            memoryview(target_bits.view(np.float32)),
            requires_grad=True,
        )

        functional.mse_loss(
            actual_input,
            actual_target,
            reduction="sum",
        ).backward()
        reference_functional.mse_loss(
            expected_input,
            expected_target,
            reduction="sum",
        ).backward()

        for name, actual, expected in (
            ("input", actual_input, expected_input),
            ("target", actual_target, expected_target),
        ):
            with self.subTest(operand=name):
                self.assertEqual(actual.grad.shape, tuple(expected.grad.shape))
                self.assertEqual(actual.grad.stride(), expected.grad.stride())
                np.testing.assert_array_equal(
                    np.asarray(actual.grad).reshape(-1).view(np.uint32),
                    expected.grad.detach().cpu().numpy().reshape(-1).view(np.uint32),
                )

    def test_unbroadcastable_shape_warning_and_error_match_pytorch_2_13(self):
        actual_input = torch.ones((2, 3))
        actual_target = torch.zeros((2, 2))
        expected_input = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        expected_target = reference_torch.zeros((2, 2), dtype=reference_torch.float32)

        with warnings.catch_warnings(record=True) as actual_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as actual_error:
                functional.mse_loss(
                    actual_input,
                    actual_target,
                    reduction="none",
                )
        with warnings.catch_warnings(record=True) as expected_warnings:
            warnings.simplefilter("always")
            with self.assertRaises(RuntimeError) as expected_error:
                reference_functional.mse_loss(
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


if __name__ == "__main__":
    unittest.main()
