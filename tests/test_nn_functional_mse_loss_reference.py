import inspect
import unittest

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
    def call_mean(module_functional, input, target, form):
        if form == "default":
            return module_functional.mse_loss(input, target)
        if form == "reduction keyword":
            return module_functional.mse_loss(input, target, reduction="mean")
        if form == "all keywords":
            return module_functional.mse_loss(
                input=input,
                target=target,
                size_average=None,
                reduce=None,
                reduction="mean",
                weight=None,
            )
        if form == "five positional":
            return module_functional.mse_loss(
                input, target, None, None, "mean"
            )
        return module_functional.mse_loss(
            input, target, None, None, "mean", None
        )

    def assert_matches(self, actual, expected, *, case):
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
            np.testing.assert_array_equal(
                np.asarray(actual).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def assert_mean_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            np.testing.assert_allclose(
                actual.item(),
                expected.item(),
                rtol=1.3e-6,
                atol=1.0e-5,
                equal_nan=True,
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

    def test_default_and_explicit_mean_layouts_storage_and_nonmutation_match(self):
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
                "default",
                "reduction keyword",
                "all keywords",
                "five positional",
                "six positional",
            ):
                actual = self.call_mean(
                    functional,
                    actual_input,
                    actual_target,
                    form,
                )
                expected = self.call_mean(
                    reference_functional,
                    expected_input,
                    expected_target,
                    form,
                )
                self.assert_mean_matches(actual, expected, case=(case, form))

                actual_repeat = self.call_mean(
                    functional,
                    actual_input,
                    actual_target,
                    form,
                )
                expected_repeat = self.call_mean(
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

    def test_mean_scalar_empty_and_extreme_bits_match_pytorch_2_13(self):
        cases = (
            ("scalar", -0.0, 2.5),
            ("underflow", [1.0e-23], [0.0]),
            ("square overflow", [np.finfo(np.float32).max], [0.0]),
            ("reduction overflow", [1.0e19] * 4, [0.0] * 4),
            ("infinite difference", [float("inf")], [-float("inf")]),
            ("indeterminate difference", [float("inf")], [float("inf")]),
            ("nan input", [float("nan")], [0.0]),
        )
        for case, input_values, target_values in cases:
            with self.subTest(case=case):
                actual = functional.mse_loss(
                    self.tensor(torch, input_values),
                    self.tensor(torch, target_values),
                )
                expected = reference_functional.mse_loss(
                    self.tensor(reference_torch, input_values),
                    self.tensor(reference_torch, target_values),
                )
                if np.isnan(expected.item()):
                    self.assertTrue(np.isnan(actual.item()))
                else:
                    np.testing.assert_array_equal(
                        np.asarray(actual).view(np.uint32),
                        expected.detach().cpu().numpy().view(np.uint32),
                    )

        actual_empty = functional.mse_loss(
            torch.zeros((2, 0, 3)),
            torch.ones((2, 0, 3)),
        )
        expected_empty = reference_functional.mse_loss(
            reference_torch.zeros((2, 0, 3)),
            reference_torch.ones((2, 0, 3)),
        )
        self.assertTrue(np.isnan(actual_empty.item()))
        self.assertTrue(np.isnan(expected_empty.item()))

    def test_mean_large_adversarial_reductions_match_pytorch_2_13(self):
        for case, element_count, input_value in (
            ("large repeated square", 5_000, 1_000.0),
            ("beyond serial float32 integer precision", 20_000_000, 1.0),
        ):
            with self.subTest(case=case):
                actual = functional.mse_loss(
                    torch.full((element_count,), input_value),
                    torch.zeros((element_count,)),
                )
                expected = reference_functional.mse_loss(
                    reference_torch.full(
                        (element_count,),
                        input_value,
                        dtype=reference_torch.float32,
                    ),
                    reference_torch.zeros(
                        (element_count,),
                        dtype=reference_torch.float32,
                    ),
                )
                np.testing.assert_array_equal(
                    np.asarray(actual).view(np.uint32),
                    expected.detach().cpu().numpy().view(np.uint32),
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
        input_values = [
            -0.0,
            0.0,
            1.0e-20,
            -1.0e-20,
            1.0,
            -1.0,
            1.0e10,
            -1.0e10,
            np.finfo(np.float32).max,
            -np.finfo(np.float32).max,
        ]
        target_values = [
            0.0,
            -0.0,
            0.0,
            0.0,
            -1.0,
            1.0,
            -1.0e10,
            1.0e10,
            0.0,
            0.0,
        ]
        actual = functional.mse_loss(
            self.tensor(torch, input_values),
            self.tensor(torch, target_values),
            reduction="none",
        )
        expected = reference_functional.mse_loss(
            self.tensor(reference_torch, input_values),
            self.tensor(reference_torch, target_values),
            reduction="none",
        )
        self.assert_matches(actual, expected, case="float32 edges")

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

            with torch.no_grad():
                actual_mean = functional.mse_loss(actual_input, actual_target)
            with reference_torch.no_grad():
                expected_mean = reference_functional.mse_loss(
                    expected_input,
                    expected_target,
                )
            self.assert_mean_matches(
                actual_mean,
                expected_mean,
                case=(input_requires_grad, target_requires_grad, "mean"),
            )


if __name__ == "__main__":
    unittest.main()
