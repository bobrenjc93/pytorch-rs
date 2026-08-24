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
class FunctionalLinearReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.linear differentials require pinned PyTorch 2.13.0"
            )

    def make_cases(self, module):
        contiguous_input = module.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            dtype=module.float32,
        )
        contiguous_weight = module.tensor(
            np.arange(12, dtype=np.float32).reshape(4, 3).tolist(),
            dtype=module.float32,
        )
        strided_input = module.tensor(
            np.arange(6, dtype=np.float32).reshape(3, 2).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        strided_weight = module.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        offset_input = module.tensor(
            np.arange(18, dtype=np.float32).reshape(3, 2, 3).tolist(),
            dtype=module.float32,
        )[1]
        offset_weight = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            dtype=module.float32,
        )[1]
        offset_strided_input = module.tensor(
            np.arange(18, dtype=np.float32).reshape(3, 3, 2).tolist(),
            dtype=module.float32,
        )[1].transpose(0, 1)
        offset_strided_weight = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )[1].transpose(0, 1)
        empty_offset_input = module.zeros((2, 0, 2), dtype=module.float32).transpose(
            0, 2
        )[1]
        return (
            ("contiguous", contiguous_input, contiguous_weight),
            ("strided", strided_input, strided_weight),
            ("offset", offset_input, offset_weight),
            (
                "offset-strided",
                offset_strided_input,
                offset_strided_weight,
            ),
            (
                "zero rows",
                module.zeros((0, 3), dtype=module.float32),
                contiguous_weight,
            ),
            (
                "zero inner",
                module.zeros((2, 0), dtype=module.float32),
                module.zeros((4, 0), dtype=module.float32),
            ),
            (
                "zero outputs",
                contiguous_input,
                module.zeros((0, 3), dtype=module.float32),
            ),
            (
                "offset zero rows",
                empty_offset_input,
                module.ones((4, 2), dtype=module.float32),
            ),
            (
                "all zero",
                module.zeros((0, 0), dtype=module.float32),
                module.zeros((0, 0), dtype=module.float32),
            ),
        )

    def make_vector_cases(self, module):
        contiguous_input = module.tensor(
            [1.0, -2.0, 3.0],
            dtype=module.float32,
        )
        contiguous_weight = module.tensor(
            np.arange(12, dtype=np.float32).reshape(4, 3).tolist(),
            dtype=module.float32,
        )
        strided_source = module.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        offset_input = module.tensor(
            np.arange(6, dtype=np.float32).reshape(2, 3).tolist(),
            dtype=module.float32,
        )[1]
        strided_weight = module.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)

        return (
            ("contiguous vector", contiguous_input, contiguous_weight),
            ("strided vector", strided_source[0], contiguous_weight),
            ("offset vector", offset_input, contiguous_weight),
            ("offset-strided vector", strided_source[1], strided_weight),
            (
                "zero features",
                module.zeros((0,), dtype=module.float32),
                module.zeros((4, 0), dtype=module.float32),
            ),
            (
                "zero outputs",
                contiguous_input,
                module.zeros((0, 3), dtype=module.float32),
            ),
            (
                "all zero",
                module.zeros((0,), dtype=module.float32),
                module.zeros((0, 0), dtype=module.float32),
            ),
        )

    def make_rank_three_cases(self, module):
        contiguous_input = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        contiguous_weight = module.tensor(
            np.arange(20, dtype=np.float32).reshape(5, 4).tolist(),
            dtype=module.float32,
        )
        strided_input = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 4, 3).tolist(),
            dtype=module.float32,
        ).transpose(1, 2)
        strided_weight = module.tensor(
            np.arange(20, dtype=np.float32).reshape(4, 5).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        offset_input = module.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4).tolist(),
            dtype=module.float32,
        )[1]
        offset_weight = module.tensor(
            np.arange(40, dtype=np.float32).reshape(2, 5, 4).tolist(),
            dtype=module.float32,
        )[1]
        offset_strided_input = module.tensor(
            np.arange(48, dtype=np.float32).reshape(2, 2, 4, 3).tolist(),
            dtype=module.float32,
        )[1].transpose(1, 2)
        offset_strided_weight = module.tensor(
            np.arange(40, dtype=np.float32).reshape(2, 4, 5).tolist(),
            dtype=module.float32,
        )[1].transpose(0, 1)
        empty_offset_input = module.zeros(
            (2, 2, 0, 4), dtype=module.float32
        ).transpose(1, 2)[1]

        return (
            ("contiguous batch", contiguous_input, contiguous_weight),
            ("strided batch", strided_input, strided_weight),
            ("offset batch", offset_input, offset_weight),
            (
                "offset-strided batch",
                offset_strided_input,
                offset_strided_weight,
            ),
            (
                "zero batch",
                module.zeros((0, 3, 4), dtype=module.float32),
                contiguous_weight,
            ),
            (
                "zero sequence",
                module.zeros((2, 0, 4), dtype=module.float32),
                contiguous_weight,
            ),
            (
                "zero inner",
                module.zeros((2, 3, 0), dtype=module.float32),
                module.zeros((5, 0), dtype=module.float32),
            ),
            (
                "zero outputs",
                contiguous_input,
                module.zeros((0, 4), dtype=module.float32),
            ),
            (
                "offset zero batch",
                empty_offset_input,
                module.ones((5, 4), dtype=module.float32),
            ),
            (
                "all zero",
                module.zeros((0, 0, 0), dtype=module.float32),
                module.zeros((0, 0), dtype=module.float32),
            ),
        )

    def make_rank_four_cases(self, module):
        contiguous_input = module.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5).tolist(),
            dtype=module.float32,
        )
        contiguous_weight = module.tensor(
            np.arange(30, dtype=np.float32).reshape(6, 5).tolist(),
            dtype=module.float32,
        )
        permuted_input = module.tensor(
            np.arange(120, dtype=np.float32).reshape(3, 4, 2, 5).tolist(),
            dtype=module.float32,
        ).permute(2, 0, 1, 3)
        strided_input = module.tensor(
            np.arange(120, dtype=np.float32).reshape(2, 3, 5, 4).tolist(),
            dtype=module.float32,
        ).transpose(2, 3)
        strided_weight = module.tensor(
            np.arange(30, dtype=np.float32).reshape(5, 6).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        offset_input = module.tensor(
            np.arange(240, dtype=np.float32).reshape(2, 2, 3, 4, 5).tolist(),
            dtype=module.float32,
        )[1]
        offset_weight = module.tensor(
            np.arange(60, dtype=np.float32).reshape(2, 6, 5).tolist(),
            dtype=module.float32,
        )[1]
        offset_strided_input = module.tensor(
            np.arange(240, dtype=np.float32).reshape(2, 2, 3, 5, 4).tolist(),
            dtype=module.float32,
        )[1].transpose(2, 3)
        offset_strided_weight = module.tensor(
            np.arange(60, dtype=np.float32).reshape(2, 5, 6).tolist(),
            dtype=module.float32,
        )[1].transpose(0, 1)
        empty_offset_input = module.zeros(
            (2, 2, 3, 0, 5), dtype=module.float32
        ).transpose(2, 3)[1]

        return (
            ("contiguous rank-four", contiguous_input, contiguous_weight),
            ("permuted rank-four", permuted_input, contiguous_weight),
            ("strided rank-four", strided_input, strided_weight),
            ("offset rank-four", offset_input, offset_weight),
            (
                "offset-strided rank-four",
                offset_strided_input,
                offset_strided_weight,
            ),
            (
                "zero outer",
                module.zeros((0, 3, 4, 5), dtype=module.float32),
                contiguous_weight,
            ),
            (
                "zero batch",
                module.zeros((2, 0, 4, 5), dtype=module.float32),
                contiguous_weight,
            ),
            (
                "zero sequence",
                module.zeros((2, 3, 0, 5), dtype=module.float32),
                contiguous_weight,
            ),
            (
                "zero inner",
                module.zeros((2, 3, 4, 0), dtype=module.float32),
                module.zeros((6, 0), dtype=module.float32),
            ),
            (
                "zero outputs",
                contiguous_input,
                module.zeros((0, 5), dtype=module.float32),
            ),
            (
                "offset zero batch",
                empty_offset_input,
                module.ones((6, 5), dtype=module.float32),
            ),
            (
                "all zero",
                module.zeros((0, 0, 0, 0), dtype=module.float32),
                module.zeros((0, 0), dtype=module.float32),
            ),
        )

    def assert_matches(self, actual, expected, *, case):
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
                np.asarray(actual, dtype=np.float32),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=0.0,
            )

    @staticmethod
    def call(module_functional, input, weight, form):
        if form == "positional":
            return module_functional.linear(input, weight)
        if form == "explicit none":
            return module_functional.linear(input, weight, None)
        return module_functional.linear(input=input, weight=weight, bias=None)

    def test_name_and_signature_surface(self):
        self.assertEqual(
            functional.linear.__name__, reference_functional.linear.__name__
        )
        signature = inspect.signature(functional.linear)
        self.assertEqual(tuple(signature.parameters), ("input", "weight", "bias"))
        self.assertIsNone(signature.parameters["bias"].default)

    def test_contiguous_strided_offset_and_zero_sized_results_match(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input, actual_weight = actual_case
            expected_name, expected_input, expected_weight = expected_case
            self.assertEqual(case, expected_name)
            for form in ("positional", "explicit none", "keywords"):
                actual = self.call(functional, actual_input, actual_weight, form)
                expected = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call(
                    functional, actual_input, actual_weight, form
                )
                expected_repeat = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_weight))
                    self.assertFalse(expected.is_set_to(expected_weight))

    def test_vector_layouts_values_metadata_and_storage_match(self):
        actual_cases = self.make_vector_cases(torch)
        expected_cases = self.make_vector_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input, actual_weight = actual_case
            expected_name, expected_input, expected_weight = expected_case
            self.assertEqual(case, expected_name)
            for form in ("positional", "explicit none", "keywords"):
                actual = self.call(functional, actual_input, actual_weight, form)
                expected = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call(
                    functional, actual_input, actual_weight, form
                )
                expected_repeat = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_weight))
                    self.assertFalse(expected.is_set_to(expected_weight))

    def test_rank_three_layouts_values_metadata_and_storage_match(self):
        actual_cases = self.make_rank_three_cases(torch)
        expected_cases = self.make_rank_three_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input, actual_weight = actual_case
            expected_name, expected_input, expected_weight = expected_case
            self.assertEqual(case, expected_name)
            for form in ("positional", "explicit none", "keywords"):
                actual = self.call(functional, actual_input, actual_weight, form)
                expected = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call(
                    functional, actual_input, actual_weight, form
                )
                expected_repeat = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_weight))
                    self.assertFalse(expected.is_set_to(expected_weight))

    def test_rank_four_layouts_values_metadata_and_storage_match(self):
        actual_cases = self.make_rank_four_cases(torch)
        expected_cases = self.make_rank_four_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input, actual_weight = actual_case
            expected_name, expected_input, expected_weight = expected_case
            self.assertEqual(case, expected_name)
            for form in ("positional", "explicit none", "keywords"):
                actual = self.call(functional, actual_input, actual_weight, form)
                expected = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call(
                    functional, actual_input, actual_weight, form
                )
                expected_repeat = self.call(
                    reference_functional, expected_input, expected_weight, form
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))
                    self.assertFalse(actual.is_set_to(actual_weight))
                    self.assertFalse(expected.is_set_to(expected_weight))

    def test_rank_four_finite_overflow_classification_matches_pytorch_2_13(self):
        maximum = np.finfo(np.float32).max
        case_metadata = (
            ("complete tiles", (1, 1, 4, 2), 16, False),
            ("column remainder", (1, 1, 4, 2), 12, False),
            ("row remainder", (1, 1, 5, 2), 16, False),
            ("batched matrix", (2, 2, 1, 2), 16, True),
        )

        actual_classifications = {}
        for case, shape, out_features, transpose in case_metadata:
            input_values = np.full(shape, maximum, dtype=np.float32)
            weight_values = np.zeros((out_features, 2), dtype=np.float32)
            weight_values[0] = (maximum, -maximum)
            actual_input = torch.tensor(input_values.tolist())
            expected_input = reference_torch.tensor(
                input_values,
                dtype=reference_torch.float32,
            )
            if transpose:
                actual_input = actual_input.transpose(0, 1)
                expected_input = expected_input.transpose(0, 1)
            actual_weight = torch.tensor(weight_values.tolist())
            expected_weight = reference_torch.tensor(
                weight_values,
                dtype=reference_torch.float32,
            )
            actual = functional.linear(actual_input, actual_weight)
            expected = reference_functional.linear(expected_input, expected_weight)
            self.assert_matches(actual, expected, case=case)

            actual_values = np.asarray(actual)
            actual_classifications[case] = actual_values
            expected_values = expected.detach().cpu().numpy()
            for classification in (np.isnan, np.isposinf, np.isneginf):
                with self.subTest(case=case, classification=classification.__name__):
                    np.testing.assert_array_equal(
                        classification(actual_values),
                        classification(expected_values),
                    )

        self.assertTrue(
            np.isposinf(actual_classifications["complete tiles"][..., 0]).all()
        )
        self.assertTrue(
            np.isposinf(actual_classifications["column remainder"][..., 0]).all()
        )
        self.assertTrue(
            np.isposinf(actual_classifications["row remainder"][..., 0]).all()
        )
        self.assertTrue(np.isnan(actual_classifications["batched matrix"][..., 0]).all())

    def test_rank_four_negative_zero_bits_match_pytorch_2_13(self):
        actual = functional.linear(
            torch.zeros((1, 1, 4, 1)),
            torch.tensor([[-1.0]] * 16),
        )
        expected = reference_functional.linear(
            reference_torch.zeros((1, 1, 4, 1), dtype=reference_torch.float32),
            -reference_torch.ones((16, 1), dtype=reference_torch.float32),
        )
        self.assert_matches(actual, expected, case="negative zero")
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint32),
            expected.numpy().view(np.uint32),
        )
        self.assertTrue(
            (np.asarray(actual).view(np.uint32) == np.uint32(0x80000000)).all()
        )

    def test_requires_grad_operands_match_inside_no_grad(self):
        for input_requires_grad, weight_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input = (
                torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    requires_grad=input_requires_grad,
                )
                .transpose(0, 1)
                .transpose(0, 1)
            )
            actual_weight = (
                torch.tensor(
                    [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                    requires_grad=weight_requires_grad,
                )
                .transpose(0, 1)
                .transpose(0, 1)
            )
            expected_input = (
                reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=reference_torch.float32,
                    requires_grad=input_requires_grad,
                )
                .transpose(0, 1)
                .transpose(0, 1)
            )
            expected_weight = (
                reference_torch.tensor(
                    [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                    dtype=reference_torch.float32,
                    requires_grad=weight_requires_grad,
                )
                .transpose(0, 1)
                .transpose(0, 1)
            )
            with torch.no_grad():
                actual = functional.linear(actual_input, actual_weight)
            with reference_torch.no_grad():
                expected = reference_functional.linear(
                    expected_input, expected_weight
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, weight_requires_grad),
            )

    def test_vector_requires_grad_operands_match_inside_no_grad(self):
        for input_requires_grad, weight_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input = torch.tensor(
                [1.0, 2.0, 3.0],
                requires_grad=input_requires_grad,
            )
            actual_weight = torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                requires_grad=weight_requires_grad,
            )
            expected_input = reference_torch.tensor(
                [1.0, 2.0, 3.0],
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_weight = reference_torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                dtype=reference_torch.float32,
                requires_grad=weight_requires_grad,
            )
            with torch.no_grad():
                actual = functional.linear(actual_input, actual_weight)
            with reference_torch.no_grad():
                expected = reference_functional.linear(
                    expected_input, expected_weight
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, weight_requires_grad),
            )

    def test_rank_three_requires_grad_operands_match_inside_no_grad(self):
        input_values = np.arange(12, dtype=np.float32).reshape(2, 2, 3).tolist()
        weight_values = [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]]
        for input_requires_grad, weight_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input = torch.tensor(
                input_values,
                requires_grad=input_requires_grad,
            )
            actual_weight = torch.tensor(
                weight_values,
                requires_grad=weight_requires_grad,
            )
            expected_input = reference_torch.tensor(
                input_values,
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_weight = reference_torch.tensor(
                weight_values,
                dtype=reference_torch.float32,
                requires_grad=weight_requires_grad,
            )
            with torch.no_grad():
                actual = functional.linear(actual_input, actual_weight)
            with reference_torch.no_grad():
                expected = reference_functional.linear(
                    expected_input, expected_weight
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, weight_requires_grad),
            )

    def test_rank_four_requires_grad_operands_match_inside_no_grad(self):
        input_values = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3).tolist()
        weight_values = [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]]
        for input_requires_grad, weight_requires_grad in (
            (True, False),
            (False, True),
            (True, True),
        ):
            actual_input = torch.tensor(
                input_values,
                requires_grad=input_requires_grad,
            )
            actual_weight = torch.tensor(
                weight_values,
                requires_grad=weight_requires_grad,
            )
            expected_input = reference_torch.tensor(
                input_values,
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_weight = reference_torch.tensor(
                weight_values,
                dtype=reference_torch.float32,
                requires_grad=weight_requires_grad,
            )
            with torch.no_grad():
                actual = functional.linear(actual_input, actual_weight)
            with reference_torch.no_grad():
                expected = reference_functional.linear(
                    expected_input, expected_weight
                )
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, weight_requires_grad),
            )

    def test_incompatible_inner_dimension_error_matches(self):
        for shape in ((2, 3), (3,), (2, 3, 4), (2, 3, 4, 6)):
            actual_input = torch.zeros(shape)
            actual_weight = torch.zeros((4, 5))
            expected_input = reference_torch.zeros(
                shape, dtype=reference_torch.float32
            )
            expected_weight = reference_torch.zeros(
                (4, 5), dtype=reference_torch.float32
            )
            with self.subTest(shape=shape):
                with self.assertRaises(Exception) as actual_raised:
                    functional.linear(actual_input, actual_weight)
                with self.assertRaises(Exception) as expected_raised:
                    reference_functional.linear(expected_input, expected_weight)
                self.assertIs(
                    type(actual_raised.exception), type(expected_raised.exception)
                )
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )

    def test_noncontiguous_rank_three_inner_dimension_error_matches(self):
        cases = (
            (
                "unfoldable leading transpose",
                torch.zeros((3, 2, 4)).transpose(0, 1),
                torch.zeros((5, 6)),
                reference_torch.zeros(
                    (3, 2, 4), dtype=reference_torch.float32
                ).transpose(0, 1),
                reference_torch.zeros((5, 6), dtype=reference_torch.float32),
            ),
            (
                "foldable permutation",
                torch.zeros((2, 3, 4)).permute(1, 2, 0),
                torch.zeros((5, 4)),
                reference_torch.zeros(
                    (2, 3, 4), dtype=reference_torch.float32
                ).permute(1, 2, 0),
                reference_torch.zeros((5, 4), dtype=reference_torch.float32),
            ),
        )
        for case, actual_input, actual_weight, expected_input, expected_weight in cases:
            with self.subTest(case=case):
                with self.assertRaises(Exception) as actual_raised:
                    functional.linear(actual_input, actual_weight)
                with self.assertRaises(Exception) as expected_raised:
                    reference_functional.linear(expected_input, expected_weight)
                self.assertIs(
                    type(actual_raised.exception), type(expected_raised.exception)
                )
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )

    def test_noncontiguous_rank_three_tracked_weight_error_matches_in_no_grad(self):
        actual_input = torch.zeros((3, 2, 4)).transpose(0, 1)
        actual_weight = torch.zeros((5, 6), requires_grad=True)
        expected_input = reference_torch.zeros(
            (3, 2, 4), dtype=reference_torch.float32
        ).transpose(0, 1)
        expected_weight = reference_torch.zeros(
            (5, 6), dtype=reference_torch.float32, requires_grad=True
        )

        with torch.no_grad():
            with self.assertRaises(Exception) as actual_raised:
                functional.linear(actual_input, actual_weight)
        with reference_torch.no_grad():
            with self.assertRaises(Exception) as expected_raised:
                reference_functional.linear(expected_input, expected_weight)
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_rank_four_layout_dependent_inner_dimension_errors_match(self):
        cases = (
            (
                "unfoldable leading transpose",
                torch.zeros((3, 2, 4, 5)).transpose(0, 1),
                reference_torch.zeros(
                    (3, 2, 4, 5), dtype=reference_torch.float32
                ).transpose(0, 1),
            ),
            (
                "unfoldable feature transpose",
                torch.zeros((2, 3, 5, 4)).transpose(2, 3),
                reference_torch.zeros(
                    (2, 3, 5, 4), dtype=reference_torch.float32
                ).transpose(2, 3),
            ),
            (
                "foldable permutation",
                torch.zeros((2, 3, 4, 5)).permute(1, 2, 3, 0),
                reference_torch.zeros(
                    (2, 3, 4, 5), dtype=reference_torch.float32
                ).permute(1, 2, 3, 0),
            ),
            (
                "offset",
                torch.zeros((2, 2, 3, 4, 5))[1],
                reference_torch.zeros(
                    (2, 2, 3, 4, 5), dtype=reference_torch.float32
                )[1],
            ),
            (
                "zero-sized",
                torch.zeros((0, 3, 4, 5)),
                reference_torch.zeros(
                    (0, 3, 4, 5), dtype=reference_torch.float32
                ),
            ),
        )
        for case, actual_input, expected_input in cases:
            actual_weight = torch.zeros((6, 7))
            expected_weight = reference_torch.zeros(
                (6, 7), dtype=reference_torch.float32
            )
            with self.subTest(case=case):
                with self.assertRaises(Exception) as actual_raised:
                    functional.linear(actual_input, actual_weight)
                with self.assertRaises(Exception) as expected_raised:
                    reference_functional.linear(expected_input, expected_weight)
                self.assertIs(
                    type(actual_raised.exception), type(expected_raised.exception)
                )
                self.assertEqual(
                    str(actual_raised.exception), str(expected_raised.exception)
                )

    def test_noncontiguous_rank_four_tracked_weight_error_matches_in_no_grad(self):
        actual_input = torch.zeros((3, 2, 4, 5)).transpose(0, 1)
        actual_weight = torch.zeros((6, 7), requires_grad=True)
        expected_input = reference_torch.zeros(
            (3, 2, 4, 5), dtype=reference_torch.float32
        ).transpose(0, 1)
        expected_weight = reference_torch.zeros(
            (6, 7), dtype=reference_torch.float32, requires_grad=True
        )

        with torch.no_grad():
            with self.assertRaises(Exception) as actual_raised:
                functional.linear(actual_input, actual_weight)
        with reference_torch.no_grad():
            with self.assertRaises(Exception) as expected_raised:
                reference_functional.linear(expected_input, expected_weight)
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))


if __name__ == "__main__":
    unittest.main()
