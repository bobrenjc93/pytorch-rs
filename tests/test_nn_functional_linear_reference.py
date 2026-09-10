import inspect
import itertools
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

    def make_matrix_bias_cases(self, module):
        bias_values = np.asarray([0.5, -1.25, 2.0, -3.5], dtype=np.float32)
        contiguous_bias = module.tensor(
            bias_values.tolist(),
            dtype=module.float32,
        )
        strided_bias = module.tensor(
            np.stack(
                (bias_values, np.full(4, 99.0, dtype=np.float32)), axis=1
            ).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)[0]
        offset_bias = module.tensor(
            np.stack((np.full(4, 99.0, dtype=np.float32), bias_values)).tolist(),
            dtype=module.float32,
        )[1]
        offset_strided_bias = module.tensor(
            np.arange(16, dtype=np.float32).reshape(2, 4, 2).tolist(),
            dtype=module.float32,
        )[1].transpose(0, 1)[0]
        empty_bias = module.zeros((0,), dtype=module.float32)
        biases = (
            contiguous_bias,
            strided_bias,
            offset_bias,
            offset_strided_bias,
            strided_bias,
            offset_bias,
            empty_bias,
            contiguous_bias,
            empty_bias,
        )
        return tuple(
            (*case, bias)
            for case, bias in zip(self.make_cases(module), biases, strict=True)
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
        bias_values = np.asarray([0.5, -1.25, 2.0, -3.5], dtype=np.float32)
        contiguous_bias = module.tensor(
            bias_values.tolist(),
            dtype=module.float32,
        )
        strided_bias = module.tensor(
            np.stack(
                (bias_values, np.full(4, 99.0, dtype=np.float32)), axis=1
            ).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)[0]
        offset_bias = module.tensor(
            np.stack((np.full(4, 99.0, dtype=np.float32), bias_values)).tolist(),
            dtype=module.float32,
        )[1]
        offset_strided_bias = module.tensor(
            np.arange(16, dtype=np.float32).reshape(2, 4, 2).tolist(),
            dtype=module.float32,
        )[1].transpose(0, 1)[0]
        empty_bias = module.zeros((0,), dtype=module.float32)

        return (
            (
                "contiguous vector",
                contiguous_input,
                contiguous_weight,
                contiguous_bias,
            ),
            (
                "strided vector",
                strided_source[0],
                contiguous_weight,
                strided_bias,
            ),
            ("offset vector", offset_input, contiguous_weight, offset_bias),
            (
                "offset-strided vector",
                strided_source[1],
                strided_weight,
                offset_strided_bias,
            ),
            (
                "zero features",
                module.zeros((0,), dtype=module.float32),
                module.zeros((4, 0), dtype=module.float32),
                strided_bias,
            ),
            (
                "zero outputs",
                contiguous_input,
                module.zeros((0, 3), dtype=module.float32),
                empty_bias,
            ),
            (
                "all zero",
                module.zeros((0,), dtype=module.float32),
                module.zeros((0, 0), dtype=module.float32),
                empty_bias,
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

    @staticmethod
    def call_with_bias(module_functional, input, weight, bias, form):
        if form == "positional":
            return module_functional.linear(input, weight, bias)
        if form == "bias keyword":
            return module_functional.linear(input, weight, bias=bias)
        return module_functional.linear(input=input, weight=weight, bias=bias)

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

    def test_rank_two_bias_values_layouts_and_storage_match(self):
        actual_cases = self.make_matrix_bias_cases(torch)
        expected_cases = self.make_matrix_bias_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_weight, actual_bias = actual_case
            expected_name, expected_input, expected_weight, expected_bias = (
                expected_case
            )
            self.assertEqual(case, expected_name)
            for form in ("positional", "bias keyword", "keywords"):
                actual = self.call_with_bias(
                    functional,
                    actual_input,
                    actual_weight,
                    actual_bias,
                    form,
                )
                expected = self.call_with_bias(
                    reference_functional,
                    expected_input,
                    expected_weight,
                    expected_bias,
                    form,
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call_with_bias(
                    functional,
                    actual_input,
                    actual_weight,
                    actual_bias,
                    form,
                )
                expected_repeat = self.call_with_bias(
                    reference_functional,
                    expected_input,
                    expected_weight,
                    expected_bias,
                    form,
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    for actual_operand, expected_operand in (
                        (actual_input, expected_input),
                        (actual_weight, expected_weight),
                        (actual_bias, expected_bias),
                    ):
                        self.assertFalse(actual.is_set_to(actual_operand))
                        self.assertFalse(expected.is_set_to(expected_operand))

    def test_vector_bias_values_layouts_and_storage_match(self):
        actual_cases = self.make_vector_cases(torch)
        expected_cases = self.make_vector_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_weight, actual_bias = actual_case
            expected_name, expected_input, expected_weight, expected_bias = (
                expected_case
            )
            self.assertEqual(case, expected_name)
            for form in ("positional", "bias keyword", "keywords"):
                actual = self.call_with_bias(
                    functional,
                    actual_input,
                    actual_weight,
                    actual_bias,
                    form,
                )
                expected = self.call_with_bias(
                    reference_functional,
                    expected_input,
                    expected_weight,
                    expected_bias,
                    form,
                )
                self.assert_matches(actual, expected, case=(case, form))

                actual_repeat = self.call_with_bias(
                    functional,
                    actual_input,
                    actual_weight,
                    actual_bias,
                    form,
                )
                expected_repeat = self.call_with_bias(
                    reference_functional,
                    expected_input,
                    expected_weight,
                    expected_bias,
                    form,
                )
                with self.subTest(case=(case, form), storage=True):
                    self.assertFalse(actual.is_set_to(actual_repeat))
                    self.assertFalse(expected.is_set_to(expected_repeat))
                    for actual_operand, expected_operand in (
                        (actual_input, expected_input),
                        (actual_weight, expected_weight),
                        (actual_bias, expected_bias),
                    ):
                        self.assertFalse(actual.is_set_to(actual_operand))
                        self.assertFalse(expected.is_set_to(expected_operand))

    def test_vector_bias_signed_zero_bits_match(self):
        cases = (
            ("negative zero product", [0.0], [[-0.0]]),
            ("negative zero input", [-0.0], [[1.0]]),
            ("positive zero product", [0.0], [[1.0]]),
            ("opposite zero signs", [-0.0], [[-1.0]]),
            ("zero features", [], [[]]),
        )
        for case, input_values, weight_values in cases:
            actual = functional.linear(
                torch.tensor(input_values),
                torch.tensor(weight_values),
                torch.tensor([-0.0]),
            )
            expected = reference_functional.linear(
                reference_torch.tensor(
                    input_values,
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor(
                    weight_values,
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor([-0.0], dtype=reference_torch.float32),
            )
            self.assert_matches(actual, expected, case=case)
            with self.subTest(case=case, sign_bits=True):
                np.testing.assert_array_equal(
                    np.asarray(actual).reshape(-1).view(np.uint32),
                    expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
                )

    def test_rank_two_bias_signed_zero_bits_match(self):
        cases = (
            (
                "negative zero product",
                [[0.0], [0.0]],
                [[-0.0]],
            ),
            (
                "negative zero input",
                [[-0.0], [-0.0]],
                [[1.0]],
            ),
            (
                "positive zero product",
                [[0.0], [0.0]],
                [[1.0]],
            ),
            (
                "opposite zero signs",
                [[-0.0], [-0.0]],
                [[-1.0]],
            ),
            (
                "zero features",
                [[], []],
                [[]],
            ),
        )
        for case, input_values, weight_values in cases:
            actual = functional.linear(
                torch.tensor(input_values),
                torch.tensor(weight_values),
                torch.tensor([-0.0]),
            )
            expected = reference_functional.linear(
                reference_torch.tensor(
                    input_values,
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor(
                    weight_values,
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor([-0.0], dtype=reference_torch.float32),
            )
            self.assert_matches(actual, expected, case=case)
            with self.subTest(case=case, sign_bits=True):
                np.testing.assert_array_equal(
                    np.asarray(actual).reshape(-1).view(np.uint32),
                    expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
                )

    def test_rank_one_and_rank_two_singleton_bias_values_and_empty_outputs_match(self):
        actual_cases = (
            (
                "vector",
                torch.tensor([1.0, 2.0, 3.0]),
                torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]]),
                torch.tensor([0.5]),
            ),
            (
                "vector zero outputs",
                torch.zeros((3,)),
                torch.zeros((0, 3)),
                torch.tensor([99.0]),
            ),
            (
                "vector all zero",
                torch.zeros((0,)),
                torch.zeros((0, 0)),
                torch.tensor([99.0]),
            ),
            (
                "matrix",
                torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                torch.tensor([[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]]),
                torch.tensor([0.5]),
            ),
            (
                "matrix zero rows",
                torch.zeros((0, 3)),
                torch.zeros((4, 3)),
                torch.tensor([99.0]),
            ),
            (
                "matrix zero outputs",
                torch.zeros((2, 3)),
                torch.zeros((0, 3)),
                torch.tensor([99.0]),
            ),
            (
                "matrix all zero",
                torch.zeros((0, 0)),
                torch.zeros((0, 0)),
                torch.tensor([99.0]),
            ),
        )
        expected_cases = (
            (
                "vector",
                reference_torch.tensor(
                    [1.0, 2.0, 3.0],
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor(
                    [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor([0.5], dtype=reference_torch.float32),
            ),
            (
                "vector zero outputs",
                reference_torch.zeros((3,), dtype=reference_torch.float32),
                reference_torch.zeros((0, 3), dtype=reference_torch.float32),
                reference_torch.tensor([99.0], dtype=reference_torch.float32),
            ),
            (
                "vector all zero",
                reference_torch.zeros((0,), dtype=reference_torch.float32),
                reference_torch.zeros((0, 0), dtype=reference_torch.float32),
                reference_torch.tensor([99.0], dtype=reference_torch.float32),
            ),
            (
                "matrix",
                reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor(
                    [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                    dtype=reference_torch.float32,
                ),
                reference_torch.tensor([0.5], dtype=reference_torch.float32),
            ),
            (
                "matrix zero rows",
                reference_torch.zeros((0, 3), dtype=reference_torch.float32),
                reference_torch.zeros((4, 3), dtype=reference_torch.float32),
                reference_torch.tensor([99.0], dtype=reference_torch.float32),
            ),
            (
                "matrix zero outputs",
                reference_torch.zeros((2, 3), dtype=reference_torch.float32),
                reference_torch.zeros((0, 3), dtype=reference_torch.float32),
                reference_torch.tensor([99.0], dtype=reference_torch.float32),
            ),
            (
                "matrix all zero",
                reference_torch.zeros((0, 0), dtype=reference_torch.float32),
                reference_torch.zeros((0, 0), dtype=reference_torch.float32),
                reference_torch.tensor([99.0], dtype=reference_torch.float32),
            ),
        )
        for actual_case, expected_case in zip(
            actual_cases,
            expected_cases,
            strict=True,
        ):
            case, actual_input, actual_weight, actual_bias = actual_case
            expected_name, expected_input, expected_weight, expected_bias = (
                expected_case
            )
            self.assertEqual(case, expected_name)
            actual = functional.linear(actual_input, actual_weight, actual_bias)
            expected = reference_functional.linear(
                expected_input,
                expected_weight,
                expected_bias,
            )
            self.assert_matches(actual, expected, case=case)

    def test_vector_layouts_values_metadata_and_storage_match(self):
        actual_cases = self.make_vector_cases(torch)
        expected_cases = self.make_vector_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input, actual_weight, _ = actual_case
            expected_name, expected_input, expected_weight, _ = expected_case
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

    @staticmethod
    def make_rank_three_bias_operands(
        module, shape, output_width, bias_width, layout
    ):
        sources = []

        def tensor(shape):
            values = (np.arange(np.prod(shape), dtype=np.float32) % 17 - 8) / 4
            source = module.tensor(
                values.tolist(), dtype=module.float32
            ).reshape(shape)
            sources.append(source)
            return source

        input = tensor((2, *shape))[1] if "offset" in layout else tensor(shape)
        weight_shape = (output_width, shape[-1])
        if "strided" in layout:
            weight_shape = weight_shape[::-1]
        weight = (
            tensor((2, *weight_shape))[1]
            if "offset" in layout
            else tensor(weight_shape)
        )
        if "strided" in layout:
            weight = weight.transpose(0, 1)
            bias = tensor((bias_width, 2)).transpose(0, 1)[1]
        else:
            bias = (
                tensor((2, bias_width))[1]
                if "offset" in layout
                else tensor((bias_width,))
            )
        return (input, weight, bias), sources

    def test_contiguous_rank_three_bias_shapes_layouts_calls_and_sources_match(
        self,
    ):
        shapes = (
            ((2, 3, 4), 5),
            ((3, 2, 5), 2),
            ((1, 1, 3), 1),
            ((0, 3, 4), 5),
            ((2, 0, 4), 5),
            ((2, 3, 0), 5),
            ((2, 3, 4), 0),
            ((0, 0, 0), 0),
        )
        for (shape, output_width), layout, singleton in itertools.product(
            shapes,
            ("contiguous", "strided", "offset", "offset-strided"),
            (False, True),
        ):
            bias_width = 1 if singleton else output_width
            actual_operands, actual_sources = (
                self.make_rank_three_bias_operands(
                    torch, shape, output_width, bias_width, layout
                )
            )
            expected_operands, expected_sources = (
                self.make_rank_three_bias_operands(
                    reference_torch, shape, output_width, bias_width, layout
                )
            )
            snapshots = [
                np.asarray(source).copy() for source in actual_sources
            ]
            expected_snapshots = [
                source.numpy().copy() for source in expected_sources
            ]
            for actual_operand, expected_operand in zip(
                actual_operands, expected_operands, strict=True
            ):
                self.assert_matches(
                    actual_operand, expected_operand, case=(shape, layout)
                )
            self.assertTrue(actual_operands[0].is_contiguous())
            for form in ("positional", "bias keyword", "keywords"):
                with self.subTest(
                    shape=shape, layout=layout, singleton=singleton, form=form
                ):
                    actual = self.call_with_bias(
                        functional, *actual_operands, form
                    )
                    expected = self.call_with_bias(
                        reference_functional, *expected_operands, form
                    )
                    self.assert_matches(actual, expected, case="rank-3 bias")
                    repeat = self.call_with_bias(
                        functional, *actual_operands, form
                    )
                    self.assertFalse(actual.is_set_to(repeat))
                    # Mutating the result must not change any backing source or
                    # another result, including when operands are offset views.
                    repeat_values = np.asarray(repeat).copy()
                    np.asarray(actual)[...] = 99.0
                    np.testing.assert_array_equal(
                        np.asarray(repeat), repeat_values
                    )
                    for source, snapshot in zip(
                        actual_sources, snapshots, strict=True
                    ):
                        np.testing.assert_array_equal(
                            np.asarray(source), snapshot
                        )
                    for source, snapshot in zip(
                        expected_sources, expected_snapshots, strict=True
                    ):
                        np.testing.assert_array_equal(source.numpy(), snapshot)

    def test_rank_three_bias_contiguous_degenerate_strides_match(self):
        # Singleton and empty axes can have noncanonical strides while the
        # tensor is still contiguous according to PyTorch's layout rules.
        for shape in ((2, 1, 4), (2, 0, 4)):
            actual_input = torch.ones(shape).transpose(0, 1)
            expected_input = reference_torch.ones(shape).transpose(0, 1)
            self.assertTrue(actual_input.is_contiguous())
            self.assertTrue(expected_input.is_contiguous())
            actual = functional.linear(
                actual_input, torch.ones((3, 4)), torch.ones((3,))
            )
            expected = reference_functional.linear(
                expected_input,
                reference_torch.ones((3, 4)),
                reference_torch.ones((3,)),
            )
            self.assert_matches(actual, expected, case=shape)

            with self.subTest(shape=shape, inner_dimension_error=True):
                with self.assertRaises(RuntimeError) as actual_raised:
                    functional.linear(
                        actual_input, torch.ones((3, 5)), torch.ones((3,))
                    )
                with self.assertRaises(RuntimeError) as expected_raised:
                    reference_functional.linear(
                        expected_input,
                        reference_torch.ones((3, 5)),
                        reference_torch.ones((3,)),
                    )
                self.assertEqual(
                    str(actual_raised.exception),
                    str(expected_raised.exception),
                )

    def test_rank_three_bias_autograd_restriction_and_no_grad_match(self):
        for flags, bias_width in itertools.product(
            itertools.product((False, True), repeat=3), (1, 5)
        ):
            if not any(flags):
                continue
            actual_operands, _ = self.make_rank_three_bias_operands(
                torch, (2, 3, 4), 5, bias_width, "offset-strided"
            )
            expected_operands, _ = self.make_rank_three_bias_operands(
                reference_torch, (2, 3, 4), 5, bias_width, "offset-strided"
            )
            for actual, expected, flag in zip(
                actual_operands, expected_operands, flags, strict=True
            ):
                actual.requires_grad_(flag)
                expected.requires_grad_(flag)
            with self.subTest(flags=flags, bias_width=bias_width):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"^linear\(\): autograd recording is not supported$",
                ):
                    functional.linear(*actual_operands)
                with torch.no_grad():
                    actual = functional.linear(*actual_operands)
                with reference_torch.no_grad():
                    expected = reference_functional.linear(*expected_operands)
                self.assert_matches(
                    actual, expected, case="rank-3 bias no_grad"
                )
                self.assertFalse(actual.requires_grad)
                self.assertTrue(actual.is_leaf)
                for operand in (*actual_operands, *expected_operands):
                    self.assertIsNone(operand.grad)
                # Leaving no_grad must restore the recording restriction.
                with self.assertRaisesRegex(
                    RuntimeError, "autograd recording is not supported"
                ):
                    functional.linear(*actual_operands)

    def test_rank_three_bias_dimension_errors_match(self):
        for shape, output_width in (
            ((2, 3, 4), 5),
            ((0, 3, 4), 5),
            ((2, 0, 4), 5),
            ((2, 3, 0), 5),
            ((2, 3, 4), 0),
            ((0, 0, 0), 0),
        ):
            for inner_width, bias_width in itertools.product(
                (shape[-1], shape[-1] + 1), (0, 1, 2, 5)
            ):
                if inner_width == shape[-1] and bias_width in (
                    1,
                    output_width,
                ):
                    continue
                with self.subTest(
                    shape=shape,
                    output_width=output_width,
                    inner_width=inner_width,
                    bias_width=bias_width,
                ):
                    with self.assertRaises(RuntimeError) as actual_raised:
                        functional.linear(
                            torch.zeros(shape),
                            torch.zeros((output_width, inner_width)),
                            torch.zeros((bias_width,)),
                        )
                    with self.assertRaises(RuntimeError) as expected_raised:
                        reference_functional.linear(
                            reference_torch.zeros(shape),
                            reference_torch.zeros((output_width, inner_width)),
                            reference_torch.zeros((bias_width,)),
                        )
                    self.assertIs(
                        type(actual_raised.exception),
                        type(expected_raised.exception),
                    )
                    self.assertEqual(
                        str(actual_raised.exception),
                        str(expected_raised.exception),
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

    def test_rank_two_bias_requires_grad_operands_match_inside_no_grad(self):
        for input_requires_grad, weight_requires_grad, bias_requires_grad in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ):
            actual_input = torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                requires_grad=input_requires_grad,
            )
            actual_weight = torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                requires_grad=weight_requires_grad,
            )
            actual_bias = torch.tensor(
                [0.5, -1.5],
                requires_grad=bias_requires_grad,
            )
            expected_input = reference_torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=reference_torch.float32,
                requires_grad=input_requires_grad,
            )
            expected_weight = reference_torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                dtype=reference_torch.float32,
                requires_grad=weight_requires_grad,
            )
            expected_bias = reference_torch.tensor(
                [0.5, -1.5],
                dtype=reference_torch.float32,
                requires_grad=bias_requires_grad,
            )
            with torch.no_grad():
                actual = functional.linear(
                    actual_input,
                    actual_weight,
                    actual_bias,
                )
            with reference_torch.no_grad():
                expected = reference_functional.linear(
                    expected_input,
                    expected_weight,
                    expected_bias,
                )
            self.assert_matches(
                actual,
                expected,
                case=(
                    input_requires_grad,
                    weight_requires_grad,
                    bias_requires_grad,
                ),
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

    def test_vector_bias_requires_grad_operands_match_inside_no_grad(self):
        for input_requires_grad, weight_requires_grad, bias_requires_grad in (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ):
            actual_input = torch.tensor(
                [1.0, 2.0, 3.0],
                requires_grad=input_requires_grad,
            )
            actual_weight = torch.tensor(
                [[1.0, 0.0, -1.0], [2.0, 3.0, 4.0]],
                requires_grad=weight_requires_grad,
            )
            actual_bias = torch.tensor(
                [0.5, -1.5],
                requires_grad=bias_requires_grad,
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
            expected_bias = reference_torch.tensor(
                [0.5, -1.5],
                dtype=reference_torch.float32,
                requires_grad=bias_requires_grad,
            )
            with torch.no_grad():
                actual = functional.linear(
                    actual_input,
                    actual_weight,
                    actual_bias,
                )
            with reference_torch.no_grad():
                expected = reference_functional.linear(
                    expected_input,
                    expected_weight,
                    expected_bias,
                )
            self.assert_matches(
                actual,
                expected,
                case=(
                    input_requires_grad,
                    weight_requires_grad,
                    bias_requires_grad,
                ),
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

    def test_incompatible_inner_dimension_error_matches(self):
        for shape in ((2, 3), (3,), (2, 3, 4)):
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

        cases = (
            (
                "vector",
                torch.zeros((3,)),
                reference_torch.zeros((3,), dtype=reference_torch.float32),
            ),
            (
                "matrix",
                torch.zeros((2, 3)),
                reference_torch.zeros((2, 3), dtype=reference_torch.float32),
            )
        )
        for case, actual_input, expected_input in cases:
            actual_weight = torch.zeros((4, 5))
            expected_weight = reference_torch.zeros(
                (4, 5),
                dtype=reference_torch.float32,
            )
            for bias_features in (4, 5):
                actual_bias = torch.zeros((bias_features,))
                expected_bias = reference_torch.zeros(
                    (bias_features,),
                    dtype=reference_torch.float32,
                )
                with self.subTest(case=case, bias_features=bias_features):
                    with self.assertRaises(Exception) as actual_raised:
                        functional.linear(actual_input, actual_weight, actual_bias)
                    with self.assertRaises(Exception) as expected_raised:
                        reference_functional.linear(
                            expected_input,
                            expected_weight,
                            expected_bias,
                        )
                    self.assertIs(
                        type(actual_raised.exception),
                        type(expected_raised.exception),
                    )
                    self.assertEqual(
                        str(actual_raised.exception),
                        str(expected_raised.exception),
                    )

    def test_rank_one_and_rank_two_bias_length_errors_match(self):
        cases = (
            ("vector", (3,), (4, 3), (0, 2, 3, 5)),
            ("vector zero outputs", (3,), (0, 3), (2,)),
            ("matrix", (2, 3), (4, 3), (0, 2, 3, 5)),
            ("matrix zero outputs", (2, 3), (0, 3), (2,)),
            ("matrix zero rows", (0, 3), (4, 3), (0, 2, 3, 5)),
        )
        for case, input_shape, weight_shape, bias_feature_cases in cases:
            actual_input = torch.zeros(input_shape)
            actual_weight = torch.zeros(weight_shape)
            expected_input = reference_torch.zeros(
                input_shape,
                dtype=reference_torch.float32,
            )
            expected_weight = reference_torch.zeros(
                weight_shape,
                dtype=reference_torch.float32,
            )
            for bias_features in bias_feature_cases:
                actual_bias = torch.zeros((bias_features,))
                expected_bias = reference_torch.zeros(
                    (bias_features,),
                    dtype=reference_torch.float32,
                )
                with self.subTest(case=case, bias_features=bias_features):
                    with self.assertRaises(Exception) as actual_raised:
                        functional.linear(actual_input, actual_weight, actual_bias)
                    with self.assertRaises(Exception) as expected_raised:
                        reference_functional.linear(
                            expected_input,
                            expected_weight,
                            expected_bias,
                        )
                    self.assertIs(
                        type(actual_raised.exception),
                        type(expected_raised.exception),
                    )
                    self.assertEqual(
                        str(actual_raised.exception),
                        str(expected_raised.exception),
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


if __name__ == "__main__":
    unittest.main()
