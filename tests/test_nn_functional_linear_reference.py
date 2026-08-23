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

    def test_metadata_documentation_and_signature_surface(self):
        self.assertEqual(
            functional.linear.__name__, reference_functional.linear.__name__
        )
        self.assertEqual(functional.linear.__doc__, reference_functional.linear.__doc__)
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

                actual_repeat = self.call(functional, actual_input, actual_weight, form)
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
                expected = reference_functional.linear(expected_input, expected_weight)
            self.assert_matches(
                actual,
                expected,
                case=(input_requires_grad, weight_requires_grad),
            )

    def test_incompatible_inner_dimension_error_matches(self):
        actual_input = torch.zeros((2, 3))
        actual_weight = torch.zeros((4, 5))
        expected_input = reference_torch.zeros((2, 3), dtype=reference_torch.float32)
        expected_weight = reference_torch.zeros((4, 5), dtype=reference_torch.float32)
        with self.assertRaises(Exception) as actual_raised:
            functional.linear(actual_input, actual_weight)
        with self.assertRaises(Exception) as expected_raised:
            reference_functional.linear(expected_input, expected_weight)
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))


if __name__ == "__main__":
    unittest.main()
