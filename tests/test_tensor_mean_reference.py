import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorMeanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.mean differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_scalar_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.asarray(actual).view(np.uint32).item(),
                expected.detach().cpu().numpy().view(np.uint32).item(),
            )

    @staticmethod
    def make_cases(module):
        dense = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = dense.transpose(0, 2)
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("negative zero", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("singleton", module.tensor([5.0], dtype=module.float32)),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_mean(source, form, module):
        if form == "default":
            return source.mean()
        if form == "positional none dim":
            return source.mean(None)
        if form == "keyword none dim":
            return source.mean(dim=None)
        if form == "none dim keepdim false":
            return source.mean(None, False)
        if form == "dtype none":
            return source.mean(dtype=None)
        if form == "dtype float32":
            return source.mean(dtype=module.float32)
        if form == "dtype float alias":
            return source.mean(dtype=module.float)
        if form == "none dim dtype float32":
            return source.mean(dim=None, keepdim=False, dtype=module.float32)
        raise AssertionError(f"unknown mean form: {form}")

    @staticmethod
    def rank_one_strided_vector(module, values, *, requires_grad=False):
        rows = len(values)
        columns = 5
        selected_column = 2
        matrix = np.full((rows, columns), np.float32(0.5), dtype=np.float32)
        matrix[:, selected_column] = np.asarray(values, dtype=np.float32)
        source = module.tensor(
            matrix.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        return source, source.transpose(0, 1)[selected_column]

    def test_values_scalar_shape_empty_and_noncontiguous_match_pytorch_2_13(self):
        forms = (
            "default",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "none dim dtype float32",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            name, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(name, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_mean(actual_input, form, torch),
                    self.call_mean(expected_input, form, reference_torch),
                    case=(name, form),
                )

    def test_autograd_empty_and_no_grad_match_pytorch_2_13(self):
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = actual_leaf.transpose(0, 1).mean(
            dim=None, keepdim=False, dtype=torch.float32
        )
        expected_loss = expected_leaf.transpose(0, 1).mean(
            dim=None, keepdim=False, dtype=reference_torch.float32
        )
        self.assert_scalar_matches(actual_loss, expected_loss, case="tracked")
        actual_loss.backward()
        expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty.transpose(0, 2).mean(None, False, dtype=None).backward()
        expected_empty.transpose(0, 2).mean(None, False, dtype=None).backward()
        self.assertEqual(actual_empty.grad.shape, tuple(expected_empty.grad.shape))
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.mean(dim=None, dtype=torch.float)
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.mean(
                dim=None, dtype=reference_torch.float
            )
        self.assert_scalar_matches(actual_untracked, expected_untracked, case="no_grad")

    def test_rank_one_transpose_selected_offset_mean_edges_match_pytorch_2_13(self):
        cases = (
            ("signed zero", [-0.0, 0.0, -0.0, 0.0]),
            ("nan", [1.0, np.nan, 2.0, -3.0]),
            ("positive infinity", [1.0, np.inf, 2.0, 3.0]),
            ("negative infinity", [1.0, -np.inf, 2.0, 3.0]),
            ("sequential cancellation", [1.0e20, -1.0e20, 3.0, -0.0]),
        )

        for case, values in cases:
            _, actual = self.rank_one_strided_vector(torch, values)
            _, expected = self.rank_one_strided_vector(reference_torch, values)
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertFalse(actual.is_contiguous())
            self.assertFalse(expected.is_contiguous())
            self.assert_scalar_matches(
                actual.mean(), expected.mean(), case=("rank-one offset", case)
            )

    def test_descriptor_contract_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0, 2.0])
        expected_tensor = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32
        )
        pairs = (
            (
                inspect.getattr_static(torch.Tensor, "mean"),
                inspect.getattr_static(reference_torch.Tensor, "mean"),
                types.MethodDescriptorType,
            ),
            (actual_tensor.mean, expected_tensor.mean, types.BuiltinMethodType),
        )
        for actual, expected, expected_type in pairs:
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

    def test_invalid_dtype_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32
        )
        cases = (
            (lambda: actual.mean(dtype=1), lambda: expected.mean(dtype=1)),
            (
                lambda: actual.mean(dtype=object()),
                lambda: expected.mean(dtype=object()),
            ),
            (
                lambda: actual.mean(dim=None, dtype=1),
                lambda: expected.mean(dim=None, dtype=1),
            ),
            (
                lambda: actual.mean(None, False, dtype=object()),
                lambda: expected.mean(None, False, dtype=object()),
            ),
            (
                lambda: actual.mean(torch.float32),
                lambda: expected.mean(reference_torch.float32),
            ),
            (lambda: actual.mean(extra=True), lambda: expected.mean(extra=True)),
            (
                lambda: actual.mean(0, False, torch.float32),
                lambda: expected.mean(0, False, reference_torch.float32),
            ),
            (lambda: actual.mean(out=None), lambda: expected.mean(out=None)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_dimension_keepdim_and_cross_dtype_reductions_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32
        )
        cases = (
            (lambda: actual.mean(0), lambda: expected.mean(0)),
            (lambda: actual.mean(dim=0), lambda: expected.mean(dim=0)),
            (
                lambda: actual.mean(0, False),
                lambda: expected.mean(0, False),
            ),
            (
                lambda: actual.mean(dim=0, keepdim=True),
                lambda: expected.mean(dim=0, keepdim=True),
            ),
            (
                lambda: actual.mean(dim=None, keepdim=True),
                lambda: expected.mean(dim=None, keepdim=True),
            ),
            (
                lambda: actual.mean(dtype=reference_torch.float64),
                lambda: expected.mean(dtype=reference_torch.float64),
            ),
            (
                lambda: actual.mean(dim=None, dtype=reference_torch.float64),
                lambda: expected.mean(dim=None, dtype=reference_torch.float64),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    actual_call()
                expected_call()


if __name__ == "__main__":
    unittest.main()
