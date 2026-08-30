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
class TensorSumReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.sum differentials require pinned PyTorch 2.13.0")

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
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_sum(source, form, module):
        if form == "default":
            return source.sum()
        if form == "positional none dim":
            return source.sum(None)
        if form == "keyword none dim":
            return source.sum(dim=None)
        if form == "none dim keepdim false":
            return source.sum(None, False)
        if form == "dtype none":
            return source.sum(dtype=None)
        if form == "dtype float32":
            return source.sum(dtype=module.float32)
        if form == "dtype float alias":
            return source.sum(dtype=module.float)
        if form == "none dim dtype float32":
            return source.sum(dim=None, keepdim=False, dtype=module.float32)
        raise AssertionError(f"unknown sum form: {form}")

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
                    self.call_sum(actual_input, form, torch),
                    self.call_sum(expected_input, form, reference_torch),
                    case=(name, form),
                )

    def test_autograd_accumulation_empty_and_no_grad_match_pytorch_2_13(self):
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = actual_leaf.transpose(0, 1).sum(
            dim=None, keepdim=False, dtype=torch.float32
        )
        expected_loss = expected_leaf.transpose(0, 1).sum(
            dim=None, keepdim=False, dtype=reference_torch.float32
        )
        self.assert_scalar_matches(actual_loss, expected_loss, case="tracked")
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty.transpose(0, 2).sum(None, False, dtype=None).backward()
        expected_empty.transpose(0, 2).sum(None, False, dtype=None).backward()
        self.assertEqual(actual_empty.grad.shape, tuple(expected_empty.grad.shape))
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.sum(dim=None, dtype=torch.float)
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.sum(
                dim=None, dtype=reference_torch.float
            )
        self.assert_scalar_matches(actual_untracked, expected_untracked, case="no_grad")

    def test_rank_9_offset_permuted_sum_cases_match_pytorch_2_13(self):
        shape = (2, 3, 2, 2, 2, 2, 2, 2, 2)
        values = (
            (np.arange(2 * np.prod(shape), dtype=np.float32) % 23) - 11
        ).reshape((2, *shape))
        actual_source = torch.tensor(values.tolist(), dtype=torch.float32)
        expected_source = reference_torch.tensor(values, dtype=reference_torch.float32)
        permutations = (
            (8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 0, 4, 6, 8, 1, 3, 5, 7),
            (1, 3, 5, 7, 0, 2, 4, 6, 8),
            (4, 1, 8, 0, 6, 2, 5, 3, 7),
        )

        for permutation in permutations:
            actual = actual_source[1].permute(permutation)
            expected = expected_source[1].permute(permutation)
            self.assertFalse(actual.is_contiguous())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assert_scalar_matches(
                actual.sum(),
                expected.sum(),
                case=("offset-permutation", permutation),
            )

        singleton_shape = (2, 1, 3, 2, 1, 2, 2, 2, 2)
        singleton_values = (
            (np.arange(2 * np.prod(singleton_shape), dtype=np.float32) % 19) - 9
        ).reshape((2, *singleton_shape))
        actual_singleton = torch.tensor(
            singleton_values.tolist(), dtype=torch.float32
        )[1].permute(2, 0, 3, 5, 4, 8, 7, 6, 1)
        expected_singleton = reference_torch.tensor(
            singleton_values, dtype=reference_torch.float32
        )[1].permute(2, 0, 3, 5, 4, 8, 7, 6, 1)
        self.assertFalse(actual_singleton.is_contiguous())
        self.assert_scalar_matches(
            actual_singleton.sum(),
            expected_singleton.sum(),
            case="singleton",
        )
        np.testing.assert_array_equal(
            np.asarray(actual_singleton.contiguous()),
            expected_singleton.contiguous().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(-actual_singleton),
            (-expected_singleton).cpu().numpy(),
        )

        actual_empty = torch.zeros((2, 0, 3, 4, 5, 2, 2, 2, 2), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3, 4, 5, 2, 2, 2, 2),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_empty_view = actual_empty.permute(4, 2, 0, 8, 7, 6, 5, 3, 1)
        expected_empty_view = expected_empty.permute(4, 2, 0, 8, 7, 6, 5, 3, 1)
        self.assert_scalar_matches(
            actual_empty_view.sum(), expected_empty_view.sum(), case="empty"
        )
        actual_empty_view.sum().backward()
        expected_empty_view.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        actual_leaf = torch.tensor(
            values.tolist(), dtype=torch.float32, requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_view = actual_leaf[1].permute(3, 1, 6, 0, 4, 8, 7, 2, 5)
        expected_view = expected_leaf[1].permute(3, 1, 6, 0, 4, 8, 7, 2, 5)
        actual_loss = actual_view.sum()
        expected_loss = expected_view.sum()
        self.assert_scalar_matches(actual_loss, expected_loss, case="tracked")
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_view.sum()
        with reference_torch.no_grad():
            expected_untracked = expected_view.sum()
        self.assert_scalar_matches(
            actual_untracked, expected_untracked, case="no_grad"
        )

    def test_descriptor_shape_and_documentation_match_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0, 2.0])
        expected_tensor = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32
        )
        pairs = (
            (
                inspect.getattr_static(torch.Tensor, "sum"),
                inspect.getattr_static(reference_torch.Tensor, "sum"),
                types.MethodDescriptorType,
            ),
            (actual_tensor.sum, expected_tensor.sum, types.BuiltinMethodType),
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
            (lambda: actual.sum(dtype=1), lambda: expected.sum(dtype=1)),
            (
                lambda: actual.sum(dtype=object()),
                lambda: expected.sum(dtype=object()),
            ),
            (
                lambda: actual.sum(dim=None, dtype=1),
                lambda: expected.sum(dim=None, dtype=1),
            ),
            (
                lambda: actual.sum(None, False, dtype=object()),
                lambda: expected.sum(None, False, dtype=object()),
            ),
            (
                lambda: actual.sum(torch.float32),
                lambda: expected.sum(reference_torch.float32),
            ),
            (lambda: actual.sum(extra=True), lambda: expected.sum(extra=True)),
            (
                lambda: actual.sum(0, False, torch.float32),
                lambda: expected.sum(0, False, reference_torch.float32),
            ),
            (lambda: actual.sum(out=None), lambda: expected.sum(out=None)),
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
            (lambda: actual.sum(0), lambda: expected.sum(0)),
            (lambda: actual.sum(dim=0), lambda: expected.sum(dim=0)),
            (
                lambda: actual.sum(0, False),
                lambda: expected.sum(0, False),
            ),
            (
                lambda: actual.sum(dim=0, keepdim=True),
                lambda: expected.sum(dim=0, keepdim=True),
            ),
            (
                lambda: actual.sum(dim=None, keepdim=True),
                lambda: expected.sum(dim=None, keepdim=True),
            ),
            (
                lambda: actual.sum(dtype=reference_torch.float64),
                lambda: expected.sum(dtype=reference_torch.float64),
            ),
            (
                lambda: actual.sum(dim=None, dtype=reference_torch.float64),
                lambda: expected.sum(dim=None, dtype=reference_torch.float64),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    actual_call()
                expected_call()


if __name__ == "__main__":
    unittest.main()
