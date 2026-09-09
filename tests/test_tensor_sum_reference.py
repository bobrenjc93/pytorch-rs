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
            raise AssertionError(
                "Tensor.sum differentials require pinned PyTorch 2.13.0"
            )

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

    def assert_tensor_matches(self, actual, expected, *, case):
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
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint32),
                expected.detach().cpu().numpy().view(np.uint32),
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
            ("singleton", module.tensor([[[7.0]]], dtype=module.float32)[0]),
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
        if form == "positional none dim keepdim true":
            return source.sum(None, True)
        if form == "mixed none dim keepdim true":
            return source.sum(None, keepdim=True)
        if form == "keyword none dim keepdim true":
            return source.sum(dim=None, keepdim=True)
        if form == "keepdim true dtype none":
            return source.sum(dim=None, keepdim=True, dtype=None)
        if form == "keepdim true dtype float32":
            return source.sum(dim=None, keepdim=True, dtype=module.float32)
        raise AssertionError(f"unknown sum form: {form}")

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

    @staticmethod
    def rank_two_dim_cases(module):
        dense = module.tensor(
            np.arange(1, 7, dtype=np.float32).reshape(2, 3).tolist(),
            dtype=module.float32,
        )
        noncontiguous = module.tensor(
            np.arange(12, dtype=np.float32).reshape(3, 4).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)
        offset = module.tensor(
            np.arange(20, dtype=np.float32).reshape(4, 5).tolist(),
            dtype=module.float32,
        ).transpose(0, 1)[2:]
        return (
            ("contiguous", dense),
            ("noncontiguous", noncontiguous),
            ("offset noncontiguous", offset),
            ("empty rows", module.zeros((0, 3), dtype=module.float32)),
            ("empty columns", module.zeros((2, 0), dtype=module.float32)),
            (
                "empty offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            (
                "singleton row",
                module.tensor([[7.0, -2.0, 5.0]], dtype=module.float32),
            ),
        )

    @staticmethod
    def rank_one_dim_cases(module):
        contiguous_base = module.tensor(
            [-5.0, 1.0, -2.0, 3.0, 4.0], dtype=module.float32
        )
        _, noncontiguous = TensorSumReferenceTests.rank_one_strided_vector(
            module, [1.0, -2.0, 3.0, -4.0]
        )
        return (
            (
                "empty",
                module.zeros((0, 5), dtype=module.float32).transpose(0, 1)[2],
            ),
            ("offset", contiguous_base[1:]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_dim_sum(source, form, module):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 0

        if form == "positional dim zero":
            return source.sum(0)
        if form == "positional dim negative":
            return source.sum(-1)
        if form == "keyword dim zero dtype none":
            return source.sum(dim=0, keepdim=False, dtype=None)
        if form == "keyword dim negative dtype none":
            return source.sum(dim=-1, keepdim=False, dtype=None)
        if form == "keyword dim zero keepdim dtype none":
            return source.sum(dim=0, keepdim=True, dtype=None)
        if form == "keyword dim negative keepdim":
            return source.sum(dim=-1, keepdim=True, dtype=None)
        if form == "integer subclass dim":
            return source.sum(IntSubclass(0))
        if form == "numpy integer dim keepdim":
            return source.sum(dim=np.int64(-1), keepdim=True)
        if form == "tuple integer protocol dim":
            return source.sum((IndexOnly(),))
        if form == "list numpy integer dim keepdim":
            return source.sum([np.int64(-1)], keepdim=True)
        if form == "dtype float32":
            return source.sum(dim=0, dtype=module.float32)
        raise AssertionError(f"unknown dim sum form: {form}")

    @staticmethod
    def call_rank_two_dim_sum(source, form, module):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 0

        if form == "positional dim zero":
            return source.sum(0)
        if form == "positional dim one":
            return source.sum(1)
        if form == "positional dim negative one":
            return source.sum(-1)
        if form == "positional dim negative two":
            return source.sum(-2)
        if form == "keyword dim zero dtype none":
            return source.sum(dim=0, keepdim=False, dtype=None)
        if form == "keyword dim one dtype none":
            return source.sum(dim=1, keepdim=False, dtype=None)
        if form == "keyword dim negative one keepdim dtype none":
            return source.sum(dim=-1, keepdim=True, dtype=None)
        if form == "keyword dim negative two keepdim dtype none":
            return source.sum(dim=-2, keepdim=True, dtype=None)
        if form == "dtype float32 dim zero":
            return source.sum(dim=0, dtype=module.float32)
        if form == "dtype float32 dim one keepdim":
            return source.sum(dim=1, keepdim=True, dtype=module.float32)
        if form == "integer subclass dim":
            return source.sum(IntSubclass(0))
        if form == "numpy integer dim keepdim":
            return source.sum(dim=np.int64(-1), keepdim=True)
        if form == "tuple integer protocol dim":
            return source.sum((IndexOnly(),))
        if form == "list numpy integer dim keepdim":
            return source.sum([np.int64(-1)], keepdim=True)
        raise AssertionError(f"unknown rank-two dim sum form: {form}")

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
            "positional none dim keepdim true",
            "mixed none dim keepdim true",
            "keyword none dim keepdim true",
            "keepdim true dtype none",
            "keepdim true dtype float32",
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

    def test_keepdim_autograd_and_no_grad_match_pytorch_2_13(self):
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_kept = actual_leaf.transpose(0, 1).sum(
            dim=None, keepdim=True, dtype=torch.float32
        )
        expected_kept = expected_leaf.transpose(0, 1).sum(
            dim=None, keepdim=True, dtype=reference_torch.float32
        )
        self.assert_scalar_matches(actual_kept, expected_kept, case="keepdim tracked")
        actual_kept.sum().backward()
        expected_kept.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_kept = actual_empty.transpose(0, 2)[1].sum(None, True)
        expected_empty_kept = expected_empty.transpose(0, 2)[1].sum(None, True)
        self.assert_scalar_matches(
            actual_empty_kept, expected_empty_kept, case="keepdim empty"
        )
        actual_empty_kept.sum().backward()
        expected_empty_kept.sum().backward()
        self.assertEqual(actual_empty.grad.shape, tuple(expected_empty.grad.shape))
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.sum(
                dim=None, keepdim=True, dtype=torch.float
            )
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.sum(
                dim=None, keepdim=True, dtype=reference_torch.float
            )
        self.assert_scalar_matches(
            actual_untracked, expected_untracked, case="keepdim no_grad"
        )

    def test_rank_one_transpose_selected_offset_sum_edges_match_pytorch_2_13(self):
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
                actual.sum(), expected.sum(), case=("rank-one offset", case)
            )

    def test_rank_one_transpose_selected_offset_sum_autograd_match_pytorch_2_13(
        self,
    ):
        actual_empty = torch.zeros((0, 5), dtype=torch.float32, requires_grad=True)
        expected_empty = reference_torch.zeros(
            (0, 5), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_view = actual_empty.transpose(0, 1)[2]
        expected_empty_view = expected_empty.transpose(0, 1)[2]
        self.assertEqual(actual_empty_view.shape, tuple(expected_empty_view.shape))
        self.assertEqual(actual_empty_view.stride(), expected_empty_view.stride())
        self.assertEqual(
            actual_empty_view.storage_offset(), expected_empty_view.storage_offset()
        )
        self.assert_scalar_matches(
            actual_empty_view.sum(),
            expected_empty_view.sum(),
            case="rank-one empty",
        )
        actual_empty_view.sum().backward()
        expected_empty_view.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        values = np.arange(1, 21, dtype=np.float32).reshape(4, 5)[:, 2]
        actual_leaf, actual_view = self.rank_one_strided_vector(
            torch, values, requires_grad=True
        )
        expected_leaf, expected_view = self.rank_one_strided_vector(
            reference_torch, values, requires_grad=True
        )
        actual_loss = actual_view.sum()
        expected_loss = expected_view.sum()
        self.assert_scalar_matches(actual_loss, expected_loss, case="rank-one tracked")
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_view.sum()
        with reference_torch.no_grad():
            expected_untracked = expected_view.sum()
        self.assert_scalar_matches(
            actual_untracked, expected_untracked, case="rank-one no_grad"
        )

    def test_rank_one_dim_reductions_match_pytorch_2_13(self):
        forms = (
            "positional dim zero",
            "positional dim negative",
            "keyword dim zero dtype none",
            "keyword dim negative dtype none",
            "keyword dim zero keepdim dtype none",
            "keyword dim negative keepdim",
            "integer subclass dim",
            "numpy integer dim keepdim",
            "tuple integer protocol dim",
            "list numpy integer dim keepdim",
            "dtype float32",
        )
        actual_cases = self.rank_one_dim_cases(torch)
        expected_cases = self.rank_one_dim_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_dim_sum(actual_input, form, torch),
                    self.call_dim_sum(expected_input, form, reference_torch),
                    case=(case, form),
                )

    def test_rank_one_dim_backward_through_sum_matches_pytorch_2_13(self):
        actual_empty = torch.zeros((0, 5), dtype=torch.float32, requires_grad=True)
        expected_empty = reference_torch.zeros(
            (0, 5), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_view = actual_empty.transpose(0, 1)[2]
        expected_empty_view = expected_empty.transpose(0, 1)[2]
        actual_empty_view.sum(dim=0).backward()
        expected_empty_view.sum(dim=0).backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.detach().cpu().numpy()
        )

        values = np.arange(1, 21, dtype=np.float32).reshape(4, 5)[:, 2]
        actual_leaf, actual_view = self.rank_one_strided_vector(
            torch, values, requires_grad=True
        )
        expected_leaf, expected_view = self.rank_one_strided_vector(
            reference_torch, values, requires_grad=True
        )
        actual_loss = actual_view.sum(dim=-1, dtype=None)
        expected_loss = expected_view.sum(dim=-1, dtype=None)
        self.assert_scalar_matches(
            actual_loss, expected_loss, case="rank-one dim tracked"
        )
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        actual_kept_leaf, actual_kept_view = self.rank_one_strided_vector(
            torch, [1.0, -2.0, 3.0], requires_grad=True
        )
        expected_kept_leaf, expected_kept_view = self.rank_one_strided_vector(
            reference_torch, [1.0, -2.0, 3.0], requires_grad=True
        )
        actual_kept = actual_kept_view.sum(dim=0, keepdim=True, dtype=None)
        expected_kept = expected_kept_view.sum(dim=0, keepdim=True, dtype=None)
        self.assert_scalar_matches(actual_kept, expected_kept, case="rank-one kept")
        actual_kept.sum().backward()
        expected_kept.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_kept_leaf.grad),
            expected_kept_leaf.grad.detach().cpu().numpy(),
        )

    def test_rank_two_single_dim_reductions_match_pytorch_2_13(self):
        forms = (
            "positional dim zero",
            "positional dim one",
            "positional dim negative one",
            "positional dim negative two",
            "keyword dim zero dtype none",
            "keyword dim one dtype none",
            "keyword dim negative one keepdim dtype none",
            "keyword dim negative two keepdim dtype none",
            "dtype float32 dim zero",
            "dtype float32 dim one keepdim",
            "integer subclass dim",
            "numpy integer dim keepdim",
            "tuple integer protocol dim",
            "list numpy integer dim keepdim",
        )
        actual_cases = self.rank_two_dim_cases(torch)
        expected_cases = self.rank_two_dim_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            self.assertEqual(actual_input.shape, tuple(expected_input.shape))
            self.assertEqual(actual_input.stride(), expected_input.stride())
            self.assertEqual(actual_input.storage_offset(), expected_input.storage_offset())
            for form in forms:
                self.assert_tensor_matches(
                    self.call_rank_two_dim_sum(actual_input, form, torch),
                    self.call_rank_two_dim_sum(expected_input, form, reference_torch),
                    case=(case, form),
                )

    def test_rank_two_single_dim_backward_through_scalar_loss_matches_pytorch_2_13(
        self,
    ):
        for dimension in (0, 1, -1, -2):
            for keepdim in (False, True):
                values = np.arange(20, dtype=np.float32).reshape(4, 5)
                actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
                expected_leaf = reference_torch.tensor(
                    values, dtype=reference_torch.float32, requires_grad=True
                )
                actual_view = actual_leaf.transpose(0, 1)[1:4]
                expected_view = expected_leaf.transpose(0, 1)[1:4]
                actual_reduced = actual_view.sum(
                    dim=dimension, keepdim=keepdim, dtype=None
                )
                expected_reduced = expected_view.sum(
                    dim=dimension, keepdim=keepdim, dtype=None
                )
                self.assert_tensor_matches(
                    actual_reduced,
                    expected_reduced,
                    case=(dimension, keepdim, "forward"),
                )
                actual_loss = actual_reduced.sum()
                expected_loss = expected_reduced.sum()
                self.assert_scalar_matches(
                    actual_loss, expected_loss, case=(dimension, keepdim, "loss")
                )
                actual_loss.backward()
                expected_loss.backward()
                np.testing.assert_array_equal(
                    np.asarray(actual_leaf.grad),
                    expected_leaf.grad.detach().cpu().numpy(),
                )

        actual_empty_rows = torch.zeros((0, 3), dtype=torch.float32, requires_grad=True)
        expected_empty_rows = reference_torch.zeros(
            (0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_rows.sum(dim=0).sum().backward()
        expected_empty_rows.sum(dim=0).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty_rows.grad),
            expected_empty_rows.grad.detach().cpu().numpy(),
        )

        actual_empty_columns = torch.zeros(
            (2, 0), dtype=torch.float32, requires_grad=True
        )
        expected_empty_columns = reference_torch.zeros(
            (2, 0), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty_columns.sum(dim=1, keepdim=True).sum().backward()
        expected_empty_columns.sum(dim=1, keepdim=True).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty_columns.grad),
            expected_empty_columns.grad.detach().cpu().numpy(),
        )

        actual_leaf = torch.ones((2, 3), dtype=torch.float32, requires_grad=True)
        expected_leaf = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_untracked = actual_leaf.sum(
                dim=1, keepdim=True, dtype=torch.float32
            )
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.sum(
                dim=1, keepdim=True, dtype=reference_torch.float32
            )
        self.assert_tensor_matches(
            actual_untracked, expected_untracked, case="no_grad"
        )
        self.assertIsNone(actual_leaf.grad)

    def test_rank_one_dim_error_ordering_matches_pytorch_2_13(self):
        actual = torch.ones((2,), dtype=torch.float32)
        expected = reference_torch.ones((2,), dtype=reference_torch.float32)
        cases = (
            (
                lambda: actual.sum(2**100, "bad"),
                lambda: expected.sum(2**100, "bad"),
            ),
            (
                lambda: actual.sum(2**100, dtype=1),
                lambda: expected.sum(2**100, dtype=1),
            ),
            (lambda: actual.sum(2**100), lambda: expected.sum(2**100)),
            (lambda: actual.sum(1), lambda: expected.sum(1)),
            (lambda: actual.sum(True), lambda: expected.sum(True)),
            (lambda: actual.sum("bad", "bad"), lambda: expected.sum("bad", "bad")),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_rank_9_offset_permuted_sum_cases_match_pytorch_2_13(self):
        shape = (2, 3, 2, 2, 2, 2, 2, 2, 2)
        values = ((np.arange(2 * np.prod(shape), dtype=np.float32) % 23) - 11).reshape(
            (2, *shape)
        )
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
        actual_singleton = torch.tensor(singleton_values.tolist(), dtype=torch.float32)[
            1
        ].permute(2, 0, 3, 5, 4, 8, 7, 6, 1)
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
        self.assert_scalar_matches(actual_untracked, expected_untracked, case="no_grad")

    def test_rank_10_offset_permuted_sum_cases_match_pytorch_2_13(self):
        shape = (2, 3, 2, 5, 2, 3, 2, 2, 2, 2)
        values = ((np.arange(2 * np.prod(shape), dtype=np.float32) % 29) - 14).reshape(
            (2, *shape)
        )
        actual_source = torch.tensor(values.tolist(), dtype=torch.float32)
        expected_source = reference_torch.tensor(values, dtype=reference_torch.float32)
        permutations = (
            (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 0, 4, 6, 8, 1, 9, 3, 5, 7),
            (1, 3, 5, 7, 9, 0, 2, 4, 6, 8),
            (4, 1, 9, 0, 6, 2, 8, 5, 3, 7),
        )

        for permutation in permutations:
            actual = actual_source[1].permute(permutation)
            expected = expected_source[1].permute(permutation)
            self.assertFalse(actual.is_contiguous())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assert_scalar_matches(
                actual.sum(),
                expected.sum(),
                case=("rank-10 offset-permutation", permutation),
            )

        singleton_shape = (2, 1, 3, 2, 1, 2, 2, 2, 2, 2)
        singleton_values = (
            (np.arange(2 * np.prod(singleton_shape), dtype=np.float32) % 19) - 9
        ).reshape((2, *singleton_shape))
        actual_singleton = torch.tensor(singleton_values.tolist(), dtype=torch.float32)[
            1
        ].permute(2, 0, 3, 5, 4, 9, 8, 7, 6, 1)
        expected_singleton = reference_torch.tensor(
            singleton_values, dtype=reference_torch.float32
        )[1].permute(2, 0, 3, 5, 4, 9, 8, 7, 6, 1)
        self.assertFalse(actual_singleton.is_contiguous())
        self.assert_scalar_matches(
            actual_singleton.sum(),
            expected_singleton.sum(),
            case="rank-10 singleton",
        )
        np.testing.assert_array_equal(
            np.asarray(actual_singleton.contiguous()),
            expected_singleton.contiguous().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(-actual_singleton),
            (-expected_singleton).cpu().numpy(),
        )

        actual_empty = torch.zeros((2, 0, 3, 4, 5, 2, 2, 2, 2, 2), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3, 4, 5, 2, 2, 2, 2, 2),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_empty_view = actual_empty.permute(4, 2, 0, 9, 8, 7, 6, 5, 3, 1)
        expected_empty_view = expected_empty.permute(4, 2, 0, 9, 8, 7, 6, 5, 3, 1)
        self.assert_scalar_matches(
            actual_empty_view.sum(), expected_empty_view.sum(), case="rank-10 empty"
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
        actual_view = actual_leaf[1].permute(3, 1, 6, 0, 4, 9, 8, 7, 2, 5)
        expected_view = expected_leaf[1].permute(3, 1, 6, 0, 4, 9, 8, 7, 2, 5)
        actual_loss = actual_view.sum()
        expected_loss = expected_view.sum()
        self.assert_scalar_matches(actual_loss, expected_loss, case="rank-10 tracked")
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
            actual_untracked, expected_untracked, case="rank-10 no_grad"
        )

    def test_rank_11_offset_permuted_sum_cases_match_pytorch_2_13(self):
        shape = (2, 3, 2, 5, 2, 3, 2, 2, 2, 2, 2)
        values = ((np.arange(2 * np.prod(shape), dtype=np.float32) % 31) - 15).reshape(
            (2, *shape)
        )
        actual_source = torch.tensor(values.tolist(), dtype=torch.float32)
        expected_source = reference_torch.tensor(values, dtype=reference_torch.float32)
        permutations = (
            (10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 0, 4, 6, 8, 10, 1, 9, 3, 5, 7),
            (1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 10),
            (4, 1, 10, 0, 6, 2, 8, 5, 9, 3, 7),
        )

        for permutation in permutations:
            actual = actual_source[1].permute(permutation)
            expected = expected_source[1].permute(permutation)
            self.assertFalse(actual.is_contiguous())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assert_scalar_matches(
                actual.sum(),
                expected.sum(),
                case=("rank-11 offset-permutation", permutation),
            )

        singleton_shape = (2, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2)
        singleton_values = (
            (np.arange(2 * np.prod(singleton_shape), dtype=np.float32) % 19) - 9
        ).reshape((2, *singleton_shape))
        actual_singleton = torch.tensor(singleton_values.tolist(), dtype=torch.float32)[
            1
        ].permute(2, 0, 3, 5, 4, 10, 9, 8, 7, 6, 1)
        expected_singleton = reference_torch.tensor(
            singleton_values, dtype=reference_torch.float32
        )[1].permute(2, 0, 3, 5, 4, 10, 9, 8, 7, 6, 1)
        self.assertFalse(actual_singleton.is_contiguous())
        self.assert_scalar_matches(
            actual_singleton.sum(),
            expected_singleton.sum(),
            case="rank-11 singleton",
        )
        np.testing.assert_array_equal(
            np.asarray(actual_singleton.contiguous()),
            expected_singleton.contiguous().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(-actual_singleton),
            (-expected_singleton).cpu().numpy(),
        )

        actual_empty = torch.zeros(
            (2, 0, 3, 4, 5, 2, 2, 2, 2, 2, 2), requires_grad=True
        )
        expected_empty = reference_torch.zeros(
            (2, 0, 3, 4, 5, 2, 2, 2, 2, 2, 2),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_empty_view = actual_empty.permute(4, 2, 0, 10, 9, 8, 7, 6, 5, 3, 1)
        expected_empty_view = expected_empty.permute(4, 2, 0, 10, 9, 8, 7, 6, 5, 3, 1)
        self.assert_scalar_matches(
            actual_empty_view.sum(), expected_empty_view.sum(), case="rank-11 empty"
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
        actual_view = actual_leaf[1].permute(3, 1, 6, 0, 4, 10, 9, 8, 7, 2, 5)
        expected_view = expected_leaf[1].permute(3, 1, 6, 0, 4, 10, 9, 8, 7, 2, 5)
        actual_loss = actual_view.sum()
        expected_loss = expected_view.sum()
        self.assert_scalar_matches(actual_loss, expected_loss, case="rank-11 tracked")
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
            actual_untracked, expected_untracked, case="rank-11 no_grad"
        )

        rank_13_shape = (2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        rank_13_values = (
            (np.arange(2 * np.prod(rank_13_shape), dtype=np.float32) % 37) - 18
        ).reshape((2, *rank_13_shape))
        actual_rank_13 = torch.tensor(rank_13_values.tolist(), dtype=torch.float32)[
            1
        ].permute(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        expected_rank_13 = reference_torch.tensor(
            rank_13_values, dtype=reference_torch.float32
        )[1].permute(12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        self.assertFalse(actual_rank_13.is_contiguous())
        self.assert_scalar_matches(
            actual_rank_13.sum(),
            expected_rank_13.sum(),
            case="rank-13 fallback",
        )

    def test_rank_12_offset_permuted_sum_cases_match_pytorch_2_13(self):
        shape = (2, 3, 2, 5, 2, 3, 2, 2, 2, 2, 2, 2)
        values = ((np.arange(2 * np.prod(shape), dtype=np.float32) % 37) - 18).reshape(
            (2, *shape)
        )
        actual_source = torch.tensor(values.tolist(), dtype=torch.float32)
        expected_source = reference_torch.tensor(values, dtype=reference_torch.float32)
        permutations = (
            (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 0, 4, 6, 8, 10, 1, 11, 9, 3, 5, 7),
            (1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10),
            (4, 1, 11, 0, 6, 2, 8, 5, 10, 9, 3, 7),
        )

        for permutation in permutations:
            actual = actual_source[1].permute(permutation)
            expected = expected_source[1].permute(permutation)
            self.assertFalse(actual.is_contiguous())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assert_scalar_matches(
                actual.sum(),
                expected.sum(),
                case=("rank-12 tensor.sum offset-permutation", permutation),
            )
            self.assert_scalar_matches(
                torch.sum(actual),
                reference_torch.sum(expected),
                case=("rank-12 torch.sum offset-permutation", permutation),
            )

        singleton_shape = (2, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2)
        singleton_values = (
            (np.arange(2 * np.prod(singleton_shape), dtype=np.float32) % 19) - 9
        ).reshape((2, *singleton_shape))
        actual_singleton = torch.tensor(singleton_values.tolist(), dtype=torch.float32)[
            1
        ].permute(2, 0, 3, 5, 4, 11, 10, 9, 8, 7, 6, 1)
        expected_singleton = reference_torch.tensor(
            singleton_values, dtype=reference_torch.float32
        )[1].permute(2, 0, 3, 5, 4, 11, 10, 9, 8, 7, 6, 1)
        self.assertFalse(actual_singleton.is_contiguous())
        self.assert_scalar_matches(
            actual_singleton.sum(),
            expected_singleton.sum(),
            case="rank-12 singleton",
        )
        np.testing.assert_array_equal(
            np.asarray(actual_singleton.contiguous()),
            expected_singleton.contiguous().cpu().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(-actual_singleton),
            (-expected_singleton).cpu().numpy(),
        )

        actual_empty = torch.zeros(
            (2, 0, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2), requires_grad=True
        )
        expected_empty = reference_torch.zeros(
            (2, 0, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2),
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_empty_view = actual_empty.permute(4, 2, 0, 11, 10, 9, 8, 7, 6, 5, 3, 1)
        expected_empty_view = expected_empty.permute(
            4, 2, 0, 11, 10, 9, 8, 7, 6, 5, 3, 1
        )
        self.assert_scalar_matches(
            actual_empty_view.sum(),
            expected_empty_view.sum(),
            case="rank-12 empty",
        )
        self.assert_scalar_matches(
            torch.sum(actual_empty_view),
            reference_torch.sum(expected_empty_view),
            case="rank-12 torch.sum empty",
        )
        actual_empty_view.sum().backward()
        expected_empty_view.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        leaf_shape = (2, 2, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2, 2)
        leaf_values = (
            (np.arange(np.prod(leaf_shape), dtype=np.float32) % 37) - 18
        ).reshape(leaf_shape)
        actual_leaf = torch.tensor(
            leaf_values.tolist(), dtype=torch.float32, requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            leaf_values, dtype=reference_torch.float32, requires_grad=True
        )
        permutation = (3, 1, 6, 0, 4, 11, 10, 9, 8, 7, 2, 5)
        actual_view = actual_leaf[1].permute(permutation)
        expected_view = expected_leaf[1].permute(permutation)
        actual_loss = torch.sum(actual_view)
        expected_loss = reference_torch.sum(expected_view)
        self.assert_scalar_matches(actual_loss, expected_loss, case="rank-12 tracked")
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = torch.sum(actual_view)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.sum(expected_view)
        self.assert_scalar_matches(
            actual_untracked, expected_untracked, case="rank-12 no_grad"
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
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
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

    def test_multi_dim_rank_device_subclass_and_cross_dtype_boundaries_remain_unsupported(
        self,
    ):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        actual_rank_three = torch.ones((1, 2, 3))
        expected_rank_three = reference_torch.ones(
            (1, 2, 3), dtype=reference_torch.float32
        )
        cases = (
            (lambda: actual.sum((0, 1)), lambda: expected.sum((0, 1))),
            (lambda: actual.sum([0, 1]), lambda: expected.sum([0, 1])),
            (lambda: actual_rank_three.sum(dim=1), lambda: expected_rank_three.sum(dim=1)),
            (
                lambda: actual.sum(dtype=reference_torch.float64),
                lambda: expected.sum(dtype=reference_torch.float64),
            ),
            (
                lambda: actual.sum(dim=None, dtype=reference_torch.float64),
                lambda: expected.sum(dim=None, dtype=reference_torch.float64),
            ),
            (
                lambda: actual.sum(dim=0, dtype=reference_torch.float64),
                lambda: expected.sum(dim=0, dtype=reference_torch.float64),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises((TypeError, NotImplementedError)):
                    actual_call()
                expected_call()

        if reference_torch.cuda.is_available():
            with self.assertRaises((NotImplementedError, RuntimeError)):
                torch.ones((2, 3), device="cuda").sum(dim=0)
            self.assertEqual(
                tuple(
                    reference_torch.ones(
                        (2, 3),
                        dtype=reference_torch.float32,
                        device="cuda",
                    ).sum(dim=0).shape
                ),
                (3,),
            )

        actual_subclass_error = None
        try:
            class ActualSubclass(torch.Tensor):
                pass

            actual_subclass = ActualSubclass(torch.ones((2, 3)))
        except TypeError as error:
            actual_subclass_error = error
            actual_subclass = None

        class ExpectedSubclass(reference_torch.Tensor):
            pass

        expected_subclass = reference_torch.Tensor._make_subclass(
            ExpectedSubclass, expected
        )
        self.assertEqual(tuple(expected_subclass.sum(dim=0).shape), (3,))
        if actual_subclass is None:
            self.assertIsInstance(actual_subclass_error, TypeError)
        else:
            with self.assertRaises(NotImplementedError):
                actual_subclass.sum(dim=0)


if __name__ == "__main__":
    unittest.main()
