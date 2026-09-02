import inspect
import re
import types
import unittest

import numpy as np
import torch_rs as torch


METHOD_DOC = """
sum(dim=None, keepdim=False, dtype=None) -> Tensor

See :func:`torch.sum`
"""

EXPECTED_OVERLOADS = (
    "but expected one of:\n"
    " * (*, torch.dtype dtype = None)\n"
    " * (tuple of ints dim, bool keepdim = False, *, "
    "torch.dtype dtype = None)\n"
)


class TensorSumTests(unittest.TestCase):
    def assert_scalar(self, value, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(value.shape, ())
            self.assertEqual(value.stride(), ())
            self.assertEqual(value.storage_offset(), 0)
            self.assertEqual(value.numel(), 1)
            self.assertTrue(value.is_contiguous())
            self.assertIs(value.dtype, torch.float32)
            self.assertEqual(value.device, torch.device("cpu"))
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.float32(value.item()).view(np.uint32).item(),
                np.float32(expected).view(np.uint32).item(),
            )

    def assert_keepdim_reduction(self, value, expected, source, *, case):
        expected_shape = (1,) * len(source.shape)
        with self.subTest(case=case, metadata=True):
            self.assertEqual(value.shape, expected_shape)
            self.assertEqual(value.stride(), (1,) * len(expected_shape))
            self.assertEqual(value.storage_offset(), 0)
            self.assertEqual(value.numel(), 1)
            self.assertTrue(value.is_contiguous())
            self.assertIs(value.dtype, torch.float32)
            self.assertEqual(value.device, torch.device("cpu"))
            self.assertFalse(value.is_set_to(source))
            if source.numel():
                self.assertNotEqual(value.data_ptr(), source.data_ptr())
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.float32(value.item()).view(np.uint32).item(),
                np.float32(expected).view(np.uint32).item(),
            )

    @staticmethod
    def supported_calls(source):
        return (
            ("default", lambda: source.sum()),
            ("positional none dim", lambda: source.sum(None)),
            ("keyword none dim", lambda: source.sum(dim=None)),
            ("none dim keepdim false", lambda: source.sum(None, False)),
            ("dtype none", lambda: source.sum(dtype=None)),
            ("dtype float32", lambda: source.sum(dtype=torch.float32)),
            ("dtype float alias", lambda: source.sum(dtype=torch.float)),
            (
                "none dim dtype float32",
                lambda: source.sum(dim=None, keepdim=False, dtype=torch.float32),
            ),
        )

    @staticmethod
    def supported_keepdim_calls(source):
        return (
            ("positional none keepdim true", lambda: source.sum(None, True)),
            ("keyword none keepdim true", lambda: source.sum(dim=None, keepdim=True)),
            (
                "positional none keyword keepdim true",
                lambda: source.sum(None, keepdim=True),
            ),
            (
                "keepdim dtype none",
                lambda: source.sum(dim=None, keepdim=True, dtype=None),
            ),
            (
                "keepdim dtype float32",
                lambda: source.sum(dim=None, keepdim=True, dtype=torch.float32),
            ),
        )

    @staticmethod
    def rank_one_strided_vector(values, *, requires_grad=False):
        rows = len(values)
        columns = 5
        selected_column = 2
        matrix = np.full((rows, columns), np.float32(0.5), dtype=np.float32)
        matrix[:, selected_column] = np.asarray(values, dtype=np.float32)
        source = torch.tensor(
            matrix.tolist(), dtype=torch.float32, requires_grad=requires_grad
        )
        return (
            source,
            source.transpose(0, 1)[selected_column],
            matrix[:, selected_column],
        )

    @staticmethod
    def sequential_float32_sum(values):
        total = np.float32(0.0)
        for value in np.asarray(values, dtype=np.float32):
            total = np.float32(total + value)
        return total

    def test_dtype_only_forms_reuse_full_reduction_values_and_metadata(self):
        dense = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        noncontiguous = dense.transpose(0, 2)
        cases = (
            ("scalar", torch.tensor(-3.5), np.float32(-3.5)),
            ("negative zero", torch.tensor(-0.0), np.float32(0.0)),
            (
                "empty",
                torch.zeros((2, 0, 3)).transpose(0, 2)[1],
                np.float32(0.0),
            ),
            ("contiguous offset", dense[1], np.float32(210.0)),
            ("offset", noncontiguous[1], np.float32(66.0)),
            ("noncontiguous", noncontiguous, np.float32(276.0)),
        )

        for name, source, expected in cases:
            for form, call in self.supported_calls(source):
                self.assert_scalar(call(), expected, case=(name, form))

    def test_keepdim_full_reduction_preserves_rank_shape_and_dtype_defaults(self):
        dense = torch.tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist())
        noncontiguous = dense.transpose(0, 2)
        cases = (
            ("scalar", torch.tensor(-3.5), np.float32(-3.5)),
            ("empty", torch.zeros((2, 0, 3)).transpose(0, 2)[1], np.float32(0.0)),
            ("singleton", torch.tensor([[[7.0]]])[0], np.float32(7.0)),
            ("contiguous offset", dense[1], np.float32(210.0)),
            ("offset", noncontiguous[1], np.float32(66.0)),
            ("noncontiguous", noncontiguous, np.float32(276.0)),
        )

        for name, source, expected in cases:
            for form, call in self.supported_keepdim_calls(source):
                self.assert_keepdim_reduction(
                    call(), expected, source, case=(name, form)
                )

    def test_keepdim_full_reduction_no_grad_and_backward_through_final_scalar_sum(
        self,
    ):
        scalar = torch.tensor(-3.0, requires_grad=True)
        scalar.sum(dim=None, keepdim=True).sum().backward()
        self.assertEqual(scalar.grad.item(), 1.0)

        singleton = torch.tensor([[[7.0]]], requires_grad=True)
        singleton[0].sum(dim=None, keepdim=True).sum().backward()
        self.assertEqual(singleton.grad.tolist(), [[[1.0]]])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.transpose(0, 2)[1].sum(dim=None, keepdim=True).sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        offset = torch.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            requires_grad=True,
        )
        offset[1].sum(dim=None, keepdim=True).sum().backward()
        expected_offset_grad = np.zeros((2, 3, 4), dtype=np.float32)
        expected_offset_grad[1] = 1.0
        np.testing.assert_array_equal(np.asarray(offset.grad), expected_offset_grad)

        noncontiguous = torch.tensor(
            [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True
        )
        noncontiguous.transpose(0, 1).sum(dim=None, keepdim=True).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(noncontiguous.grad),
            np.ones((2, 3), dtype=np.float32),
        )

        with torch.no_grad():
            untracked = noncontiguous.transpose(0, 1).sum(dim=None, keepdim=True)
        self.assertEqual(untracked.shape, (1, 1))
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

    def test_dtype_forms_preserve_autograd_accumulation_and_empty_gradients(self):
        leaf = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], requires_grad=True)
        loss = leaf.transpose(0, 1).sum(dim=None, keepdim=False, dtype=torch.float32)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        self.assertEqual(leaf.grad.tolist(), [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty.transpose(0, 2).sum(None, False, dtype=None).backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        with torch.no_grad():
            untracked = leaf.sum(dim=None, dtype=torch.float)
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertTrue(leaf.sum(None, dtype=torch.float32).requires_grad)

    def test_rank_one_transpose_selected_offset_sum_edges(self):
        cases = (
            ("signed zero", [-0.0, 0.0, -0.0, 0.0]),
            ("nan", [1.0, np.nan, 2.0, -3.0]),
            ("positive infinity", [1.0, np.inf, 2.0, 3.0]),
            ("negative infinity", [1.0, -np.inf, 2.0, 3.0]),
            ("sequential cancellation", [1.0e20, -1.0e20, 3.0, -0.0]),
        )

        for case, values in cases:
            _, view, selected = self.rank_one_strided_vector(values)
            self.assertEqual(view.shape, (len(values),))
            self.assertEqual(view.stride(), (5,))
            self.assertEqual(view.storage_offset(), 2)
            self.assertFalse(view.is_contiguous())
            self.assert_scalar(
                view.sum(),
                self.sequential_float32_sum(selected),
                case=("rank-one transpose-selected offset", case),
            )

    def test_rank_one_transpose_selected_offset_sum_empty_no_grad_and_repeated_backward(
        self,
    ):
        empty = torch.zeros((0, 5), requires_grad=True)
        empty_view = empty.transpose(0, 1)[2]
        self.assertEqual(empty_view.shape, (0,))
        self.assertEqual(empty_view.stride(), (5,))
        self.assertEqual(empty_view.storage_offset(), 2)
        self.assert_scalar(empty_view.sum(), np.float32(0.0), case="rank-one empty")
        empty_view.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [])

        leaf, view, selected = self.rank_one_strided_vector(
            np.arange(1, 21, dtype=np.float32).reshape(4, 5)[:, 2],
            requires_grad=True,
        )
        loss = view.sum()
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        expected_gradient = np.zeros((4, 5), dtype=np.float32)
        expected_gradient[:, 2] = 2.0
        np.testing.assert_array_equal(np.asarray(leaf.grad), expected_gradient)

        with torch.no_grad():
            untracked = view.sum()
        self.assert_scalar(
            untracked,
            self.sequential_float32_sum(selected),
            case="rank-one no_grad",
        )
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

    def test_rank_11_offset_permuted_sum_cases_cover_boundary_behaviors(self):
        shape = (2, 3, 2, 5, 2, 3, 2, 2, 2, 2, 2)
        values = ((np.arange(2 * np.prod(shape), dtype=np.float32) % 31) - 15).reshape(
            (2, *shape)
        )
        source = torch.tensor(values.tolist(), dtype=torch.float32)
        permutations = (
            (10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 0, 4, 6, 8, 10, 1, 9, 3, 5, 7),
            (1, 3, 5, 7, 9, 0, 2, 4, 6, 8, 10),
            (4, 1, 10, 0, 6, 2, 8, 5, 9, 3, 7),
        )

        for permutation in permutations:
            actual = source[1].permute(permutation)
            expected = np.transpose(values[1], permutation).sum(dtype=np.float32)
            self.assertFalse(actual.is_contiguous())
            self.assertNotEqual(actual.storage_offset(), 0)
            self.assert_scalar(
                actual.sum(),
                expected,
                case=("rank-11 offset-permutation", permutation),
            )

        singleton_shape = (2, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2)
        singleton_values = (
            (np.arange(2 * np.prod(singleton_shape), dtype=np.float32) % 19) - 9
        ).reshape((2, *singleton_shape))
        singleton_permutation = (2, 0, 3, 5, 4, 10, 9, 8, 7, 6, 1)
        singleton = torch.tensor(singleton_values.tolist(), dtype=torch.float32)[
            1
        ].permute(singleton_permutation)
        self.assertFalse(singleton.is_contiguous())
        self.assert_scalar(
            singleton.sum(),
            np.transpose(singleton_values[1], singleton_permutation).sum(
                dtype=np.float32
            ),
            case="rank-11 singleton",
        )
        np.testing.assert_array_equal(
            np.asarray(singleton.contiguous()),
            np.ascontiguousarray(
                np.transpose(singleton_values[1], singleton_permutation)
            ),
        )
        np.testing.assert_array_equal(
            np.asarray(-singleton),
            -np.transpose(singleton_values[1], singleton_permutation),
        )

        empty = torch.zeros((2, 0, 3, 4, 5, 2, 2, 2, 2, 2, 2), requires_grad=True)
        empty_view = empty.permute(4, 2, 0, 10, 9, 8, 7, 6, 5, 3, 1)
        self.assert_scalar(empty_view.sum(), np.float32(0.0), case="rank-11 empty")
        empty_view.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        leaf_shape = (2, 2, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2)
        leaf_values = (
            (np.arange(np.prod(leaf_shape), dtype=np.float32) % 31) - 15
        ).reshape(leaf_shape)
        leaf = torch.tensor(
            leaf_values.tolist(),
            dtype=torch.float32,
            requires_grad=True,
        )
        view = leaf[1].permute(3, 1, 6, 0, 4, 10, 9, 8, 7, 2, 5)
        loss = view.sum()
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        gradient = np.asarray(leaf.grad)
        np.testing.assert_array_equal(gradient[0], np.zeros_like(gradient[0]))
        np.testing.assert_array_equal(gradient[1], np.full_like(gradient[1], 2.0))

        with torch.no_grad():
            untracked = view.sum()
        self.assert_scalar(
            untracked,
            np.transpose(leaf_values[1], (3, 1, 6, 0, 4, 10, 9, 8, 7, 2, 5)).sum(
                dtype=np.float32
            ),
            case="rank-11 no_grad",
        )
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

        rank_13_shape = (2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        rank_13_values = (
            (np.arange(2 * np.prod(rank_13_shape), dtype=np.float32) % 37) - 18
        ).reshape((2, *rank_13_shape))
        rank_13_permutation = (12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
        rank_13 = torch.tensor(rank_13_values.tolist(), dtype=torch.float32)[1].permute(
            rank_13_permutation
        )
        self.assertFalse(rank_13.is_contiguous())
        self.assert_scalar(
            rank_13.sum(),
            np.transpose(rank_13_values[1], rank_13_permutation).sum(dtype=np.float32),
            case="rank-13 boundary fallback",
        )

    def test_rank_12_offset_permuted_sum_cases_cover_boundary_behaviors(self):
        shape = (2, 3, 2, 5, 2, 3, 2, 2, 2, 2, 2, 2)
        values = ((np.arange(2 * np.prod(shape), dtype=np.float32) % 37) - 18).reshape(
            (2, *shape)
        )
        source = torch.tensor(values.tolist(), dtype=torch.float32)
        permutations = (
            (11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
            (2, 0, 4, 6, 8, 10, 1, 11, 9, 3, 5, 7),
            (1, 3, 5, 7, 9, 11, 0, 2, 4, 6, 8, 10),
            (4, 1, 11, 0, 6, 2, 8, 5, 10, 9, 3, 7),
        )

        for permutation in permutations:
            actual = source[1].permute(permutation)
            expected = np.transpose(values[1], permutation).sum(dtype=np.float32)
            self.assertFalse(actual.is_contiguous())
            self.assertNotEqual(actual.storage_offset(), 0)
            self.assert_scalar(
                actual.sum(),
                expected,
                case=("rank-12 tensor.sum offset-permutation", permutation),
            )
            self.assert_scalar(
                torch.sum(actual),
                expected,
                case=("rank-12 torch.sum offset-permutation", permutation),
            )

        singleton_shape = (2, 1, 3, 2, 1, 2, 2, 2, 2, 2, 2, 2)
        singleton_values = (
            (np.arange(2 * np.prod(singleton_shape), dtype=np.float32) % 19) - 9
        ).reshape((2, *singleton_shape))
        singleton_permutation = (2, 0, 3, 5, 4, 11, 10, 9, 8, 7, 6, 1)
        singleton = torch.tensor(singleton_values.tolist(), dtype=torch.float32)[
            1
        ].permute(singleton_permutation)
        self.assertFalse(singleton.is_contiguous())
        self.assert_scalar(
            singleton.sum(),
            np.transpose(singleton_values[1], singleton_permutation).sum(
                dtype=np.float32
            ),
            case="rank-12 singleton",
        )
        np.testing.assert_array_equal(
            np.asarray(singleton.contiguous()),
            np.ascontiguousarray(
                np.transpose(singleton_values[1], singleton_permutation)
            ),
        )
        np.testing.assert_array_equal(
            np.asarray(-singleton),
            -np.transpose(singleton_values[1], singleton_permutation),
        )

        empty = torch.zeros((2, 0, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2), requires_grad=True)
        empty_view = empty.permute(4, 2, 0, 11, 10, 9, 8, 7, 6, 5, 3, 1)
        self.assert_scalar(empty_view.sum(), np.float32(0.0), case="rank-12 empty")
        self.assert_scalar(
            torch.sum(empty_view), np.float32(0.0), case="rank-12 torch.sum empty"
        )
        empty_view.sum().backward()
        self.assertEqual(empty.grad.shape, empty.shape)
        self.assertEqual(empty.grad.tolist(), [[], []])

        leaf_shape = (2, 2, 3, 4, 5, 2, 2, 2, 2, 2, 2, 2, 2)
        leaf_values = (
            (np.arange(np.prod(leaf_shape), dtype=np.float32) % 37) - 18
        ).reshape(leaf_shape)
        leaf = torch.tensor(
            leaf_values.tolist(),
            dtype=torch.float32,
            requires_grad=True,
        )
        view = leaf[1].permute(3, 1, 6, 0, 4, 11, 10, 9, 8, 7, 2, 5)
        loss = torch.sum(view)
        self.assertTrue(loss.requires_grad)
        self.assertFalse(loss.is_leaf)
        loss.backward()
        loss.backward()
        gradient = np.asarray(leaf.grad)
        np.testing.assert_array_equal(gradient[0], np.zeros_like(gradient[0]))
        np.testing.assert_array_equal(gradient[1], np.full_like(gradient[1], 2.0))

        with torch.no_grad():
            untracked = torch.sum(view)
        self.assert_scalar(
            untracked,
            np.transpose(leaf_values[1], (3, 1, 6, 0, 4, 11, 10, 9, 8, 7, 2, 5)).sum(
                dtype=np.float32
            ),
            case="rank-12 no_grad",
        )
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

    def test_descriptor_documentation_and_unbound_dtype_calls(self):
        tensor = torch.tensor([1.0, 2.0])
        descriptor = inspect.getattr_static(torch.Tensor, "sum")
        bound = tensor.sum

        self.assertIs(type(descriptor), types.MethodDescriptorType)
        self.assertIs(type(bound), types.BuiltinMethodType)
        for callable_object in (descriptor, bound):
            self.assertEqual(callable_object.__name__, "sum")
            self.assertEqual(callable_object.__doc__, METHOD_DOC)
            self.assertIsNone(callable_object.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)

        self.assertEqual(descriptor(tensor).item(), 3.0)
        self.assertEqual(descriptor(tensor, dtype=None).item(), 3.0)
        self.assertEqual(descriptor(tensor, dtype=torch.float32).item(), 3.0)

    def test_invalid_dtype_and_argument_errors_match_the_pytorch_overload(self):
        tensor = torch.ones((2, 3))
        invalid = "sum() received an invalid combination of arguments - got "
        cases = (
            (
                lambda: tensor.sum(dtype=1),
                f"{invalid}(dtype=int, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(dtype=object()),
                f"{invalid}(dtype=object, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(dim=None, dtype=1),
                "sum(): argument 'dtype' must be torch.dtype, not int",
            ),
            (
                lambda: tensor.sum(None, False, dtype=object()),
                "sum(): argument 'dtype' must be torch.dtype, not object",
            ),
            (
                lambda: tensor.sum(torch.float32),
                f"{invalid}(torch.dtype), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(extra=True),
                f"{invalid}(extra=bool, ), {EXPECTED_OVERLOADS}",
            ),
            (
                lambda: tensor.sum(0, False, torch.float32),
                "sum() takes from 1 to 2 positional arguments but 3 were given",
            ),
        )
        for call, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
                    call()

    def test_dimension_keepdim_and_out_forms_remain_unsupported(self):
        tensor = torch.ones((2, 3))
        cases = (
            ("positional dim", lambda: tensor.sum(0)),
            ("keyword dim", lambda: tensor.sum(dim=0)),
            ("positional keepdim", lambda: tensor.sum(0, False)),
            ("keyword keepdim", lambda: tensor.sum(dim=0, keepdim=True)),
            ("out", lambda: tensor.sum(out=None)),
        )
        for case, call in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    TypeError,
                    r"^sum\(\) received an invalid combination of arguments",
                ):
                    call()


if __name__ == "__main__":
    unittest.main()
