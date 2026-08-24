import sys
import unittest

import numpy as np
import torch_rs as torch


class TransposeBackwardTests(unittest.TestCase):
    gradient_pattern = np.array(
        [
            0x00000000,
            0x80000000,
            0x3F800001,
            0xBF000001,
            0x41234567,
            0xC1234567,
            0x00800000,
            0x80800000,
            0x3EAAAAAB,
            0xBEAAAAAB,
        ],
        dtype=np.uint32,
    ).view(np.float32)

    def patterned_values(self, shape):
        elements = int(np.prod(shape, dtype=np.int64))
        return np.resize(self.gradient_pattern, elements).reshape(shape)

    def tensor(self, values, *, requires_grad=False):
        if values.size == 0:
            return torch.zeros(values.shape, requires_grad=requires_grad)
        return torch.tensor(values.tolist(), requires_grad=requires_grad)

    def rank_two_case(self, kind, shape):
        rows, columns = shape
        if kind == "direct":
            leaf_values = np.zeros(shape, dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            return leaf, leaf, lambda gradient: gradient
        if kind == "offset":
            leaf_values = np.zeros((2, rows, columns), dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            source = leaf[1]

            def to_leaf_gradient(gradient):
                expected = np.zeros(leaf_values.shape, dtype=np.float32)
                expected[1] = gradient
                return expected

            return leaf, source, to_leaf_gradient
        if kind == "noncontiguous":
            leaf_values = np.zeros((columns, rows), dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            source = leaf.transpose(0, 1)
            return leaf, source, lambda gradient: gradient.T.copy()
        if kind == "offset noncontiguous":
            leaf_values = np.zeros((columns, rows, 2), dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            source = leaf.transpose(0, 2)[1]

            def to_leaf_gradient(gradient):
                expected = np.zeros(leaf_values.shape, dtype=np.float32)
                expected[:, :, 1] = gradient.T
                return expected

            return leaf, source, to_leaf_gradient
        raise AssertionError(f"unknown rank-two case: {kind}")

    def batched_case(self, kind, shape):
        batch_shape = shape[:-2]
        rows, columns = shape[-2:]
        if kind == "direct":
            leaf_values = np.zeros(shape, dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            return leaf, leaf, lambda gradient: gradient
        if kind == "offset":
            leaf_values = np.zeros((2, *shape), dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            source = leaf[1]

            def to_leaf_gradient(gradient):
                expected = np.zeros(leaf_values.shape, dtype=np.float32)
                expected[1] = gradient
                return expected

            return leaf, source, to_leaf_gradient
        if kind == "noncontiguous":
            leaf_values = np.zeros((*batch_shape, columns, rows), dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            source = leaf.transpose(-2, -1)
            return leaf, source, lambda gradient: np.swapaxes(gradient, -2, -1).copy()
        if kind == "offset noncontiguous":
            leaf_values = np.zeros((columns, *batch_shape, rows, 2), dtype=np.float32)
            leaf = self.tensor(leaf_values, requires_grad=True)
            source = leaf.transpose(0, -1)[1]

            def to_leaf_gradient(gradient):
                expected = np.zeros(leaf_values.shape, dtype=np.float32)
                expected[..., 1] = np.moveaxis(gradient, -1, 0)
                return expected

            return leaf, source, to_leaf_gradient
        raise AssertionError(f"unknown batched case: {kind}")

    def assert_gradient_bits(self, actual, expected):
        actual_values = np.asarray(actual)
        self.assertEqual(actual_values.shape, expected.shape)
        np.testing.assert_array_equal(
            actual_values.view(np.uint32), expected.view(np.uint32)
        )

    def exercise_backward(self, leaf, source, operation, to_leaf_gradient):
        transformed = operation(source)
        weights = self.patterned_values(transformed.shape)
        loss = (transformed * self.tensor(weights)).sum()
        expected_once = to_leaf_gradient(np.swapaxes(weights, -2, -1).copy())

        loss.backward()
        self.assert_gradient_bits(leaf.grad, expected_once)
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        transformed = operation(source)
        (transformed * self.tensor(weights)).sum().backward()
        expected_twice = expected_once + expected_once
        self.assert_gradient_bits(leaf.grad, expected_twice)

        reusable_loss = operation(source).sum()
        source_ones = np.ones(source.shape, dtype=np.float32)
        expected_reusable = to_leaf_gradient(source_ones)
        reusable_loss.backward()
        expected_twice += expected_reusable
        self.assert_gradient_bits(leaf.grad, expected_twice)
        reusable_loss.backward()
        expected_twice += expected_reusable
        self.assert_gradient_bits(leaf.grad, expected_twice)

    def test_rank_two_fast_path_preserves_bits_accumulation_and_graph_freeing(self):
        cases = (
            ("empty rows", "direct", (0, 7)),
            ("empty columns", "direct", (5, 0)),
            ("singleton", "direct", (1, 1)),
            ("singleton axis", "direct", (1, 7)),
            ("awkward rectangular", "direct", (7, 11)),
            ("offset", "offset", (5, 7)),
            ("noncontiguous", "noncontiguous", (5, 7)),
            ("offset noncontiguous", "offset noncontiguous", (5, 7)),
        )
        operations = (
            ("Tensor.t", lambda tensor: tensor.t()),
            ("torch.t", torch.t),
            (
                "Tensor.transpose",
                lambda tensor: tensor.transpose(0, 1),
            ),
            (
                "torch.transpose",
                lambda tensor: torch.transpose(tensor, 0, 1),
            ),
            (
                "Tensor.permute",
                lambda tensor: tensor.permute(1, 0),
            ),
            (
                "torch.permute",
                lambda tensor: torch.permute(tensor, (1, 0)),
            ),
        )

        for case, kind, shape in cases:
            for operation_name, operation in operations:
                with self.subTest(case=case, operation=operation_name):
                    leaf, source, to_leaf_gradient = self.rank_two_case(kind, shape)
                    self.exercise_backward(leaf, source, operation, to_leaf_gradient)

    def test_batched_fast_path_preserves_bits_accumulation_and_graph_lifecycle(self):
        cases = (
            ("rank three rectangular", "direct", (2, 7, 11)),
            ("rank four singleton", "direct", (2, 1, 1, 7)),
            ("rank five rectangular", "direct", (2, 3, 2, 5, 7)),
            ("empty rows", "direct", (2, 3, 0, 7)),
            ("empty columns", "direct", (2, 5, 0)),
            ("empty batch", "direct", (2, 0, 3, 5)),
            ("offset", "offset", (2, 3, 5)),
            ("noncontiguous", "noncontiguous", (2, 3, 5, 7)),
            ("offset noncontiguous", "offset noncontiguous", (2, 3, 5)),
        )

        for case, kind, shape in cases:
            rank = len(shape)
            permutation = (*range(rank - 2), rank - 1, rank - 2)
            operations = (
                (
                    "Tensor.transpose",
                    lambda tensor: tensor.transpose(-2, -1),
                ),
                (
                    "torch.transpose",
                    lambda tensor: torch.transpose(tensor, rank - 2, rank - 1),
                ),
                (
                    "Tensor.swapdims",
                    lambda tensor: tensor.swapdims(-2, -1),
                ),
                (
                    "torch.swapdims",
                    lambda tensor: torch.swapdims(tensor, -2, -1),
                ),
                (
                    "Tensor.swapaxes",
                    lambda tensor: tensor.swapaxes(-2, -1),
                ),
                (
                    "torch.swapaxes",
                    lambda tensor: torch.swapaxes(tensor, -2, -1),
                ),
                ("Tensor.mT", lambda tensor: tensor.mT),
                ("Tensor.mH", lambda tensor: tensor.mH),
                (
                    "Tensor.adjoint",
                    lambda tensor: tensor.adjoint(),
                ),
                (
                    "torch.adjoint",
                    lambda tensor: torch.adjoint(tensor),
                ),
                (
                    "Tensor.permute",
                    lambda tensor: tensor.permute(permutation),
                ),
                (
                    "torch.permute",
                    lambda tensor: torch.permute(tensor, permutation),
                ),
                (
                    "Tensor.movedim",
                    lambda tensor: tensor.movedim(-1, -2),
                ),
                (
                    "torch.movedim",
                    lambda tensor: torch.movedim(tensor, -1, -2),
                ),
                (
                    "Tensor.moveaxis",
                    lambda tensor: tensor.moveaxis(-1, -2),
                ),
                (
                    "torch.moveaxis",
                    lambda tensor: torch.moveaxis(tensor, -1, -2),
                ),
            )
            for operation_name, operation in operations:
                with self.subTest(case=case, operation=operation_name):
                    leaf, source, to_leaf_gradient = self.batched_case(kind, shape)
                    self.exercise_backward(leaf, source, operation, to_leaf_gradient)

    def test_rank_three_and_higher_general_permutations_keep_generic_backward_behavior(self):
        cases = (
            ((2, 3, 5), (2, 0, 1)),
            ((2, 3, 5, 7), (2, 0, 3, 1)),
            ((2, 3, 2, 5, 7), (4, 2, 0, 3, 1)),
        )
        for shape, permutation in cases:
            with self.subTest(shape=shape, permutation=permutation):
                leaf_values = np.zeros(shape, dtype=np.float32)
                leaf = self.tensor(leaf_values, requires_grad=True)
                transformed = leaf.permute(permutation)
                weights = self.patterned_values(transformed.shape)
                loss = (transformed * self.tensor(weights)).sum()
                inverse = np.argsort(permutation)
                expected_once = weights.transpose(*inverse).copy()

                loss.backward()
                self.assert_gradient_bits(leaf.grad, expected_once)
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    loss.backward()

                (leaf.permute(permutation) * self.tensor(weights)).sum().backward()
                self.assert_gradient_bits(leaf.grad, expected_once + expected_once)

    def test_extreme_empty_rank_two_transposes_do_not_iterate_zero_elements(self):
        operations = (
            lambda tensor: tensor.t(),
            lambda tensor: tensor.transpose(0, 1),
            lambda tensor: tensor.permute(1, 0),
        )
        for shape in ((sys.maxsize, 0), (0, sys.maxsize)):
            for operation in operations:
                with self.subTest(shape=shape, operation=operation):
                    leaf = torch.zeros(shape, requires_grad=True)
                    loss = operation(leaf).sum()
                    loss.backward()
                    loss.backward()
                    self.assertEqual(leaf.grad.shape, shape)
                    self.assertEqual(leaf.grad.numel(), 0)

    def test_extreme_empty_batched_transposes_do_not_iterate_zero_elements(self):
        operations = (
            lambda tensor: tensor.transpose(-2, -1),
            lambda tensor: tensor.permute(0, 2, 1),
            lambda tensor: tensor.mT,
        )
        for shape in (
            (sys.maxsize, 0, 2),
            (2, 0, sys.maxsize),
            (2, sys.maxsize, 0),
        ):
            for operation in operations:
                with self.subTest(shape=shape, operation=operation):
                    leaf = torch.zeros(shape, requires_grad=True)
                    loss = operation(leaf).sum()
                    loss.backward()
                    loss.backward()
                    self.assertEqual(leaf.grad.shape, shape)
                    self.assertEqual(leaf.grad.numel(), 0)


if __name__ == "__main__":
    unittest.main()
