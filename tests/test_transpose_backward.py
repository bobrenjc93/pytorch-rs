import sys
import unittest

import numpy as np
import torch_rs as torch


class RankTwoTransposeBackwardTests(unittest.TestCase):
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
        expected_once = to_leaf_gradient(weights.T.copy())

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
            ("Tensor.transpose", lambda tensor: tensor.transpose(0, 1)),
            ("torch.transpose", lambda tensor: torch.transpose(tensor, 0, 1)),
            ("Tensor.permute", lambda tensor: tensor.permute(1, 0)),
            ("torch.permute", lambda tensor: torch.permute(tensor, (1, 0))),
        )

        for case, kind, shape in cases:
            for operation_name, operation in operations:
                with self.subTest(case=case, operation=operation_name):
                    leaf, source, to_leaf_gradient = self.rank_two_case(kind, shape)
                    self.exercise_backward(
                        leaf, source, operation, to_leaf_gradient
                    )

    def test_rank_three_permutation_keeps_the_generic_backward_behavior(self):
        leaf_values = np.zeros((2, 3, 5), dtype=np.float32)
        leaf = self.tensor(leaf_values, requires_grad=True)
        transformed = leaf.permute(2, 0, 1)
        weights = self.patterned_values(transformed.shape)
        loss = (transformed * self.tensor(weights)).sum()
        expected_once = weights.transpose(1, 2, 0).copy()

        loss.backward()
        self.assert_gradient_bits(leaf.grad, expected_once)
        with self.assertRaisesRegex(
            RuntimeError, "backward through the graph a second time"
        ):
            loss.backward()

        (leaf.permute(2, 0, 1) * self.tensor(weights)).sum().backward()
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


if __name__ == "__main__":
    unittest.main()
