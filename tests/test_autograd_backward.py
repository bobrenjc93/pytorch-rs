import copy
import importlib
import inspect
import operator
import pickle
import re
import subprocess
import sys
import types
import typing
import unittest
from collections.abc import Sequence

import torch_rs as torch


class CustomSequence(Sequence):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


def wrap_root(root, sequence_type):
    if sequence_type is None:
        return root
    return sequence_type((root,))


def default_grad_tensors(sequence_type):
    if sequence_type is None:
        return None
    return sequence_type((None,))


class AutogradBackwardTests(unittest.TestCase):
    def test_single_root_calls_return_none_and_accumulate_gradients(self):
        calls = (
            lambda loss: torch.autograd.backward(loss),
            lambda loss: torch.autograd.backward(tensors=loss),
            lambda loss: torch.autograd.backward(
                loss,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            ),
            lambda loss: torch.autograd.backward(
                loss, None, False, False, None, None
            ),
            lambda loss: torch.autograd.backward(
                loss, None, operator.index(False), 0
            ),
            lambda loss: torch.autograd.backward(loss, (None,)),
            lambda loss: torch.autograd.backward(
                loss, grad_tensors=[None]
            ),
        )

        for sequence_type in (None, tuple, list):
            for case, call in enumerate(calls):
                with self.subTest(sequence_type=sequence_type, case=case):
                    leaf = torch.tensor([2.0, -3.0], requires_grad=True)
                    loss = (leaf * leaf).sum()
                    roots = wrap_root(loss, sequence_type)
                    self.assertIsNone(call(roots))
                    self.assertEqual(leaf.grad.tolist(), [4.0, -6.0])

    def test_empty_root_calls_are_non_mutating_noops(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "empty tuple grad_tensors",
                lambda roots: torch.autograd.backward(roots, ()),
            ),
            (
                "empty list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[]
                ),
            ),
            (
                "tuple singleton None grad_tensors",
                lambda roots: torch.autograd.backward(roots, (None,)),
            ),
            (
                "list singleton None grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[None]
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                    leaf.sum().backward()
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                    loss = (leaf * leaf).sum()

                    self.assertIsNone(call(root_sequence_type()))
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

                    loss.backward()
                    self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_two_leaf_roots_return_none_and_accumulate_unit_gradients(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "tuple grad_tensors",
                lambda roots: torch.autograd.backward(roots, (None, None)),
            ),
            (
                "list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[None, None]
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    scalar_leaf = torch.tensor(2.0, requires_grad=True)
                    strided_leaf = torch.tensor(
                        [[3.0]], requires_grad=True
                    )
                    self.assertEqual(strided_leaf.shape, (1, 1))
                    self.assertEqual(strided_leaf.stride(), (1, 1))

                    roots = root_sequence_type((scalar_leaf, strided_leaf))
                    self.assertIsNone(call(roots))
                    self.assertEqual(scalar_leaf.grad.item(), 1.0)
                    self.assertEqual(strided_leaf.grad.tolist(), [[1.0]])

    def test_three_leaf_roots_return_none_and_accumulate_unit_gradients(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "tuple grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, (None, None, None)
                ),
            ),
            (
                "list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[None, None, None]
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    scalar_leaf = torch.tensor(2.0, requires_grad=True)
                    vector_leaf = torch.tensor([3.0], requires_grad=True)
                    matrix_leaf = torch.tensor([[4.0]], requires_grad=True)
                    roots = root_sequence_type(
                        (scalar_leaf, vector_leaf, matrix_leaf)
                    )

                    self.assertIsNone(call(roots))
                    self.assertEqual(scalar_leaf.grad.item(), 1.0)
                    self.assertEqual(vector_leaf.grad.tolist(), [1.0])
                    self.assertEqual(matrix_leaf.grad.tolist(), [[1.0]])

    def test_four_leaf_roots_return_none_and_accumulate_unit_gradients(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "tuple grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, (None, None, None, None)
                ),
            ),
            (
                "list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[None, None, None, None]
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    scalar_leaf = torch.tensor(2.0, requires_grad=True)
                    vector_leaf = torch.tensor([3.0], requires_grad=True)
                    matrix_leaf = torch.tensor([[4.0]], requires_grad=True)
                    rank_three_leaf = torch.tensor(
                        [[[5.0]]], requires_grad=True
                    )
                    roots = root_sequence_type(
                        (
                            scalar_leaf,
                            vector_leaf,
                            matrix_leaf,
                            rank_three_leaf,
                        )
                    )

                    self.assertIsNone(call(roots))
                    self.assertEqual(scalar_leaf.grad.item(), 1.0)
                    self.assertEqual(vector_leaf.grad.tolist(), [1.0])
                    self.assertEqual(matrix_leaf.grad.tolist(), [[1.0]])
                    self.assertEqual(rank_three_leaf.grad.tolist(), [[[1.0]]])

    def test_five_leaf_roots_return_none_and_accumulate_unit_gradients(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "tuple grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, (None, None, None, None, None)
                ),
            ),
            (
                "list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, grad_tensors=[None, None, None, None, None]
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    leaves = (
                        torch.tensor(2.0, requires_grad=True),
                        torch.tensor([3.0], requires_grad=True),
                        torch.tensor([[4.0]], requires_grad=True),
                        torch.tensor([[[5.0]]], requires_grad=True),
                        torch.tensor([[[[6.0]]]], requires_grad=True),
                    )
                    roots = root_sequence_type(leaves)

                    self.assertIsNone(call(roots))
                    self.assertEqual(leaves[0].grad.item(), 1.0)
                    self.assertEqual(leaves[1].grad.tolist(), [1.0])
                    self.assertEqual(leaves[2].grad.tolist(), [[1.0]])
                    self.assertEqual(leaves[3].grad.tolist(), [[[1.0]]])
                    self.assertEqual(leaves[4].grad.tolist(), [[[[1.0]]]])

    def test_six_leaf_roots_return_none_and_accumulate_unit_gradients(self):
        calls = (
            ("positional", lambda roots: torch.autograd.backward(roots)),
            (
                "keyword",
                lambda roots: torch.autograd.backward(tensors=roots),
            ),
            (
                "explicit defaults",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                ),
            ),
            (
                "positional defaults",
                lambda roots: torch.autograd.backward(
                    roots, None, False, False, None, None
                ),
            ),
            (
                "integer false",
                lambda roots: torch.autograd.backward(roots, None, 0, 0),
            ),
            (
                "tuple grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots, (None, None, None, None, None, None)
                ),
            ),
            (
                "list grad_tensors",
                lambda roots: torch.autograd.backward(
                    roots,
                    grad_tensors=[None, None, None, None, None, None],
                ),
            ),
        )

        for root_sequence_type in (tuple, list):
            for label, call in calls:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=label
                ):
                    leaves = (
                        torch.tensor(2.0, requires_grad=True),
                        torch.tensor([3.0], requires_grad=True),
                        torch.tensor([[4.0]], requires_grad=True),
                        torch.tensor([[[5.0]]], requires_grad=True),
                        torch.tensor([[[[6.0]]]], requires_grad=True),
                        torch.tensor([[[[[7.0]]]]], requires_grad=True),
                    )
                    roots = root_sequence_type(leaves)

                    self.assertIsNone(call(roots))
                    self.assertEqual(leaves[0].grad.item(), 1.0)
                    self.assertEqual(leaves[1].grad.tolist(), [1.0])
                    self.assertEqual(leaves[2].grad.tolist(), [[1.0]])
                    self.assertEqual(leaves[3].grad.tolist(), [[[1.0]]])
                    self.assertEqual(leaves[4].grad.tolist(), [[[[1.0]]]])
                    self.assertEqual(
                        leaves[5].grad.tolist(), [[[[[1.0]]]]]
                    )

    def test_duplicate_two_leaf_roots_are_repeatable_and_accumulate_twice(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    leaf = torch.tensor([4.0], requires_grad=True)
                    roots = root_sequence_type((leaf, leaf))
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None))
                    )

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertEqual(leaf.grad.tolist(), [2.0])
                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertEqual(leaf.grad.tolist(), [4.0])

    def test_duplicate_roots_aggregate_before_existing_leaf_gradient(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    leaf = torch.tensor([1.0], requires_grad=True)
                    (leaf * 16_777_216.0).backward()
                    existing_gradient = leaf.grad
                    roots = root_sequence_type((leaf, leaf))
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None))
                    )

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertIs(leaf.grad, existing_gradient)
                    self.assertEqual(leaf.grad.tolist(), [16_777_218.0])

    def test_three_roots_aggregate_duplicates_before_existing_gradients(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    duplicate = torch.tensor([1.0], requires_grad=True)
                    distinct = torch.tensor([[2.0]], requires_grad=True)
                    (duplicate * 16_777_216.0).backward()
                    existing_gradient = duplicate.grad
                    roots = root_sequence_type(
                        (duplicate, distinct, duplicate)
                    )
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None, None))
                    )

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertIs(duplicate.grad, existing_gradient)
                    self.assertEqual(
                        duplicate.grad.tolist(), [16_777_218.0]
                    )
                    self.assertEqual(distinct.grad.tolist(), [[1.0]])

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertEqual(
                        duplicate.grad.tolist(), [16_777_220.0]
                    )
                    self.assertEqual(distinct.grad.tolist(), [[2.0]])

    def test_four_roots_aggregate_duplicates_before_existing_gradients(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    duplicate = torch.tensor([1.0], requires_grad=True)
                    distinct = torch.tensor([[2.0]], requires_grad=True)
                    (duplicate * 16_777_216.0).backward()
                    existing_gradient = duplicate.grad
                    roots = root_sequence_type(
                        (duplicate, distinct, duplicate, duplicate)
                    )
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None, None, None))
                    )

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertIs(duplicate.grad, existing_gradient)
                    self.assertEqual(
                        duplicate.grad.tolist(), [16_777_220.0]
                    )
                    self.assertEqual(distinct.grad.tolist(), [[1.0]])

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertEqual(
                        duplicate.grad.tolist(), [16_777_224.0]
                    )
                    self.assertEqual(distinct.grad.tolist(), [[2.0]])

    def test_five_roots_aggregate_duplicates_before_existing_gradients(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    duplicate = torch.tensor([1.0], requires_grad=True)
                    distinct = torch.tensor([[2.0]], requires_grad=True)
                    (duplicate * 16_777_216.0).backward()
                    existing_gradient = duplicate.grad
                    roots = root_sequence_type(
                        (
                            duplicate,
                            distinct,
                            duplicate,
                            duplicate,
                            duplicate,
                        )
                    )
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type(
                            (None, None, None, None, None)
                        )
                    )

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertIs(duplicate.grad, existing_gradient)
                    self.assertEqual(
                        duplicate.grad.tolist(), [16_777_220.0]
                    )
                    self.assertEqual(distinct.grad.tolist(), [[1.0]])

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertEqual(
                        duplicate.grad.tolist(), [16_777_224.0]
                    )
                    self.assertEqual(distinct.grad.tolist(), [[2.0]])

    def test_six_duplicate_roots_aggregate_before_existing_gradient(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    leaf = torch.tensor([1.0], requires_grad=True)
                    (leaf * 16_777_216.0).backward()
                    existing_gradient = leaf.grad
                    roots = root_sequence_type((leaf,) * 6)
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None,) * 6)
                    )

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertIs(leaf.grad, existing_gradient)
                    self.assertEqual(leaf.grad.tolist(), [16_777_222.0])

                    self.assertIsNone(
                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                    )
                    self.assertEqual(leaf.grad.tolist(), [16_777_228.0])

    def test_no_grad_view_second_root_is_rejected_before_first_mutates(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    first = torch.tensor(3.0, requires_grad=True)
                    source = torch.tensor(
                        [[1.0, 2.0]], requires_grad=True
                    )
                    with torch.no_grad():
                        invalid = source.transpose(0, 1)[1]
                    self.assertEqual(invalid.shape, (1,))
                    self.assertEqual(invalid.stride(), (2,))
                    self.assertEqual(invalid.storage_offset(), 1)
                    self.assertTrue(invalid.requires_grad)
                    self.assertTrue(invalid.is_leaf)
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None))
                    )

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "element 1 of tensors does not require grad",
                    ):
                        torch.autograd.backward(
                            root_sequence_type((first, invalid)),
                            grad_tensors=grad_tensors,
                        )
                    self.assertIsNone(first.grad)
                    self.assertIsNone(invalid.grad)
                    self.assertIsNone(source.grad)

                    first.backward()
                    source.sum().backward()
                    self.assertEqual(first.grad.item(), 1.0)
                    self.assertEqual(source.grad.tolist(), [[1.0, 1.0]])

    def test_no_grad_view_third_root_is_rejected_before_any_root_mutates(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    first = torch.tensor(3.0, requires_grad=True)
                    second = torch.tensor([4.0], requires_grad=True)
                    source = torch.tensor(
                        [[1.0, 2.0]], requires_grad=True
                    )
                    with torch.no_grad():
                        invalid = source.transpose(0, 1)[1]
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None, None))
                    )

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "element 2 of tensors does not require grad",
                    ):
                        torch.autograd.backward(
                            root_sequence_type((first, second, invalid)),
                            grad_tensors=grad_tensors,
                        )
                    self.assertIsNone(first.grad)
                    self.assertIsNone(second.grad)
                    self.assertIsNone(invalid.grad)
                    self.assertIsNone(source.grad)

                    first.backward()
                    second.backward()
                    source.sum().backward()
                    self.assertEqual(first.grad.item(), 1.0)
                    self.assertEqual(second.grad.tolist(), [1.0])
                    self.assertEqual(source.grad.tolist(), [[1.0, 1.0]])

    def test_no_grad_view_fourth_root_is_rejected_before_any_root_mutates(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    first = torch.tensor(3.0, requires_grad=True)
                    second = torch.tensor([4.0], requires_grad=True)
                    third = torch.tensor([[5.0]], requires_grad=True)
                    source = torch.tensor(
                        [[1.0, 2.0]], requires_grad=True
                    )
                    with torch.no_grad():
                        invalid = source.transpose(0, 1)[1]
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None, None, None, None))
                    )

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "element 3 of tensors does not require grad",
                    ):
                        torch.autograd.backward(
                            root_sequence_type(
                                (first, second, third, invalid)
                            ),
                            grad_tensors=grad_tensors,
                        )
                    self.assertIsNone(first.grad)
                    self.assertIsNone(second.grad)
                    self.assertIsNone(third.grad)
                    self.assertIsNone(invalid.grad)
                    self.assertIsNone(source.grad)

                    first.backward()
                    second.backward()
                    third.backward()
                    source.sum().backward()
                    self.assertEqual(first.grad.item(), 1.0)
                    self.assertEqual(second.grad.tolist(), [1.0])
                    self.assertEqual(third.grad.tolist(), [[1.0]])
                    self.assertEqual(source.grad.tolist(), [[1.0, 1.0]])

    def test_no_grad_view_fifth_root_is_rejected_before_any_root_mutates(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    valid = [
                        torch.tensor(3.0, requires_grad=True),
                        torch.tensor([4.0], requires_grad=True),
                        torch.tensor([[5.0]], requires_grad=True),
                        torch.tensor([[[6.0]]], requires_grad=True),
                    ]
                    source = torch.tensor(
                        [[1.0, 2.0]], requires_grad=True
                    )
                    with torch.no_grad():
                        invalid = source.transpose(0, 1)[1]
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type(
                            (None, None, None, None, None)
                        )
                    )

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "element 4 of tensors does not require grad",
                    ):
                        torch.autograd.backward(
                            root_sequence_type((*valid, invalid)),
                            grad_tensors=grad_tensors,
                        )
                    self.assertTrue(all(root.grad is None for root in valid))
                    self.assertIsNone(invalid.grad)
                    self.assertIsNone(source.grad)

                    for root in valid:
                        root.backward()
                    source.sum().backward()
                    self.assertEqual(valid[0].grad.item(), 1.0)
                    self.assertEqual(valid[1].grad.tolist(), [1.0])
                    self.assertEqual(valid[2].grad.tolist(), [[1.0]])
                    self.assertEqual(valid[3].grad.tolist(), [[[1.0]]])
                    self.assertEqual(source.grad.tolist(), [[1.0, 1.0]])

    def test_no_grad_view_sixth_root_is_rejected_before_any_root_mutates(self):
        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    valid = [
                        torch.tensor(3.0, requires_grad=True),
                        torch.tensor([4.0], requires_grad=True),
                        torch.tensor([[5.0]], requires_grad=True),
                        torch.tensor([[[6.0]]], requires_grad=True),
                        torch.tensor([[[[7.0]]]], requires_grad=True),
                    ]
                    source = torch.tensor(
                        [[1.0, 2.0]], requires_grad=True
                    )
                    with torch.no_grad():
                        invalid = source.transpose(0, 1)[1]
                    grad_tensors = (
                        None
                        if grad_sequence_type is None
                        else grad_sequence_type((None,) * 6)
                    )

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "element 5 of tensors does not require grad",
                    ):
                        torch.autograd.backward(
                            root_sequence_type((*valid, invalid)),
                            grad_tensors=grad_tensors,
                        )
                    self.assertTrue(all(root.grad is None for root in valid))
                    self.assertIsNone(invalid.grad)
                    self.assertIsNone(source.grad)

                    for root in valid:
                        root.backward()
                    source.sum().backward()
                    self.assertEqual(valid[0].grad.item(), 1.0)
                    self.assertEqual(valid[1].grad.tolist(), [1.0])
                    self.assertEqual(valid[2].grad.tolist(), [[1.0]])
                    self.assertEqual(valid[3].grad.tolist(), [[[1.0]]])
                    self.assertEqual(valid[4].grad.tolist(), [[[[1.0]]]])
                    self.assertEqual(source.grad.tolist(), [[1.0, 1.0]])

    def test_graph_reuse_freeing_and_accumulation_follow_tensor_backward(self):
        for sequence_type in (None, tuple, list):
            with self.subTest(sequence_type=sequence_type):
                reusable_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
                reusable_loss = reusable_leaf.transpose(0, 0).sum()
                torch.autograd.backward(wrap_root(reusable_loss, sequence_type))
                torch.autograd.backward(wrap_root(reusable_loss, sequence_type))
                self.assertEqual(reusable_leaf.grad.tolist(), [2.0, 2.0])

                scalar_leaf = torch.tensor(7.0, requires_grad=True)
                torch.autograd.backward(wrap_root(scalar_leaf, sequence_type))
                torch.autograd.backward(wrap_root(scalar_leaf, sequence_type))
                self.assertEqual(scalar_leaf.grad.item(), 2.0)

                freed_leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                freed_loss = (freed_leaf * freed_leaf).sum()
                torch.autograd.backward(wrap_root(freed_loss, sequence_type))
                self.assertEqual(freed_leaf.grad.tolist(), [4.0, 6.0])
                with self.assertRaisesRegex(
                    RuntimeError, "backward through the graph a second time"
                ):
                    torch.autograd.backward(wrap_root(freed_loss, sequence_type))

                torch.autograd.backward(
                    wrap_root((freed_leaf * freed_leaf).sum(), sequence_type)
                )
                self.assertEqual(freed_leaf.grad.tolist(), [8.0, 12.0])

    def test_singleton_none_grad_tensors_preserve_backward_semantics(self):
        for root_sequence_type in (None, tuple, list):
            for grad_sequence_type in (tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    grad_tensors = default_grad_tensors(grad_sequence_type)

                    reusable_leaf = torch.tensor(
                        [1.0, 2.0], requires_grad=True
                    )
                    reusable_loss = reusable_leaf.transpose(0, 0).sum()
                    torch.autograd.backward(
                        wrap_root(reusable_loss, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    torch.autograd.backward(
                        wrap_root(reusable_loss, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(
                        reusable_leaf.grad.tolist(), [2.0, 2.0]
                    )

                    scalar_leaf = torch.tensor(7.0, requires_grad=True)
                    torch.autograd.backward(
                        wrap_root(scalar_leaf, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    torch.autograd.backward(
                        wrap_root(scalar_leaf, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(scalar_leaf.grad.item(), 2.0)

                    freed_leaf = torch.tensor(
                        [2.0, 3.0], requires_grad=True
                    )
                    freed_leaf.sum().backward()
                    freed_loss = (freed_leaf * freed_leaf).sum()
                    torch.autograd.backward(
                        wrap_root(freed_loss, root_sequence_type),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(freed_leaf.grad.tolist(), [5.0, 7.0])
                    with self.assertRaisesRegex(
                        RuntimeError, "backward through the graph a second time"
                    ):
                        torch.autograd.backward(
                            wrap_root(freed_loss, root_sequence_type),
                            grad_tensors=grad_tensors,
                        )
                    torch.autograd.backward(
                        wrap_root(
                            (freed_leaf * freed_leaf).sum(),
                            root_sequence_type,
                        ),
                        grad_tensors=grad_tensors,
                    )
                    self.assertEqual(
                        freed_leaf.grad.tolist(), [9.0, 13.0]
                    )

                    nonscalar_leaf = torch.tensor(
                        [2.0, 3.0], requires_grad=True
                    )
                    nonscalar = nonscalar_leaf * nonscalar_leaf
                    with self.assertRaisesRegex(
                        RuntimeError, "implicitly created only for scalar"
                    ):
                        torch.autograd.backward(
                            wrap_root(nonscalar, root_sequence_type),
                            grad_tensors=grad_tensors,
                        )
                    self.assertIsNone(nonscalar_leaf.grad)
                    nonscalar.sum().backward()
                    self.assertEqual(
                        nonscalar_leaf.grad.tolist(), [4.0, 6.0]
                    )

    def test_tensor_backward_errors_are_preserved(self):
        for sequence_type in (None, tuple, list):
            with self.subTest(sequence_type=sequence_type):
                with self.assertRaisesRegex(RuntimeError, "does not require grad"):
                    torch.autograd.backward(
                        wrap_root(torch.tensor(1.0), sequence_type)
                    )
                with self.assertRaisesRegex(
                    RuntimeError, "implicitly created only for scalar"
                ):
                    torch.autograd.backward(
                        wrap_root(
                            torch.tensor([1.0, 2.0], requires_grad=True),
                            sequence_type,
                        )
                    )

    def test_unsupported_forms_fail_before_gradients_or_graph_state_change(self):
        root_error = (
            "torch_rs.autograd.backward only supports an exact native Tensor, "
            "directly or in an exact tuple or list containing at most six "
            "exact native Tensors"
        )
        two_root_error = (
            "torch_rs.autograd.backward only supports two roots when both "
            "are one-element native leaf Tensors requiring gradients"
        )
        unsupported = (
            (
                "non-leaf tuple roots",
                NotImplementedError,
                two_root_error,
                lambda leaf, loss: torch.autograd.backward((loss, loss)),
            ),
            (
                "non-leaf list roots",
                NotImplementedError,
                two_root_error,
                lambda leaf, loss: torch.autograd.backward([loss, loss]),
            ),
            (
                "seven tuple roots",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    (loss, loss, loss, loss, loss, loss, loss)
                ),
            ),
            (
                "seven list roots",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    [loss, loss, loss, loss, loss, loss, loss]
                ),
            ),
            (
                "custom sequence",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    CustomSequence((loss,))
                ),
            ),
            (
                "six-root custom sequence",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    CustomSequence((loss, loss, loss, loss, loss, loss))
                ),
            ),
            (
                "empty custom sequence",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    CustomSequence(())
                ),
            ),
            (
                "tuple subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    TupleSubclass((loss,))
                ),
            ),
            (
                "empty tuple subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(TupleSubclass()),
            ),
            (
                "list subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(
                    ListSubclass([loss])
                ),
            ),
            (
                "empty list subclass",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward(ListSubclass()),
            ),
            (
                "non-tensor singleton",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward((object(),)),
            ),
            (
                "non-tensor pair",
                TypeError,
                root_error,
                lambda leaf, loss: torch.autograd.backward((loss, object())),
            ),
            (
                "retained graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support retain_graph=True",
                lambda leaf, loss: torch.autograd.backward(
                    loss, retain_graph=True
                ),
            ),
            (
                "higher-order graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support create_graph=True",
                lambda leaf, loss: torch.autograd.backward(
                    loss, create_graph=True
                ),
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf, loss: torch.autograd.backward(
                    loss, grad_variables=torch.tensor(1.0)
                ),
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf, loss: torch.autograd.backward(loss, inputs=leaf),
            ),
        )

        for label, error_type, message, call in unsupported:
            with self.subTest(label=label):
                leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                leaf.sum().backward()
                self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                loss = (leaf * leaf).sum()
                with self.assertRaisesRegex(
                    error_type, f"^{re.escape(message)}$"
                ):
                    call(leaf, loss)
                self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])

                loss.backward()
                self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_two_root_eligibility_is_prevalidated_for_both_roots(self):
        message = (
            "torch_rs.autograd.backward only supports two roots when both "
            "are one-element native leaf Tensors requiring gradients"
        )

        for root_sequence_type in (tuple, list):
            for invalid_position in (0, 1):
                for invalid_kind in (
                    "non-leaf",
                    "does not require grad",
                    "multiple elements",
                ):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        invalid_position=invalid_position,
                        invalid_kind=invalid_kind,
                    ):
                        valid = torch.tensor([2.0], requires_grad=True)
                        source = None
                        if invalid_kind == "non-leaf":
                            source = torch.tensor(3.0, requires_grad=True)
                            invalid = source * source
                        elif invalid_kind == "does not require grad":
                            invalid = torch.tensor(3.0)
                        else:
                            invalid = torch.tensor(
                                [3.0, 4.0], requires_grad=True
                            )

                        roots = [valid, invalid]
                        if invalid_position == 0:
                            roots.reverse()
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots)
                            )

                        self.assertIsNone(valid.grad)
                        if invalid_kind == "non-leaf":
                            self.assertIsNone(source.grad)
                            invalid.backward()
                            self.assertEqual(source.grad.item(), 6.0)
                        elif invalid_kind == "multiple elements":
                            self.assertIsNone(invalid.grad)
                            invalid.sum().backward()
                            self.assertEqual(invalid.grad.tolist(), [1.0, 1.0])
                        valid.backward()
                        self.assertEqual(valid.grad.tolist(), [1.0])

    def test_three_root_eligibility_is_prevalidated_for_every_root(self):
        message = (
            "torch_rs.autograd.backward only supports three roots when all "
            "three are one-element native leaf Tensors requiring gradients"
        )

        for root_sequence_type in (tuple, list):
            for invalid_position in range(3):
                for invalid_kind in (
                    "non-leaf",
                    "does not require grad",
                    "multiple elements",
                ):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        invalid_position=invalid_position,
                        invalid_kind=invalid_kind,
                    ):
                        valid = [
                            torch.tensor([2.0], requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                        ]
                        source = None
                        if invalid_kind == "non-leaf":
                            source = torch.tensor(4.0, requires_grad=True)
                            invalid = source * source
                        elif invalid_kind == "does not require grad":
                            invalid = torch.tensor(4.0)
                        else:
                            invalid = torch.tensor(
                                [4.0, 5.0], requires_grad=True
                            )

                        roots = valid.copy()
                        roots.insert(invalid_position, invalid)
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots)
                            )

                        self.assertTrue(
                            all(root.grad is None for root in valid)
                        )
                        if invalid_kind == "non-leaf":
                            self.assertIsNone(source.grad)
                            invalid.backward()
                            self.assertEqual(source.grad.item(), 8.0)
                        elif invalid_kind == "multiple elements":
                            self.assertIsNone(invalid.grad)
                            invalid.sum().backward()
                            self.assertEqual(
                                invalid.grad.tolist(), [1.0, 1.0]
                            )
                        for root in valid:
                            root.backward()
                            self.assertEqual(root.grad.tolist(), [1.0])

    def test_four_root_eligibility_is_prevalidated_for_every_root(self):
        message = (
            "torch_rs.autograd.backward only supports four roots when all "
            "four are one-element native leaf Tensors requiring gradients"
        )

        for root_sequence_type in (tuple, list):
            for invalid_position in range(4):
                for invalid_kind in (
                    "non-leaf",
                    "does not require grad",
                    "multiple elements",
                ):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        invalid_position=invalid_position,
                        invalid_kind=invalid_kind,
                    ):
                        valid = [
                            torch.tensor([2.0], requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                            torch.tensor([4.0], requires_grad=True),
                        ]
                        source = None
                        if invalid_kind == "non-leaf":
                            source = torch.tensor(5.0, requires_grad=True)
                            invalid = source * source
                        elif invalid_kind == "does not require grad":
                            invalid = torch.tensor(5.0)
                        else:
                            invalid = torch.tensor(
                                [5.0, 6.0], requires_grad=True
                            )

                        roots = valid.copy()
                        roots.insert(invalid_position, invalid)
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots)
                            )

                        self.assertTrue(
                            all(root.grad is None for root in valid)
                        )
                        if invalid_kind == "non-leaf":
                            self.assertIsNone(source.grad)
                            invalid.backward()
                            self.assertEqual(source.grad.item(), 10.0)
                        elif invalid_kind == "multiple elements":
                            self.assertIsNone(invalid.grad)
                            invalid.sum().backward()
                            self.assertEqual(
                                invalid.grad.tolist(), [1.0, 1.0]
                            )
                        for root in valid:
                            root.backward()
                            self.assertEqual(root.grad.tolist(), [1.0])

    def test_five_root_eligibility_is_prevalidated_for_every_root(self):
        message = (
            "torch_rs.autograd.backward only supports five roots when all "
            "five are one-element native leaf Tensors requiring gradients"
        )

        for root_sequence_type in (tuple, list):
            for invalid_position in range(5):
                for invalid_kind in (
                    "non-leaf",
                    "does not require grad",
                    "multiple elements",
                ):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        invalid_position=invalid_position,
                        invalid_kind=invalid_kind,
                    ):
                        valid = [
                            torch.tensor([2.0], requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                            torch.tensor([4.0], requires_grad=True),
                            torch.tensor([5.0], requires_grad=True),
                        ]
                        source = None
                        if invalid_kind == "non-leaf":
                            source = torch.tensor(6.0, requires_grad=True)
                            invalid = source * source
                        elif invalid_kind == "does not require grad":
                            invalid = torch.tensor(6.0)
                        else:
                            invalid = torch.tensor(
                                [6.0, 7.0], requires_grad=True
                            )

                        roots = valid.copy()
                        roots.insert(invalid_position, invalid)
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots)
                            )

                        self.assertTrue(
                            all(root.grad is None for root in valid)
                        )
                        if invalid_kind == "non-leaf":
                            self.assertIsNone(source.grad)
                            invalid.backward()
                            self.assertEqual(source.grad.item(), 12.0)
                        elif invalid_kind == "multiple elements":
                            self.assertIsNone(invalid.grad)
                            invalid.sum().backward()
                            self.assertEqual(
                                invalid.grad.tolist(), [1.0, 1.0]
                            )
                        for root in valid:
                            root.backward()
                            self.assertEqual(root.grad.tolist(), [1.0])

    def test_six_root_eligibility_is_prevalidated_for_every_root(self):
        message = (
            "torch_rs.autograd.backward only supports six roots when all "
            "six are one-element native leaf Tensors requiring gradients"
        )

        for root_sequence_type in (tuple, list):
            for invalid_position in range(6):
                for invalid_kind in (
                    "non-leaf",
                    "does not require grad",
                    "multiple elements",
                ):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        invalid_position=invalid_position,
                        invalid_kind=invalid_kind,
                    ):
                        valid = [
                            torch.tensor([2.0], requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                            torch.tensor([4.0], requires_grad=True),
                            torch.tensor([5.0], requires_grad=True),
                            torch.tensor([6.0], requires_grad=True),
                        ]
                        source = None
                        if invalid_kind == "non-leaf":
                            source = torch.tensor(7.0, requires_grad=True)
                            invalid = source * source
                        elif invalid_kind == "does not require grad":
                            invalid = torch.tensor(7.0)
                        else:
                            invalid = torch.tensor(
                                [7.0, 8.0], requires_grad=True
                            )

                        roots = valid.copy()
                        roots.insert(invalid_position, invalid)
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots)
                            )

                        self.assertTrue(
                            all(root.grad is None for root in valid)
                        )
                        if invalid_kind == "non-leaf":
                            self.assertIsNone(source.grad)
                            invalid.backward()
                            self.assertEqual(source.grad.item(), 14.0)
                        elif invalid_kind == "multiple elements":
                            self.assertIsNone(invalid.grad)
                            invalid.sum().backward()
                            self.assertEqual(
                                invalid.grad.tolist(), [1.0, 1.0]
                            )
                        for root in valid:
                            root.backward()
                            self.assertEqual(root.grad.tolist(), [1.0])

    def test_non_default_grad_tensors_forms_are_rejected_before_backward(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("tuple with tensor", lambda: (torch.tensor(1.0),)),
            ("list with tensor", lambda: [torch.tensor(1.0)]),
            ("empty tuple", tuple),
            ("empty list", list),
            ("multiple tuple", lambda: (None, None)),
            ("multiple list", lambda: [None, None]),
            ("custom sequence", lambda: CustomSequence((None,))),
            ("tuple subclass", lambda: TupleSubclass((None,))),
            ("list subclass", lambda: ListSubclass([None])),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for sequence_type in (None, tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    sequence_type=sequence_type, grad_tensors=label
                ):
                    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                    leaf.sum().backward()
                    loss = (leaf * leaf).sum()
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            wrap_root(loss, sequence_type),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                    loss.backward()
                    self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_two_leaf_roots_reject_nonmatching_or_concrete_gradients(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("empty tuple", tuple),
            ("empty list", list),
            ("singleton tuple", lambda: (None,)),
            ("singleton list", lambda: [None]),
            ("first concrete", lambda: (torch.tensor(1.0), None)),
            ("second concrete", lambda: [None, torch.tensor(1.0)]),
            ("three tuple", lambda: (None, None, None)),
            ("three list", lambda: [None, None, None]),
            ("custom sequence", lambda: CustomSequence((None, None))),
            ("tuple subclass", lambda: TupleSubclass((None, None))),
            ("list subclass", lambda: ListSubclass([None, None])),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    first = torch.tensor(2.0, requires_grad=True)
                    second = torch.tensor([3.0], requires_grad=True)
                    roots = root_sequence_type((first, second))
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            roots, grad_tensors=make_grad_tensors()
                        )
                    self.assertIsNone(first.grad)
                    self.assertIsNone(second.grad)

                    torch.autograd.backward(roots, (None, None))
                    self.assertEqual(first.grad.item(), 1.0)
                    self.assertEqual(second.grad.tolist(), [1.0])

    def test_three_leaf_roots_reject_nonmatching_or_concrete_gradients(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("empty tuple", tuple),
            ("empty list", list),
            ("singleton tuple", lambda: (None,)),
            ("pair list", lambda: [None, None]),
            ("first concrete", lambda: (torch.tensor(1.0), None, None)),
            ("second concrete", lambda: [None, torch.tensor(1.0), None]),
            ("third concrete", lambda: (None, None, torch.tensor(1.0))),
            ("four tuple", lambda: (None, None, None, None)),
            ("four list", lambda: [None, None, None, None]),
            (
                "custom sequence",
                lambda: CustomSequence((None, None, None)),
            ),
            (
                "tuple subclass",
                lambda: TupleSubclass((None, None, None)),
            ),
            (
                "list subclass",
                lambda: ListSubclass([None, None, None]),
            ),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    roots = [
                        torch.tensor(2.0, requires_grad=True),
                        torch.tensor([3.0], requires_grad=True),
                        torch.tensor([[4.0]], requires_grad=True),
                    ]
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertTrue(
                        all(root.grad is None for root in roots)
                    )

                    torch.autograd.backward(
                        root_sequence_type(roots),
                        (None, None, None),
                    )
                    self.assertEqual(roots[0].grad.item(), 1.0)
                    self.assertEqual(roots[1].grad.tolist(), [1.0])
                    self.assertEqual(roots[2].grad.tolist(), [[1.0]])

    def test_four_leaf_roots_reject_nonmatching_or_concrete_gradients(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("empty tuple", tuple),
            ("empty list", list),
            ("singleton tuple", lambda: (None,)),
            ("pair list", lambda: [None, None]),
            ("triple tuple", lambda: (None, None, None)),
            (
                "first concrete",
                lambda: (torch.tensor(1.0), None, None, None),
            ),
            (
                "second concrete",
                lambda: [None, torch.tensor(1.0), None, None],
            ),
            (
                "third concrete",
                lambda: (None, None, torch.tensor(1.0), None),
            ),
            (
                "fourth concrete",
                lambda: [None, None, None, torch.tensor(1.0)],
            ),
            ("five tuple", lambda: (None, None, None, None, None)),
            ("five list", lambda: [None, None, None, None, None]),
            (
                "custom sequence",
                lambda: CustomSequence((None, None, None, None)),
            ),
            (
                "tuple subclass",
                lambda: TupleSubclass((None, None, None, None)),
            ),
            (
                "list subclass",
                lambda: ListSubclass([None, None, None, None]),
            ),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    roots = [
                        torch.tensor(2.0, requires_grad=True),
                        torch.tensor([3.0], requires_grad=True),
                        torch.tensor([[4.0]], requires_grad=True),
                        torch.tensor([[[5.0]]], requires_grad=True),
                    ]
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertTrue(
                        all(root.grad is None for root in roots)
                    )

                    torch.autograd.backward(
                        root_sequence_type(roots),
                        (None, None, None, None),
                    )
                    self.assertEqual(roots[0].grad.item(), 1.0)
                    self.assertEqual(roots[1].grad.tolist(), [1.0])
                    self.assertEqual(roots[2].grad.tolist(), [[1.0]])
                    self.assertEqual(roots[3].grad.tolist(), [[[1.0]]])

    def test_five_leaf_roots_reject_nonmatching_or_concrete_gradients(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("empty tuple", tuple),
            ("empty list", list),
            ("singleton tuple", lambda: (None,)),
            ("pair list", lambda: [None, None]),
            ("triple tuple", lambda: (None, None, None)),
            ("four list", lambda: [None, None, None, None]),
            (
                "first concrete",
                lambda: (torch.tensor(1.0), None, None, None, None),
            ),
            (
                "second concrete",
                lambda: [None, torch.tensor(1.0), None, None, None],
            ),
            (
                "third concrete",
                lambda: (None, None, torch.tensor(1.0), None, None),
            ),
            (
                "fourth concrete",
                lambda: [None, None, None, torch.tensor(1.0), None],
            ),
            (
                "fifth concrete",
                lambda: (None, None, None, None, torch.tensor(1.0)),
            ),
            ("six tuple", lambda: (None, None, None, None, None, None)),
            ("six list", lambda: [None, None, None, None, None, None]),
            (
                "custom sequence",
                lambda: CustomSequence((None, None, None, None, None)),
            ),
            (
                "tuple subclass",
                lambda: TupleSubclass((None, None, None, None, None)),
            ),
            (
                "list subclass",
                lambda: ListSubclass([None, None, None, None, None]),
            ),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    roots = [
                        torch.tensor(2.0, requires_grad=True),
                        torch.tensor([3.0], requires_grad=True),
                        torch.tensor([[4.0]], requires_grad=True),
                        torch.tensor([[[5.0]]], requires_grad=True),
                        torch.tensor([[[[6.0]]]], requires_grad=True),
                    ]
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertTrue(
                        all(root.grad is None for root in roots)
                    )

                    torch.autograd.backward(
                        root_sequence_type(roots),
                        (None, None, None, None, None),
                    )
                    self.assertEqual(roots[0].grad.item(), 1.0)
                    self.assertEqual(roots[1].grad.tolist(), [1.0])
                    self.assertEqual(roots[2].grad.tolist(), [[1.0]])
                    self.assertEqual(roots[3].grad.tolist(), [[[1.0]]])
                    self.assertEqual(roots[4].grad.tolist(), [[[[1.0]]]])

    def test_six_leaf_roots_reject_nonmatching_or_concrete_gradients(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("empty tuple", tuple),
            ("empty list", list),
            ("singleton tuple", lambda: (None,)),
            ("pair list", lambda: [None, None]),
            ("triple tuple", lambda: (None, None, None)),
            ("four list", lambda: [None, None, None, None]),
            ("five tuple", lambda: (None, None, None, None, None)),
            (
                "first concrete",
                lambda: (
                    torch.tensor(1.0),
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ),
            (
                "second concrete",
                lambda: [
                    None,
                    torch.tensor(1.0),
                    None,
                    None,
                    None,
                    None,
                ],
            ),
            (
                "third concrete",
                lambda: (
                    None,
                    None,
                    torch.tensor(1.0),
                    None,
                    None,
                    None,
                ),
            ),
            (
                "fourth concrete",
                lambda: [
                    None,
                    None,
                    None,
                    torch.tensor(1.0),
                    None,
                    None,
                ],
            ),
            (
                "fifth concrete",
                lambda: (
                    None,
                    None,
                    None,
                    None,
                    torch.tensor(1.0),
                    None,
                ),
            ),
            (
                "sixth concrete",
                lambda: [
                    None,
                    None,
                    None,
                    None,
                    None,
                    torch.tensor(1.0),
                ],
            ),
            ("seven tuple", lambda: (None,) * 7),
            ("seven list", lambda: [None] * 7),
            (
                "custom sequence",
                lambda: CustomSequence((None,) * 6),
            ),
            (
                "tuple subclass",
                lambda: TupleSubclass((None,) * 6),
            ),
            (
                "list subclass",
                lambda: ListSubclass([None] * 6),
            ),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    roots = [
                        torch.tensor(2.0, requires_grad=True),
                        torch.tensor([3.0], requires_grad=True),
                        torch.tensor([[4.0]], requires_grad=True),
                        torch.tensor([[[5.0]]], requires_grad=True),
                        torch.tensor([[[[6.0]]]], requires_grad=True),
                        torch.tensor([[[[[7.0]]]]], requires_grad=True),
                    ]
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertTrue(
                        all(root.grad is None for root in roots)
                    )

                    torch.autograd.backward(
                        root_sequence_type(roots),
                        (None, None, None, None, None, None),
                    )
                    self.assertEqual(roots[0].grad.item(), 1.0)
                    self.assertEqual(roots[1].grad.tolist(), [1.0])
                    self.assertEqual(roots[2].grad.tolist(), [[1.0]])
                    self.assertEqual(roots[3].grad.tolist(), [[[1.0]]])
                    self.assertEqual(roots[4].grad.tolist(), [[[[1.0]]]])
                    self.assertEqual(
                        roots[5].grad.tolist(), [[[[[1.0]]]]]
                    )

    def test_empty_roots_reject_non_default_grad_tensors(self):
        grad_tensors = (
            ("tensor", lambda: torch.tensor(1.0)),
            ("tuple with tensor", lambda: (torch.tensor(1.0),)),
            ("list with tensor", lambda: [torch.tensor(1.0)]),
            ("multiple tuple", lambda: (None, None)),
            ("multiple list", lambda: [None, None]),
            ("custom empty sequence", lambda: CustomSequence(())),
            ("custom singleton None", lambda: CustomSequence((None,))),
            ("empty tuple subclass", TupleSubclass),
            ("empty list subclass", ListSubclass),
        )
        message = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )

        for root_sequence_type in (tuple, list):
            for label, make_grad_tensors in grad_tensors:
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_tensors=label,
                ):
                    leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                    leaf.sum().backward()
                    loss = (leaf * leaf).sum()
                    with self.assertRaisesRegex(
                        NotImplementedError, f"^{re.escape(message)}$"
                    ):
                        torch.autograd.backward(
                            root_sequence_type(),
                            grad_tensors=make_grad_tensors(),
                        )
                    self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                    loss.backward()
                    self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_graph_option_conversion_errors_are_non_mutating(self):
        cases = (
            ("retain_graph", 0.5),
            ("create_graph", None),
        )
        for grad_sequence_type in (None, tuple, list):
            for name, value in cases:
                with self.subTest(
                    grad_sequence_type=grad_sequence_type,
                    name=name,
                    value=value,
                ):
                    leaf = torch.tensor(2.0, requires_grad=True)
                    loss = leaf * leaf
                    with self.assertRaises(TypeError) as raised:
                        torch.autograd.backward(
                            loss,
                            grad_tensors=default_grad_tensors(
                                grad_sequence_type
                            ),
                            **{name: value},
                        )
                    self.assertEqual(
                        str(raised.exception),
                        f"'{type(value).__name__}' object cannot be "
                        "interpreted as an integer",
                    )
                    self.assertIsNone(leaf.grad)
                    loss.backward()
                    self.assertEqual(leaf.grad.item(), 4.0)

    def test_two_leaf_roots_validate_all_options_before_backward(self):
        cases = (
            (
                "retain_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda first: {"retain_graph": True},
            ),
            (
                "create_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda first: {"create_graph": True},
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda first: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda first: {"inputs": first},
            ),
            (
                "retain_graph conversion",
                TypeError,
                "'float' object cannot be interpreted as an integer",
                lambda first: {"retain_graph": 0.5},
            ),
            (
                "create_graph conversion",
                TypeError,
                "'NoneType' object cannot be interpreted as an integer",
                lambda first: {"create_graph": None},
            ),
        )

        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (tuple, list):
                for label, error_type, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        option=label,
                    ):
                        first = torch.tensor(2.0, requires_grad=True)
                        second = torch.tensor([3.0], requires_grad=True)
                        roots = root_sequence_type((first, second))
                        grad_tensors = grad_sequence_type((None, None))
                        with self.assertRaisesRegex(
                            error_type, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                roots,
                                grad_tensors=grad_tensors,
                                **make_options(first),
                            )
                        self.assertIsNone(first.grad)
                        self.assertIsNone(second.grad)

                        torch.autograd.backward(
                            roots, grad_tensors=grad_tensors
                        )
                        self.assertEqual(first.grad.item(), 1.0)
                        self.assertEqual(second.grad.tolist(), [1.0])

    def test_four_leaf_roots_validate_all_options_before_backward(self):
        cases = (
            (
                "retain_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda first: {"retain_graph": True},
            ),
            (
                "create_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda first: {"create_graph": True},
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda first: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda first: {"inputs": first},
            ),
            (
                "retain_graph conversion",
                TypeError,
                "'float' object cannot be interpreted as an integer",
                lambda first: {"retain_graph": 0.5},
            ),
            (
                "create_graph conversion",
                TypeError,
                "'NoneType' object cannot be interpreted as an integer",
                lambda first: {"create_graph": None},
            ),
        )

        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (tuple, list):
                for label, error_type, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        option=label,
                    ):
                        roots = [
                            torch.tensor(2.0, requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                            torch.tensor([[4.0]], requires_grad=True),
                            torch.tensor([[[5.0]]], requires_grad=True),
                        ]
                        grad_tensors = grad_sequence_type(
                            (None, None, None, None)
                        )
                        with self.assertRaisesRegex(
                            error_type, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots),
                                grad_tensors=grad_tensors,
                                **make_options(roots[0]),
                            )
                        self.assertTrue(
                            all(root.grad is None for root in roots)
                        )

                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=grad_tensors,
                        )
                        self.assertEqual(roots[0].grad.item(), 1.0)
                        self.assertEqual(roots[1].grad.tolist(), [1.0])
                        self.assertEqual(roots[2].grad.tolist(), [[1.0]])
                        self.assertEqual(roots[3].grad.tolist(), [[[1.0]]])

    def test_five_leaf_roots_validate_all_options_before_backward(self):
        cases = (
            (
                "retain_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda first: {"retain_graph": True},
            ),
            (
                "create_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda first: {"create_graph": True},
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda first: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda first: {"inputs": first},
            ),
            (
                "retain_graph conversion",
                TypeError,
                "'float' object cannot be interpreted as an integer",
                lambda first: {"retain_graph": 0.5},
            ),
            (
                "create_graph conversion",
                TypeError,
                "'NoneType' object cannot be interpreted as an integer",
                lambda first: {"create_graph": None},
            ),
        )

        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (tuple, list):
                for label, error_type, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        option=label,
                    ):
                        roots = [
                            torch.tensor(2.0, requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                            torch.tensor([[4.0]], requires_grad=True),
                            torch.tensor([[[5.0]]], requires_grad=True),
                            torch.tensor([[[[6.0]]]], requires_grad=True),
                        ]
                        grad_tensors = grad_sequence_type(
                            (None, None, None, None, None)
                        )
                        with self.assertRaisesRegex(
                            error_type, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots),
                                grad_tensors=grad_tensors,
                                **make_options(roots[0]),
                            )
                        self.assertTrue(
                            all(root.grad is None for root in roots)
                        )

                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=grad_tensors,
                        )
                        self.assertEqual(roots[0].grad.item(), 1.0)
                        self.assertEqual(roots[1].grad.tolist(), [1.0])
                        self.assertEqual(roots[2].grad.tolist(), [[1.0]])
                        self.assertEqual(roots[3].grad.tolist(), [[[1.0]]])
                        self.assertEqual(roots[4].grad.tolist(), [[[[1.0]]]])

    def test_six_leaf_roots_validate_all_options_before_backward(self):
        cases = (
            (
                "retain_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda first: {"retain_graph": True},
            ),
            (
                "create_graph",
                NotImplementedError,
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda first: {"create_graph": True},
            ),
            (
                "grad_variables",
                NotImplementedError,
                "torch_rs.autograd.backward does not support grad_variables",
                lambda first: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                NotImplementedError,
                "torch_rs.autograd.backward does not support inputs",
                lambda first: {"inputs": first},
            ),
            (
                "retain_graph conversion",
                TypeError,
                "'float' object cannot be interpreted as an integer",
                lambda first: {"retain_graph": 0.5},
            ),
            (
                "create_graph conversion",
                TypeError,
                "'NoneType' object cannot be interpreted as an integer",
                lambda first: {"create_graph": None},
            ),
        )

        for root_sequence_type in (tuple, list):
            for grad_sequence_type in (tuple, list):
                for label, error_type, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        option=label,
                    ):
                        roots = [
                            torch.tensor(2.0, requires_grad=True),
                            torch.tensor([3.0], requires_grad=True),
                            torch.tensor([[4.0]], requires_grad=True),
                            torch.tensor([[[5.0]]], requires_grad=True),
                            torch.tensor([[[[6.0]]]], requires_grad=True),
                            torch.tensor([[[[[7.0]]]]], requires_grad=True),
                        ]
                        grad_tensors = grad_sequence_type((None,) * 6)
                        with self.assertRaisesRegex(
                            error_type, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(roots),
                                grad_tensors=grad_tensors,
                                **make_options(roots[0]),
                            )
                        self.assertTrue(
                            all(root.grad is None for root in roots)
                        )

                        torch.autograd.backward(
                            root_sequence_type(roots),
                            grad_tensors=grad_tensors,
                        )
                        self.assertEqual(roots[0].grad.item(), 1.0)
                        self.assertEqual(roots[1].grad.tolist(), [1.0])
                        self.assertEqual(roots[2].grad.tolist(), [[1.0]])
                        self.assertEqual(roots[3].grad.tolist(), [[[1.0]]])
                        self.assertEqual(roots[4].grad.tolist(), [[[[1.0]]]])
                        self.assertEqual(
                            roots[5].grad.tolist(), [[[[[1.0]]]]]
                        )

    def test_singleton_none_grad_tensors_reach_later_option_validation(self):
        cases = (
            (
                "retain_graph",
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda leaf: {"retain_graph": True},
            ),
            (
                "create_graph",
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda leaf: {"create_graph": True},
            ),
            (
                "grad_variables",
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf: {"inputs": leaf},
            ),
        )
        for root_sequence_type in (None, tuple, list):
            for grad_sequence_type in (tuple, list):
                for label, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        option=label,
                    ):
                        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                        leaf.sum().backward()
                        loss = (leaf * leaf).sum()
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                wrap_root(loss, root_sequence_type),
                                grad_tensors=default_grad_tensors(
                                    grad_sequence_type
                                ),
                                **make_options(leaf),
                            )
                        self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                        loss.backward()
                        self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_empty_roots_reject_non_default_options(self):
        cases = (
            (
                "retain_graph",
                "torch_rs.autograd.backward does not support "
                "retain_graph=True",
                lambda leaf: {"retain_graph": True},
            ),
            (
                "create_graph",
                "torch_rs.autograd.backward does not support "
                "create_graph=True",
                lambda leaf: {"create_graph": True},
            ),
            (
                "grad_variables",
                "torch_rs.autograd.backward does not support grad_variables",
                lambda leaf: {"grad_variables": torch.tensor(1.0)},
            ),
            (
                "inputs",
                "torch_rs.autograd.backward does not support inputs",
                lambda leaf: {"inputs": leaf},
            ),
        )
        supported_grad_tensors = (None, (), [], (None,), [None])

        for root_sequence_type in (tuple, list):
            for grad_tensors in supported_grad_tensors:
                for label, message, make_options in cases:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_tensors=grad_tensors,
                        option=label,
                    ):
                        leaf = torch.tensor([2.0, 3.0], requires_grad=True)
                        leaf.sum().backward()
                        loss = (leaf * leaf).sum()
                        with self.assertRaisesRegex(
                            NotImplementedError, f"^{re.escape(message)}$"
                        ):
                            torch.autograd.backward(
                                root_sequence_type(),
                                grad_tensors=grad_tensors,
                                **make_options(leaf),
                            )
                        self.assertEqual(leaf.grad.tolist(), [1.0, 1.0])
                        loss.backward()
                        self.assertEqual(leaf.grad.tolist(), [5.0, 7.0])

    def test_root_and_gradient_validation_precede_graph_options(self):
        root_error = (
            "torch_rs.autograd.backward only supports an exact native Tensor, "
            "directly or in an exact tuple or list containing at most six "
            "exact native Tensors"
        )
        two_root_error = (
            "torch_rs.autograd.backward only supports two roots when both "
            "are one-element native leaf Tensors requiring gradients"
        )
        three_root_error = (
            "torch_rs.autograd.backward only supports three roots when all "
            "three are one-element native leaf Tensors requiring gradients"
        )
        four_root_error = (
            "torch_rs.autograd.backward only supports four roots when all "
            "four are one-element native leaf Tensors requiring gradients"
        )
        five_root_error = (
            "torch_rs.autograd.backward only supports five roots when all "
            "five are one-element native leaf Tensors requiring gradients"
        )
        six_root_error = (
            "torch_rs.autograd.backward only supports six roots when all "
            "six are one-element native leaf Tensors requiring gradients"
        )
        gradient_error = (
            "torch_rs.autograd.backward does not support explicit gradients"
        )
        leaf = torch.tensor(2.0, requires_grad=True)
        loss = leaf * leaf

        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(two_root_error)}$"
        ):
            torch.autograd.backward(
                (loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(three_root_error)}$"
        ):
            torch.autograd.backward(
                (loss, loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(four_root_error)}$"
        ):
            torch.autograd.backward(
                (loss, loss, loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(five_root_error)}$"
        ):
            torch.autograd.backward(
                (loss, loss, loss, loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(six_root_error)}$"
        ):
            torch.autograd.backward(
                (loss, loss, loss, loss, loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(TypeError, f"^{re.escape(root_error)}$"):
            torch.autograd.backward(
                (loss, loss, loss, loss, loss, loss, loss),
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(gradient_error)}$"
        ):
            torch.autograd.backward(
                (), grad_tensors=(torch.tensor(1.0),), retain_graph=True
            )
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(gradient_error)}$"
        ):
            torch.autograd.backward(
                loss,
                grad_tensors=(torch.tensor(1.0),),
                retain_graph=True,
            )
        first = torch.tensor(3.0, requires_grad=True)
        second = torch.tensor([4.0], requires_grad=True)
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(gradient_error)}$"
        ):
            torch.autograd.backward(
                (first, second),
                grad_tensors=(torch.tensor(1.0), None),
                retain_graph=True,
            )
        six_roots = [
            torch.tensor(5.0, requires_grad=True),
            torch.tensor([6.0], requires_grad=True),
            torch.tensor([[7.0]], requires_grad=True),
            torch.tensor([[[8.0]]], requires_grad=True),
            torch.tensor([[[[9.0]]]], requires_grad=True),
            torch.tensor([[[[[10.0]]]]], requires_grad=True),
        ]
        with self.assertRaisesRegex(
            NotImplementedError, f"^{re.escape(gradient_error)}$"
        ):
            torch.autograd.backward(
                tuple(six_roots),
                grad_tensors=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    torch.tensor(1.0),
                ),
                retain_graph=True,
            )
        self.assertIsNone(first.grad)
        self.assertIsNone(second.grad)
        self.assertTrue(all(root.grad is None for root in six_roots))
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertEqual(leaf.grad.item(), 4.0)

    def test_metadata_imports_copying_pickling_and_exports(self):
        module = importlib.import_module("torch_rs.autograd")
        function = module.backward

        self.assertIs(module, torch.autograd)
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "backward")
        self.assertEqual(function.__qualname__, "backward")
        self.assertEqual(function.__module__, "torch_rs.autograd")
        self.assertEqual(
            tuple(inspect.signature(function).parameters),
            (
                "tensors",
                "grad_tensors",
                "retain_graph",
                "create_graph",
                "grad_variables",
                "inputs",
            ),
        )
        self.assertEqual(function.__defaults__, (None, None, False, None, None))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(
            tuple(function.__annotations__),
            (
                "tensors",
                "grad_tensors",
                "retain_graph",
                "create_graph",
                "grad_variables",
                "inputs",
                "return",
            ),
        )
        self.assertIs(
            typing.get_args(function.__annotations__["tensors"])[0],
            torch.Tensor,
        )
        self.assertIs(function.__annotations__["create_graph"], bool)
        self.assertIsNone(function.__annotations__["return"])
        self.assertIn("Compute the sum of gradients", function.__doc__)

        self.assertEqual(module.__all__.count("backward"), 1)
        wildcard_namespace = {}
        exec("from torch_rs.autograd import *", wildcard_namespace)
        self.assertIs(wildcard_namespace["backward"], function)
        explicit_namespace = {}
        exec("from torch_rs.autograd import backward", explicit_namespace)
        self.assertIs(explicit_namespace["backward"], function)

        self.assertFalse(hasattr(torch, "backward"))
        self.assertNotIn("backward", torch.__all__)
        self.assertFalse(hasattr(module, "grad"))

        self.assertIs(copy.copy(function), function)
        self.assertIs(copy.deepcopy(function), function)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(
                    pickle.loads(pickle.dumps(function, protocol=protocol)),
                    function,
                )

    def test_signature_binding_errors_are_non_mutating(self):
        leaf = torch.tensor(2.0, requires_grad=True)
        loss = leaf * leaf
        calls = (
            lambda: torch.autograd.backward(),
            lambda: torch.autograd.backward(loss, tensors=loss),
            lambda: torch.autograd.backward(loss, unexpected=True),
            lambda: torch.autograd.backward(
                loss, None, None, False, None, None, None
            ),
        )
        for case, call in enumerate(calls):
            with self.subTest(case=case), self.assertRaises(TypeError):
                call()
        self.assertIsNone(leaf.grad)
        loss.backward()
        self.assertEqual(leaf.grad.item(), 4.0)

    def test_importing_and_calling_does_not_import_pytorch(self):
        script = r"""
import sys

class RejectPytorchImport:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError(f"PyTorch import was attempted: {fullname}")
        return None

sys.meta_path.insert(0, RejectPytorchImport())
import torch_rs as torch
from torch_rs.autograd import backward

leaf = torch.tensor(2.0, requires_grad=True)
assert backward(leaf * leaf) is None
assert leaf.grad.item() == 4.0
tuple_leaf = torch.tensor(3.0, requires_grad=True)
assert backward((tuple_leaf * tuple_leaf,)) is None
assert tuple_leaf.grad.item() == 6.0
list_leaf = torch.tensor(4.0, requires_grad=True)
assert backward([list_leaf * list_leaf]) is None
assert list_leaf.grad.item() == 8.0
tuple_grad_leaf = torch.tensor(5.0, requires_grad=True)
assert backward(tuple_grad_leaf * tuple_grad_leaf, (None,)) is None
assert tuple_grad_leaf.grad.item() == 10.0
list_grad_leaf = torch.tensor(6.0, requires_grad=True)
assert backward((list_grad_leaf * list_grad_leaf,), [None]) is None
assert list_grad_leaf.grad.item() == 12.0
assert backward(()) is None
assert backward([], ()) is None
assert backward((), [None]) is None
pair_scalar = torch.tensor(7.0, requires_grad=True)
pair_strided = torch.tensor([[8.0]], requires_grad=True)
assert backward((pair_scalar, pair_strided), [None, None]) is None
assert pair_scalar.grad.item() == 1.0
assert pair_strided.grad.tolist() == [[1.0]]
duplicate = torch.tensor([9.0], requires_grad=True)
assert backward([duplicate, duplicate]) is None
assert duplicate.grad.tolist() == [2.0]
triple_duplicate = torch.tensor([10.0], requires_grad=True)
triple_distinct = torch.tensor([[11.0]], requires_grad=True)
assert backward(
    (triple_duplicate, triple_distinct, triple_duplicate),
    [None, None, None],
) is None
assert triple_duplicate.grad.tolist() == [2.0]
assert triple_distinct.grad.tolist() == [[1.0]]
quadruple_duplicate = torch.tensor([12.0], requires_grad=True)
quadruple_distinct = torch.tensor([[[13.0]]], requires_grad=True)
assert backward(
    [
        quadruple_duplicate,
        quadruple_distinct,
        quadruple_duplicate,
        quadruple_duplicate,
    ],
    (None, None, None, None),
) is None
assert quadruple_duplicate.grad.tolist() == [3.0]
assert quadruple_distinct.grad.tolist() == [[[1.0]]]
quintuple_duplicate = torch.tensor([14.0], requires_grad=True)
quintuple_distinct = torch.tensor([[[[15.0]]]], requires_grad=True)
assert backward(
    (
        quintuple_duplicate,
        quintuple_distinct,
        quintuple_duplicate,
        quintuple_duplicate,
        quintuple_duplicate,
    ),
    [None, None, None, None, None],
) is None
assert quintuple_duplicate.grad.tolist() == [4.0]
assert quintuple_distinct.grad.tolist() == [[[[1.0]]]]
sextuple_duplicate = torch.tensor([16.0], requires_grad=True)
sextuple_distinct = torch.tensor([[[[[17.0]]]]], requires_grad=True)
assert backward(
    [
        sextuple_duplicate,
        sextuple_distinct,
        sextuple_duplicate,
        sextuple_duplicate,
        sextuple_duplicate,
        sextuple_duplicate,
    ],
    (None, None, None, None, None, None),
) is None
assert sextuple_duplicate.grad.tolist() == [5.0]
assert sextuple_distinct.grad.tolist() == [[[[[1.0]]]]]
assert not hasattr(torch.autograd, "grad")
assert not hasattr(torch, "backward")
assert not any(name == "torch" or name.startswith("torch.") for name in sys.modules)
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
