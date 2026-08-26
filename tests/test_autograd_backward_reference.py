import copy
import importlib
import inspect
import pickle
import types
import unittest
from collections.abc import Sequence

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class CustomSequence(Sequence):
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


class MaterializationCountingSequence(CustomSequence):
    def __init__(self, values):
        super().__init__(values)
        self.materializations = 0

    def __iter__(self):
        self.materializations += 1
        return iter(self.values)


class ListSubclass(list):
    pass


class TupleSubclass(tuple):
    pass


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class AutogradBackwardReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "autograd.backward differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))
        self.assertEqual(actual_raised.exception.args, expected_raised.exception.args)

    @staticmethod
    def roots(root, sequence_type):
        if sequence_type is None:
            return root
        return sequence_type((root,))

    @staticmethod
    def default_grad_tensors(sequence_type):
        if sequence_type is None:
            return None
        return sequence_type((None,))

    @staticmethod
    def run_custom_sequence_backward(module, roots, gradient_form):
        if gradient_form == "omitted":
            return module.autograd.backward(roots)
        if gradient_form == "None":
            return module.autograd.backward(roots, grad_tensors=None)
        if gradient_form == "tuple":
            return module.autograd.backward(
                roots, grad_tensors=(None,) * len(roots)
            )
        if gradient_form == "list":
            return module.autograd.backward(
                roots, grad_tensors=[None] * len(roots)
            )
        if gradient_form == "singleton tuple":
            return module.autograd.backward(roots, grad_tensors=(None,))
        if gradient_form == "singleton list":
            return module.autograd.backward(roots, grad_tensors=[None])
        raise AssertionError(f"unknown gradient form: {gradient_form}")

    def custom_sequence_outcome(
        self, module, root_sequence_type, root_count, gradient_form
    ):
        leaf = module.tensor([2.0], requires_grad=True)
        roots = root_sequence_type((leaf,) * root_count)
        first_result = self.run_custom_sequence_backward(
            module, roots, gradient_form
        )

        if root_count == 0:
            untouched = leaf.grad is None
            leaf.backward()
            return first_result, untouched, np.asarray(leaf.grad).copy()

        first_gradient = leaf.grad
        first_values = np.asarray(first_gradient).copy()
        second_result = self.run_custom_sequence_backward(
            module, roots, gradient_form
        )
        return (
            first_result,
            second_result,
            first_values,
            leaf.grad is first_gradient,
            np.asarray(leaf.grad).copy(),
        )

    @staticmethod
    def materialization_outcome(module):
        leaf = module.tensor([3.0], requires_grad=True)
        roots = MaterializationCountingSequence((leaf, leaf, leaf))
        first_result = module.autograd.backward(roots)
        first_count = roots.materializations
        first_gradient = leaf.grad
        first_values = np.asarray(first_gradient).copy()
        second_result = module.autograd.backward(roots)
        return (
            first_result,
            first_count,
            first_values,
            second_result,
            roots.materializations,
            leaf.grad is first_gradient,
            np.asarray(leaf.grad).copy(),
        )

    def supported_outcome(self, module, form, sequence_type):
        leaf = module.tensor([2.0, -3.0], requires_grad=True)
        loss = (leaf * leaf).sum()
        roots = self.roots(loss, sequence_type)
        if form == "positional":
            result = module.autograd.backward(roots)
        elif form == "keyword":
            result = module.autograd.backward(tensors=roots)
        elif form == "explicit defaults":
            result = module.autograd.backward(
                roots,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            )
        elif form == "positional defaults":
            result = module.autograd.backward(
                roots, None, False, False, None, None
            )
        elif form == "integer false":
            result = module.autograd.backward(roots, None, 0, 0)
        elif form == "tuple grad_tensors":
            result = module.autograd.backward(roots, (None,))
        elif form == "list grad_tensors":
            result = module.autograd.backward(roots, grad_tensors=[None])
        else:
            raise AssertionError(f"unknown form: {form}")
        return result, np.asarray(leaf.grad).copy()

    def empty_outcome(self, module, form, sequence_type):
        leaf = module.tensor([2.0, 3.0], requires_grad=True)
        leaf.sum().backward()
        initial_gradient = np.asarray(leaf.grad).copy()
        loss = (leaf * leaf).sum()
        roots = sequence_type()

        if form == "positional":
            result = module.autograd.backward(roots)
        elif form == "keyword":
            result = module.autograd.backward(tensors=roots)
        elif form == "explicit defaults":
            result = module.autograd.backward(
                roots,
                grad_tensors=None,
                retain_graph=None,
                create_graph=False,
                grad_variables=None,
                inputs=None,
            )
        elif form == "positional defaults":
            result = module.autograd.backward(
                roots, None, False, False, None, None
            )
        elif form == "integer false":
            result = module.autograd.backward(roots, None, 0, 0)
        elif form == "empty tuple grad_tensors":
            result = module.autograd.backward(roots, ())
        elif form == "empty list grad_tensors":
            result = module.autograd.backward(roots, grad_tensors=[])
        elif form == "tuple singleton None grad_tensors":
            result = module.autograd.backward(roots, (None,))
        elif form == "list singleton None grad_tensors":
            result = module.autograd.backward(roots, grad_tensors=[None])
        else:
            raise AssertionError(f"unknown form: {form}")

        gradient_after_empty_backward = np.asarray(leaf.grad).copy()
        loss.backward()
        return (
            result,
            initial_gradient,
            gradient_after_empty_backward,
            np.asarray(leaf.grad).copy(),
        )

    @staticmethod
    def run_two_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(roots, (None, None))
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None, None]
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_three_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(roots, (None, None, None))
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None, None, None]
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_four_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(
                roots, (None, None, None, None)
            )
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None, None, None, None]
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_five_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(
                roots, (None, None, None, None, None)
            )
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None, None, None, None, None]
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_six_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(
                roots, (None, None, None, None, None, None)
            )
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None, None, None, None, None, None]
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_seven_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(roots, (None,) * 7)
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None] * 7
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_eight_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(roots, (None,) * 8)
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None] * 8
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_nine_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(roots, (None,) * 9)
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None] * 9
            )
        raise AssertionError(f"unknown form: {form}")

    @staticmethod
    def run_ten_root_backward(module, roots, form):
        if form == "omitted":
            return module.autograd.backward(roots)
        if form == "explicit None":
            return module.autograd.backward(roots, grad_tensors=None)
        if form == "tuple grad_tensors":
            return module.autograd.backward(roots, (None,) * 10)
        if form == "list grad_tensors":
            return module.autograd.backward(
                roots, grad_tensors=[None] * 10
            )
        raise AssertionError(f"unknown form: {form}")

    def two_leaf_outcome(self, module, root_sequence_type, form):
        scalar_leaf = module.tensor(2.0, requires_grad=True)
        strided_leaf = module.tensor([[3.0]], requires_grad=True)
        roots = root_sequence_type((scalar_leaf, strided_leaf))

        first_result = self.run_two_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(scalar_leaf.grad).copy(),
            np.asarray(strided_leaf.grad).copy(),
        )
        second_result = self.run_two_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(scalar_leaf.grad).copy(),
            np.asarray(strided_leaf.grad).copy(),
        )
        return (
            first_result,
            second_result,
            tuple(strided_leaf.shape),
            strided_leaf.stride(),
            first_gradients,
            second_gradients,
        )

    def duplicate_leaf_outcome(self, module, root_sequence_type, form):
        leaf = module.tensor([4.0], requires_grad=True)
        roots = root_sequence_type((leaf, leaf))

        first_result = self.run_two_root_backward(module, roots, form)
        first_gradient = np.asarray(leaf.grad).copy()
        second_result = self.run_two_root_backward(module, roots, form)
        second_gradient = np.asarray(leaf.grad).copy()
        return first_result, second_result, first_gradient, second_gradient

    def precision_duplicate_outcome(self, module, root_sequence_type, form):
        leaf = module.tensor([1.0], requires_grad=True)
        (leaf * 16_777_216.0).backward()
        existing_gradient = leaf.grad
        roots = root_sequence_type((leaf, leaf))

        result = self.run_two_root_backward(module, roots, form)
        return (
            result,
            leaf.grad is existing_gradient,
            np.asarray(leaf.grad).copy(),
        )

    def rejected_no_grad_view_outcome(
        self, module, root_sequence_type, form
    ):
        first = module.tensor(3.0, requires_grad=True)
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((first, invalid))

        try:
            self.run_two_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = (
            first.grad is None,
            invalid.grad is None,
            source.grad is None,
        )
        first.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            first.grad.item(),
            np.asarray(source.grad).copy(),
        )

    def three_leaf_outcome(self, module, root_sequence_type, form):
        scalar_leaf = module.tensor(2.0, requires_grad=True)
        vector_leaf = module.tensor([3.0], requires_grad=True)
        matrix_leaf = module.tensor([[4.0]], requires_grad=True)
        roots = root_sequence_type(
            (scalar_leaf, vector_leaf, matrix_leaf)
        )

        first_result = self.run_three_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy()
            for root in (scalar_leaf, vector_leaf, matrix_leaf)
        )
        second_result = self.run_three_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy()
            for root in (scalar_leaf, vector_leaf, matrix_leaf)
        )
        return first_result, second_result, first_gradients, second_gradients

    def three_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type((duplicate, distinct, duplicate))

        result = self.run_three_root_backward(module, roots, form)
        return (
            result,
            duplicate.grad is existing_gradient,
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )

    def rejected_no_grad_view_third_outcome(
        self, module, root_sequence_type, form
    ):
        first = module.tensor(3.0, requires_grad=True)
        second = module.tensor([4.0], requires_grad=True)
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((first, second, invalid))

        try:
            self.run_three_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = (
            first.grad is None,
            second.grad is None,
            invalid.grad is None,
            source.grad is None,
        )
        first.backward()
        second.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            first.grad.item(),
            np.asarray(second.grad).copy(),
            np.asarray(source.grad).copy(),
        )

    def four_leaf_outcome(self, module, root_sequence_type, form):
        scalar_leaf = module.tensor(2.0, requires_grad=True)
        vector_leaf = module.tensor([3.0], requires_grad=True)
        matrix_leaf = module.tensor([[4.0]], requires_grad=True)
        rank_three_leaf = module.tensor([[[5.0]]], requires_grad=True)
        leaves = (scalar_leaf, vector_leaf, matrix_leaf, rank_three_leaf)
        roots = root_sequence_type(leaves)

        first_result = self.run_four_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_four_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        return first_result, second_result, first_gradients, second_gradients

    def four_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (duplicate, distinct, duplicate, duplicate)
        )

        first_result = self.run_four_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_four_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_fourth_outcome(
        self, module, root_sequence_type, form
    ):
        first = module.tensor(3.0, requires_grad=True)
        second = module.tensor([4.0], requires_grad=True)
        third = module.tensor([[5.0]], requires_grad=True)
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((first, second, third, invalid))

        try:
            self.run_four_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = (
            first.grad is None,
            second.grad is None,
            third.grad is None,
            invalid.grad is None,
            source.grad is None,
        )
        first.backward()
        second.backward()
        third.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            first.grad.item(),
            np.asarray(second.grad).copy(),
            np.asarray(third.grad).copy(),
            np.asarray(source.grad).copy(),
        )

    def five_leaf_outcome(self, module, root_sequence_type, form):
        leaves = (
            module.tensor(2.0, requires_grad=True),
            module.tensor([3.0], requires_grad=True),
            module.tensor([[4.0]], requires_grad=True),
            module.tensor([[[5.0]]], requires_grad=True),
            module.tensor([[[[6.0]]]], requires_grad=True),
        )
        roots = root_sequence_type(leaves)

        first_result = self.run_five_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_five_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        return first_result, second_result, first_gradients, second_gradients

    def five_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (duplicate, distinct, duplicate, duplicate, duplicate)
        )

        first_result = self.run_five_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_five_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_fifth_outcome(
        self, module, root_sequence_type, form
    ):
        valid = (
            module.tensor(3.0, requires_grad=True),
            module.tensor([4.0], requires_grad=True),
            module.tensor([[5.0]], requires_grad=True),
            module.tensor([[[6.0]]], requires_grad=True),
        )
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((*valid, invalid))

        try:
            self.run_five_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = tuple(root.grad is None for root in valid) + (
            invalid.grad is None,
            source.grad is None,
        )
        for root in valid:
            root.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            valid[0].grad.item(),
            *(np.asarray(root.grad).copy() for root in valid[1:]),
            np.asarray(source.grad).copy(),
        )

    def six_leaf_outcome(self, module, root_sequence_type, form):
        leaves = (
            module.tensor(2.0, requires_grad=True),
            module.tensor([3.0], requires_grad=True),
            module.tensor([[4.0]], requires_grad=True),
            module.tensor([[[5.0]]], requires_grad=True),
            module.tensor([[[[6.0]]]], requires_grad=True),
            module.tensor([[[[[7.0]]]]], requires_grad=True),
        )
        roots = root_sequence_type(leaves)

        first_result = self.run_six_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_six_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        return first_result, second_result, first_gradients, second_gradients

    def six_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (
                duplicate,
                distinct,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
            )
        )

        first_result = self.run_six_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_six_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_sixth_outcome(
        self, module, root_sequence_type, form
    ):
        valid = (
            module.tensor(3.0, requires_grad=True),
            module.tensor([4.0], requires_grad=True),
            module.tensor([[5.0]], requires_grad=True),
            module.tensor([[[6.0]]], requires_grad=True),
            module.tensor([[[[7.0]]]], requires_grad=True),
        )
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((*valid, invalid))

        try:
            self.run_six_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = tuple(root.grad is None for root in valid) + (
            invalid.grad is None,
            source.grad is None,
        )
        for root in valid:
            root.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            valid[0].grad.item(),
            *(np.asarray(root.grad).copy() for root in valid[1:]),
            np.asarray(source.grad).copy(),
        )

    def seven_leaf_outcome(self, module, root_sequence_type, form):
        leaves = (
            module.tensor(2.0, requires_grad=True),
            module.tensor([3.0], requires_grad=True),
            module.tensor([[4.0]], requires_grad=True),
            module.tensor([[[5.0]]], requires_grad=True),
            module.tensor([[[[6.0]]]], requires_grad=True),
            module.tensor([[[[[7.0]]]]], requires_grad=True),
            module.tensor([[[[[[8.0]]]]]], requires_grad=True),
        )
        roots = root_sequence_type(leaves)

        first_result = self.run_seven_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_seven_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        return first_result, second_result, first_gradients, second_gradients

    def seven_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (
                duplicate,
                distinct,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
            )
        )

        first_result = self.run_seven_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_seven_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_seventh_outcome(
        self, module, root_sequence_type, form
    ):
        valid = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(6)
        )
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((*valid, invalid))

        try:
            self.run_seven_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = tuple(root.grad is None for root in valid) + (
            invalid.grad is None,
            source.grad is None,
        )
        for root in valid:
            root.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            valid[0].grad.item(),
            *(np.asarray(root.grad).copy() for root in valid[1:]),
            np.asarray(source.grad).copy(),
        )

    def eight_leaf_outcome(self, module, root_sequence_type, form):
        leaves = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(8)
        )
        roots = root_sequence_type(leaves)

        first_result = self.run_eight_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_eight_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        return first_result, second_result, first_gradients, second_gradients

    def eight_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (
                duplicate,
                distinct,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
            )
        )

        first_result = self.run_eight_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_eight_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_eighth_outcome(
        self, module, root_sequence_type, form
    ):
        valid = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(7)
        )
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((*valid, invalid))

        try:
            self.run_eight_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = tuple(root.grad is None for root in valid) + (
            invalid.grad is None,
            source.grad is None,
        )
        for root in valid:
            root.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            valid[0].grad.item(),
            *(np.asarray(root.grad).copy() for root in valid[1:]),
            np.asarray(source.grad).copy(),
        )

    def nine_leaf_outcome(self, module, root_sequence_type, form):
        leaves = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(9)
        )
        roots = root_sequence_type(leaves)

        first_result = self.run_nine_root_backward(module, roots, form)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_nine_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        return first_result, second_result, first_gradients, second_gradients

    def nine_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (
                duplicate,
                distinct,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
            )
        )

        first_result = self.run_nine_root_backward(module, roots, form)
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_nine_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_ninth_outcome(
        self, module, root_sequence_type, form
    ):
        valid = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(8)
        )
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((*valid, invalid))

        try:
            self.run_nine_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = tuple(root.grad is None for root in valid) + (
            invalid.grad is None,
            source.grad is None,
        )
        for root in valid:
            root.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            valid[0].grad.item(),
            *(np.asarray(root.grad).copy() for root in valid[1:]),
            np.asarray(source.grad).copy(),
        )

    def ten_leaf_outcome(self, module, root_sequence_type, form):
        leaves = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(10)
        )
        roots = root_sequence_type(leaves)

        first_result = self.run_ten_root_backward(module, roots, form)
        gradients = tuple(root.grad for root in leaves)
        first_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        second_result = self.run_ten_root_backward(module, roots, form)
        second_gradients = tuple(
            np.asarray(root.grad).copy() for root in leaves
        )
        gradient_identities = tuple(
            root.grad is gradient
            for root, gradient in zip(leaves, gradients)
        )
        return (
            first_result,
            second_result,
            first_gradients,
            second_gradients,
            gradient_identities,
        )

    def ten_duplicate_outcome(self, module, root_sequence_type, form):
        duplicate = module.tensor([1.0], requires_grad=True)
        distinct = module.tensor([[2.0]], requires_grad=True)
        (duplicate * 16_777_216.0).backward()
        existing_gradient = duplicate.grad
        roots = root_sequence_type(
            (
                duplicate,
                distinct,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
                duplicate,
            )
        )

        first_result = self.run_ten_root_backward(module, roots, form)
        distinct_gradient = distinct.grad
        first_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        second_result = self.run_ten_root_backward(module, roots, form)
        second_gradients = (
            np.asarray(duplicate.grad).copy(),
            np.asarray(distinct.grad).copy(),
        )
        return (
            first_result,
            second_result,
            duplicate.grad is existing_gradient,
            distinct.grad is distinct_gradient,
            first_gradients,
            second_gradients,
        )

    def rejected_no_grad_view_tenth_outcome(
        self, module, root_sequence_type, form
    ):
        valid = tuple(
            module.tensor([float(index)], requires_grad=True)
            for index in range(9)
        )
        source = module.tensor([[1.0, 2.0]], requires_grad=True)
        with module.no_grad():
            invalid = source.transpose(0, 1)[1]
        roots = root_sequence_type((*valid, invalid))

        try:
            self.run_ten_root_backward(module, roots, form)
        except RuntimeError as error:
            failure = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a no_grad view cannot seed backward")
        untouched = tuple(root.grad is None for root in valid) + (
            invalid.grad is None,
            source.grad is None,
        )
        for root in valid:
            root.backward()
        source.sum().backward()
        return (
            failure,
            tuple(invalid.shape),
            invalid.stride(),
            invalid.storage_offset(),
            invalid.requires_grad,
            invalid.is_leaf,
            untouched,
            valid[0].grad.item(),
            *(np.asarray(root.grad).copy() for root in valid[1:]),
            np.asarray(source.grad).copy(),
        )

    def graph_outcome(self, module, root_sequence_type, grad_sequence_type=None):
        grad_tensors = self.default_grad_tensors(grad_sequence_type)
        reusable_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        reusable_loss = reusable_leaf.transpose(0, 0).sum()
        module.autograd.backward(
            self.roots(reusable_loss, root_sequence_type),
            grad_tensors=grad_tensors,
        )
        module.autograd.backward(
            self.roots(reusable_loss, root_sequence_type),
            grad_tensors=grad_tensors,
        )

        scalar_leaf = module.tensor(7.0, requires_grad=True)
        module.autograd.backward(
            self.roots(scalar_leaf, root_sequence_type),
            grad_tensors=grad_tensors,
        )
        module.autograd.backward(
            self.roots(scalar_leaf, root_sequence_type),
            grad_tensors=grad_tensors,
        )

        freed_leaf = module.tensor([2.0, 3.0], requires_grad=True)
        freed_loss = (freed_leaf * freed_leaf).sum()
        module.autograd.backward(
            self.roots(freed_loss, root_sequence_type),
            grad_tensors=grad_tensors,
        )
        first_gradient = np.asarray(freed_leaf.grad).copy()
        try:
            module.autograd.backward(
                self.roots(freed_loss, root_sequence_type),
                grad_tensors=grad_tensors,
            )
        except RuntimeError as error:
            repeated_error = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a value-dependent graph must be freed")
        module.autograd.backward(
            self.roots(
                (freed_leaf * freed_leaf).sum(), root_sequence_type
            ),
            grad_tensors=grad_tensors,
        )

        return (
            np.asarray(reusable_leaf.grad).copy(),
            scalar_leaf.grad.item(),
            first_gradient,
            repeated_error,
            np.asarray(freed_leaf.grad).copy(),
        )

    def test_single_root_default_calls_match_pytorch_2_13(self):
        for sequence_type in (None, tuple, list):
            for form in (
                "positional",
                "keyword",
                "explicit defaults",
                "positional defaults",
                "integer false",
                "tuple grad_tensors",
                "list grad_tensors",
            ):
                with self.subTest(sequence_type=sequence_type, form=form):
                    actual_result, actual_gradient = self.supported_outcome(
                        torch, form, sequence_type
                    )
                    expected_result, expected_gradient = self.supported_outcome(
                        reference_torch, form, sequence_type
                    )
                    self.assertIsNone(actual_result)
                    self.assertIsNone(expected_result)
                    np.testing.assert_array_equal(
                        actual_gradient, expected_gradient
                    )

    def test_empty_root_noops_match_pytorch_2_13(self):
        forms = (
            "positional",
            "keyword",
            "explicit defaults",
            "positional defaults",
            "integer false",
            "empty tuple grad_tensors",
            "empty list grad_tensors",
            "tuple singleton None grad_tensors",
            "list singleton None grad_tensors",
        )
        for sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(sequence_type=sequence_type, form=form):
                    actual = self.empty_outcome(torch, form, sequence_type)
                    expected = self.empty_outcome(
                        reference_torch, form, sequence_type
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(expected[0])
                    np.testing.assert_array_equal(actual[1], actual[2])
                    np.testing.assert_array_equal(expected[1], expected[2])
                    for actual_gradient, expected_gradient in zip(
                        actual[1:], expected[1:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_custom_root_sequences_match_pytorch_2_13(self):
        for root_sequence_type in (
            CustomSequence,
            TupleSubclass,
            ListSubclass,
        ):
            for root_count in range(11):
                gradient_forms = ["omitted", "None", "tuple", "list"]
                if root_count == 0:
                    gradient_forms.extend(
                        ("singleton tuple", "singleton list")
                    )
                for gradient_form in gradient_forms:
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        root_count=root_count,
                        gradient_form=gradient_form,
                    ):
                        actual = self.custom_sequence_outcome(
                            torch,
                            root_sequence_type,
                            root_count,
                            gradient_form,
                        )
                        expected = self.custom_sequence_outcome(
                            reference_torch,
                            root_sequence_type,
                            root_count,
                            gradient_form,
                        )
                        self.assertEqual(actual[:2], expected[:2])
                        np.testing.assert_array_equal(actual[2], expected[2])
                        if root_count:
                            self.assertEqual(actual[3], expected[3])
                            np.testing.assert_array_equal(
                                actual[4], expected[4]
                            )

    def test_custom_root_sequence_materialization_matches_pytorch_2_13(self):
        actual = self.materialization_outcome(torch)
        expected = self.materialization_outcome(reference_torch)
        self.assertEqual(actual[:2], expected[:2])
        np.testing.assert_array_equal(actual[2], expected[2])
        self.assertEqual(actual[3:6], expected[3:6])
        np.testing.assert_array_equal(actual[6], expected[6])

    def test_two_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.two_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.two_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2:4], expected[2:4])
                    for actual_gradients, expected_gradients in zip(
                        actual[4:], expected[4:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_three_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.three_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.three_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_four_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.four_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.four_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_five_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.five_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.five_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_six_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.six_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.six_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_seven_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.seven_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.seven_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_eight_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.eight_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.eight_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_nine_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.nine_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.nine_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:], expected[2:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_ten_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.ten_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.ten_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    for actual_gradients, expected_gradients in zip(
                        actual[2:4], expected[2:4]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )
                    self.assertEqual(actual[4], expected[4])

    def test_three_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.three_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.three_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(expected[0])
                    self.assertEqual(actual[1], expected[1])
                    np.testing.assert_array_equal(actual[2], expected[2])
                    np.testing.assert_array_equal(actual[3], expected[3])

    def test_four_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.four_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.four_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2], expected[2])
                    for actual_gradients, expected_gradients in zip(
                        actual[3:], expected[3:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_five_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.five_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.five_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2], expected[2])
                    for actual_gradients, expected_gradients in zip(
                        actual[3:], expected[3:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_six_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.six_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.six_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2], expected[2])
                    for actual_gradients, expected_gradients in zip(
                        actual[3:], expected[3:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_seven_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.seven_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.seven_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2], expected[2])
                    for actual_gradients, expected_gradients in zip(
                        actual[3:], expected[3:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_eight_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.eight_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.eight_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2], expected[2])
                    for actual_gradients, expected_gradients in zip(
                        actual[3:], expected[3:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_nine_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.nine_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.nine_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2], expected[2])
                    for actual_gradients, expected_gradients in zip(
                        actual[3:], expected[3:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_ten_roots_with_duplicates_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.ten_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.ten_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    self.assertEqual(actual[2:4], expected[2:4])
                    for actual_gradients, expected_gradients in zip(
                        actual[4:], expected[4:]
                    ):
                        for actual_gradient, expected_gradient in zip(
                            actual_gradients, expected_gradients
                        ):
                            np.testing.assert_array_equal(
                                actual_gradient, expected_gradient
                            )

    def test_duplicate_two_leaf_roots_match_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.duplicate_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.duplicate_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(actual[1])
                    self.assertIsNone(expected[0])
                    self.assertIsNone(expected[1])
                    np.testing.assert_array_equal(actual[2], expected[2])
                    np.testing.assert_array_equal(actual[3], expected[3])

    def test_duplicate_root_precision_boundary_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.precision_duplicate_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.precision_duplicate_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertIsNone(actual[0])
                    self.assertIsNone(expected[0])
                    self.assertEqual(actual[1], expected[1])
                    np.testing.assert_array_equal(actual[2], expected[2])

    def test_no_grad_view_second_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    np.testing.assert_array_equal(actual[8], expected[8])

    def test_no_grad_view_third_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_third_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_third_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    np.testing.assert_array_equal(actual[8], expected[8])
                    np.testing.assert_array_equal(actual[9], expected[9])

    def test_no_grad_view_fourth_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_fourth_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_fourth_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    np.testing.assert_array_equal(actual[8], expected[8])
                    np.testing.assert_array_equal(actual[9], expected[9])
                    np.testing.assert_array_equal(actual[10], expected[10])

    def test_no_grad_view_fifth_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_fifth_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_fifth_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    for actual_gradient, expected_gradient in zip(
                        actual[8:], expected[8:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_no_grad_view_sixth_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_sixth_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_sixth_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    for actual_gradient, expected_gradient in zip(
                        actual[8:], expected[8:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_no_grad_view_seventh_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_seventh_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_seventh_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    for actual_gradient, expected_gradient in zip(
                        actual[8:], expected[8:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_no_grad_view_eighth_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_eighth_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_eighth_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    for actual_gradient, expected_gradient in zip(
                        actual[8:], expected[8:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_no_grad_view_ninth_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_ninth_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_ninth_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    for actual_gradient, expected_gradient in zip(
                        actual[8:], expected[8:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_no_grad_view_tenth_root_failure_matches_pytorch_2_13(self):
        forms = (
            "omitted",
            "explicit None",
            "tuple grad_tensors",
            "list grad_tensors",
        )
        for root_sequence_type in (tuple, list):
            for form in forms:
                with self.subTest(
                    root_sequence_type=root_sequence_type, form=form
                ):
                    actual = self.rejected_no_grad_view_tenth_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.rejected_no_grad_view_tenth_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual[:8], expected[:8])
                    for actual_gradient, expected_gradient in zip(
                        actual[8:], expected[8:]
                    ):
                        np.testing.assert_array_equal(
                            actual_gradient, expected_gradient
                        )

    def test_accumulation_graph_reuse_and_freeing_match_pytorch_2_13(self):
        for root_sequence_type in (
            None,
            tuple,
            list,
            CustomSequence,
            TupleSubclass,
            ListSubclass,
        ):
            for grad_sequence_type in (None, tuple, list):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    grad_sequence_type=grad_sequence_type,
                ):
                    actual = self.graph_outcome(
                        torch, root_sequence_type, grad_sequence_type
                    )
                    expected = self.graph_outcome(
                        reference_torch,
                        root_sequence_type,
                        grad_sequence_type,
                    )
                    np.testing.assert_array_equal(actual[0], expected[0])
                    self.assertEqual(actual[1], expected[1])
                    np.testing.assert_array_equal(actual[2], expected[2])
                    self.assertEqual(actual[3], expected[3])
                    np.testing.assert_array_equal(actual[4], expected[4])

    def test_native_engine_errors_match_pytorch_2_13(self):
        for root_sequence_type in (None, tuple, list):
            for grad_sequence_type in (None, tuple, list):
                actual_grad_tensors = self.default_grad_tensors(
                    grad_sequence_type
                )
                expected_grad_tensors = self.default_grad_tensors(
                    grad_sequence_type
                )
                cases = (
                    (
                        lambda: torch.autograd.backward(
                            self.roots(
                                torch.tensor(1.0), root_sequence_type
                            ),
                            grad_tensors=actual_grad_tensors,
                        ),
                        lambda: reference_torch.autograd.backward(
                            self.roots(
                                reference_torch.tensor(1.0),
                                root_sequence_type,
                            ),
                            grad_tensors=expected_grad_tensors,
                        ),
                    ),
                    (
                        lambda: torch.autograd.backward(
                            self.roots(
                                torch.tensor(
                                    [1.0, 2.0], requires_grad=True
                                ),
                                root_sequence_type,
                            ),
                            grad_tensors=actual_grad_tensors,
                        ),
                        lambda: reference_torch.autograd.backward(
                            self.roots(
                                reference_torch.tensor(
                                    [1.0, 2.0], requires_grad=True
                                ),
                                root_sequence_type,
                            ),
                            grad_tensors=expected_grad_tensors,
                        ),
                    ),
                )
                for case, (actual_call, expected_call) in enumerate(cases):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        case=case,
                    ):
                        self.assert_error_matches(actual_call, expected_call)

    def test_graph_option_conversion_errors_match_and_do_not_mutate(self):
        for root_sequence_type in (None, tuple, list):
            for grad_sequence_type in (None, tuple, list):
                for name, value in (
                    ("retain_graph", 0.5),
                    ("create_graph", None),
                ):
                    with self.subTest(
                        root_sequence_type=root_sequence_type,
                        grad_sequence_type=grad_sequence_type,
                        name=name,
                        value=value,
                    ):
                        actual_leaf = torch.tensor(2.0, requires_grad=True)
                        actual_loss = actual_leaf * actual_leaf
                        expected_leaf = reference_torch.tensor(
                            2.0, requires_grad=True
                        )
                        expected_loss = expected_leaf * expected_leaf
                        actual_grad_tensors = self.default_grad_tensors(
                            grad_sequence_type
                        )
                        expected_grad_tensors = self.default_grad_tensors(
                            grad_sequence_type
                        )
                        self.assert_error_matches(
                            lambda: torch.autograd.backward(
                                self.roots(
                                    actual_loss, root_sequence_type
                                ),
                                grad_tensors=actual_grad_tensors,
                                **{name: value},
                            ),
                            lambda: reference_torch.autograd.backward(
                                self.roots(
                                    expected_loss, root_sequence_type
                                ),
                                grad_tensors=expected_grad_tensors,
                                **{name: value},
                            ),
                        )
                        self.assertIsNone(actual_leaf.grad)
                        self.assertIsNone(expected_leaf.grad)
                        actual_loss.backward()
                        expected_loss.backward()
                        self.assertEqual(
                            actual_leaf.grad.item(),
                            expected_leaf.grad.item(),
                        )

    def test_signature_binding_errors_match_and_do_not_mutate(self):
        actual_leaf = torch.tensor(2.0, requires_grad=True)
        actual_loss = actual_leaf * actual_leaf
        expected_leaf = reference_torch.tensor(2.0, requires_grad=True)
        expected_loss = expected_leaf * expected_leaf
        cases = (
            (
                lambda: torch.autograd.backward(),
                lambda: reference_torch.autograd.backward(),
            ),
            (
                lambda: torch.autograd.backward(
                    actual_loss, tensors=actual_loss
                ),
                lambda: reference_torch.autograd.backward(
                    expected_loss, tensors=expected_loss
                ),
            ),
            (
                lambda: torch.autograd.backward(actual_loss, unexpected=True),
                lambda: reference_torch.autograd.backward(
                    expected_loss, unexpected=True
                ),
            ),
            (
                lambda: torch.autograd.backward(
                    actual_loss, None, None, False, None, None, None
                ),
                lambda: reference_torch.autograd.backward(
                    expected_loss, None, None, False, None, None, None
                ),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)

    def test_metadata_documentation_annotations_imports_and_pickle_match(self):
        actual_module = importlib.import_module("torch_rs.autograd")
        expected_module = importlib.import_module("torch.autograd")
        actual = actual_module.backward
        expected = expected_module.backward

        self.assertIs(type(actual), types.FunctionType)
        self.assertIs(type(expected), types.FunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(
            actual.__module__.replace("torch_rs", "torch"), expected.__module__
        )
        self.assertEqual(
            str(inspect.signature(actual)).replace("torch_rs", "torch"),
            str(inspect.signature(expected)),
        )
        self.assertEqual(
            str(actual.__annotations__).replace("torch_rs", "torch"),
            str(expected.__annotations__),
        )
        self.assertEqual(actual.__defaults__, expected.__defaults__)
        self.assertEqual(actual.__kwdefaults__, expected.__kwdefaults__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(
            hasattr(actual, "__text_signature__"),
            hasattr(expected, "__text_signature__"),
        )
        self.assertEqual(
            actual_module.__all__.count("backward"),
            expected_module.__all__.count("backward"),
        )

        for module, function in (
            (actual_module, actual),
            (expected_module, expected),
        ):
            wildcard_namespace = {}
            exec(f"from {module.__name__} import *", wildcard_namespace)
            self.assertIs(wildcard_namespace["backward"], function)
            self.assertIs(copy.copy(function), function)
            self.assertIs(copy.deepcopy(function), function)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(module=module.__name__, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(function, protocol=protocol)),
                        function,
                    )

        self.assertFalse(hasattr(torch, "backward"))
        self.assertFalse(hasattr(reference_torch, "backward"))
        self.assertFalse(hasattr(torch.autograd, "grad"))
        self.assertTrue(hasattr(reference_torch.autograd, "grad"))


if __name__ == "__main__":
    unittest.main()
