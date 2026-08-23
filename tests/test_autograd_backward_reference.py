import copy
import importlib
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


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

    def two_leaf_outcome(self, module, root_sequence_type, form):
        scalar_leaf = module.tensor(2.0, requires_grad=True)
        strided_leaf = module.tensor([3.0], requires_grad=True)
        roots = root_sequence_type((scalar_leaf, strided_leaf))

        def backward(current_roots):
            if form == "omitted":
                return module.autograd.backward(current_roots)
            if form == "keyword roots":
                return module.autograd.backward(tensors=current_roots)
            if form == "none":
                return module.autograd.backward(
                    current_roots, grad_tensors=None
                )
            if form == "explicit defaults":
                return module.autograd.backward(
                    current_roots,
                    grad_tensors=None,
                    retain_graph=None,
                    create_graph=False,
                    grad_variables=None,
                    inputs=None,
                )
            if form == "positional defaults":
                return module.autograd.backward(
                    current_roots, None, False, False, None, None
                )
            if form == "integer false":
                return module.autograd.backward(current_roots, None, 0, 0)
            if form == "tuple":
                return module.autograd.backward(
                    current_roots, grad_tensors=(None, None)
                )
            if form == "list":
                return module.autograd.backward(
                    current_roots, grad_tensors=[None, None]
                )
            raise AssertionError(f"unknown form: {form}")

        first_result = backward(roots)
        second_result = backward(roots)

        duplicate_leaf = module.tensor([[5.0]], requires_grad=True)
        duplicate_result = backward(
            root_sequence_type((duplicate_leaf, duplicate_leaf))
        )
        return (
            first_result,
            second_result,
            (
                tuple(strided_leaf.shape),
                strided_leaf.stride(),
                strided_leaf.requires_grad,
                strided_leaf.is_leaf,
            ),
            scalar_leaf.grad.item(),
            strided_leaf.grad.tolist(),
            duplicate_result,
            duplicate_leaf.grad.tolist(),
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

    def test_two_leaf_roots_match_pytorch_2_13(self):
        for root_sequence_type in (tuple, list):
            for form in (
                "omitted",
                "keyword roots",
                "none",
                "explicit defaults",
                "positional defaults",
                "integer false",
                "tuple",
                "list",
            ):
                with self.subTest(
                    root_sequence_type=root_sequence_type,
                    form=form,
                ):
                    actual = self.two_leaf_outcome(
                        torch, root_sequence_type, form
                    )
                    expected = self.two_leaf_outcome(
                        reference_torch, root_sequence_type, form
                    )
                    self.assertEqual(actual, expected)

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

    def test_accumulation_graph_reuse_and_freeing_match_pytorch_2_13(self):
        for root_sequence_type in (None, tuple, list):
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
