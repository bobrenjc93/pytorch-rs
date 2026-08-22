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
        else:
            raise AssertionError(f"unknown form: {form}")
        return result, np.asarray(leaf.grad).copy()

    def graph_outcome(self, module, sequence_type):
        reusable_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        reusable_loss = reusable_leaf.transpose(0, 0).sum()
        module.autograd.backward(self.roots(reusable_loss, sequence_type))
        module.autograd.backward(self.roots(reusable_loss, sequence_type))

        scalar_leaf = module.tensor(7.0, requires_grad=True)
        module.autograd.backward(self.roots(scalar_leaf, sequence_type))
        module.autograd.backward(self.roots(scalar_leaf, sequence_type))

        freed_leaf = module.tensor([2.0, 3.0], requires_grad=True)
        freed_loss = (freed_leaf * freed_leaf).sum()
        module.autograd.backward(self.roots(freed_loss, sequence_type))
        first_gradient = np.asarray(freed_leaf.grad).copy()
        try:
            module.autograd.backward(self.roots(freed_loss, sequence_type))
        except RuntimeError as error:
            repeated_error = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a value-dependent graph must be freed")
        module.autograd.backward(
            self.roots((freed_leaf * freed_leaf).sum(), sequence_type)
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

    def test_accumulation_graph_reuse_and_freeing_match_pytorch_2_13(self):
        for sequence_type in (None, tuple, list):
            with self.subTest(sequence_type=sequence_type):
                actual = self.graph_outcome(torch, sequence_type)
                expected = self.graph_outcome(reference_torch, sequence_type)
                np.testing.assert_array_equal(actual[0], expected[0])
                self.assertEqual(actual[1], expected[1])
                np.testing.assert_array_equal(actual[2], expected[2])
                self.assertEqual(actual[3], expected[3])
                np.testing.assert_array_equal(actual[4], expected[4])

    def test_native_engine_errors_match_pytorch_2_13(self):
        for sequence_type in (None, tuple, list):
            cases = (
                (
                    lambda: torch.autograd.backward(
                        self.roots(torch.tensor(1.0), sequence_type)
                    ),
                    lambda: reference_torch.autograd.backward(
                        self.roots(
                            reference_torch.tensor(1.0), sequence_type
                        )
                    ),
                ),
                (
                    lambda: torch.autograd.backward(
                        self.roots(
                            torch.tensor(
                                [1.0, 2.0], requires_grad=True
                            ),
                            sequence_type,
                        )
                    ),
                    lambda: reference_torch.autograd.backward(
                        self.roots(
                            reference_torch.tensor(
                                [1.0, 2.0], requires_grad=True
                            ),
                            sequence_type,
                        )
                    ),
                ),
            )
            for case, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(sequence_type=sequence_type, case=case):
                    self.assert_error_matches(actual_call, expected_call)

    def test_graph_option_conversion_errors_match_and_do_not_mutate(self):
        for sequence_type in (None, tuple, list):
            for name, value in (("retain_graph", 0.5), ("create_graph", None)):
                with self.subTest(
                    sequence_type=sequence_type, name=name, value=value
                ):
                    actual_leaf = torch.tensor(2.0, requires_grad=True)
                    actual_loss = actual_leaf * actual_leaf
                    expected_leaf = reference_torch.tensor(
                        2.0, requires_grad=True
                    )
                    expected_loss = expected_leaf * expected_leaf
                    self.assert_error_matches(
                        lambda: torch.autograd.backward(
                            self.roots(actual_loss, sequence_type),
                            **{name: value},
                        ),
                        lambda: reference_torch.autograd.backward(
                            self.roots(expected_loss, sequence_type),
                            **{name: value},
                        ),
                    )
                    self.assertIsNone(actual_leaf.grad)
                    self.assertIsNone(expected_leaf.grad)
                    actual_loss.backward()
                    expected_loss.backward()
                    self.assertEqual(
                        actual_leaf.grad.item(), expected_leaf.grad.item()
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
