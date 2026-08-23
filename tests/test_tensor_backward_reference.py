import copy
import inspect
import operator
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


class IndexValue:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorBackwardReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.backward differentials require pinned PyTorch 2.13.0"
            )

    def call_supported_form(self, module, loss, form):
        if form == "omitted":
            return loss.backward()
        if form == "gradient positional":
            return loss.backward(None)
        if form == "gradient keyword":
            return loss.backward(gradient=None)
        if form == "all positional none":
            return loss.backward(None, None, False, None)
        if form == "all positional false":
            return loss.backward(None, False, False, None)
        if form == "all keywords":
            return loss.backward(
                gradient=None,
                retain_graph=None,
                create_graph=False,
                inputs=None,
            )
        if form == "integer false":
            return loss.backward(None, IndexValue(0), IndexValue(0), None)
        if form == "unbound keyword self":
            return module.Tensor.backward(
                self=loss,
                gradient=None,
                retain_graph=operator.index(False),
                create_graph=0,
                inputs=None,
            )
        raise AssertionError(f"unknown form: {form}")

    def supported_outcome(self, module, form):
        leaf = module.tensor([2.0, -3.0], requires_grad=True)
        loss = (leaf * leaf).sum()
        result = self.call_supported_form(module, loss, form)
        return result, np.asarray(leaf.grad).copy()

    def test_default_equivalent_calls_match_pytorch_2_13(self):
        for form in (
            "omitted",
            "gradient positional",
            "gradient keyword",
            "all positional none",
            "all positional false",
            "all keywords",
            "integer false",
            "unbound keyword self",
        ):
            with self.subTest(form=form):
                actual_result, actual_gradient = self.supported_outcome(
                    torch, form
                )
                expected_result, expected_gradient = self.supported_outcome(
                    reference_torch, form
                )
                self.assertIsNone(actual_result)
                self.assertIsNone(expected_result)
                np.testing.assert_array_equal(
                    actual_gradient, expected_gradient
                )

    def graph_outcome(self, module):
        reusable_leaf = module.tensor([1.0, 2.0], requires_grad=True)
        reusable_loss = reusable_leaf.transpose(0, 0).sum()
        reusable_loss.backward(retain_graph=False)
        reusable_loss.backward(
            gradient=None,
            retain_graph=None,
            create_graph=False,
            inputs=None,
        )

        scalar_leaf = module.tensor(7.0, requires_grad=True)
        scalar_leaf.backward(None, False, False, None)
        scalar_leaf.backward()

        freed_leaf = module.tensor([2.0, 3.0], requires_grad=True)
        freed_loss = (freed_leaf * freed_leaf).sum()
        freed_loss.backward(gradient=None, retain_graph=None)
        first_gradient = np.asarray(freed_leaf.grad).copy()
        try:
            freed_loss.backward(None, False, False, None)
        except RuntimeError as error:
            repeated_error = (type(error).__name__, str(error), error.args)
        else:
            raise AssertionError("a value-dependent graph must be freed")
        (freed_leaf * freed_leaf).sum().backward(inputs=None)

        return (
            np.asarray(reusable_leaf.grad).copy(),
            scalar_leaf.grad.item(),
            first_gradient,
            repeated_error,
            np.asarray(freed_leaf.grad).copy(),
        )

    def test_accumulation_graph_reuse_and_freeing_match_pytorch_2_13(self):
        actual = self.graph_outcome(torch)
        expected = self.graph_outcome(reference_torch)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        np.testing.assert_array_equal(actual[2], expected[2])
        self.assertEqual(actual[3], expected[3])
        np.testing.assert_array_equal(actual[4], expected[4])

    def error(self, call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error), error.args
        self.fail("Tensor.backward unexpectedly accepted an invalid call")

    def test_native_scalar_errors_match_pytorch_2_13(self):
        cases = (
            lambda module: module.tensor(1.0).backward(
                gradient=None,
                retain_graph=False,
                create_graph=False,
                inputs=None,
            ),
            lambda module: module.tensor(
                [1.0, 2.0], requires_grad=True
            ).backward(None, None, False, None),
        )
        for case, make_call in enumerate(cases):
            with self.subTest(case=case):
                self.assertEqual(
                    self.error(lambda: make_call(torch)),
                    self.error(lambda: make_call(reference_torch)),
                )

    def test_option_conversion_errors_match_and_do_not_mutate(self):
        for name, value in (("retain_graph", 0.5), ("create_graph", None)):
            with self.subTest(name=name):
                actual_leaf = torch.tensor(2.0, requires_grad=True)
                actual_loss = actual_leaf * actual_leaf
                expected_leaf = reference_torch.tensor(
                    2.0, requires_grad=True
                )
                expected_loss = expected_leaf * expected_leaf
                self.assertEqual(
                    self.error(lambda: actual_loss.backward(**{name: value})),
                    self.error(
                        lambda: expected_loss.backward(**{name: value})
                    ),
                )
                self.assertIsNone(actual_leaf.grad)
                self.assertIsNone(expected_leaf.grad)
                actual_loss.backward()
                expected_loss.backward()
                self.assertEqual(
                    actual_leaf.grad.item(), expected_leaf.grad.item()
                )

    def callable_contract(self, module):
        tensor = module.tensor(1.0, requires_grad=True)
        function = inspect.getattr_static(module.Tensor, "backward")
        bound = tensor.backward
        return {
            "function_type": type(function).__name__,
            "bound_type": type(bound).__name__,
            "function_name": function.__name__,
            "function_qualname": function.__qualname__,
            "function_module": function.__module__.replace("torch_rs", "torch"),
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "bound_module": bound.__module__.replace("torch_rs", "torch"),
            "doc": function.__doc__,
            "bound_doc": bound.__doc__,
            "annotations": function.__annotations__,
            "bound_annotations": bound.__annotations__,
            "defaults": function.__defaults__,
            "kwdefaults": function.__kwdefaults__,
            "has_text_signature": hasattr(function, "__text_signature__"),
            "bound_has_text_signature": hasattr(bound, "__text_signature__"),
            "signatures": (
                str(inspect.signature(function)),
                str(inspect.signature(bound)),
            ),
            "owned_by_tensor": "backward" in module.Tensor.__dict__,
            "absent_from_bases": all(
                "backward" not in owner.__dict__
                for owner in module.Tensor.__mro__[1:]
            ),
            "module_tensor_identity": module._tensor.Tensor is module.Tensor,
            "module_function_identity": (
                module._tensor.Tensor.backward is function
            ),
            "copies_are_identical": (
                copy.copy(function) is function,
                copy.deepcopy(function) is function,
                pickle.loads(pickle.dumps(function)) is function,
            ),
            "types_match": (
                type(function) is types.FunctionType,
                type(bound) is types.MethodType,
            ),
        }

    def test_signature_documentation_and_ownership_match_pytorch_2_13(self):
        self.assertEqual(
            self.callable_contract(torch),
            self.callable_contract(reference_torch),
        )

    def binding_errors(self, module):
        leaf = module.tensor(2.0, requires_grad=True)
        loss = leaf * leaf
        function = inspect.getattr_static(module.Tensor, "backward")
        calls = (
            lambda: function(),
            lambda: loss.backward(None, gradient=None),
            lambda: loss.backward(unexpected=True),
            lambda: loss.backward(None, None, False, None, None),
        )
        errors = tuple(self.error(call) for call in calls)
        return errors, leaf.grad

    def test_signature_binding_errors_match_and_do_not_mutate(self):
        actual_errors, actual_grad = self.binding_errors(torch)
        expected_errors, expected_grad = self.binding_errors(reference_torch)
        self.assertEqual(actual_errors, expected_errors)
        self.assertIsNone(actual_grad)
        self.assertIsNone(expected_grad)


if __name__ == "__main__":
    unittest.main()
