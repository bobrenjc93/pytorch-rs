import importlib
import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn as nn
import torch_rs.nn.functional as functional


FUNCTION_DOC = """relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
if sys.version_info >= (3, 13):
    FUNCTION_DOC = inspect.cleandoc(FUNCTION_DOC) + "\n"


class FunctionalReluTests(unittest.TestCase):
    def assert_tensor_matches(self, actual, expected):
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(actual).reshape(-1).view(np.uint32),
            np.asarray(expected).reshape(-1).view(np.uint32),
        )

    def test_imports_signature_and_documentation(self):
        imported_nn = importlib.import_module("torch_rs.nn")
        imported_functional = importlib.import_module("torch_rs.nn.functional")
        from torch_rs.nn import functional as from_nn
        from torch_rs.nn.functional import relu

        self.assertIs(torch.nn, nn)
        self.assertIs(nn, imported_nn)
        self.assertIs(nn.functional, functional)
        self.assertIs(functional, imported_functional)
        self.assertIs(from_nn, functional)
        self.assertIs(relu, functional.relu)
        self.assertNotIn("nn", torch.__all__)
        self.assertFalse(hasattr(nn, "__all__"))
        self.assertFalse(hasattr(functional, "__all__"))
        self.assertIsNone(nn.__doc__)
        self.assertEqual(functional.__doc__, "Functional interface.")

        function = functional.relu
        signature = inspect.signature(function)
        parameters = tuple(signature.parameters.values())
        self.assertIs(type(function), types.FunctionType)
        self.assertEqual(function.__name__, "relu")
        self.assertEqual(function.__qualname__, "relu")
        self.assertEqual(function.__module__, "torch_rs.nn.functional")
        self.assertEqual(function.__defaults__, (False,))
        self.assertIsNone(function.__kwdefaults__)
        self.assertFalse(hasattr(function, "__text_signature__"))
        self.assertEqual(function.__doc__, FUNCTION_DOC)
        self.assertEqual(tuple(signature.parameters), ("input", "inplace"))
        self.assertIs(parameters[0].annotation, torch.Tensor)
        self.assertIs(parameters[1].annotation, bool)
        self.assertIs(parameters[1].default, False)
        self.assertIs(signature.return_annotation, torch.Tensor)

    def test_out_of_place_forms_delegate_to_native_relu(self):
        storage = torch.tensor(
            [
                [9.0, 9.0, 9.0, 9.0],
                [-1.0, 2.0, -0.0, 3.0],
                [4.0, -5.0, 6.0, -7.0],
            ]
        )
        offset = storage[1]
        cases = (
            torch.tensor(-0.0),
            torch.zeros((2, 0, 3)).transpose(0, 2)[1],
            offset,
            offset.reshape(2, 2).transpose(0, 1),
        )
        for case, source in enumerate(cases):
            expected = torch.relu(source)
            calls = (
                lambda: functional.relu(source),
                lambda: functional.relu(source, False),
                lambda: functional.relu(input=source, inplace=False),
            )
            for form, call in enumerate(calls):
                with self.subTest(case=case, form=form):
                    actual = call()
                    self.assertIsNot(actual, source)
                    self.assertFalse(actual.is_set_to(source))
                    self.assert_tensor_matches(actual, expected)

    def test_autograd_and_no_grad_reuse_native_behavior(self):
        leaf = torch.tensor(
            [
                [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]],
                [[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]],
            ],
            requires_grad=True,
        )
        source = leaf[1].transpose(0, 1)
        output = functional.relu(source)
        self.assertTrue(output.requires_grad)
        self.assertFalse(output.is_leaf)
        self.assertEqual(output.stride(), (1, 3))
        output.sum().backward()
        self.assertEqual(
            leaf.grad.tolist(),
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            ],
        )

        untracked_leaf = torch.tensor(
            [[-1.0, 2.0], [0.0, 3.0]], requires_grad=True
        )
        with torch.no_grad():
            untracked = functional.relu(untracked_leaf.transpose(0, 1))
        self.assertFalse(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)
        self.assertEqual(untracked.tolist(), [[0.0, 0.0], [2.0, 3.0]])
        self.assertIsNone(untracked_leaf.grad)

    def test_inplace_true_fails_before_mutating_the_input(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        values_before = np.asarray(leaf.detach()).copy().view(np.uint32)
        metadata_before = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
            source.requires_grad,
            source.is_leaf,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch_rs\\.nn\\.functional\\.relu does not support inplace=True$",
        ):
            functional.relu(source, inplace=True)

        np.testing.assert_array_equal(
            np.asarray(leaf.detach()).view(np.uint32), values_before
        )
        self.assertEqual(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.requires_grad,
                source.is_leaf,
            ),
            metadata_before,
        )
        self.assertIsNone(leaf.grad)


if __name__ == "__main__":
    unittest.main()
