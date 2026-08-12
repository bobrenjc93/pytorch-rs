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
class TopLevelDetachReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_detach_matches(self, actual_source, expected_source, *, keyword):
        if keyword:
            actual = torch.detach(input=actual_source)
            expected = reference_torch.detach(input=expected_source)
        else:
            actual = torch.detach(actual_source)
            expected = reference_torch.detach(expected_source)

        self.assertIsNot(actual, actual_source)
        self.assertIsNot(expected, expected_source)
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertIs(actual.dtype, torch.float32)
        self.assertEqual(actual.device, torch.device("cpu"))
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertFalse(actual.requires_grad)
        np.testing.assert_array_equal(np.asarray(actual), expected.numpy())
        self.assertEqual(
            expected.untyped_storage().data_ptr(),
            expected_source.untyped_storage().data_ptr(),
        )

    def test_positional_keyword_layout_and_alias_semantics_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        cases = (
            (
                torch.tensor(3.0, requires_grad=True),
                reference_torch.tensor(3.0, requires_grad=True),
            ),
            (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True),
                reference_torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
                ),
            ),
            (
                (actual_leaf * 2.0).transpose(0, 1)[1],
                (expected_leaf * 2.0).transpose(0, 1)[1],
            ),
            (
                torch.zeros((2, 0, 3), requires_grad=True).transpose(0, 2)[1],
                reference_torch.zeros((2, 0, 3), requires_grad=True)
                .transpose(0, 2)[1],
            ),
        )
        for case, (actual_source, expected_source) in enumerate(cases):
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    self.assert_detach_matches(
                        actual_source, expected_source, keyword=keyword
                    )
            self.assertEqual(actual_source.requires_grad, expected_source.requires_grad)

    def test_autograd_boundary_preserves_source_graph_like_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
            )
            source = (leaf * 3.0).transpose(0, 1)[1]
            detached = module.detach(source)
            detached_loss = (detached * detached).sum()
            source.sum().backward()
            outcomes.append(
                (
                    source.requires_grad,
                    detached.requires_grad,
                    detached_loss.requires_grad,
                    np.asarray(leaf.grad).copy(),
                )
            )

        self.assertEqual(outcomes[0][:3], outcomes[1][:3])
        np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])

    def test_callable_metadata_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.detach
        expected = reference_torch.detach
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertTrue(callable(actual))
        self.assertTrue(callable(expected))
        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)

    def test_binding_and_non_tensor_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.detach(), lambda: reference_torch.detach()),
            (
                lambda: torch.detach(actual, actual),
                lambda: reference_torch.detach(expected, expected),
            ),
            (
                lambda: torch.detach(actual, input=actual),
                lambda: reference_torch.detach(expected, input=expected),
            ),
            (
                lambda: torch.detach(foo=actual),
                lambda: reference_torch.detach(foo=expected),
            ),
            (
                lambda: torch.detach(actual, extra=True),
                lambda: reference_torch.detach(expected, extra=True),
            ),
            (lambda: torch.detach(None), lambda: reference_torch.detach(None)),
            (lambda: torch.detach(input=1), lambda: reference_torch.detach(input=1)),
            (lambda: torch.detach([]), lambda: reference_torch.detach([])),
            (
                lambda: torch.detach(1, extra=True),
                lambda: reference_torch.detach(1, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
