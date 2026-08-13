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
class TensorTypeAsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("type_as differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_identity_cases(self, module):
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            module.tensor(
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=module.float32
            ).transpose(0, 1)[1],
        )

    def test_identity_layout_and_storage_semantics_match_pytorch_2_13(self):
        actual_cases = self.make_identity_cases(torch)
        expected_cases = self.make_identity_cases(reference_torch)
        actual_other = torch.tensor([8.0], requires_grad=True)
        expected_other = reference_torch.tensor(
            [8.0], dtype=reference_torch.float32, requires_grad=True
        )

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    actual_metadata = (
                        actual.shape,
                        actual.stride(),
                        actual.storage_offset(),
                        actual.requires_grad,
                        actual.is_leaf,
                    )
                    expected_metadata = (
                        tuple(expected.shape),
                        expected.stride(),
                        expected.storage_offset(),
                        expected.requires_grad,
                        expected.is_leaf,
                    )
                    if keyword:
                        actual_result = actual.type_as(other=actual_other)
                        expected_result = expected.type_as(other=expected_other)
                    else:
                        actual_result = actual.type_as(actual_other)
                        expected_result = expected.type_as(expected_other)

                    self.assertIs(actual_result, actual)
                    self.assertIs(expected_result, expected)
                    self.assertEqual(actual_metadata, expected_metadata)
                    self.assertEqual(
                        (
                            actual_result.shape,
                            actual_result.stride(),
                            actual_result.storage_offset(),
                            actual_result.requires_grad,
                            actual_result.is_leaf,
                        ),
                        actual_metadata,
                    )
                    self.assertTrue(actual_result.is_set_to(actual.detach()))
                    self.assertEqual(
                        expected_result.untyped_storage().data_ptr(),
                        expected.untyped_storage().data_ptr(),
                    )
                    np.testing.assert_array_equal(
                        np.asarray(actual_result), expected_result.numpy()
                    )

    def test_autograd_graph_identity_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = (leaf * 3.0).transpose(0, 1)[1]
            other = module.zeros((2,), dtype=module.float32, requires_grad=True)
            graph_before = (
                source.requires_grad,
                source.is_leaf,
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                other.grad,
            )
            result = source.type_as(other=other)
            graph_after = (
                source.requires_grad,
                source.is_leaf,
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                other.grad,
            )
            result.sum().backward()
            outcomes.append(
                (
                    result is source,
                    graph_before,
                    graph_after,
                    np.asarray(leaf.grad).copy(),
                    other.grad,
                )
            )

        self.assertEqual(outcomes[0][:3], outcomes[1][:3])
        np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])
        self.assertIsNone(outcomes[0][4])
        self.assertIsNone(outcomes[1][4])

    def test_descriptor_documentation_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor(
            [1.0], dtype=reference_torch.float32
        )
        actual_other = torch.tensor([2.0])
        expected_other = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32
        )
        actual_descriptor = inspect.getattr_static(torch.Tensor, "type_as")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "type_as"
        )

        for actual, expected, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (
                actual_tensor.type_as,
                expected_tensor.type_as,
                types.BuiltinMethodType,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertIs(actual_descriptor(actual_tensor, actual_other), actual_tensor)
        self.assertIs(
            expected_descriptor(expected_tensor, expected_other), expected_tensor
        )

    def test_binding_and_type_error_precedence_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_other = torch.tensor([2.0])
        expected_other = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32
        )
        actual_descriptor = inspect.getattr_static(torch.Tensor, "type_as")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "type_as"
        )
        array = np.zeros((2, 3), dtype=np.float32)
        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(1, actual_other),
                lambda: expected_descriptor(1, expected_other),
            ),
            (lambda: actual.type_as(), lambda: expected.type_as()),
            (
                lambda: actual.type_as(actual_other, actual_other),
                lambda: expected.type_as(expected_other, expected_other),
            ),
            (
                lambda: actual.type_as(actual_other, other=actual_other),
                lambda: expected.type_as(expected_other, other=expected_other),
            ),
            (
                lambda: actual.type_as(foo=actual_other),
                lambda: expected.type_as(foo=expected_other),
            ),
            (
                lambda: actual.type_as(actual_other, extra=True),
                lambda: expected.type_as(expected_other, extra=True),
            ),
            (lambda: actual.type_as(1), lambda: expected.type_as(1)),
            (lambda: actual.type_as(None), lambda: expected.type_as(None)),
            (lambda: actual.type_as([]), lambda: expected.type_as([])),
            (lambda: actual.type_as(array), lambda: expected.type_as(array)),
            (
                lambda: actual.type_as(other=1),
                lambda: expected.type_as(other=1),
            ),
            (
                lambda: actual.type_as(other=None),
                lambda: expected.type_as(other=None),
            ),
            (
                lambda: actual.type_as(other=[]),
                lambda: expected.type_as(other=[]),
            ),
            (
                lambda: actual.type_as(**{"other": 1, "extra": True}),
                lambda: expected.type_as(**{"other": 1, "extra": True}),
            ),
            (
                lambda: actual.type_as(**{"extra": True, "other": 1}),
                lambda: expected.type_as(**{"extra": True, "other": 1}),
            ),
            (
                lambda: actual.type_as(1, other=actual_other),
                lambda: expected.type_as(1, other=expected_other),
            ),
            (
                lambda: actual.type_as(1, extra=True),
                lambda: expected.type_as(1, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
