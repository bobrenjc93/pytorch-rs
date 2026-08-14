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
class TensorPositiveReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("positive differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_identity_cases(self, module):
        base = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            strided[1],
            strided,
            module.tensor(memoryview(special_bits.view(np.float32))),
        )

    def test_identity_bits_layout_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_identity_cases(torch)
        expected_cases = self.make_identity_cases(reference_torch)

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
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
                actual_pointer = actual.data_ptr()
                expected_pointer = expected.data_ptr()
                actual_detached = actual.detach()
                expected_detached = expected.detach()

                actual_method_result = actual.positive()
                expected_method_result = expected.positive()
                actual_result = +actual
                expected_result = +expected

                self.assertIs(actual_method_result, actual)
                self.assertIs(expected_method_result, expected)
                self.assertIs(actual_result, actual)
                self.assertIs(expected_result, expected)
                self.assertEqual(actual_metadata, expected_metadata)
                self.assertEqual(actual_result.data_ptr(), actual_pointer)
                self.assertEqual(expected_result.data_ptr(), expected_pointer)
                self.assertTrue(actual_result.is_set_to(actual_detached))
                self.assertTrue(expected_result.is_set_to(expected_detached))
                np.testing.assert_array_equal(
                    np.asarray(actual_result).reshape(-1).view(np.uint32),
                    expected_result.numpy().reshape(-1).view(np.uint32),
                )

    def test_leaf_and_non_leaf_autograd_identity_matches_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            leaf_method_result = leaf.positive()
            leaf_result = +leaf
            source = (leaf_result * 3.0).transpose(0, 1)[1]
            graph_before = (
                source.requires_grad,
                source.is_leaf,
                tuple(source.shape),
                source.stride(),
                source.storage_offset(),
            )
            pointer = source.data_ptr()
            method_result = source.positive()
            result = +source
            graph_after = (
                result.requires_grad,
                result.is_leaf,
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
            )
            result.sum().backward()
            outcomes.append(
                (
                    leaf_method_result is leaf,
                    leaf_result is leaf,
                    method_result is source,
                    result is source,
                    result.data_ptr() == pointer,
                    graph_before,
                    graph_after,
                    np.asarray(leaf.grad).copy(),
                )
            )

        self.assertEqual(outcomes[0][:-1], outcomes[1][:-1])
        np.testing.assert_array_equal(outcomes[0][-1], outcomes[1][-1])

    def test_descriptor_documentation_and_signature_match_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "positive")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "positive")
        actual_operator_descriptor = inspect.getattr_static(torch.Tensor, "__pos__")
        expected_operator_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "__pos__"
        )

        self.assertIs(actual_operator_descriptor, actual_descriptor)
        self.assertIs(expected_operator_descriptor, expected_descriptor)
        self.assertIs(torch.Tensor.__dict__["__pos__"], actual_descriptor)
        self.assertIs(
            reference_torch.Tensor.__dict__["__pos__"], expected_descriptor
        )
        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(AttributeError):
                inspect.getattr_static(descriptor.__objclass__, "__pos__")

        for actual, expected, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (
                actual_tensor.positive,
                expected_tensor.positive,
                types.BuiltinMethodType,
            ),
            (
                actual_tensor.__pos__,
                expected_tensor.__pos__,
                types.BuiltinMethodType,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__qualname__, expected.__qualname__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertIs(actual_descriptor(actual_tensor), actual_tensor)
        self.assertIs(expected_descriptor(expected_tensor), expected_tensor)

    def test_invalid_call_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "positive")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "positive")
        actual_bound = actual.positive
        expected_bound = expected.positive
        actual_operator_bound = actual.__pos__
        expected_operator_bound = expected.__pos__
        cases = (
            (lambda: actual.positive(1), lambda: expected.positive(1)),
            (lambda: actual.positive(1, 2), lambda: expected.positive(1, 2)),
            (lambda: actual.positive(dim=0), lambda: expected.positive(dim=0)),
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (
                lambda: actual_bound(unexpected=True),
                lambda: expected_bound(unexpected=True),
            ),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
            (
                lambda: actual_descriptor(self=actual),
                lambda: expected_descriptor(self=expected),
            ),
            (lambda: actual.__pos__(1), lambda: expected.__pos__(1)),
            (lambda: actual.__pos__(1, 2), lambda: expected.__pos__(1, 2)),
            (lambda: actual.__pos__(dim=0), lambda: expected.__pos__(dim=0)),
            (lambda: actual_operator_bound(1), lambda: expected_operator_bound(1)),
            (
                lambda: actual_operator_bound(unexpected=True),
                lambda: expected_operator_bound(unexpected=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
