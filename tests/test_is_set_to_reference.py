import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorIsSetToReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("is_set_to differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_layout_cases(self, module):
        source = module.tensor(
            [
                [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]],
                [[8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]],
                [[16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0]],
            ],
            dtype=module.float32,
        )
        return (
            (source, source),
            (source, source.detach()),
            (source, source.reshape((3, 2, 4))),
            (source, source.transpose(1, 1)),
            (source, source.transpose(0, 2).transpose(0, 2)),
            (source, source.clone()),
            (source, source.reshape((6, 4))),
            (source, source.transpose(0, 2)),
            (source[0], source[0]),
            (source[0], source[1]),
        )

    def test_storage_and_layout_results_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for case, ((actual_left, actual_right), (expected_left, expected_right)) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(actual_left.shape, tuple(expected_left.shape))
                self.assertEqual(actual_left.stride(), expected_left.stride())
                self.assertEqual(
                    actual_left.storage_offset(), expected_left.storage_offset()
                )
                self.assertEqual(actual_right.shape, tuple(expected_right.shape))
                self.assertEqual(actual_right.stride(), expected_right.stride())
                actual = actual_left.is_set_to(actual_right)
                expected = expected_left.is_set_to(expected_right)
                self.assertIs(type(actual), bool)
                self.assertEqual(actual, expected)

    def make_edge_cases(self, module):
        scalar = module.tensor(3.0, dtype=module.float32, requires_grad=True)
        empty = module.zeros(
            (2, 0, 3), dtype=module.float32, requires_grad=True
        )
        empty_base = module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)
        offset_empty = empty_base[1]
        extreme_empty = (
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2)
        )
        return (
            (scalar, scalar.detach()),
            (scalar, scalar.clone()),
            (empty, empty.detach()),
            (empty, empty.clone()),
            (empty, empty.transpose(0, 2)),
            (offset_empty, empty_base[1]),
            (offset_empty, offset_empty.detach()),
            (extreme_empty, extreme_empty.detach()),
            (extreme_empty, extreme_empty.clone()),
        )

    def test_scalar_empty_and_offset_results_match_pytorch_2_13(self):
        actual_cases = self.make_edge_cases(torch)
        expected_cases = self.make_edge_cases(reference_torch)
        for case, ((actual_left, actual_right), (expected_left, expected_right)) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(
                    actual_left.is_set_to(actual_right),
                    expected_left.is_set_to(expected_right),
                )

    def test_autograd_graph_is_unchanged_like_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            tracked = (leaf * 2.0).transpose(0, 1)
            detached = tracked.detach()
            graph_before = (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                tracked.requires_grad,
                tracked.is_leaf,
                detached.requires_grad,
                detached.is_leaf,
            )
            result = tracked.is_set_to(detached)
            graph_after = (
                leaf.requires_grad,
                leaf.is_leaf,
                leaf.grad,
                tracked.requires_grad,
                tracked.is_leaf,
                detached.requires_grad,
                detached.is_leaf,
            )
            tracked.sum().backward()
            outcomes.append((result, graph_before, graph_after, leaf.grad.tolist()))

        self.assertEqual(outcomes[0], outcomes[1])

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "is_set_to")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "is_set_to"
        )

        self.assertFalse(hasattr(torch, "is_set_to"))
        self.assertFalse(hasattr(reference_torch, "is_set_to"))
        for actual, expected, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (
                actual_tensor.is_set_to,
                expected_tensor.is_set_to,
                types.BuiltinMethodType,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            self.assertTrue(callable(actual))
            self.assertTrue(callable(expected))
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertEqual(
            actual_descriptor(actual_tensor, actual_tensor),
            expected_descriptor(expected_tensor, expected_tensor),
        )
        self.assertEqual(
            actual_tensor.is_set_to(tensor=actual_tensor),
            expected_tensor.is_set_to(tensor=expected_tensor),
        )

    def test_binding_and_non_tensor_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: actual.is_set_to(), lambda: expected.is_set_to()),
            (
                lambda: actual.is_set_to(actual, actual),
                lambda: expected.is_set_to(expected, expected),
            ),
            (
                lambda: actual.is_set_to(actual, tensor=actual),
                lambda: expected.is_set_to(expected, tensor=expected),
            ),
            (
                lambda: actual.is_set_to(other=actual),
                lambda: expected.is_set_to(other=expected),
            ),
            (
                lambda: actual.is_set_to(actual, extra=True),
                lambda: expected.is_set_to(expected, extra=True),
            ),
            (lambda: actual.is_set_to(1), lambda: expected.is_set_to(1)),
            (lambda: actual.is_set_to(None), lambda: expected.is_set_to(None)),
            (
                lambda: actual.is_set_to(tensor=[]),
                lambda: expected.is_set_to(tensor=[]),
            ),
            (
                lambda: actual.is_set_to(np.zeros((2, 3), dtype=np.float32)),
                lambda: expected.is_set_to(
                    np.zeros((2, 3), dtype=np.float32)
                ),
            ),
            (
                lambda: actual.is_set_to(**{"tensor": 1, "extra": True}),
                lambda: expected.is_set_to(**{"tensor": 1, "extra": True}),
            ),
            (
                lambda: actual.is_set_to(**{"extra": True, "tensor": 1}),
                lambda: expected.is_set_to(**{"extra": True, "tensor": 1}),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
