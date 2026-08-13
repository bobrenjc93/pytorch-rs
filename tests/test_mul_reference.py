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
class TensorMulReferenceTests(unittest.TestCase):
    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_broadcast_views_empties_and_real_scalars_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])

        calls = (
            (
                "tensor positional",
                actual_left.mul(actual_right),
                expected_left.mul(expected_right),
            ),
            (
                "tensor keyword",
                actual_left.mul(other=actual_right),
                expected_left.mul(other=expected_right),
            ),
            (
                "offset scalar positional",
                actual_left[1].mul(-2.5),
                expected_left[1].mul(-2.5),
            ),
            (
                "offset scalar keyword",
                actual_left[1].mul(other=np.float32(-0.0)),
                expected_left[1].mul(other=np.float32(-0.0)),
            ),
            (
                "numpy integer scalar",
                actual_left.mul(np.int64(3)),
                expected_left.mul(np.int64(3)),
            ),
        )
        for case, actual, expected in calls:
            self.assert_matches(actual, expected, case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        actual_broadcast = torch.ones((1, 1, 2))
        expected_broadcast = reference_torch.ones((1, 1, 2))
        self.assert_matches(
            actual_empty.mul(other=actual_broadcast),
            expected_empty.mul(other=expected_broadcast),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.tensor(values).mul(-0.0),
            reference_torch.tensor(values).mul(-0.0),
            case="signed zero and non-finites",
        )

    def test_autograd_shared_operands_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )
        actual_output = actual_left.transpose(0, 1).mul(
            other=actual_right.transpose(0, 1)
        )
        expected_output = expected_left.transpose(0, 1).mul(
            other=expected_right.transpose(0, 1)
        )
        self.assert_matches(actual_output, expected_output, case="tracked views")
        actual_output.sum().backward()
        expected_output.sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad), expected_left.grad.numpy()
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad), expected_right.grad.numpy()
        )

        actual_shared = torch.tensor([2.0, -3.0], requires_grad=True)
        expected_shared = reference_torch.tensor([2.0, -3.0], requires_grad=True)
        actual_shared.mul(actual_shared).sum().backward()
        expected_shared.mul(expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.mul(other=torch.ones((1, 1, 3))).sum().backward()
        expected_empty.mul(other=reference_torch.ones((1, 1, 3))).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = actual_no_grad.transpose(0, 1).mul(other=2.0)
        with reference_torch.no_grad():
            expected_untracked = expected_no_grad.transpose(0, 1).mul(other=2.0)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(actual_no_grad.mul(2.0).requires_grad)
        self.assertTrue(expected_no_grad.mul(2.0).requires_grad)

    def test_descriptor_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "mul")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "mul")

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "mul")
            self.assertIsNone(descriptor.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(descriptor)
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)

        for bound in (actual.mul, expected.mul):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "mul")
            self.assertIsNone(bound.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(bound)

        self.assert_matches(
            actual_descriptor(actual, other=actual),
            expected_descriptor(expected, other=expected),
            case="unbound call",
        )
        cases = (
            (lambda: actual.mul(), lambda: expected.mul()),
            (lambda: actual.mul(actual, actual), lambda: expected.mul(expected, expected)),
            (
                lambda: actual.mul(actual, other=actual),
                lambda: expected.mul(expected, other=expected),
            ),
            (
                lambda: actual.mul(actual, out=actual),
                lambda: expected.mul(expected, out=expected),
            ),
            (lambda: actual.mul(wat=actual), lambda: expected.mul(wat=expected)),
            (lambda: actual.mul([]), lambda: expected.mul([])),
            (lambda: actual.mul(other=None), lambda: expected.mul(other=None)),
            (
                lambda: actual.mul([], out=actual),
                lambda: expected.mul([], out=expected),
            ),
            (
                lambda: actual.mul(np.uint64(2**63)),
                lambda: expected.mul(np.uint64(2**63)),
            ),
            (lambda: actual.mul(2**64), lambda: expected.mul(2**64)),
            (
                lambda: actual.mul(-(2**63) - 1),
                lambda: expected.mul(-(2**63) - 1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor(1, actual)


if __name__ == "__main__":
    unittest.main()
