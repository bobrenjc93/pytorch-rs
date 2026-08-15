import inspect
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

if __package__:
    from .signature_utils import assert_no_argument_signature
else:
    from signature_utils import assert_no_argument_signature

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorTMethodReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.cpu().detach().numpy()
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_seeded_views_rank_errors_and_warnings_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x7_213)
        shapes = [(), (0,), (1,), (4,), (0, 3), (2, 0), (2, 3)]
        for _ in range(32):
            rank = int(rng.integers(0, 3))
            shapes.append(tuple(int(size) for size in rng.integers(0, 6, rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-3.0, 3.0, elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual = torch.zeros(shape, requires_grad=True)
                expected = reference_torch.zeros(
                    shape, dtype=reference_torch.float32, requires_grad=True
                )
            else:
                data = values.item() if shape == () else values.tolist()
                actual = torch.tensor(data, requires_grad=True)
                expected = reference_torch.tensor(
                    values, dtype=reference_torch.float32, requires_grad=True
                )

            with warnings.catch_warnings(record=True) as actual_warnings:
                warnings.simplefilter("always")
                actual_t = actual.t()
            with warnings.catch_warnings(record=True) as expected_warnings:
                warnings.simplefilter("always")
                expected_t = expected.t()
            self.assertEqual(actual_warnings, [])
            self.assertEqual(expected_warnings, [])
            self.assertIsNot(actual_t, actual)
            self.assertIsNot(expected_t, expected)
            self.assert_matches(actual_t, expected_t, case=(case, "t"))
            self.assert_matches(actual_t.t(), expected_t.t(), case=(case, "t.t"))

        for rank in (3, 4, 65):
            actual = torch.zeros((0,) + (1,) * (rank - 1))
            expected = reference_torch.zeros((0,) + (1,) * (rank - 1))
            self.assert_error_matches(actual.t, expected.t)

    def test_offset_empty_and_autograd_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = actual_leaf.transpose(0, 2)[1].t()
        expected = expected_leaf.transpose(0, 2)[1].t()
        self.assert_matches(actual, expected, case="offset")

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)[1].t()
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)[1].t()
        self.assert_matches(actual_empty, expected_empty, case="empty-offset")

        actual_weights = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_weights = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        )
        (actual * actual_weights).sum().backward()
        (expected * expected_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        actual_scalar = torch.tensor(2.0, requires_grad=True)
        expected_scalar = reference_torch.tensor(2.0, requires_grad=True)
        (actual_scalar.t() * 5.0).backward()
        (expected_scalar.t() * 5.0).backward()
        self.assertEqual(actual_scalar.grad.item(), expected_scalar.grad.item())

    def test_no_argument_descriptor_and_bound_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3))
        expected = reference_torch.zeros((2, 3))
        actual_descriptor = inspect.getattr_static(torch.Tensor, "t")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "t")
        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "t")
            assert_no_argument_signature(self, descriptor, "(self, /)")
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        for bound in (actual.t, expected.t):
            self.assertIs(type(bound), types.BuiltinMethodType)
            assert_no_argument_signature(self, bound, "()")

        self.assert_matches(
            actual_descriptor(actual), expected_descriptor(expected), case="unbound-call"
        )
        actual_bound = actual.t
        expected_bound = expected.t
        for actual_call, expected_call in (
            (lambda: actual_bound(1), lambda: expected_bound(1)),
            (lambda: actual_bound(dim=0), lambda: expected_bound(dim=0)),
            (
                lambda: actual_bound(input=actual),
                lambda: expected_bound(input=expected),
            ),
            (lambda: actual_bound(1, 2), lambda: expected_bound(1, 2)),
        ):
            self.assert_error_matches(actual_call, expected_call)

        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor(1)


if __name__ == "__main__":
    unittest.main()
