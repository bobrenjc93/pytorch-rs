import inspect
import sys
import types
import unittest

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
class UnaryNegationReferenceTests(unittest.TestCase):
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

    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_matches(self, actual, expected, *, case):
        self.assert_metadata_matches(actual, expected, case=case)
        with self.subTest(case=case):
            actual_bits = np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32)
            expected_bits = (
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            )
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_values_layouts_signed_zero_and_non_finites_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        actual_cases = [("scalar", torch.tensor(2.5))]
        expected_cases = [("scalar", reference_torch.tensor(2.5))]
        for shape in ((0,), (1, 0), (0, 1), (1, 0, 1), (2, 0, 3)):
            actual_cases.append((shape, torch.zeros(shape)))
            expected_cases.append((shape, reference_torch.zeros(shape)))

        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        actual_base = torch.tensor(values.tolist())
        expected_base = reference_torch.tensor(values)
        actual_dense_view = actual_base.transpose(0, 2)
        expected_dense_view = expected_base.transpose(0, 2)
        actual_cases.extend(
            (
                ("transposed dense view", actual_dense_view),
                ("offset non-dense view", actual_dense_view[1]),
            )
        )
        expected_cases.extend(
            (
                ("transposed dense view", expected_dense_view),
                ("offset non-dense view", expected_dense_view[1]),
            )
        )

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
        special_values = memoryview(special_bits.view(np.float32))
        actual_cases.append(
            ("signed zeros and non-finites", torch.tensor(special_values))
        )
        expected_cases.append(
            (
                "signed zeros and non-finites",
                reference_torch.tensor(special_values),
            )
        )

        for (actual_case, actual), (expected_case, expected) in zip(
            actual_cases, expected_cases
        ):
            self.assertEqual(actual_case, expected_case)
            for operation, actual_output, expected_output in (
                ("operator", -actual, -expected),
                ("method", actual.neg(), expected.neg()),
                ("negative alias", actual.negative(), expected.negative()),
            ):
                self.assert_matches(
                    actual_output,
                    expected_output,
                    case=(actual_case, operation),
                )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        weights = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual_weights = torch.tensor(weights)
        expected_weights = reference_torch.tensor(weights)

        actual_output = actual_leaf.transpose(0, 1).negative()
        expected_output = expected_leaf.transpose(0, 1).negative()
        self.assert_matches(actual_output, expected_output, case="tracked view")
        (actual_output * actual_weights).sum().backward()
        (expected_output * expected_weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad).view(np.uint32),
            expected_leaf.grad.numpy().view(np.uint32),
        )

        actual_empty = torch.tensor([[]], requires_grad=True)
        expected_empty = reference_torch.tensor([[]], requires_grad=True)
        actual_empty_output = actual_empty.negative()
        expected_empty_output = expected_empty.negative()
        self.assert_matches(
            actual_empty_output,
            expected_empty_output,
            case="tracked empty",
        )
        actual_empty_output.sum().backward()
        expected_empty_output.sum().backward()
        self.assert_matches(
            actual_empty.grad,
            expected_empty.grad,
            case="empty gradient",
        )

        actual_repeated = torch.tensor([2.0, 3.0], requires_grad=True)
        expected_repeated = reference_torch.tensor(
            [2.0, 3.0], requires_grad=True
        )
        actual_repeated_loss = actual_repeated.negative().sum()
        expected_repeated_loss = expected_repeated.negative().sum()
        actual_repeated_loss.backward()
        expected_repeated_loss.backward()
        actual_repeated_loss.backward()
        expected_repeated_loss.backward()
        self.assert_matches(
            actual_repeated.grad,
            expected_repeated.grad,
            case="repeated backward",
        )

        actual_shared = torch.tensor([5.0, 7.0], requires_grad=True)
        expected_shared = reference_torch.tensor(
            [5.0, 7.0], requires_grad=True
        )
        actual_shared_negative = actual_shared.negative()
        expected_shared_negative = expected_shared.negative()
        actual_shared_roots = (
            actual_shared_negative.sum(),
            actual_shared_negative.sum(),
        )
        expected_shared_roots = (
            expected_shared_negative.sum(),
            expected_shared_negative.sum(),
        )
        for actual_root, expected_root in zip(
            actual_shared_roots, expected_shared_roots
        ):
            actual_root.backward()
            expected_root.backward()
        self.assert_matches(
            actual_shared.grad,
            expected_shared.grad,
            case="shared roots",
        )

        nan_bits = np.asarray((0x7FC1_2345, 0xFFC5_4321), dtype=np.uint32)
        actual_nan_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_nan_leaf = reference_torch.tensor(
            [1.0, 2.0], requires_grad=True
        )
        actual_nan_weights = torch.tensor(memoryview(nan_bits.view(np.float32)))
        expected_nan_weights = reference_torch.tensor(
            memoryview(nan_bits.view(np.float32))
        )
        (actual_nan_leaf.negative() * actual_nan_weights).sum().backward()
        (expected_nan_leaf.negative() * expected_nan_weights).sum().backward()
        self.assert_matches(
            actual_nan_leaf.grad,
            expected_nan_leaf.grad,
            case="NaN upstream gradient",
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.transpose(0, 1).negative()
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.transpose(0, 1).negative()
        self.assert_matches(
            actual_untracked,
            expected_untracked,
            case="no_grad view",
        )
        self.assertTrue(actual_leaf.negative().requires_grad)
        self.assertTrue(expected_leaf.negative().requires_grad)

    def test_denormal_flush_mode_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        subnormal_bits = np.asarray((0x0000_0001, 0x8000_0001), dtype=np.uint32)
        values = memoryview(subnormal_bits.view(np.float32))
        actual = torch.tensor(values)
        expected = reference_torch.tensor(values)

        # Stable PyTorch supplies the process-wide FTZ/DAZ switch; construct
        # both inputs first and always restore the default mode for later tests.
        reference_torch.set_flush_denormal(False)
        try:
            self.assertTrue(reference_torch.set_flush_denormal(True))
            actual_output = actual.negative()
            expected_output = expected.negative()
        finally:
            reference_torch.set_flush_denormal(False)

        self.assert_matches(
            actual_output,
            expected_output,
            case="denormal flushing",
        )
        np.testing.assert_array_equal(
            np.asarray(actual_output).view(np.uint32),
            np.asarray((0x8000_0001, 0x0000_0001), dtype=np.uint32),
        )

    def test_extreme_empty_layouts_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for shape in (
            (0, sys.maxsize, 3),
            (0, 2, sys.maxsize, sys.maxsize),
            (0, 1, 2, 1 << 61),
            (2, 0, sys.maxsize),
            (7, 2, 3, 0, 1 << 60),
        ):
            actual = torch.zeros((0,)).reshape(shape)
            expected = reference_torch.zeros((0,)).reshape(shape)
            with self.subTest(shape=shape):
                try:
                    expected_output = expected.negative()
                except Exception as expected_error:
                    with self.assertRaises(type(expected_error)) as actual_raised:
                        actual.negative()
                    self.assertEqual(
                        str(actual_raised.exception), str(expected_error)
                    )
                else:
                    actual_output = actual.negative()
                    self.assert_metadata_matches(
                        actual_output, expected_output, case=shape
                    )
                    self.assertEqual(
                        actual_output.tolist(), expected_output.tolist()
                    )

    def test_method_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "neg")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "neg")

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "neg")
            assert_no_argument_signature(self, descriptor, "(self, /)")
        self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
        self.assertEqual(
            actual_descriptor.__qualname__, expected_descriptor.__qualname__
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )

        for bound in (actual.neg, expected.neg):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "neg")
            assert_no_argument_signature(self, bound, "()")

        self.assert_matches(
            actual_descriptor(actual),
            expected_descriptor(expected),
            case="unbound call",
        )
        for actual_call, expected_call in (
            (lambda: actual.neg(1), lambda: expected.neg(1)),
            (lambda: actual.neg(1, 2), lambda: expected.neg(1, 2)),
            (lambda: actual.neg(dim=0), lambda: expected.neg(dim=0)),
            (lambda: actual.neg(input=actual), lambda: expected.neg(input=expected)),
            (
                lambda: actual_descriptor(actual, 1),
                lambda: expected_descriptor(expected, 1),
            ),
        ):
            self.assert_error_matches(actual_call, expected_call)

        for descriptor in (actual_descriptor, expected_descriptor):
            with self.assertRaises(TypeError):
                descriptor()
            with self.assertRaises(TypeError):
                descriptor(1)

    def test_negative_descriptor_documentation_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "negative")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "negative"
        )
        actual_bound = actual.negative
        expected_bound = expected.negative

        for actual_callable, expected_callable, expected_type in (
            (
                actual_descriptor,
                expected_descriptor,
                types.MethodDescriptorType,
            ),
            (actual_bound, expected_bound, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__qualname__, expected_callable.__qualname__
            )
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            expected_signature = (
                "(self, /)"
                if expected_type is types.MethodDescriptorType
                else "()"
            )
            assert_no_argument_signature(self, actual_callable, expected_signature)
            assert_no_argument_signature(self, expected_callable, expected_signature)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assert_matches(
            actual_descriptor(actual),
            expected_descriptor(expected),
            case="negative unbound call",
        )

        for actual_call, expected_call in (
            (lambda: actual.negative(1), lambda: expected.negative(1)),
            (lambda: actual.negative(1, 2), lambda: expected.negative(1, 2)),
            (lambda: actual.negative(dim=0), lambda: expected.negative(dim=0)),
            (
                lambda: actual.negative(input=actual),
                lambda: expected.negative(input=expected),
            ),
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
        ):
            self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
