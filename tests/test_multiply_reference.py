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
class TensorMultiplyReferenceTests(unittest.TestCase):
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
                actual_left.multiply(actual_right),
                expected_left.multiply(expected_right),
            ),
            (
                "tensor keyword",
                actual_left.multiply(other=actual_right),
                expected_left.multiply(other=expected_right),
            ),
            (
                "tensor x2 keyword",
                actual_left.multiply(x2=actual_right),
                expected_left.multiply(x2=expected_right),
            ),
            (
                "offset scalar positional",
                actual_left[1].multiply(-2.5),
                expected_left[1].multiply(-2.5),
            ),
            (
                "offset scalar keyword",
                actual_left[1].multiply(other=np.float32(-0.0)),
                expected_left[1].multiply(other=np.float32(-0.0)),
            ),
            (
                "offset scalar x2 keyword",
                actual_left[1].multiply(x2=np.float32(-2.5)),
                expected_left[1].multiply(x2=np.float32(-2.5)),
            ),
            (
                "numpy integer scalar",
                actual_left.multiply(np.int64(3)),
                expected_left.multiply(np.int64(3)),
            ),
            (
                "mul x2 keyword",
                actual_left.mul(x2=actual_right),
                expected_left.mul(x2=expected_right),
            ),
        )
        for case, actual, expected in calls:
            self.assert_matches(actual, expected, case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        actual_broadcast = torch.ones((1, 1, 2))
        expected_broadcast = reference_torch.ones((1, 1, 2))
        self.assert_matches(
            actual_empty.multiply(other=actual_broadcast),
            expected_empty.multiply(other=expected_broadcast),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.tensor(values).multiply(-0.0),
            reference_torch.tensor(values).multiply(-0.0),
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
        actual_output = actual_left.transpose(0, 1).multiply(
            other=actual_right.transpose(0, 1)
        )
        expected_output = expected_left.transpose(0, 1).multiply(
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
        actual_shared.multiply(actual_shared).sum().backward()
        expected_shared.multiply(expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.multiply(other=torch.ones((1, 1, 3))).sum().backward()
        expected_empty.multiply(
            other=reference_torch.ones((1, 1, 3))
        ).sum().backward()
        self.assert_matches(actual_empty.grad, expected_empty.grad, case="empty gradient")

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = actual_no_grad.transpose(0, 1).multiply(other=2.0)
        with reference_torch.no_grad():
            expected_untracked = expected_no_grad.transpose(0, 1).multiply(other=2.0)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(actual_no_grad.multiply(2.0).requires_grad)
        self.assertTrue(expected_no_grad.multiply(2.0).requires_grad)

    def test_descriptor_metadata_and_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptor = inspect.getattr_static(torch.Tensor, "multiply")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "multiply")

        for descriptor in (actual_descriptor, expected_descriptor):
            self.assertIs(type(descriptor), types.MethodDescriptorType)
            self.assertEqual(descriptor.__name__, "multiply")
            self.assertIsNone(descriptor.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(descriptor)
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

        for bound in (actual.multiply, expected.multiply):
            self.assertIs(type(bound), types.BuiltinMethodType)
            self.assertEqual(bound.__name__, "multiply")
            self.assertIsNone(bound.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(bound)
        self.assertEqual(actual.multiply.__qualname__, expected.multiply.__qualname__)

        self.assert_matches(
            actual_descriptor(actual, other=actual),
            expected_descriptor(expected, other=expected),
            case="unbound call",
        )
        cases = (
            (lambda: actual.multiply(), lambda: expected.multiply()),
            (
                lambda: actual.multiply(actual, actual),
                lambda: expected.multiply(expected, expected),
            ),
            (
                lambda: actual.multiply(actual, other=actual),
                lambda: expected.multiply(expected, other=expected),
            ),
            (
                lambda: actual.multiply(actual, out=actual),
                lambda: expected.multiply(expected, out=expected),
            ),
            (
                lambda: actual.multiply(other=actual, wat=actual),
                lambda: expected.multiply(other=expected, wat=expected),
            ),
            (
                lambda: actual.multiply(wat=actual),
                lambda: expected.multiply(wat=expected),
            ),
            (lambda: actual.multiply([]), lambda: expected.multiply([])),
            (
                lambda: actual.multiply(other=None),
                lambda: expected.multiply(other=None),
            ),
            (
                lambda: actual.multiply(x2=[]),
                lambda: expected.multiply(x2=[]),
            ),
            (
                lambda: actual.multiply([], out=actual),
                lambda: expected.multiply([], out=expected),
            ),
            (
                lambda: actual.multiply(np.uint64(2**63)),
                lambda: expected.multiply(np.uint64(2**63)),
            ),
            (lambda: actual.multiply(2**64), lambda: expected.multiply(2**64)),
            (
                lambda: actual.multiply(-(2**63) - 1),
                lambda: expected.multiply(-(2**63) - 1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        self.assert_error_matches(
            lambda: actual.mul(x2=actual, wat=actual),
            lambda: expected.mul(x2=expected, wat=expected),
        )

        descriptor_cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(1, actual),
                lambda: expected_descriptor(1, expected),
            ),
            (
                lambda: actual_descriptor(self=actual, other=actual),
                lambda: expected_descriptor(self=expected, other=expected),
            ),
        )
        for actual_call, expected_call in descriptor_cases:
            self.assert_error_matches(actual_call, expected_call)

    def test_invalid_sequence_subclasses_and_keyword_order_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        class NamedList(list):
            pass

        class NamedTuple(tuple):
            pass

        class IterationBombList(list):
            def __iter__(self):
                raise RuntimeError("list iteration must not be invoked")

        class IterationBombTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration must not be invoked")

        class ProtocolList(list):
            def __iter__(self):
                raise RuntimeError("list iteration must not be invoked")

            def __len__(self):
                self.calls.append("len")
                return 1

            def __getitem__(self, index):
                self.calls.append(("getitem", index))
                return 3.5

        class ProtocolTuple(tuple):
            def __iter__(self):
                raise RuntimeError("tuple iteration must not be invoked")

            def __len__(self):
                self.calls.append("len")
                return 1

            def __getitem__(self, index):
                self.calls.append(("getitem", index))
                return 3.5

        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        for value in (
            NamedList([1, "x"]),
            NamedTuple((1, "x")),
            IterationBombList([1, "x"]),
            IterationBombTuple((1, "x")),
        ):
            with self.subTest(sequence=type(value).__name__):
                self.assert_error_matches(
                    lambda value=value: actual.multiply(value),
                    lambda value=value: expected.multiply(value),
                )

        for sequence_type in (ProtocolList, ProtocolTuple):
            actual_value = sequence_type([1, "x"])
            expected_value = sequence_type([1, "x"])
            actual_value.calls = []
            expected_value.calls = []
            with self.subTest(protocol_sequence=sequence_type.__name__):
                self.assert_error_matches(
                    lambda: actual.multiply(actual_value),
                    lambda: expected.multiply(expected_value),
                )
                self.assertEqual(actual_value.calls, expected_value.calls)

        self.assert_error_matches(
            lambda: actual.multiply(a=actual, b=actual, d=actual),
            lambda: expected.multiply(a=expected, b=expected, d=expected),
        )

        actual_keywords = {f"key{index}": actual for index in range(14)}
        expected_keywords = {f"key{index}": expected for index in range(14)}
        self.assert_error_matches(
            lambda: actual.multiply(**actual_keywords),
            lambda: expected.multiply(**expected_keywords),
        )

        actual_keywords = {f"key{index}": actual for index in range(258)}
        expected_keywords = {f"key{index}": expected for index in range(258)}
        self.assert_error_matches(
            lambda: actual.multiply(**actual_keywords),
            lambda: expected.multiply(**expected_keywords),
        )


if __name__ == "__main__":
    unittest.main()
