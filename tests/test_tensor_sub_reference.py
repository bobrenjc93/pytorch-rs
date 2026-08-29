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
class TensorSubtractionMethodReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_supported_tensor_scalar_layouts_and_empty_outputs_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [3.0], [4.0]])
        expected_right = reference_torch.tensor([[2.0], [3.0], [4.0]])

        cases = (
            (
                "sub tensor positional",
                actual_left.sub(actual_right),
                expected_left.sub(expected_right),
            ),
            (
                "sub tensor keyword",
                actual_left.sub(other=actual_right, alpha=1),
                expected_left.sub(other=expected_right, alpha=1),
            ),
            (
                "sub tensor x2 keyword",
                actual_left.sub(x2=actual_right, alpha=np.float32(1.0)),
                expected_left.sub(x2=expected_right, alpha=np.float32(1.0)),
            ),
            (
                "subtract tensor positional",
                actual_left.subtract(actual_right),
                expected_left.subtract(expected_right),
            ),
            (
                "subtract tensor keyword",
                actual_left.subtract(other=actual_right, alpha=1.0),
                expected_left.subtract(other=expected_right, alpha=1.0),
            ),
            (
                "subtract tensor x2 keyword",
                actual_left.subtract(x2=actual_right, alpha=np.uint64(1)),
                expected_left.subtract(x2=expected_right, alpha=np.uint64(1)),
            ),
            (
                "sub scalar",
                actual_left[1].sub(np.float32(-0.0)),
                expected_left[1].sub(np.float32(-0.0)),
            ),
            (
                "subtract scalar",
                actual_left[1].subtract(other=np.bool_(False)),
                expected_left[1].subtract(other=np.bool_(False)),
            ),
        )
        for case, actual, expected in cases:
            self.assert_matches(actual, expected, case=case)

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        actual_broadcast = torch.ones((1, 1, 2))
        expected_broadcast = reference_torch.ones((1, 1, 2))
        self.assert_matches(
            actual_empty.sub(actual_broadcast),
            expected_empty.sub(expected_broadcast),
            case="strided broadcast empty",
        )

        special_bits = np.asarray(
            (0x7FC1_2345, 0x7F80_0000, 0xFF80_0000, 0x0000_0000, 0x8000_0000),
            dtype=np.uint32,
        )
        actual_values = memoryview(special_bits.view(np.float32))
        expected_values = memoryview(special_bits.view(np.float32))
        self.assert_matches(
            torch.tensor(actual_values).subtract(-0.0),
            reference_torch.tensor(expected_values).subtract(-0.0),
            case="signed zero and non-finites",
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
        expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
        actual_right = torch.tensor([[5.0], [7.0], [11.0]], requires_grad=True)
        expected_right = reference_torch.tensor(
            [[5.0], [7.0], [11.0]], requires_grad=True
        )

        actual_output = actual_left.transpose(0, 1).sub(
            other=actual_right.transpose(0, 1)
        )
        expected_output = expected_left.transpose(0, 1).sub(
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
        actual_shared.subtract(actual_shared).sum().backward()
        expected_shared.subtract(expected_shared).sum().backward()
        self.assert_matches(
            actual_shared.grad, expected_shared.grad, case="shared operand gradient"
        )

        actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
        expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
        with torch.no_grad():
            actual_untracked = actual_no_grad.transpose(0, 1).subtract(2.0)
        with reference_torch.no_grad():
            expected_untracked = expected_no_grad.transpose(0, 1).subtract(2.0)
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad view")
        self.assertTrue(actual_no_grad.sub(2.0).requires_grad)
        self.assertTrue(expected_no_grad.sub(2.0).requires_grad)

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])

        for name in ("sub", "subtract"):
            with self.subTest(name=name):
                actual_descriptor = inspect.getattr_static(torch.Tensor, name)
                expected_descriptor = inspect.getattr_static(reference_torch.Tensor, name)
                self.assertIs(type(actual_descriptor), types.MethodDescriptorType)
                self.assertEqual(type(actual_descriptor), type(expected_descriptor))
                self.assertEqual(actual_descriptor.__name__, expected_descriptor.__name__)
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
                self.assertEqual(actual_descriptor.__doc__, expected_descriptor.__doc__)
                self.assertIsNone(actual_descriptor.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(actual_descriptor)

                actual_bound = getattr(actual, name)
                expected_bound = getattr(expected, name)
                self.assertIs(type(actual_bound), types.BuiltinMethodType)
                self.assertEqual(type(actual_bound), type(expected_bound))
                self.assertEqual(actual_bound.__name__, expected_bound.__name__)
                self.assertEqual(actual_bound.__qualname__, expected_bound.__qualname__)
                self.assertIsNone(actual_bound.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(actual_bound)

                self.assert_matches(
                    actual_descriptor(actual, other=actual),
                    expected_descriptor(expected, other=expected),
                    case=("unbound", name),
                )

    def test_torch_function_dispatch_matches_pytorch_2_13_for_supported_schema(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        marker = object()

        class ActualRecordingMode(torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        class ExpectedRecordingMode(reference_torch.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for name in ("sub", "subtract"):
            actual = torch.tensor([2.0])
            expected = reference_torch.tensor([2.0])
            actual_other = torch.tensor([1.0])
            expected_other = reference_torch.tensor([1.0])
            actual_descriptor = inspect.getattr_static(torch.Tensor, name)
            expected_descriptor = inspect.getattr_static(reference_torch.Tensor, name)

            for actual_call, expected_call in (
                (
                    lambda: getattr(actual, name)(actual_other),
                    lambda: getattr(expected, name)(expected_other),
                ),
                (
                    lambda: getattr(actual, name)(other=actual_other, alpha=1),
                    lambda: getattr(expected, name)(other=expected_other, alpha=1),
                ),
                (
                    lambda: getattr(actual, name)(1.0),
                    lambda: getattr(expected, name)(1.0),
                ),
            ):
                with self.subTest(name=name, call=actual_call):
                    actual_mode = ActualRecordingMode()
                    expected_mode = ExpectedRecordingMode()
                    with actual_mode:
                        self.assertIs(actual_call(), marker)
                    with expected_mode:
                        self.assertIs(expected_call(), marker)
                    actual_record = actual_mode.calls[0]
                    expected_record = expected_mode.calls[0]
                    self.assertIs(actual_record[0], actual_descriptor)
                    self.assertIs(expected_record[0], expected_descriptor)
                    self.assertEqual(actual_record[1], expected_record[1])
                    self.assertEqual(len(actual_record[2]), len(expected_record[2]))
                    self.assertEqual(actual_record[3] is None, expected_record[3] is None)
                    if actual_record[3] is not None:
                        self.assertEqual(tuple(actual_record[3]), tuple(expected_record[3]))


if __name__ == "__main__":
    unittest.main()
