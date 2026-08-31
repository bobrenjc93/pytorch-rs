import copy
import importlib
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorDivisionMethodReferenceTests(unittest.TestCase):
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
        actual_right = torch.tensor([[2.0], [-0.0], [float("inf")]])
        expected_right = reference_torch.tensor([[2.0], [-0.0], [float("inf")]])

        for name in ("div", "divide"):
            cases = (
                (
                    "tensor positional",
                    getattr(actual_left, name)(actual_right),
                    getattr(expected_left, name)(expected_right),
                ),
                (
                    "tensor keyword",
                    getattr(actual_left, name)(other=actual_right),
                    getattr(expected_left, name)(other=expected_right),
                ),
                (
                    "tensor x2 keyword",
                    getattr(actual_left, name)(x2=actual_right),
                    getattr(expected_left, name)(x2=expected_right),
                ),
                (
                    "explicit true division",
                    getattr(actual_left, name)(actual_right, rounding_mode=None),
                    getattr(expected_left, name)(expected_right, rounding_mode=None),
                ),
                (
                    "offset scalar positional",
                    getattr(actual_left[1], name)(-2.5),
                    getattr(expected_left[1], name)(-2.5),
                ),
                (
                    "offset scalar keyword",
                    getattr(actual_left[1], name)(other=np.float32(-0.0)),
                    getattr(expected_left[1], name)(other=np.float32(-0.0)),
                ),
                (
                    "numpy integer scalar",
                    getattr(actual_left, name)(np.int64(3)),
                    getattr(expected_left, name)(np.int64(3)),
                ),
            )
            for case, actual, expected in cases:
                self.assert_matches(actual, expected, case=(name, case))

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        actual_broadcast = torch.ones((1, 1, 2))
        expected_broadcast = reference_torch.ones((1, 1, 2))
        for name in ("div", "divide"):
            self.assert_matches(
                getattr(actual_empty, name)(actual_broadcast),
                getattr(expected_empty, name)(expected_broadcast),
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
            ),
            dtype=np.uint32,
        )
        actual_special = torch.tensor(memoryview(special_bits.view(np.float32)))
        expected_special = reference_torch.tensor(memoryview(special_bits.view(np.float32)))
        actual_divisors = torch.tensor(
            [1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0]
        )
        expected_divisors = reference_torch.tensor(
            [1.0, -1.0, 0.0, -0.0, float("inf"), float("-inf"), 2.0]
        )
        for name in ("div", "divide"):
            self.assert_matches(
                getattr(actual_special, name)(actual_divisors),
                getattr(expected_special, name)(expected_divisors),
                case=(name, "signed zero nan infinity"),
            )

    def test_no_grad_matches_pytorch_2_13_for_autograd_operands(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for name in ("div", "divide"):
            actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            actual_right = torch.tensor([[5.0], [7.0]], requires_grad=True)
            expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
            expected_right = reference_torch.tensor([[5.0], [7.0]], requires_grad=True)
            with torch.no_grad():
                actual_output = getattr(actual_left.transpose(0, 1), name)(
                    actual_right.transpose(0, 1)
                )
                actual_scalar = getattr(actual_left, name)(2.0)
            with reference_torch.no_grad():
                expected_output = getattr(expected_left.transpose(0, 1), name)(
                    expected_right.transpose(0, 1)
                )
                expected_scalar = getattr(expected_left, name)(2.0)
            self.assert_matches(actual_output, expected_output, case=(name, "no_grad tensor"))
            self.assert_matches(actual_scalar, expected_scalar, case=(name, "no_grad scalar"))

    def test_descriptor_metadata_copy_pickle_and_reload_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_descriptors = {
            name: inspect.getattr_static(torch.Tensor, name)
            for name in ("div", "divide")
        }
        expected_descriptors = {
            name: inspect.getattr_static(reference_torch.Tensor, name)
            for name in ("div", "divide")
        }

        for name in ("div", "divide"):
            actual_descriptor = actual_descriptors[name]
            expected_descriptor = expected_descriptors[name]
            with self.subTest(name=name, descriptor=True):
                self.assertIs(type(actual_descriptor), type(expected_descriptor))
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
                self.assertIs(copy.copy(actual_descriptor), actual_descriptor)
                self.assertIs(copy.deepcopy(actual_descriptor), actual_descriptor)

            actual_bound = getattr(actual, name)
            expected_bound = getattr(expected, name)
            with self.subTest(name=name, bound=True):
                self.assertIs(type(actual_bound), type(expected_bound))
                self.assertEqual(actual_bound.__name__, expected_bound.__name__)
                self.assertEqual(actual_bound.__qualname__, expected_bound.__qualname__)
                self.assertEqual(actual_bound.__doc__, expected_bound.__doc__)
                self.assertIsNone(actual_bound.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(actual_bound)
                self.assertIs(copy.copy(actual_bound), actual_bound)
                self.assertIs(copy.deepcopy(actual_bound), actual_bound)

            self.assert_matches(
                actual_descriptor(actual, other=actual),
                expected_descriptor(expected, other=expected),
                case=(name, "unbound call"),
            )
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(pickle.dumps(actual_descriptor, protocol)),
                        actual_descriptor,
                    )
                    self.assertIs(
                        pickle.loads(pickle.dumps(expected_descriptor, protocol)),
                        expected_descriptor,
                    )

        actual_reloaded = importlib.reload(torch)
        self.assertIs(actual_reloaded, torch)
        for name, descriptor in actual_descriptors.items():
            self.assertIs(inspect.getattr_static(torch.Tensor, name), descriptor)

    def test_supported_argument_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: actual.div(), lambda: expected.div()),
            (lambda: actual.div(actual, actual), lambda: expected.div(expected, expected)),
            (
                lambda: actual.div(actual, other=actual),
                lambda: expected.div(expected, other=expected),
            ),
            (
                lambda: actual.div(actual, dtype=torch.float32),
                lambda: expected.div(expected, dtype=reference_torch.float32),
            ),
            (
                lambda: actual.div(other=None),
                lambda: expected.div(other=None),
            ),
            (lambda: actual.div([]), lambda: expected.div([])),
            (lambda: actual.divide(), lambda: expected.divide()),
            (
                lambda: actual.divide(actual, actual),
                lambda: expected.divide(expected, expected),
            ),
            (
                lambda: actual.divide(actual, other=actual),
                lambda: expected.divide(expected, other=expected),
            ),
        )
        for index, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(index=index):
                self.assert_error_matches(actual_call, expected_call)

        for name in ("div", "divide"):
            self.assertEqual(hasattr(torch.Tensor, f"{name}_"), False)
            self.assertEqual(hasattr(reference_torch.Tensor, f"{name}_"), True)


if __name__ == "__main__":
    unittest.main()
