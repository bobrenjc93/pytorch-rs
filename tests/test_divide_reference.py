import copy
import inspect
import math
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
class TensorDivideMethodReferenceTests(unittest.TestCase):
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
        actual_right = torch.tensor([[2.0], [4.0], [8.0]])
        expected_right = reference_torch.tensor([[2.0], [4.0], [8.0]])

        for name in ("div", "divide"):
            calls = (
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
                    "rounding none",
                    getattr(actual_left, name)(
                        other=actual_right, rounding_mode=None
                    ),
                    getattr(expected_left, name)(
                        other=expected_right, rounding_mode=None
                    ),
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
                    "offset scalar x2 keyword",
                    getattr(actual_left[1], name)(x2=np.float32(-2.5)),
                    getattr(expected_left[1], name)(x2=np.float32(-2.5)),
                ),
                (
                    "numpy integer scalar",
                    getattr(actual_left, name)(np.int64(4)),
                    getattr(expected_left, name)(np.int64(4)),
                ),
            )
            for case, actual, expected in calls:
                self.assert_matches(actual, expected, case=(name, case))

            actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
            expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
            actual_broadcast = torch.ones((1, 1, 2))
            expected_broadcast = reference_torch.ones((1, 1, 2))
            self.assert_matches(
                getattr(actual_empty, name)(other=actual_broadcast),
                getattr(expected_empty, name)(other=expected_broadcast),
                case=(name, "strided broadcast empty"),
            )

            numerator = torch.tensor(
                [
                    math.nan,
                    math.inf,
                    -math.inf,
                    math.inf,
                    -math.inf,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    0.0,
                    -0.0,
                ]
            )
            expected_numerator = reference_torch.tensor(
                [
                    math.nan,
                    math.inf,
                    -math.inf,
                    math.inf,
                    -math.inf,
                    1.0,
                    -1.0,
                    1.0,
                    -1.0,
                    0.0,
                    -0.0,
                    0.0,
                    -0.0,
                ]
            )
            denominator = torch.tensor(
                [
                    1.0,
                    math.inf,
                    -math.inf,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.0,
                    2.0,
                    2.0,
                    -2.0,
                    -2.0,
                ]
            )
            expected_denominator = reference_torch.tensor(
                [
                    1.0,
                    math.inf,
                    -math.inf,
                    2.0,
                    2.0,
                    0.0,
                    0.0,
                    -0.0,
                    -0.0,
                    2.0,
                    2.0,
                    -2.0,
                    -2.0,
                ]
            )
            self.assert_matches(
                getattr(numerator, name)(denominator),
                getattr(expected_numerator, name)(expected_denominator),
                case=(name, "ieee tensor values"),
            )

    def test_rounding_modes_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_left = torch.tensor(
            [
                math.nan,
                math.inf,
                -math.inf,
                math.inf,
                -math.inf,
                1.0,
                -1.0,
                1.0,
                -1.0,
                0.0,
                -0.0,
                0.0,
                -0.0,
            ]
        )
        expected_left = reference_torch.tensor(
            [
                math.nan,
                math.inf,
                -math.inf,
                math.inf,
                -math.inf,
                1.0,
                -1.0,
                1.0,
                -1.0,
                0.0,
                -0.0,
                0.0,
                -0.0,
            ]
        )
        actual_right = torch.tensor(
            [
                1.0,
                math.inf,
                -math.inf,
                2.0,
                2.0,
                0.0,
                0.0,
                -0.0,
                -0.0,
                2.0,
                2.0,
                -2.0,
                -2.0,
            ]
        )
        expected_right = reference_torch.tensor(
            [
                1.0,
                math.inf,
                -math.inf,
                2.0,
                2.0,
                0.0,
                0.0,
                -0.0,
                -0.0,
                2.0,
                2.0,
                -2.0,
                -2.0,
            ]
        )

        for name in ("div", "divide"):
            for rounding_mode in (
                "floor",
                "trunc",
                b"floor",
                b"trunc",
                np.str_("floor"),
                np.bytes_(b"trunc"),
            ):
                self.assert_matches(
                    getattr(actual_left, name)(
                        actual_right, rounding_mode=rounding_mode
                    ),
                    getattr(expected_left, name)(
                        expected_right, rounding_mode=rounding_mode
                    ),
                    case=(name, "tensor rounding", rounding_mode),
                )

            actual_scalar_left = torch.tensor([1.5, -1.5, 5.0, -5.0])
            expected_scalar_left = reference_torch.tensor([1.5, -1.5, 5.0, -5.0])
            for rounding_mode in ("floor", "trunc"):
                self.assert_matches(
                    getattr(actual_scalar_left, name)(
                        2.0, rounding_mode=rounding_mode
                    ),
                    getattr(expected_scalar_left, name)(
                        2.0, rounding_mode=rounding_mode
                    ),
                    case=(name, "scalar rounding", rounding_mode),
                )

            actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
            expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
            actual_broadcast = torch.ones((1, 1, 2))
            expected_broadcast = reference_torch.ones((1, 1, 2))
            for rounding_mode in ("floor", "trunc"):
                self.assert_matches(
                    getattr(actual_empty, name)(
                        actual_broadcast, rounding_mode=rounding_mode
                    ),
                    getattr(expected_empty, name)(
                        expected_broadcast, rounding_mode=rounding_mode
                    ),
                    case=(name, "empty broadcast rounding", rounding_mode),
                )

    def test_descriptor_metadata_copy_and_pickle_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])

        for name in ("div", "divide"):
            actual_descriptor = inspect.getattr_static(torch.Tensor, name)
            expected_descriptor = inspect.getattr_static(reference_torch.Tensor, name)

            for descriptor in (actual_descriptor, expected_descriptor):
                self.assertIs(type(descriptor), types.MethodDescriptorType)
                self.assertEqual(descriptor.__name__, name)
                self.assertIsNone(descriptor.__text_signature__)
                self.assertFalse(hasattr(descriptor, "__module__"))
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

            for bound in (getattr(actual, name), getattr(expected, name)):
                self.assertIs(type(bound), types.BuiltinMethodType)
                self.assertEqual(bound.__name__, name)
                self.assertIsNone(bound.__module__)
                self.assertIsNone(bound.__text_signature__)
                with self.assertRaises(ValueError):
                    inspect.signature(bound)

            self.assert_matches(
                actual_descriptor(actual, other=actual),
                expected_descriptor(expected, other=expected),
                case=(name, "unbound call"),
            )
            self.assertIs(copy.copy(actual_descriptor), actual_descriptor)
            self.assertIs(copy.deepcopy(actual_descriptor), actual_descriptor)
            owner = actual_descriptor.__reduce__()[1][0]
            self.assertEqual(owner.__name__, "TensorBase")
            self.assertEqual(owner.__module__, "torch._C")
            self.assertIs(getattr(owner, name), actual_descriptor)
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(name=name, protocol=protocol):
                    self.assertIs(
                        pickle.loads(
                            pickle.dumps(actual_descriptor, protocol=protocol)
                        ),
                        actual_descriptor,
                    )

    def test_pytorch_matching_rejections(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])

        for name in ("div", "divide"):
            cases = (
                (
                    lambda name=name: getattr(actual, name)(actual, out=actual),
                    lambda name=name: getattr(expected, name)(expected, out=expected),
                ),
                (
                    lambda name=name: getattr(actual, name)(actual, out=None),
                    lambda name=name: getattr(expected, name)(expected, out=None),
                ),
                (
                    lambda name=name: getattr(actual, name)(actual, dtype=torch.float32),
                    lambda name=name: getattr(expected, name)(
                        expected, dtype=reference_torch.float32
                    ),
                ),
                (
                    lambda name=name: getattr(actual, name)(actual, rounding_mode=1),
                    lambda name=name: getattr(expected, name)(
                        expected, rounding_mode=1
                    ),
                ),
                (
                    lambda name=name: getattr(actual, name)(
                        actual, rounding_mode="bad"
                    ),
                    lambda name=name: getattr(expected, name)(
                        expected, rounding_mode="bad"
                    ),
                ),
                (
                    lambda name=name: getattr(actual, name)(actual, actual),
                    lambda name=name: getattr(expected, name)(expected, expected),
                ),
                (
                    lambda name=name: inspect.getattr_static(torch.Tensor, name)(),
                    lambda name=name: inspect.getattr_static(
                        reference_torch.Tensor, name
                    )(),
                ),
            )
            for actual_call, expected_call in cases:
                with self.subTest(name=name, actual_call=actual_call):
                    self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
