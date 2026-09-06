import copy
import inspect
import pickle
import re
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


MATCHING_DENSE_LEFT_BITS = np.asarray(
    (
        0x0000_0000,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC1_2345,
        0x3F80_0000,
        0x7F81_2345,
        0xFF81_2345,
    ),
    dtype=np.uint32,
)
MATCHING_DENSE_RIGHT_BITS = np.asarray(
    (
        0x8000_0000,
        0x0000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FCA_BCDE,
        0x7FCA_BCDE,
        0xFF85_6789,
        0x7F85_6789,
    ),
    dtype=np.uint32,
)


def matching_dense_view(module, bits, *, offset=False, requires_grad=False):
    storage_bits = bits
    if offset:
        storage_bits = np.concatenate([np.zeros(bits.size, dtype=np.uint32), bits])
    tensor = module.tensor(
        memoryview(storage_bits.view(np.float32)),
        requires_grad=requires_grad,
    )
    if offset:
        return tensor.reshape((2, 2, 4))[1].transpose(0, 1)
    return tensor.reshape((2, 4)).transpose(0, 1)


def tensor_bits(tensor):
    return np.asarray(tensor).reshape(-1).view(np.uint32).copy()


def tensor_from_bits(module, bits):
    return module.tensor(memoryview(np.asarray(bits, dtype=np.uint32).view(np.float32)))


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorSubMethodReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "Tensor.sub differentials require pinned PyTorch 2.13.0"
            )

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

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_values_layouts_ieee_empties_and_argument_forms_match_pytorch_2_13(self):
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
                "positional tensors",
                lambda name: getattr(actual_left, name)(actual_right),
                lambda name: getattr(expected_left, name)(expected_right),
            ),
            (
                "other keyword",
                lambda name: getattr(actual_left, name)(other=actual_right),
                lambda name: getattr(expected_left, name)(other=expected_right),
            ),
            (
                "x2 alias",
                lambda name: getattr(actual_left, name)(x2=actual_right),
                lambda name: getattr(expected_left, name)(x2=expected_right),
            ),
            (
                "default alpha",
                lambda name: getattr(actual_left, name)(actual_right, alpha=1),
                lambda name: getattr(expected_left, name)(expected_right, alpha=1),
            ),
            (
                "default numpy alpha",
                lambda name: getattr(actual_left, name)(
                    actual_right, alpha=np.int64(1)
                ),
                lambda name: getattr(expected_left, name)(
                    expected_right, alpha=np.int64(1)
                ),
            ),
        )
        for name in ("sub", "subtract"):
            for case, actual_call, expected_call in calls:
                self.assert_matches(
                    actual_call(name), expected_call(name), case=(name, case)
                )

        actual_offset = actual_left[1]
        expected_offset = expected_left[1]
        for scalar in (-2, 2.5, np.bool_(True), np.int64(3), np.float32(-0.0)):
            for name in ("sub", "subtract"):
                self.assert_matches(
                    getattr(actual_offset, name)(scalar),
                    getattr(expected_offset, name)(scalar),
                    case=(name, "offset scalar", type(scalar).__name__, scalar),
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    expected_positional = getattr(expected_offset, name)(scalar, 1)
                self.assert_matches(
                    getattr(actual_offset, name)(scalar, 1),
                    expected_positional,
                    case=(
                        name,
                        "offset scalar positional alpha",
                        type(scalar).__name__,
                        scalar,
                    ),
                )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_legacy = expected_left.sub(1, expected_right)
        self.assert_matches(
            actual_left.sub(1, actual_right),
            expected_legacy,
            case=("sub", "legacy positional alpha tensor other"),
        )

        actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
        expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
        for name in ("sub", "subtract"):
            self.assert_matches(
                getattr(actual_empty, name)(torch.ones((1, 1, 2))),
                getattr(expected_empty, name)(reference_torch.ones((1, 1, 2))),
                case=(name, "strided broadcast empty"),
            )

        special_bits = np.asarray(
            (0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC1_2345),
            dtype=np.uint32,
        )
        values = memoryview(special_bits.view(np.float32))
        for name in ("sub", "subtract"):
            self.assert_matches(
                getattr(torch.tensor(values), name)(torch.zeros((5,))),
                getattr(reference_torch.tensor(values), name)(
                    reference_torch.zeros((5,))
                ),
                case=(name, "signed zero and non-finites"),
            )

    def test_numeric_alpha_forms_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[[1.0, -0.0], [float("inf"), -4.0], [5.0, float("nan")]]]
        ).transpose(0, 2)
        expected_left = reference_torch.tensor(
            [[[1.0, -0.0], [float("inf"), -4.0], [5.0, float("nan")]]]
        ).transpose(0, 2)
        actual_right = torch.tensor([[2.0], [-0.0], [float("-inf")]])
        expected_right = reference_torch.tensor([[2.0], [-0.0], [float("-inf")]])

        for name in ("sub", "subtract"):
            actual_method = getattr(actual_left, name)
            expected_method = getattr(expected_left, name)
            actual_left_before = tensor_bits(actual_left)
            actual_right_before = tensor_bits(actual_right)
            actual_fused_left = tensor_from_bits(torch, [0xD032_7A78])
            expected_fused_left = tensor_from_bits(reference_torch, [0xD032_7A78])
            actual_fused_right = tensor_from_bits(torch, [0xD5F4_4919])
            expected_fused_right = tensor_from_bits(reference_torch, [0xD5F4_4919])
            fused_scalar = np.asarray([0xD5F4_4919], dtype=np.uint32).view(
                np.float32
            )[0]
            actual_integer_zero_left = tensor_from_bits(
                torch, [0x8000_0000, 0x8000_0000]
            )
            expected_integer_zero_left = tensor_from_bits(
                reference_torch, [0x8000_0000, 0x8000_0000]
            )
            actual_integer_zero_right = tensor_from_bits(
                torch, [0x0000_0000, 0x8000_0000]
            )
            expected_integer_zero_right = tensor_from_bits(
                reference_torch, [0x0000_0000, 0x8000_0000]
            )
            alpha_nan = np.asarray([0x7F8A_BCDE], dtype=np.uint32).view(np.float32)[0]
            actual_nan_left = tensor_from_bits(
                torch, [0x3F80_0000, 0x7FC1_2345]
            )
            expected_nan_left = tensor_from_bits(
                reference_torch, [0x3F80_0000, 0x7FC1_2345]
            )
            actual_nan_right = tensor_from_bits(
                torch, [0x4000_0000, 0x4000_0000]
            )
            expected_nan_right = tensor_from_bits(
                reference_torch, [0x4000_0000, 0x4000_0000]
            )
            actual_right_nan = tensor_from_bits(torch, [0x7F81_2345])
            expected_right_nan = tensor_from_bits(reference_torch, [0x7F81_2345])
            cases = (
                (
                    "keyword tensor alpha",
                    lambda actual_method=actual_method: actual_method(
                        actual_right, alpha=np.float32(2.0)
                    ),
                    lambda expected_method=expected_method: expected_method(
                        expected_right, alpha=np.float32(2.0)
                    ),
                ),
                (
                    "tensor scalar alpha",
                    lambda name=name: getattr(actual_left[1], name)(
                        np.float32(-0.0), alpha=2.0
                    ),
                    lambda name=name: getattr(expected_left[1], name)(
                        np.float32(-0.0), alpha=2.0
                    ),
                ),
                (
                    "positional scalar alpha",
                    lambda name=name: getattr(actual_left[1], name)(2.0, 3.0),
                    lambda name=name: getattr(expected_left[1], name)(2.0, 3.0),
                ),
                (
                    "fractional fused alpha",
                    lambda name=name: getattr(actual_fused_left, name)(
                        actual_fused_right, alpha=np.float32(0.1)
                    ),
                    lambda name=name: getattr(expected_fused_left, name)(
                        expected_fused_right, alpha=np.float32(0.1)
                    ),
                ),
                (
                    "fractional fused scalar other",
                    lambda name=name: getattr(actual_fused_left, name)(
                        fused_scalar, alpha=np.float32(0.1)
                    ),
                    lambda name=name: getattr(expected_fused_left, name)(
                        fused_scalar, alpha=np.float32(0.1)
                    ),
                ),
                (
                    "nonfinite alpha nan precedence",
                    lambda name=name: getattr(actual_nan_left, name)(
                        actual_nan_right, alpha=alpha_nan
                    ),
                    lambda name=name: getattr(expected_nan_left, name)(
                        expected_nan_right, alpha=alpha_nan
                    ),
                ),
                (
                    "finite alpha right nan precedence",
                    lambda name=name: getattr(
                        tensor_from_bits(torch, [0x3F80_0000]), name
                    )(actual_right_nan, alpha=np.float32(0.1)),
                    lambda name=name: getattr(
                        tensor_from_bits(reference_torch, [0x3F80_0000]), name
                    )(expected_right_nan, alpha=np.float32(0.1)),
                ),
                (
                    "nan alpha right nan precedence",
                    lambda name=name: getattr(
                        tensor_from_bits(torch, [0x3F80_0000]), name
                    )(actual_right_nan, alpha=alpha_nan),
                    lambda name=name: getattr(
                        tensor_from_bits(reference_torch, [0x3F80_0000]), name
                    )(expected_right_nan, alpha=alpha_nan),
                ),
                (
                    "nan alpha scalar right nan precedence",
                    lambda name=name: getattr(
                        tensor_from_bits(torch, [0x3F80_0000]), name
                    )(
                        np.asarray([0x7F81_2345], dtype=np.uint32).view(np.float32)[
                            0
                        ],
                        alpha=alpha_nan,
                    ),
                    lambda name=name: getattr(
                        tensor_from_bits(reference_torch, [0x3F80_0000]), name
                    )(
                        np.asarray([0x7F81_2345], dtype=np.uint32).view(np.float32)[
                            0
                        ],
                        alpha=alpha_nan,
                    ),
                ),
                (
                    "nonfinite alpha scalar other",
                    lambda name=name: getattr(
                        tensor_from_bits(torch, [0x3F80_0000]), name
                    )(np.float32(2.0), alpha=alpha_nan),
                    lambda name=name: getattr(
                        tensor_from_bits(reference_torch, [0x3F80_0000]), name
                    )(np.float32(2.0), alpha=alpha_nan),
                ),
            )
            for case, actual_call, expected_call in cases:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    expected = expected_call()
                self.assert_matches(actual_call(), expected, case=(name, case))
            for integer_zero_alpha in (0, np.int64(0), np.uint64(0)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    expected_tensor = getattr(expected_integer_zero_left, name)(
                        expected_integer_zero_right,
                        alpha=integer_zero_alpha,
                    )
                    expected_tensor_scalar = getattr(
                        tensor_from_bits(reference_torch, [0x8000_0000]), name
                    )(
                        np.float32(0.0),
                        alpha=integer_zero_alpha,
                    )
                self.assert_matches(
                    getattr(actual_integer_zero_left, name)(
                        actual_integer_zero_right,
                        alpha=integer_zero_alpha,
                    ),
                    expected_tensor,
                    case=(name, "integer zero tensor", type(integer_zero_alpha).__name__),
                )
                self.assert_matches(
                    getattr(tensor_from_bits(torch, [0x8000_0000]), name)(
                        np.float32(0.0),
                        alpha=integer_zero_alpha,
                    ),
                    expected_tensor_scalar,
                    case=(name, "integer zero scalar", type(integer_zero_alpha).__name__),
                )
            np.testing.assert_array_equal(tensor_bits(actual_left), actual_left_before)
            np.testing.assert_array_equal(tensor_bits(actual_right), actual_right_before)

            actual_grad_left = torch.tensor([[1.0, 2.0]], requires_grad=True)
            expected_grad_left = reference_torch.tensor(
                [[1.0, 2.0]], requires_grad=True
            )
            actual_grad_right = torch.tensor(
                [[3.0], [4.0], [5.0]], requires_grad=True
            )
            expected_grad_right = reference_torch.tensor(
                [[3.0], [4.0], [5.0]], requires_grad=True
            )
            getattr(actual_grad_left.transpose(0, 1), name)(
                actual_grad_right.transpose(0, 1),
                alpha=2.5,
            ).sum().backward()
            getattr(expected_grad_left.transpose(0, 1), name)(
                expected_grad_right.transpose(0, 1),
                alpha=2.5,
            ).sum().backward()
            np.testing.assert_array_equal(
                np.asarray(actual_grad_left.grad),
                expected_grad_left.grad.numpy(),
            )
            np.testing.assert_array_equal(
                np.asarray(actual_grad_right.grad),
                expected_grad_right.grad.numpy(),
            )

            actual_tensor_scalar_grad = torch.tensor([2.0, -3.0], requires_grad=True)
            expected_tensor_scalar_grad = reference_torch.tensor(
                [2.0, -3.0], requires_grad=True
            )
            getattr(actual_tensor_scalar_grad, name)(4.0, alpha=2.5).sum().backward()
            getattr(expected_tensor_scalar_grad, name)(4.0, alpha=2.5).sum().backward()
            np.testing.assert_array_equal(
                np.asarray(actual_tensor_scalar_grad.grad),
                expected_tensor_scalar_grad.grad.numpy(),
            )

            actual_nan_grad_left = torch.tensor([1.0], requires_grad=True)
            expected_nan_grad_left = reference_torch.tensor([1.0], requires_grad=True)
            actual_nan_grad_right = torch.tensor([2.0], requires_grad=True)
            expected_nan_grad_right = reference_torch.tensor([2.0], requires_grad=True)
            getattr(actual_nan_grad_left, name)(
                actual_nan_grad_right, alpha=alpha_nan
            ).sum().backward()
            getattr(expected_nan_grad_left, name)(
                expected_nan_grad_right, alpha=alpha_nan
            ).sum().backward()
            np.testing.assert_array_equal(
                tensor_bits(actual_nan_grad_right.grad),
                expected_nan_grad_right.grad.detach().numpy().view(np.uint32),
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_legacy = expected_left.sub(2.0, expected_right)
        self.assert_matches(
            actual_left.sub(2.0, actual_right),
            expected_legacy,
            case=("sub", "legacy positional tensor alpha"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_keyword_other = expected_left.sub(2.0, other=expected_right)
        self.assert_matches(
            actual_left.sub(2.0, other=actual_right),
            expected_keyword_other,
            case=("sub", "legacy keyword other positional alpha"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected_keyword_x2 = expected_left.sub(2.0, x2=expected_right)
        self.assert_matches(
            actual_left.sub(2.0, x2=actual_right),
            expected_keyword_x2,
            case=("sub", "legacy keyword x2 positional alpha"),
        )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        for name in ("sub", "subtract"):
            actual_left = torch.tensor([[2.0, 3.0]], requires_grad=True)
            expected_left = reference_torch.tensor([[2.0, 3.0]], requires_grad=True)
            actual_right = torch.tensor(
                [[5.0], [7.0], [11.0]], requires_grad=True
            )
            expected_right = reference_torch.tensor(
                [[5.0], [7.0], [11.0]], requires_grad=True
            )
            actual_output = getattr(actual_left.transpose(0, 1), name)(
                actual_right.transpose(0, 1)
            )
            expected_output = getattr(expected_left.transpose(0, 1), name)(
                expected_right.transpose(0, 1)
            )
            self.assert_matches(actual_output, expected_output, case=(name, "views"))
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
            getattr(actual_shared, name)(actual_shared).sum().backward()
            getattr(expected_shared, name)(expected_shared).sum().backward()
            self.assert_matches(
                actual_shared.grad,
                expected_shared.grad,
                case=(name, "shared operand gradient"),
            )

            actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
            expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
            getattr(actual_empty, name)(torch.ones((1, 1, 3))).sum().backward()
            getattr(expected_empty, name)(reference_torch.ones((1, 1, 3))).sum().backward()
            self.assert_matches(
                actual_empty.grad, expected_empty.grad, case=(name, "empty gradient")
            )

            actual_no_grad = torch.tensor([[1.0, 2.0]], requires_grad=True)
            expected_no_grad = reference_torch.tensor([[1.0, 2.0]], requires_grad=True)
            with torch.no_grad():
                actual_untracked = getattr(actual_no_grad.transpose(0, 1), name)(2.0)
            with reference_torch.no_grad():
                expected_untracked = getattr(expected_no_grad.transpose(0, 1), name)(
                    2.0
                )
            self.assert_matches(
                actual_untracked, expected_untracked, case=(name, "no_grad view")
            )

    def test_same_shape_matching_dense_views_match_pytorch_2_13(self):
        for name in ("sub", "subtract"):
            for offset in (False, True):
                actual_left = matching_dense_view(
                    torch, MATCHING_DENSE_LEFT_BITS, offset=offset
                )
                actual_right = matching_dense_view(
                    torch, MATCHING_DENSE_RIGHT_BITS, offset=offset
                )
                expected_left = matching_dense_view(
                    reference_torch, MATCHING_DENSE_LEFT_BITS, offset=offset
                )
                expected_right = matching_dense_view(
                    reference_torch, MATCHING_DENSE_RIGHT_BITS, offset=offset
                )
                actual_left_before = tensor_bits(actual_left)
                actual_right_before = tensor_bits(actual_right)

                self.assert_matches(
                    getattr(actual_left, name)(actual_right),
                    getattr(expected_left, name)(expected_right),
                    case=(name, "offset transposed" if offset else "transposed"),
                )
                np.testing.assert_array_equal(tensor_bits(actual_left), actual_left_before)
                np.testing.assert_array_equal(tensor_bits(actual_right), actual_right_before)

            actual_empty = torch.zeros((2, 0, 3)).transpose(0, 2)
            expected_empty = reference_torch.zeros((2, 0, 3)).transpose(0, 2)
            self.assert_matches(
                getattr(actual_empty, name)(actual_empty),
                getattr(expected_empty, name)(expected_empty),
                case=(name, "empty transposed"),
            )

            actual_no_grad_left = matching_dense_view(
                torch, MATCHING_DENSE_LEFT_BITS, offset=True, requires_grad=True
            )
            actual_no_grad_right = matching_dense_view(
                torch, MATCHING_DENSE_RIGHT_BITS, offset=True, requires_grad=True
            )
            expected_no_grad_left = matching_dense_view(
                reference_torch,
                MATCHING_DENSE_LEFT_BITS,
                offset=True,
                requires_grad=True,
            )
            expected_no_grad_right = matching_dense_view(
                reference_torch,
                MATCHING_DENSE_RIGHT_BITS,
                offset=True,
                requires_grad=True,
            )
            with torch.no_grad():
                actual_untracked = getattr(actual_no_grad_left, name)(
                    actual_no_grad_right
                )
            with reference_torch.no_grad():
                expected_untracked = getattr(expected_no_grad_left, name)(
                    expected_no_grad_right
                )
            self.assert_matches(
                actual_untracked,
                expected_untracked,
                case=(name, "no_grad offset transposed"),
            )

    def test_same_shape_matching_dense_active_autograd_matches_pytorch_2_13(self):
        values = np.arange(8, dtype=np.float32).reshape(2, 4)
        other_values = (16.0 - values).astype(np.float32)
        for name in ("sub", "subtract"):
            actual_left_leaf = torch.tensor(values.tolist(), requires_grad=True)
            actual_right_leaf = torch.tensor(other_values.tolist(), requires_grad=True)
            expected_left_leaf = reference_torch.tensor(
                values.tolist(), requires_grad=True
            )
            expected_right_leaf = reference_torch.tensor(
                other_values.tolist(), requires_grad=True
            )
            actual_output = getattr(actual_left_leaf.transpose(0, 1), name)(
                actual_right_leaf.transpose(0, 1)
            )
            expected_output = getattr(expected_left_leaf.transpose(0, 1), name)(
                expected_right_leaf.transpose(0, 1)
            )

            self.assert_matches(
                actual_output, expected_output, case=(name, "tracked transposed")
            )
            actual_output.sum().backward()
            expected_output.sum().backward()
            np.testing.assert_array_equal(
                np.asarray(actual_left_leaf.grad), expected_left_leaf.grad.numpy()
            )
            np.testing.assert_array_equal(
                np.asarray(actual_right_leaf.grad), expected_right_leaf.grad.numpy()
            )

    @staticmethod
    def dispatch_observation(module, method_name):
        left = module.tensor([2.0])
        right = module.tensor([3.0])
        descriptor = inspect.getattr_static(module.Tensor, method_name)
        marker = object()
        mode_observations = []

        class RecordingMode(module.overrides.TorchFunctionMode):
            def __init__(self, result=marker):
                self.calls = []
                self.result = result

            def __torch_function__(self, func, types, args=(), kwargs=None):
                self.calls.append((func, types, args, kwargs))
                return self.result

        for call, keyword_names in (
            (lambda: getattr(left, method_name)(right), None),
            (lambda: getattr(left, method_name)(4.0), None),
            (lambda: getattr(left, method_name)(other=right), ("other",)),
            (lambda: getattr(left, method_name)(x2=right), ("x2",)),
            (lambda: getattr(left, method_name)(right, alpha=True), ("alpha",)),
            (lambda: getattr(left, method_name)(right, alpha=1e39), ("alpha",)),
        ):
            mode = RecordingMode()
            with mode:
                result = call()
            func, dispatch_types, args, kwargs = mode.calls[0]
            mode_observations.append(
                (
                    result is marker,
                    func is descriptor,
                    dispatch_types == (),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword_names,
                )
            )

        override_observations = []

        class Override:
            calls = []

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                cls.calls.append((func, types, args, kwargs))
                return marker

        for call, keyword in (
            (lambda value: getattr(left, method_name)(value), None),
            (lambda value: getattr(left, method_name)(value, alpha=1e39), None),
            (lambda value: getattr(left, method_name)(right, alpha=value), "alpha"),
        ):
            value = Override()
            Override.calls.clear()
            result = call(value)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is descriptor,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    keyword is not None
                    and kwargs is not None
                    and kwargs[keyword] is value,
                )
            )

        if method_name == "sub":
            value = Override()
            Override.calls.clear()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                result = left.sub(value, right)
            func, dispatch_types, args, kwargs = Override.calls[0]
            override_observations.append(
                (
                    result is marker,
                    func is descriptor,
                    tuple(item.__name__ for item in dispatch_types),
                    len(args),
                    kwargs is None,
                    None if kwargs is None else tuple(kwargs),
                    args[1] is value,
                )
            )

        invalid_observations = []
        for call in (
            lambda: getattr(left, method_name)([]),
            lambda: getattr(left, method_name)(right, out=right),
        ):
            invalid_mode = RecordingMode()
            try:
                with invalid_mode:
                    call()
            except Exception as error:
                invalid_observations.append(
                    (type(error).__name__, str(error), len(invalid_mode.calls))
                )

        return mode_observations, override_observations, invalid_observations

    def test_torch_function_mode_and_operand_dispatch_match_pytorch_2_13(self):
        for name in ("sub", "subtract"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.dispatch_observation(torch, name),
                    self.dispatch_observation(reference_torch, name),
                )

    @staticmethod
    def callable_contract(module, method_name):
        descriptor = inspect.getattr_static(module.Tensor, method_name)
        tensor = module.tensor([1.0])
        bound = getattr(tensor, method_name)
        try:
            inspect.signature(descriptor)
        except Exception as error:
            signature_error = (
                type(error).__name__,
                re.sub(r"0x[0-9a-f]+", "0x...", str(error)),
            )
        else:
            signature_error = None
        return {
            "descriptor_type": type(descriptor).__name__,
            "descriptor_is_method_descriptor": type(descriptor)
            is types.MethodDescriptorType,
            "bound_type": type(bound).__name__,
            "bound_is_builtin_method": type(bound) is types.BuiltinMethodType,
            "name": descriptor.__name__,
            "qualname": descriptor.__qualname__,
            "bound_name": bound.__name__,
            "bound_qualname": bound.__qualname__,
            "descriptor_doc": descriptor.__doc__,
            "bound_doc": bound.__doc__,
            "text_signature": descriptor.__text_signature__,
            "bound_text_signature": bound.__text_signature__,
            "signature_error": signature_error,
            "objclass_name": descriptor.__objclass__.__name__,
            "objclass_module": descriptor.__objclass__.__module__,
            "has_descriptor_module": hasattr(descriptor, "__module__"),
            "bound_module": bound.__module__,
            "descriptor_copy_identity": copy.copy(descriptor) is descriptor,
            "descriptor_deepcopy_identity": copy.deepcopy(descriptor) is descriptor,
            "bound_copy_identity": copy.copy(bound) is bound,
            "bound_deepcopy_identity": copy.deepcopy(bound) is bound,
            "descriptor_pickle_identity": tuple(
                pickle.loads(pickle.dumps(descriptor, protocol)) is descriptor
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ),
        }

    def test_callable_metadata_documentation_copy_and_pickle_match_pytorch_2_13(self):
        for name in ("sub", "subtract"):
            with self.subTest(name=name):
                self.assertEqual(
                    self.callable_contract(torch, name),
                    self.callable_contract(reference_torch, name),
                )

    def test_common_binding_and_scalar_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        actual_alias = torch.tensor([3.0])
        expected_alias = reference_torch.tensor([3.0])
        for name in ("sub", "subtract"):
            actual_method = getattr(actual, name)
            expected_method = getattr(expected, name)
            cases = (
                (lambda: actual_method(), lambda: expected_method()),
                (lambda: actual_method([]), lambda: expected_method([])),
                (lambda: actual_method(None), lambda: expected_method(None)),
                (
                    lambda: actual_method(actual, out=actual),
                    lambda: expected_method(expected, out=expected),
                ),
                (
                    lambda: actual_method(x2=actual_alias, other=actual),
                    lambda: expected_method(x2=expected_alias, other=expected),
                ),
                (
                    lambda: actual_method(other=actual, x2=actual_alias),
                    lambda: expected_method(other=expected, x2=expected_alias),
                ),
                (lambda: actual_method(True), lambda: expected_method(True)),
                (
                    lambda: actual_method(np.uint64(2**63)),
                    lambda: expected_method(np.uint64(2**63)),
                ),
                (lambda: actual_method(2**64), lambda: expected_method(2**64)),
                (
                    lambda: actual_method(-(2**63) - 1),
                    lambda: expected_method(-(2**63) - 1),
                ),
            )
            for index, (actual_call, expected_call) in enumerate(cases):
                with self.subTest(name=name, case=index):
                    self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
