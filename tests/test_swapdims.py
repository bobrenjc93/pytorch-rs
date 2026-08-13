import inspect
import sys
import unittest
import warnings

import numpy as np
import torch_rs as torch


class TensorSwapdimsTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, *, shape, stride, offset):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected, dtype=np.float32)
        )

    def test_positional_keyword_and_negative_dimensions_reuse_transpose_views(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        source = base.transpose(0, 2)[1]
        expected_source = values.transpose(2, 1, 0)[1]

        for swapped in (
            source.swapdims(0, -1),
            source.swapdims(dim0=0, dim1=-1),
            source.swapdims(dim1=-1, dim0=0),
        ):
            with self.subTest(swapped=swapped):
                self.assert_tensor(
                    swapped,
                    expected_source.swapaxes(0, -1),
                    shape=(2, 3),
                    stride=(12, 4),
                    offset=1,
                )
                self.assertIs(swapped.dtype, source.dtype)
                self.assertEqual(swapped.device, source.device)
                self.assertIsNot(swapped, source)

        unchanged = source.swapdims(1, -1)
        self.assert_tensor(
            unchanged,
            expected_source,
            shape=source.shape,
            stride=source.stride(),
            offset=source.storage_offset(),
        )
        self.assertIsNot(unchanged, source)

    def test_scalar_empty_and_extreme_metadata(self):
        scalar = torch.tensor([2.5, 3.5])[1]
        for dim0, dim1 in ((0, 0), (-1, -1), (0, -1), (-1, 0)):
            with self.subTest(kind="scalar", dim0=dim0, dim1=dim1):
                swapped = scalar.swapdims(dim0, dim1)
                self.assert_tensor(
                    swapped,
                    3.5,
                    shape=(),
                    stride=(),
                    offset=1,
                )

        empty = torch.zeros((4, 2, 0, 3)).transpose(0, 3)[2]
        swapped_empty = empty.swapdims(-3, -1)
        self.assertEqual(swapped_empty.shape, (4, 0, 2))
        self.assertEqual(
            swapped_empty.stride(), (empty.stride()[2], empty.stride()[1], empty.stride()[0])
        )
        self.assertEqual(swapped_empty.storage_offset(), empty.storage_offset())
        self.assertEqual(swapped_empty.numel(), 0)
        self.assertEqual(swapped_empty.tolist(), [[], [], [], []])

        extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        for call in (
            lambda: extreme.swapdims(1, 3),
            lambda: extreme.swapdims(dim0=-3, dim1=-1),
        ):
            with self.subTest(kind="overflow"):
                with self.assertRaisesRegex(
                    RuntimeError, "^numel: integer multiplication overflow$"
                ):
                    call()
        self.assertEqual(extreme.swapdims(1, 1).numel(), 0)

    def test_dimensions_accept_pytorch_integer_types_and_report_ranges(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            tensor.swapdims(IntSubclass(0), np.int64(-1)).shape, (4, 3, 2)
        )
        self.assertEqual(
            tensor.swapdims(dim0=np.uint32(1), dim1=2).shape, (2, 4, 3)
        )

        for dimension in (True, False, 1.0, "1", None, IndexOnly()):
            with self.subTest(invalid_type=dimension):
                with self.assertRaisesRegex(TypeError, "must be int"):
                    tensor.swapdims(dimension, 0)
        for dimension in (2**100, -(2**100)):
            with self.subTest(overflow=dimension):
                with self.assertRaisesRegex(
                    ValueError, "^Overflow when unpacking long long$"
                ):
                    tensor.swapdims(dimension, 0)

        scalar = torch.tensor(1.0)
        for dimension in (-2, 1):
            with self.subTest(scalar_dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-1, 0\], but got {dimension}\)$",
                ):
                    scalar.swapdims(dimension, 0)
        for dimension in (-4, 3):
            with self.subTest(tensor_dimension=dimension):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-3, 2\], but got {dimension}\)$",
                ):
                    tensor.swapdims(0, dimension)

    def test_binding_errors_and_validation_precedence_match_pytorch(self):
        class UserOverflow(np.int64):
            def __index__(self):
                raise OverflowError("user overflow")

        tensor = torch.zeros((2, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            timedelta = np.timedelta64(1)
        cases = (
            (
                lambda: tensor.swapdims(),
                'swapdims() missing 2 required positional argument: "dim0", "dim1"',
            ),
            (
                lambda: tensor.swapdims(0),
                'swapdims() missing 1 required positional arguments: "dim1"',
            ),
            (
                lambda: tensor.swapdims(0, 1, 2),
                "swapdims() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.swapdims(dim1=0),
                'swapdims() missing 2 required positional argument: "dim0", "dim1"',
            ),
            (
                lambda: tensor.swapdims(0, 1, unexpected=None),
                "swapdims() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.swapdims(0, dim0=1, dim1=0),
                "swapdims() got multiple values for argument 'dim0'",
            ),
            (
                lambda: tensor.swapdims(1.5),
                "swapdims(): argument 'dim0' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.swapdims(dim0=np.bool_(True)),
                "swapdims(): argument 'dim0' must be int, not numpy.bool",
            ),
            (
                lambda: tensor.swapdims(1.5, 0, unexpected=None),
                "swapdims(): argument 'dim0' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.swapdims(2**100, unexpected=None),
                'swapdims() missing 1 required positional arguments: "dim1"',
            ),
            (
                lambda: tensor.swapdims(2**100, 0, unexpected=None),
                "swapdims() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.swapdims(2**100, 1.5),
                "swapdims(): argument 'dim1' (position 2) must be int, not float",
            ),
            (
                lambda: tensor.swapdims(timedelta, 0),
                "'numpy.timedelta64' object cannot be interpreted as an integer",
            ),
        )
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaises(TypeError) as raised:
                    call()
                self.assertEqual(str(raised.exception), expected)

        with self.assertRaisesRegex(
            ValueError, "^Overflow when unpacking long long$"
        ):
            tensor.swapdims(np.uint64(2**63), 0)
        with self.assertRaisesRegex(
            TypeError,
            "^'numpy.timedelta64' object cannot be interpreted as an integer$",
        ):
            tensor.swapdims(2**100, timedelta)
        with self.assertRaisesRegex(
            ValueError, "^Overflow when unpacking long long$"
        ):
            tensor.swapdims(timedelta, 2**100)
        with self.assertRaisesRegex(OverflowError, "^user overflow$"):
            tensor.swapdims(2**100, UserOverflow(0))

    def test_dimension_values_convert_dim1_before_dim0(self):
        state = {"dim1_converted": False, "calls": []}

        class StatefulInteger(np.int64):
            def __new__(cls, role):
                value = np.int64.__new__(cls, 0)
                value.role = role
                return value

            def __index__(self):
                state["calls"].append(self.role)
                if self.role == "dim1":
                    state["dim1_converted"] = True
                    return 1
                return 0 if state["dim1_converted"] else 2

        output = torch.zeros((2, 3, 4)).swapdims(
            StatefulInteger("dim0"), StatefulInteger("dim1")
        )
        self.assertEqual(state["calls"], ["dim1", "dim0"])
        self.assertEqual(output.shape, (3, 2, 4))
        self.assertEqual(output.stride(), (4, 12, 1))

    def test_autograd_uses_inverse_swap_and_no_grad_policy(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 3, 2)
        leaf = torch.tensor(values.tolist(), requires_grad=True)
        swapped = leaf.swapdims(0, -1)

        self.assertTrue(swapped.requires_grad)
        self.assertFalse(swapped.is_leaf)
        (swapped * torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_allclose(
            np.asarray(leaf.grad), weights.swapaxes(0, -1), rtol=0.0, atol=0.0
        )

        identity_leaf = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        identity = identity_leaf.swapdims(1, 1)
        (identity * torch.tensor([[2.0, 3.0], [5.0, 7.0]])).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(identity_leaf.grad), [[2.0, 3.0], [5.0, 7.0]]
        )

        with torch.no_grad():
            untracked = leaf.swapdims(0, 1)
        self.assertTrue(untracked.requires_grad)
        self.assertTrue(untracked.is_leaf)

    def test_public_descriptor_matches_pytorch_shape(self):
        descriptor = torch.Tensor.swapdims
        bound = torch.zeros((1,)).swapdims
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            descriptor.__doc__,
            "\nswapdims(dim0, dim1) -> Tensor\n\nSee :func:`torch.swapdims`\n",
        )
        for callable_object in (descriptor, bound):
            with self.subTest(callable_object=callable_object):
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)


if __name__ == "__main__":
    unittest.main()
