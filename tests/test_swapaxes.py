import inspect
import sys
import unittest
import warnings

import numpy as np
import torch_rs as torch


class TensorSwapaxesTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, *, shape, stride, offset):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected, dtype=np.float32)
        )

    def test_positional_keyword_and_negative_axes_reuse_transpose_views(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        source = base.transpose(0, 2)[1]
        expected_source = values.transpose(2, 1, 0)[1]

        for swapped in (
            source.swapaxes(0, -1),
            source.swapaxes(axis0=0, axis1=-1),
            source.swapaxes(axis1=-1, axis0=0),
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

        unchanged = source.swapaxes(1, -1)
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
        for axis0, axis1 in ((0, 0), (-1, -1), (0, -1), (-1, 0)):
            with self.subTest(kind="scalar", axis0=axis0, axis1=axis1):
                swapped = scalar.swapaxes(axis0, axis1)
                self.assert_tensor(swapped, 3.5, shape=(), stride=(), offset=1)

        empty = torch.zeros((4, 2, 0, 3)).transpose(0, 3)[2]
        swapped_empty = empty.swapaxes(-3, -1)
        self.assertEqual(swapped_empty.shape, (4, 0, 2))
        self.assertEqual(
            swapped_empty.stride(),
            (empty.stride()[2], empty.stride()[1], empty.stride()[0]),
        )
        self.assertEqual(swapped_empty.storage_offset(), empty.storage_offset())
        self.assertEqual(swapped_empty.numel(), 0)
        self.assertEqual(swapped_empty.tolist(), [[], [], [], []])

        extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        for call in (
            lambda: extreme.swapaxes(1, 3),
            lambda: extreme.swapaxes(axis0=-3, axis1=-1),
        ):
            with self.subTest(kind="overflow"):
                with self.assertRaisesRegex(
                    RuntimeError, "^numel: integer multiplication overflow$"
                ):
                    call()
        self.assertEqual(extreme.swapaxes(1, 1).numel(), 0)

    def test_axis_types_and_ranges_match_pytorch(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        tensor = torch.zeros((2, 3, 4))
        self.assertEqual(
            tensor.swapaxes(IntSubclass(0), np.int64(-1)).shape, (4, 3, 2)
        )
        self.assertEqual(
            tensor.swapaxes(axis0=np.uint32(1), axis1=2).shape, (2, 4, 3)
        )

        for axis in (True, False, 1.0, "1", None, IndexOnly()):
            with self.subTest(invalid_type=axis):
                with self.assertRaisesRegex(TypeError, "must be int"):
                    tensor.swapaxes(axis, 0)
        for axis in (2**100, -(2**100)):
            with self.subTest(overflow=axis):
                with self.assertRaisesRegex(
                    ValueError, "^Overflow when unpacking long long$"
                ):
                    tensor.swapaxes(axis, 0)

        scalar = torch.tensor(1.0)
        for axis in (-2, 1):
            with self.subTest(scalar_axis=axis):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-1, 0\], but got {axis}\)$",
                ):
                    scalar.swapaxes(axis, 0)
        for axis in (-4, 3):
            with self.subTest(tensor_axis=axis):
                with self.assertRaisesRegex(
                    IndexError,
                    rf"^Dimension out of range \(expected to be in range of \[-3, 2\], but got {axis}\)$",
                ):
                    tensor.swapaxes(0, axis)

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
                lambda: tensor.swapaxes(),
                'swapaxes() missing 2 required positional argument: "axis0", "axis1"',
            ),
            (
                lambda: tensor.swapaxes(0),
                'swapaxes() missing 1 required positional arguments: "axis1"',
            ),
            (
                lambda: tensor.swapaxes(0, 1, 2),
                "swapaxes() takes 2 positional arguments but 3 were given",
            ),
            (
                lambda: tensor.swapaxes(axis1=0),
                'swapaxes() missing 2 required positional argument: "axis0", "axis1"',
            ),
            (
                lambda: tensor.swapaxes(dim0=0, dim1=1),
                'swapaxes() missing 2 required positional argument: "axis0", "axis1"',
            ),
            (
                lambda: tensor.swapaxes(0, 1, unexpected=None),
                "swapaxes() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.swapaxes(0, axis0=1, axis1=0),
                "swapaxes() got multiple values for argument 'axis0'",
            ),
            (
                lambda: tensor.swapaxes(1.5),
                "swapaxes(): argument 'axis0' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.swapaxes(axis0=np.bool_(True)),
                "swapaxes(): argument 'axis0' must be int, not numpy.bool",
            ),
            (
                lambda: tensor.swapaxes(1.5, 0, unexpected=None),
                "swapaxes(): argument 'axis0' (position 1) must be int, not float",
            ),
            (
                lambda: tensor.swapaxes(2**100, unexpected=None),
                'swapaxes() missing 1 required positional arguments: "axis1"',
            ),
            (
                lambda: tensor.swapaxes(2**100, 0, unexpected=None),
                "swapaxes() got an unexpected keyword argument 'unexpected'",
            ),
            (
                lambda: tensor.swapaxes(2**100, 1.5),
                "swapaxes(): argument 'axis1' (position 2) must be int, not float",
            ),
            (
                lambda: tensor.swapaxes(timedelta, 0),
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
            tensor.swapaxes(np.uint64(2**63), 0)
        with self.assertRaisesRegex(
            TypeError,
            "^'numpy.timedelta64' object cannot be interpreted as an integer$",
        ):
            tensor.swapaxes(2**100, timedelta)
        with self.assertRaisesRegex(
            ValueError, "^Overflow when unpacking long long$"
        ):
            tensor.swapaxes(timedelta, 2**100)
        with self.assertRaisesRegex(OverflowError, "^user overflow$"):
            tensor.swapaxes(2**100, UserOverflow(0))

    def test_axis_values_convert_axis1_before_axis0(self):
        state = {"axis1_converted": False, "calls": []}

        class StatefulInteger(np.int64):
            def __new__(cls, role):
                value = np.int64.__new__(cls, 0)
                value.role = role
                return value

            def __index__(self):
                state["calls"].append(self.role)
                if self.role == "axis1":
                    state["axis1_converted"] = True
                    return 1
                return 0 if state["axis1_converted"] else 2

        output = torch.zeros((2, 3, 4)).swapaxes(
            StatefulInteger("axis0"), StatefulInteger("axis1")
        )
        self.assertEqual(state["calls"], ["axis1", "axis0"])
        self.assertEqual(output.shape, (3, 2, 4))
        self.assertEqual(output.stride(), (4, 12, 1))

    def test_autograd_and_no_grad_match_swapdims(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 3, 2)
        axes_leaf = torch.tensor(values.tolist(), requires_grad=True)
        dims_leaf = torch.tensor(values.tolist(), requires_grad=True)
        axes_view = axes_leaf.swapaxes(0, -1)
        dims_view = dims_leaf.swapdims(0, -1)

        self.assertEqual(axes_view.shape, dims_view.shape)
        self.assertEqual(axes_view.stride(), dims_view.stride())
        self.assertEqual(axes_view.requires_grad, dims_view.requires_grad)
        self.assertEqual(axes_view.is_leaf, dims_view.is_leaf)
        axes_view.mul(torch.tensor(weights.tolist())).sum().backward()
        dims_view.mul(torch.tensor(weights.tolist())).sum().backward()
        np.testing.assert_array_equal(np.asarray(axes_leaf.grad), np.asarray(dims_leaf.grad))

        with torch.no_grad():
            axes_untracked = axes_leaf.swapaxes(0, 1)
            dims_untracked = dims_leaf.swapdims(0, 1)
        self.assertEqual(axes_untracked.requires_grad, dims_untracked.requires_grad)
        self.assertEqual(axes_untracked.is_leaf, dims_untracked.is_leaf)

    def test_public_descriptor_matches_pytorch_shape(self):
        descriptor = torch.Tensor.swapaxes
        bound = torch.zeros((1,)).swapaxes
        self.assertIsNone(descriptor.__text_signature__)
        self.assertIsNone(bound.__text_signature__)
        self.assertEqual(
            descriptor.__doc__,
            "\nswapaxes(axis0, axis1) -> Tensor\n\nSee :func:`torch.swapaxes`\n",
        )
        for callable_object in (descriptor, bound):
            with self.subTest(callable_object=callable_object):
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)


if __name__ == "__main__":
    unittest.main()
