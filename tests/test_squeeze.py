import gc
import unittest

import numpy as np
import torch_rs as torch


class SqueezeTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, shape, stride):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected, dtype=np.float32))

    def test_method_and_top_level_squeeze_only_view_metadata(self):
        values = np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3, 1)
        source = torch.tensor(values.tolist())

        views = (
            source.squeeze(),
            torch.squeeze(source),
            torch.squeeze(input=source),
            source.squeeze((0, 2, -1)),
            source.squeeze([0, 2, -1]),
            torch.squeeze(source, dim=(0, 2, 4)),
            torch.squeeze(input=source, axis=[0, -3, -1]),
        )
        for view in views:
            with self.subTest(view=view):
                self.assert_tensor(view, values.reshape(2, 3), (2, 3), (3, 1))
                self.assertEqual(view.storage_offset(), source.storage_offset())
                self.assertIs(view.dtype, source.dtype)
                self.assertEqual(view.device, source.device)
                self.assertTrue(view.is_contiguous())

        leading = source.squeeze(0)
        self.assert_tensor(leading, values.reshape(2, 1, 3, 1), (2, 1, 3, 1), (3, 3, 1, 1))
        unchanged = source.squeeze(1)
        self.assertEqual(unchanged.shape, source.shape)
        self.assertEqual(unchanged.stride(), source.stride())
        self.assertEqual(source.squeeze(()).shape, source.shape)
        self.assertEqual(source.squeeze([]).stride(), source.stride())

    def test_method_accepts_variadic_dims_and_keywords_match_pytorch(self):
        source = torch.zeros((1, 3, 1, 1))
        for view in (
            source.squeeze(0, 2, 3),
            source.squeeze(dim=(0, 2, 3)),
            source.squeeze(axis=[0, 2, 3]),
        ):
            self.assertEqual(view.shape, (3,))
            self.assertEqual(view.stride(), (1,))

        self.assertEqual(torch.squeeze(source, 0).shape, (3, 1, 1))
        self.assertEqual(torch.squeeze(source, dim=[0, 2]).shape, (3, 1))
        with self.assertRaises(TypeError):
            torch.squeeze(source, 0, 2)
        with self.assertRaises(TypeError):
            source.squeeze(0, dim=2)
        with self.assertRaises(TypeError):
            torch.squeeze(source, 0, dim=2)

    def test_non_contiguous_offset_view_and_stride_aware_consumers(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 1, 3, 4)
        source = torch.tensor(values.tolist())
        view = source.transpose(0, 3)[1].squeeze()
        del source
        gc.collect()

        expected = values.transpose(3, 1, 2, 0)[1].squeeze()
        self.assert_tensor(view, expected, (3, 2), (4, 12))
        self.assertEqual(view.storage_offset(), 1)
        self.assertFalse(view.is_contiguous())

        self.assert_tensor(view.transpose(0, 1), expected.T, (2, 3), (12, 4))
        self.assert_tensor(view.reshape(-1), expected.reshape(-1), (6,), (1,))
        self.assert_tensor(view.clone(), expected.copy(), (3, 2), (1, 3))
        self.assert_tensor(view + 1.25, expected + 1.25, (3, 2), (1, 3))
        self.assertAlmostEqual(view.sum().item(), float(expected.sum()))
        self.assertEqual(view.tolist(), expected.tolist())
        np.testing.assert_array_equal(np.asarray(view), expected)
        self.assertEqual(repr(view), "tensor([1.0, 13.0, 5.0, 17.0, 9.0, 21.0], shape=[3, 2])")

    def test_scalars_zero_sizes_negative_dims_and_high_ranks(self):
        scalar = torch.tensor(2.5)
        for view in (
            scalar.squeeze(),
            scalar.squeeze(0),
            scalar.squeeze(-1),
            scalar.squeeze((0,)),
            torch.squeeze(scalar),
        ):
            self.assertEqual(view.shape, ())
            self.assertEqual(view.stride(), ())
            self.assertEqual(view.item(), 2.5)

        empty = torch.zeros((1, 0, 1, 2)).squeeze()
        self.assertEqual(empty.shape, (0, 2))
        self.assertEqual(empty.stride(), (2, 1))
        self.assertEqual(empty.tolist(), [])
        self.assertTrue(empty.is_contiguous())

        high_rank = torch.zeros((1,) * 65)
        self.assertEqual(high_rank.squeeze().shape, ())
        self.assertEqual(len(high_rank.squeeze(0).shape), 64)
        with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims"):
            high_rank.squeeze(())

    def test_invalid_dimensions_types_and_bindings_have_public_errors(self):
        source = torch.zeros((1, 2, 1))
        for call in (
            lambda: source.squeeze((0, -3)),
            lambda: torch.squeeze(source, [0, 0]),
        ):
            with self.subTest(call=call):
                with self.assertRaisesRegex(
                    RuntimeError, "dim 0 appears multiple times in the list of dims"
                ):
                    call()

        for dimension in (-4, 3):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(IndexError, "Dimension out of range"):
                    source.squeeze(dimension)

        invalid_calls = (
            lambda: source.squeeze(None),
            lambda: torch.squeeze(source, None),
            lambda: source.squeeze(True),
            lambda: source.squeeze(1.5),
            lambda: source.squeeze(dim=np.float64(1)),
            lambda: torch.squeeze(source, dim="1"),
            lambda: torch.squeeze(),
            lambda: torch.squeeze(source, 0, 1),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

        for dimension in (2**100, -(2**100)):
            with self.assertRaisesRegex(ValueError, "Overflow when unpacking long long"):
                source.squeeze(dimension)
        with self.assertRaisesRegex(
            TypeError, "failed to unpack.*Overflow when unpacking long long"
        ):
            source.squeeze([2**100])

        with self.assertRaises(TypeError) as raised:
            torch.squeeze(np.zeros((1,), dtype=np.float32))
        self.assertEqual(
            str(raised.exception),
            "squeeze(): argument 'input' (position 1) must be Tensor, not numpy.ndarray",
        )
        with self.assertRaises(TypeError) as raised:
            source.squeeze(torch.float32)
        self.assertIn("torch.dtype", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
