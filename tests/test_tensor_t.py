import inspect
import sys
import unittest
import warnings

import numpy as np
import torch_rs as torch


T_WARNING = (
    "The use of `x.T` on tensors of dimension other than 2 to reverse their "
    "shape is deprecated and it will throw an error in a future release. "
    "Consider `x.mT` to transpose batches of matrices or "
    "`x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the "
    "dimensions of a tensor."
)
T_SCALAR_WARNING = (
    "Tensor.T is deprecated on 0-D tensors. This function is the identity in "
    "these cases."
)
MT_SCALAR_WARNING = (
    "Tensor.mT is deprecated on 0-D tensors. This function is the identity in "
    "these cases."
)


def stable_warning_message(message):
    return message.split(" (Triggered internally at ", 1)[0]


class TensorTransposePropertyTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, shape, stride, offset=0):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        self.assertEqual(actual.storage_offset(), offset)
        np.testing.assert_array_equal(
            np.asarray(actual), np.asarray(expected, dtype=np.float32)
        )

    def test_00_t_and_mt_warnings_are_once_only_and_point_to_the_caller(self):
        high_rank = torch.zeros((2, 3, 4))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            high_rank.T
            high_rank.T
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(stable_warning_message(str(caught[0].message)), T_WARNING)
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(caught[0].lineno, warning_line)

        scalar = torch.tensor(2.5)
        for attribute, expected_message in (
            ("T", T_SCALAR_WARNING),
            ("mT", MT_SCALAR_WARNING),
        ):
            with self.subTest(attribute=attribute):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    warning_line = inspect.currentframe().f_lineno + 1
                    getattr(scalar, attribute)
                    getattr(scalar, attribute)
                self.assertEqual(len(caught), 1)
                self.assertIs(caught[0].category, UserWarning)
                self.assertEqual(
                    stable_warning_message(str(caught[0].message)), expected_message
                )
                self.assertEqual(caught[0].filename, __file__)
                self.assertEqual(caught[0].lineno, warning_line)

    def test_values_metadata_composition_and_transpose_equivalence(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reversed_view = source.T
        matrix_view = source.mT

        self.assert_tensor(
            reversed_view,
            values.transpose(2, 1, 0),
            (4, 3, 2),
            (1, 4, 12),
        )
        self.assert_tensor(
            matrix_view,
            values.swapaxes(-2, -1),
            (2, 4, 3),
            (12, 1, 4),
        )
        self.assertIs(reversed_view.dtype, source.dtype)
        self.assertEqual(reversed_view.device, source.device)
        self.assertFalse(reversed_view.is_contiguous())
        self.assertEqual(matrix_view.shape, source.transpose(-2, -1).shape)
        self.assertEqual(matrix_view.stride(), source.transpose(-2, -1).stride())
        self.assertEqual(matrix_view.tolist(), source.transpose(-2, -1).tolist())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restored_t = reversed_view.T
        restored_mt = matrix_view.mT
        for restored in (restored_t, restored_mt):
            self.assertIsNot(restored, source)
            self.assert_tensor(restored, values, source.shape, source.stride())

        matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(matrix.T.tolist(), matrix.mT.tolist())
        self.assertEqual(matrix.T.tolist(), matrix.transpose(0, 1).tolist())
        self.assertEqual(matrix.T.stride(), matrix.mT.stride())

    def test_scalar_vector_errors_identity_and_read_only_descriptors(self):
        scalar = torch.tensor(3.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scalar_t = scalar.T
            scalar_mt = scalar.mT
        self.assertIsNot(scalar_t, scalar)
        self.assertIs(scalar_mt, scalar)
        self.assertEqual(scalar_t.shape, ())
        self.assertEqual(scalar_t.stride(), ())
        self.assertEqual(scalar_t.item(), 3.5)

        vector = torch.tensor([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vector_t = vector.T
        self.assertIsNot(vector_t, vector)
        self.assert_tensor(vector_t, [1.0, 2.0, 3.0], (3,), (1,))
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\.mT is only supported on matrices or batches of matrices\. Got 1-D tensor\.$",
        ):
            vector.mT

        for tensor in (scalar, vector, torch.zeros((2, 3))):
            for attribute in ("T", "mT"):
                with self.subTest(shape=tensor.shape, attribute=attribute):
                    with self.assertRaisesRegex(
                        AttributeError,
                        rf"attribute '{attribute}'.*not writable",
                    ):
                        setattr(tensor, attribute, tensor)

    def test_offset_singleton_empty_high_rank_and_lifetime_views(self):
        values = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        base = torch.tensor(values.tolist()).transpose(0, 3)[1]
        expected = values.transpose(3, 1, 2, 0)[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reversed_view = base.T
        matrix_view = base.mT
        self.assert_tensor(
            reversed_view,
            expected.transpose(2, 1, 0),
            (2, 4, 3),
            (60, 5, 20),
            1,
        )
        self.assert_tensor(
            matrix_view,
            expected.swapaxes(-2, -1),
            (3, 2, 4),
            (20, 60, 5),
            1,
        )

        singleton = torch.zeros((1, 2, 1, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            singleton_t = singleton.T
        self.assertEqual(singleton_t.shape, (3, 1, 2, 1))
        self.assertEqual(singleton_t.stride(), (1, 3, 3, 6))
        self.assertFalse(singleton_t.is_contiguous())

        empty = torch.zeros((2, 0, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            empty_t = empty.T
        self.assertEqual(empty_t.shape, (3, 0, 2))
        self.assertEqual(empty_t.stride(), (1, 3, 3))
        self.assertEqual(empty_t.numel(), 0)
        self.assertTrue(empty_t.is_contiguous())

        high_rank_shape = [1] * 96
        high_rank_shape[3] = 2
        high_rank_shape[47] = 0
        high_rank_shape[91] = 3
        high_rank = torch.zeros(tuple(high_rank_shape))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            high_rank_t = high_rank.T
        self.assertEqual(high_rank_t.shape, tuple(reversed(high_rank_shape)))
        self.assertEqual(high_rank_t.stride(), tuple(reversed(high_rank.stride())))

        def view_after_source_drops():
            source = torch.tensor(values.tolist())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return source.T

        np.testing.assert_array_equal(
            np.asarray(view_after_source_drops()), values.transpose(3, 2, 1, 0)
        )

    def test_stride_aware_consumers_and_view_combinations(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            view = source.T
        expected = values.transpose(2, 1, 0)

        self.assertEqual(view.tolist(), expected.tolist())
        np.testing.assert_array_equal(np.asarray(view), expected)
        self.assertEqual(
            repr(view),
            "tensor([0.0, 12.0, 4.0, 16.0, 8.0, 20.0, 1.0, 13.0, 5.0, 17.0, 9.0, 21.0, 2.0, 14.0, 6.0, 18.0, 10.0, 22.0, 3.0, 15.0, 7.0, 19.0, 11.0, 23.0], shape=[4, 3, 2])",
        )

        self.assert_tensor(view.clone(), expected, view.shape, view.stride())
        self.assert_tensor(view.contiguous(), expected, view.shape, (6, 2, 1))
        self.assert_tensor(view + 1.25, expected + 1.25, view.shape, view.stride())
        self.assertEqual(view.sum().item(), np.float32(expected.sum()).item())
        self.assert_tensor(view.reshape(-1), expected.reshape(-1), (24,), (1,))
        self.assert_tensor(view.flatten(), expected.reshape(-1), (24,), (1,))
        self.assertEqual(view.transpose(0, 2).shape, source.shape)

        combined_source = torch.tensor(
            np.arange(6, dtype=np.float32).reshape(1, 2, 1, 3).tolist()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            combined = combined_source.T.squeeze().flatten().contiguous()
        self.assertEqual(combined.shape, (6,))
        self.assertEqual(combined.stride(), (1,))
        self.assertEqual(combined.tolist(), [0.0, 3.0, 1.0, 4.0, 2.0, 5.0])

    def test_extreme_zero_sized_boundaries(self):
        maximum = sys.maxsize
        source = torch.zeros((maximum, 0, maximum))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reversed_view = source.T
        self.assertEqual(reversed_view.shape, (maximum, 0, maximum))
        self.assertEqual(reversed_view.stride(), (1, maximum, maximum))
        self.assertEqual(reversed_view.numel(), 0)
        with self.assertRaisesRegex(
            RuntimeError, r"^numel: integer multiplication overflow$"
        ):
            source.mT

        offset = torch.zeros((maximum, 0, 1))[maximum - 1]
        self.assertEqual(offset.storage_offset(), maximum - 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            offset_t = offset.T
        for view in (offset_t, offset.mT):
            self.assertEqual(view.shape, (1, 0))
            self.assertEqual(view.stride(), (1, 1))
            self.assertEqual(view.storage_offset(), maximum - 1)
            self.assertEqual(view.tolist(), [[]])


if __name__ == "__main__":
    unittest.main()
