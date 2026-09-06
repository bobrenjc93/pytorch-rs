import unittest

import numpy as np

import torch_rs as torch


class TensorRangeSliceTests(unittest.TestCase):
    def assert_dropout_probability_node(self, value, node_name):
        with self.assertRaisesRegex(ValueError, rf"grad_fn=<{node_name}>"):
            torch.nn.functional.dropout(None, p=value, training=False)

    def assert_data_pointer_matches_offset(self, source, selected):
        if selected.numel() == 0:
            self.assertEqual(selected.data_ptr(), 0)
            return
        offset_delta = selected.storage_offset() - source.storage_offset()
        self.assertEqual(
            selected.data_ptr(),
            source.data_ptr() + offset_delta * selected.element_size(),
        )

    def test_direct_range_slices_preserve_metadata_values_and_aliasing(self):
        values = np.arange(24, dtype=np.float32).reshape(4, 3, 2)
        source = torch.tensor(values.tolist())

        cases = (
            (slice(1, 3), values[1:3], (2, 3, 2), (6, 2, 1), 6),
            (slice(None, -1), values[:-1], (3, 3, 2), (6, 2, 1), 0),
            (slice(-3, None), values[-3:], (3, 3, 2), (6, 2, 1), 6),
            (slice(None, None, 1), values[::1], (4, 3, 2), (6, 2, 1), 0),
            (slice(3, 1), values[3:1], (0, 3, 2), (6, 2, 1), 18),
        )
        for index, expected_values, shape, stride, offset in cases:
            with self.subTest(index=index):
                selected = source[index]
                self.assertEqual(selected.tolist(), expected_values.tolist())
                self.assertEqual(selected.shape, shape)
                self.assertEqual(selected.stride(), stride)
                self.assertEqual(selected.storage_offset(), offset)
                self.assertTrue(selected.is_set_to(source[index]))
                self.assert_data_pointer_matches_offset(source, selected)

    def test_transposed_offset_and_empty_sources_preserve_view_metadata(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist())
        transposed = base.transpose(0, 1)
        transposed_selected = transposed[1:3]
        self.assertEqual(transposed_selected.tolist(), values.transpose(1, 0, 2)[1:3].tolist())
        self.assertEqual(transposed_selected.shape, (2, 2, 4))
        self.assertEqual(transposed_selected.stride(), (4, 12, 1))
        self.assertEqual(transposed_selected.storage_offset(), 4)
        self.assert_data_pointer_matches_offset(transposed, transposed_selected)

        offset = base[1]
        offset_selected = offset[1:3]
        self.assertEqual(offset_selected.tolist(), values[1, 1:3].tolist())
        self.assertEqual(offset_selected.shape, (2, 4))
        self.assertEqual(offset_selected.stride(), (4, 1))
        self.assertEqual(offset_selected.storage_offset(), 16)
        self.assert_data_pointer_matches_offset(offset, offset_selected)

        empty_middle = torch.zeros((3, 0, 4))
        empty_selected = empty_middle[2:3]
        self.assertEqual(empty_selected.shape, (1, 0, 4))
        self.assertEqual(empty_selected.stride(), (4, 4, 1))
        self.assertEqual(empty_selected.storage_offset(), 8)
        self.assertEqual(empty_selected.data_ptr(), 0)
        self.assertEqual(empty_selected.tolist(), [[]])

        empty_range = torch.zeros((5, 2))[4:1]
        self.assertEqual(empty_range.shape, (0, 2))
        self.assertEqual(empty_range.stride(), (2, 1))
        self.assertEqual(empty_range.storage_offset(), 8)
        self.assertEqual(empty_range.data_ptr(), 0)

        huge_start = base[10**5000:]
        self.assertEqual(huge_start.tolist(), [])
        self.assertEqual(huge_start.shape, (0, 3, 4))
        self.assertEqual(huge_start.stride(), (12, 4, 1))
        self.assertEqual(huge_start.storage_offset(), 24)
        self.assertEqual(huge_start.data_ptr(), 0)

    def test_integer_prefix_trailing_range_slice(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())
        selected = source[1, 1:3]

        self.assertEqual(selected.tolist(), values[1, 1:3].tolist())
        self.assertEqual(selected.shape, (2, 4))
        self.assertEqual(selected.stride(), (4, 1))
        self.assertEqual(selected.storage_offset(), 16)
        self.assert_data_pointer_matches_offset(source, selected)

    def test_range_slice_backward_through_sum(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        leaf = torch.tensor(values.reshape(-1).tolist(), requires_grad=True)
        source = (leaf * 2.0).reshape(2, 3, 4).transpose(0, 1)
        selected = source[1:3]

        self.assertTrue(selected.requires_grad)
        self.assertFalse(selected.is_leaf)
        self.assertEqual(selected.output_nr, 0)
        selected.sum().backward()

        expected_gradient = np.zeros_like(values)
        expected_gradient[:, 1:3, :] = 2.0
        np.testing.assert_array_equal(
            np.asarray(leaf.grad).reshape(values.shape),
            expected_gradient,
        )

        diagnostic_leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        self.assert_dropout_probability_node(diagnostic_leaf[1:2], "SliceBackward0")

    def test_full_span_tuple_range_slices_reuse_alias_and_select_nodes(self):
        self.assert_dropout_probability_node(
            torch.tensor([2.0], requires_grad=True)[(slice(None, None, 1),)],
            "AliasBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([2.0], requires_grad=True)[(slice(0, 1),)],
            "AliasBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([2.0], requires_grad=True)[
                (slice(np.int64(0), None, 1),)
            ],
            "AliasBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([[2.0]], requires_grad=True)[0, slice(None, None, 1)],
            "SelectBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([[2.0]], requires_grad=True)[0, slice(0, 1)],
            "SelectBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([[2.0]], requires_grad=True)[
                0, slice(np.int64(0), None, 1)
            ],
            "SelectBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([2.0], requires_grad=True)[(slice(-1, None),)],
            "SliceBackward0",
        )
        self.assert_dropout_probability_node(
            torch.tensor([[2.0]], requires_grad=True)[0, slice(-1, None)],
            "SliceBackward0",
        )

    def test_unsupported_slice_forms_stay_unsupported(self):
        source = torch.zeros((3, 4, 5))
        unsupported = (
            slice(None, None, 2),
            slice(None, None, -1),
            slice(1, 3, 2),
            (0, slice(None, None, 2)),
            (slice(1, 3), slice(None)),
            (slice(None), slice(1, 3)),
            (slice(1, 3), 0),
            ([0, 1],),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaises(IndexError):
                    source[index]


if __name__ == "__main__":
    unittest.main()
