import gc
import unittest

import numpy as np
import torch_rs as torch


class TensorBareEllipsisTests(unittest.TestCase):
    def tensor_bits(self, tensor):
        return np.asarray(tensor).reshape(-1).view(np.uint32)

    def assert_metadata_alias(self, source, alias):
        self.assertIsNot(alias, source)
        self.assertEqual(alias.shape, source.shape)
        self.assertEqual(alias.stride(), source.stride())
        self.assertEqual(alias.storage_offset(), source.storage_offset())
        self.assertEqual(alias.numel(), source.numel())
        self.assertEqual(alias.is_contiguous(), source.is_contiguous())
        self.assertIs(alias.dtype, source.dtype)
        self.assertEqual(alias.device, source.device)
        self.assertEqual(alias.data_ptr(), source.data_ptr())
        self.assertTrue(alias.is_set_to(source))
        np.testing.assert_array_equal(
            self.tensor_bits(alias), self.tensor_bits(source)
        )

    def test_scalar_empty_offset_and_noncontiguous_inputs_return_distinct_aliases(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist(), dtype=torch.float32)
        cases = (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), dtype=torch.float32).transpose(0, 2)[1],
            ),
            ("offset", base[1]),
            ("noncontiguous-offset", base.transpose(0, 2)[1]),
        )

        for case, source in cases:
            with self.subTest(case=case):
                self.assert_metadata_alias(source, source[...])

    def test_alias_autograd_gradients_and_source_lifetime(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        alias = source[...]
        self.assert_metadata_alias(source, alias)
        self.assertTrue(alias.requires_grad)
        self.assertFalse(alias.is_leaf)
        self.assertEqual(alias.output_nr, 0)

        del source
        gc.collect()
        weights = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        (alias * weights).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(leaf.grad),
            np.asarray([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]]),
        )

        scalar = torch.tensor(-2.0, requires_grad=True)
        scalar_alias = scalar[...]
        self.assert_metadata_alias(scalar, scalar_alias)
        self.assertTrue(scalar_alias.requires_grad)
        self.assertFalse(scalar_alias.is_leaf)
        (scalar_alias * 7.0).backward()
        self.assertEqual(scalar.grad.item(), 7.0)

        empty = torch.zeros((2, 0, 3), requires_grad=True)
        empty_alias = empty[...]
        self.assert_metadata_alias(empty, empty_alias)
        empty_alias.sum().backward()
        self.assertEqual(empty.grad.shape, (2, 0, 3))
        self.assertEqual(empty.grad.tolist(), [[], []])

    def test_no_grad_alias_is_a_leaf_and_outlives_its_source(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = leaf.transpose(0, 1)
        with torch.no_grad():
            alias = source[...]

        self.assert_metadata_alias(source, alias)
        self.assertTrue(alias.requires_grad)
        self.assertTrue(alias.is_leaf)
        self.assertEqual(alias.output_nr, 0)
        del source, leaf
        gc.collect()
        self.assertEqual(alias.tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_other_index_forms_remain_narrowly_unsupported(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(tensor[1].tolist(), [3.0, 4.0])
        self.assertEqual(tensor[1, 0].item(), 3.0)
        self.assertTrue(tensor[()].is_set_to(tensor))

        unsupported = (
            slice(None),
            None,
            (Ellipsis,),
            (Ellipsis, 0),
            (0, Ellipsis),
        )
        for index in unsupported:
            with self.subTest(index=repr(index)):
                with self.assertRaisesRegex(IndexError, "only integers"):
                    tensor[index]
