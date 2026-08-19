import gc
import unittest

import numpy as np
import torch_rs as torch


class CloneChannelsLastTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        values = np.ascontiguousarray(np.asarray(tensor))
        return values.reshape(-1).view(np.uint32)

    @staticmethod
    def clone_operations():
        return (
            ("method", lambda tensor: tensor.clone(memory_format=torch.channels_last)),
            (
                "function",
                lambda tensor: torch.clone(
                    input=tensor, memory_format=torch.channels_last
                ),
            ),
        )

    def make_layout_cases(self):
        patterns = np.array(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x3F80_0000,
            ],
            dtype=np.uint32,
        )
        bits = np.resize(patterns, 144)
        contiguous = torch.tensor(
            memoryview(bits[:48].view(np.float32)), dtype=torch.float32
        ).reshape((2, 3, 2, 4))
        offset = torch.tensor(
            memoryview(bits.view(np.float32)), dtype=torch.float32
        ).reshape((3, 2, 3, 2, 4))[1]
        strided = contiguous.transpose(0, 3)
        empty = torch.zeros((2, 0, 4, 5), dtype=torch.float32)
        return (
            ("contiguous", contiguous, (24, 1, 12, 3)),
            ("offset", offset, (24, 1, 12, 3)),
            ("strided", strided, (12, 1, 6, 3)),
            ("empty", empty, (0, 1, 0, 0)),
        )

    def test_method_and_function_materialize_canonical_independent_copies(self):
        for case, source, expected_stride in self.make_layout_cases():
            expected_bits = self.tensor_bits(source).copy()
            for operation_name, operation in self.clone_operations():
                with self.subTest(case=case, operation=operation_name):
                    copied = operation(source)
                    self.assertIsNot(copied, source)
                    self.assertFalse(copied.is_set_to(source))
                    self.assertEqual(copied.shape, source.shape)
                    self.assertEqual(copied.stride(), expected_stride)
                    self.assertEqual(copied.storage_offset(), 0)
                    self.assertTrue(
                        copied.is_contiguous(memory_format=torch.channels_last)
                    )
                    self.assertIs(copied.dtype, source.dtype)
                    self.assertEqual(copied.device, source.device)
                    np.testing.assert_array_equal(
                        self.tensor_bits(copied), expected_bits
                    )

    def test_autograd_no_grad_and_source_lifetime(self):
        for operation_name, operation in self.clone_operations():
            with self.subTest(operation=operation_name, mode="autograd"):
                leaf = torch.ones((2, 3, 2, 4), requires_grad=True)
                source = (leaf * 3.0).transpose(0, 3)
                copied = operation(source)
                self.assertEqual(copied.stride(), (12, 1, 6, 3))
                self.assertTrue(copied.requires_grad)
                self.assertFalse(copied.is_leaf)
                self.assertFalse(copied.is_set_to(source))
                del source
                gc.collect()

                copied.sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(leaf.grad), np.full((2, 3, 2, 4), 3.0)
                )

            with self.subTest(operation=operation_name, mode="no_grad"):
                leaf = torch.ones((2, 3, 2, 4), requires_grad=True)
                source = (leaf * 3.0).transpose(0, 3)
                with torch.no_grad():
                    copied = operation(source)
                self.assertEqual(copied.stride(), (12, 1, 6, 3))
                self.assertFalse(copied.requires_grad)
                self.assertTrue(copied.is_leaf)
                self.assertIs(copied.dtype, source.dtype)
                self.assertEqual(copied.device, source.device)
                del source, leaf
                gc.collect()
                np.testing.assert_array_equal(
                    np.asarray(copied), np.full((4, 3, 2, 2), 3.0)
                )

    def test_rank_errors_and_channels_last_3d_behavior(self):
        for rank, source in (
            (0, torch.tensor(1.0)),
            (3, torch.ones((2, 3, 4))),
            (5, torch.ones((1, 2, 3, 4, 5))),
        ):
            operations = (
                lambda tensor=source: tensor.clone(
                    memory_format=torch.channels_last
                ),
                lambda tensor=source: torch.clone(
                    tensor, memory_format=torch.channels_last
                ),
            )
            for operation in operations:
                with self.subTest(rank=rank, operation=operation):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "required rank 4 tensor to use channels_last format",
                    ):
                        operation()

        rank_five = torch.ones((1, 2, 3, 4, 5))
        for operation in (
            lambda: rank_five.clone(memory_format=torch.channels_last_3d),
            lambda: torch.clone(rank_five, memory_format=torch.channels_last_3d),
        ):
            with self.subTest(memory_format="channels_last_3d", operation=operation):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "clone with memory format torch.channels_last_3d is not supported",
                ):
                    operation()


if __name__ == "__main__":
    unittest.main()
