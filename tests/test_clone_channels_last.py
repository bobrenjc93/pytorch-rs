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
    def clone_operations(memory_format):
        return (
            ("method", lambda tensor: tensor.clone(memory_format=memory_format)),
            (
                "function",
                lambda tensor: torch.clone(
                    input=tensor, memory_format=memory_format
                ),
            ),
        )

    def make_layout_suites(self):
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
        bits = np.resize(patterns, 720)
        contiguous_2d = torch.tensor(
            memoryview(bits[:48].view(np.float32)), dtype=torch.float32
        ).reshape((2, 3, 2, 4))
        offset_2d = torch.tensor(
            memoryview(bits[:144].view(np.float32)), dtype=torch.float32
        ).reshape((3, 2, 3, 2, 4))[1]
        strided_2d = contiguous_2d.transpose(0, 3)
        empty_2d = torch.zeros((2, 0, 4, 5), dtype=torch.float32)

        contiguous_3d = torch.tensor(
            memoryview(bits[:240].view(np.float32)), dtype=torch.float32
        ).reshape((2, 3, 2, 4, 5))
        offset_3d = torch.tensor(
            memoryview(bits.view(np.float32)), dtype=torch.float32
        ).reshape((3, 2, 3, 2, 4, 5))[1]
        strided_3d = contiguous_3d.transpose(0, 4)
        empty_3d = torch.zeros((2, 0, 4, 5, 6), dtype=torch.float32)
        return (
            (
                torch.channels_last,
                (
                    ("contiguous", contiguous_2d, (24, 1, 12, 3)),
                    ("offset", offset_2d, (24, 1, 12, 3)),
                    ("strided", strided_2d, (12, 1, 6, 3)),
                    ("empty", empty_2d, (0, 1, 0, 0)),
                ),
            ),
            (
                torch.channels_last_3d,
                (
                    ("contiguous", contiguous_3d, (120, 1, 60, 15, 3)),
                    ("offset", offset_3d, (120, 1, 60, 15, 3)),
                    ("strided", strided_3d, (48, 1, 24, 6, 3)),
                    ("empty", empty_3d, (0, 1, 0, 0, 0)),
                ),
            ),
        )

    def test_method_and_function_materialize_canonical_independent_copies(self):
        for memory_format, cases in self.make_layout_suites():
            for case, source, expected_stride in cases:
                expected_bits = self.tensor_bits(source).copy()
                for operation_name, operation in self.clone_operations(memory_format):
                    with self.subTest(
                        memory_format=memory_format,
                        case=case,
                        operation=operation_name,
                    ):
                        copied = operation(source)
                        self.assertIsNot(copied, source)
                        self.assertFalse(copied.is_set_to(source))
                        if source.numel() != 0:
                            self.assertNotEqual(
                                copied.data_ptr(), source.data_ptr()
                            )
                        self.assertEqual(copied.shape, source.shape)
                        self.assertEqual(copied.stride(), expected_stride)
                        self.assertEqual(copied.storage_offset(), 0)
                        self.assertTrue(
                            copied.is_contiguous(memory_format=memory_format)
                        )
                        self.assertIs(copied.dtype, source.dtype)
                        self.assertEqual(copied.device, source.device)
                        np.testing.assert_array_equal(
                            self.tensor_bits(copied), expected_bits
                        )

    def test_autograd_no_grad_and_source_lifetime(self):
        suites = (
            (torch.channels_last, (2, 3, 2, 4), (0, 3), (12, 1, 6, 3)),
            (
                torch.channels_last_3d,
                (2, 3, 2, 4, 5),
                (0, 4),
                (48, 1, 24, 6, 3),
            ),
        )
        for memory_format, shape, dimensions, expected_stride in suites:
            for operation_name, operation in self.clone_operations(memory_format):
                with self.subTest(
                    memory_format=memory_format,
                    operation=operation_name,
                    mode="autograd",
                ):
                    leaf = torch.ones(shape, requires_grad=True)
                    source = (leaf * 3.0).transpose(*dimensions)
                    copied = operation(source)
                    self.assertEqual(copied.stride(), expected_stride)
                    self.assertTrue(copied.requires_grad)
                    self.assertFalse(copied.is_leaf)
                    self.assertFalse(copied.is_set_to(source))
                    del source
                    gc.collect()

                    copied.sum().backward()
                    np.testing.assert_array_equal(
                        np.asarray(leaf.grad), np.full(shape, 3.0)
                    )

                with self.subTest(
                    memory_format=memory_format,
                    operation=operation_name,
                    mode="no_grad",
                ):
                    leaf = torch.ones(shape, requires_grad=True)
                    source = (leaf * 3.0).transpose(*dimensions)
                    expected_shape = source.shape
                    with torch.no_grad():
                        copied = operation(source)
                    self.assertEqual(copied.stride(), expected_stride)
                    self.assertFalse(copied.requires_grad)
                    self.assertTrue(copied.is_leaf)
                    self.assertIs(copied.dtype, source.dtype)
                    self.assertEqual(copied.device, source.device)
                    del source, leaf
                    gc.collect()
                    np.testing.assert_array_equal(
                        np.asarray(copied), np.full(expected_shape, 3.0)
                    )

    def test_rank_errors(self):
        suites = (
            (
                torch.channels_last,
                4,
                ((), (2, 3, 4), (1, 2, 3, 4, 5)),
            ),
            (
                torch.channels_last_3d,
                5,
                ((), (2, 3, 4, 5), (1, 2, 3, 4, 5, 6)),
            ),
        )
        for memory_format, expected_rank, shapes in suites:
            format_name = str(memory_format).removeprefix("torch.")
            for shape in shapes:
                source = torch.ones(shape)
                for operation_name, operation in self.clone_operations(memory_format):
                    with self.subTest(
                        memory_format=memory_format,
                        shape=shape,
                        operation=operation_name,
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            f"required rank {expected_rank} tensor to use "
                            f"{format_name} format",
                        ):
                            operation(source)


if __name__ == "__main__":
    unittest.main()
