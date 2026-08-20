import gc
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CloneChannelsLastReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "channels-last clone differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_bits(tensor):
        values = np.ascontiguousarray(np.asarray(tensor))
        return values.reshape(-1).view(np.uint32)

    @staticmethod
    def clone(module, tensor, functional, memory_format_name):
        memory_format = getattr(module, memory_format_name)
        if functional:
            return module.clone(input=tensor, memory_format=memory_format)
        return tensor.clone(memory_format=memory_format)

    def make_layout_suites(self, module):
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
        contiguous_2d = module.tensor(
            memoryview(bits[:48].view(np.float32)), dtype=module.float32
        ).reshape((2, 3, 2, 4))
        offset_2d = module.tensor(
            memoryview(bits[:144].view(np.float32)), dtype=module.float32
        ).reshape((3, 2, 3, 2, 4))[1]
        strided_2d = contiguous_2d.transpose(0, 3)
        empty_2d = module.zeros((2, 0, 4, 5), dtype=module.float32)

        contiguous_3d = module.tensor(
            memoryview(bits[:240].view(np.float32)), dtype=module.float32
        ).reshape((2, 3, 2, 4, 5))
        offset_3d = module.tensor(
            memoryview(bits.view(np.float32)), dtype=module.float32
        ).reshape((3, 2, 3, 2, 4, 5))[1]
        strided_3d = contiguous_3d.transpose(0, 4)
        empty_3d = module.zeros((2, 0, 4, 5, 6), dtype=module.float32)
        return (
            (
                "channels_last",
                (
                    (contiguous_2d, (24, 1, 12, 3)),
                    (offset_2d, (24, 1, 12, 3)),
                    (strided_2d, (12, 1, 6, 3)),
                    (empty_2d, (0, 1, 0, 0)),
                ),
            ),
            (
                "channels_last_3d",
                (
                    (contiguous_3d, (120, 1, 60, 15, 3)),
                    (offset_3d, (120, 1, 60, 15, 3)),
                    (strided_3d, (48, 1, 24, 6, 3)),
                    (empty_3d, (0, 1, 0, 0, 0)),
                ),
            ),
        )

    def test_layout_values_and_independence_match_pytorch_2_13(self):
        actual_suites = self.make_layout_suites(torch)
        expected_suites = self.make_layout_suites(reference_torch)
        for (memory_format_name, actual_cases), (
            expected_memory_format_name,
            expected_cases,
        ) in zip(actual_suites, expected_suites, strict=True):
            self.assertEqual(memory_format_name, expected_memory_format_name)
            actual_memory_format = getattr(torch, memory_format_name)
            expected_memory_format = getattr(reference_torch, memory_format_name)
            for case, (
                (actual_source, canonical_stride),
                (expected_source, expected_canonical_stride),
            ) in enumerate(
                zip(actual_cases, expected_cases, strict=True)
            ):
                self.assertEqual(canonical_stride, expected_canonical_stride)
                for functional in (False, True):
                    with self.subTest(
                        memory_format=memory_format_name,
                        case=case,
                        functional=functional,
                    ):
                        actual = self.clone(
                            torch,
                            actual_source,
                            functional,
                            memory_format_name,
                        )
                        expected = self.clone(
                            reference_torch,
                            expected_source,
                            functional,
                            memory_format_name,
                        )
                        self.assertEqual(actual.shape, tuple(expected.shape))
                        self.assertEqual(actual.stride(), canonical_stride)
                        self.assertEqual(actual.stride(), expected.stride())
                        self.assertEqual(
                            actual.storage_offset(), expected.storage_offset()
                        )
                        self.assertEqual(actual.storage_offset(), 0)
                        self.assertEqual(
                            actual.is_contiguous(memory_format=actual_memory_format),
                            expected.is_contiguous(
                                memory_format=expected_memory_format
                            ),
                        )
                        self.assertTrue(
                            actual.is_contiguous(memory_format=actual_memory_format)
                        )
                        self.assertEqual(
                            actual.is_set_to(actual_source),
                            expected.is_set_to(expected_source),
                        )
                        self.assertFalse(actual.is_set_to(actual_source))
                        if actual_source.numel() != 0:
                            self.assertNotEqual(
                                actual.data_ptr(), actual_source.data_ptr()
                            )
                            self.assertNotEqual(
                                expected.data_ptr(), expected_source.data_ptr()
                            )
                        self.assertEqual(
                            actual.dtype is actual_source.dtype,
                            expected.dtype is expected_source.dtype,
                        )
                        self.assertEqual(str(actual.device), str(expected.device))
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual), self.tensor_bits(expected)
                        )
                        np.testing.assert_array_equal(
                            self.tensor_bits(actual), self.tensor_bits(actual_source)
                        )

    def autograd_outcome(
        self, module, functional, memory_format_name, shape, dimensions
    ):
        leaf = module.ones(shape, dtype=module.float32, requires_grad=True)
        source = (leaf * 3.0).transpose(*dimensions)
        copied = self.clone(module, source, functional, memory_format_name)
        metadata = (
            copied.shape,
            copied.stride(),
            copied.storage_offset(),
            copied.requires_grad,
            copied.is_leaf,
            copied.is_set_to(source),
            copied.dtype is source.dtype,
            str(copied.device),
        )
        copied_bits = self.tensor_bits(copied.detach()).copy()
        del source
        gc.collect()
        copied.sum().backward()
        gradient_bits = self.tensor_bits(leaf.grad).copy()
        return metadata, copied_bits, gradient_bits

    def no_grad_outcome(
        self, module, functional, memory_format_name, shape, dimensions
    ):
        leaf = module.ones(shape, dtype=module.float32, requires_grad=True)
        source = (leaf * 3.0).transpose(*dimensions)
        with module.no_grad():
            copied = self.clone(module, source, functional, memory_format_name)
        metadata = (
            copied.shape,
            copied.stride(),
            copied.storage_offset(),
            copied.requires_grad,
            copied.is_leaf,
            copied.dtype is source.dtype,
            str(copied.device),
        )
        del source, leaf
        gc.collect()
        return metadata, self.tensor_bits(copied).copy()

    def test_autograd_no_grad_and_source_lifetime_match_pytorch_2_13(self):
        suites = (
            ("channels_last", (2, 3, 2, 4), (0, 3)),
            ("channels_last_3d", (2, 3, 2, 4, 5), (0, 4)),
        )
        for memory_format_name, shape, dimensions in suites:
            for functional in (False, True):
                with self.subTest(
                    memory_format=memory_format_name,
                    functional=functional,
                    mode="autograd",
                ):
                    actual_metadata, actual_values, actual_gradient = (
                        self.autograd_outcome(
                            torch,
                            functional,
                            memory_format_name,
                            shape,
                            dimensions,
                        )
                    )
                    expected_metadata, expected_values, expected_gradient = (
                        self.autograd_outcome(
                            reference_torch,
                            functional,
                            memory_format_name,
                            shape,
                            dimensions,
                        )
                    )
                    self.assertEqual(actual_metadata, expected_metadata)
                    np.testing.assert_array_equal(actual_values, expected_values)
                    np.testing.assert_array_equal(
                        actual_gradient, expected_gradient
                    )

                with self.subTest(
                    memory_format=memory_format_name,
                    functional=functional,
                    mode="no_grad",
                ):
                    actual_metadata, actual_values = self.no_grad_outcome(
                        torch,
                        functional,
                        memory_format_name,
                        shape,
                        dimensions,
                    )
                    expected_metadata, expected_values = self.no_grad_outcome(
                        reference_torch,
                        functional,
                        memory_format_name,
                        shape,
                        dimensions,
                    )
                    self.assertEqual(actual_metadata, expected_metadata)
                    np.testing.assert_array_equal(actual_values, expected_values)

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_rank_errors_match_pytorch_2_13(self):
        suites = (
            ("channels_last", ((), (2, 3, 4), (1, 2, 3, 4, 5))),
            (
                "channels_last_3d",
                ((), (2, 3, 4, 5), (1, 2, 3, 4, 5, 6)),
            ),
        )
        for memory_format_name, shapes in suites:
            for shape in shapes:
                actual = torch.ones(shape, dtype=torch.float32)
                expected = reference_torch.ones(
                    shape, dtype=reference_torch.float32
                )
                for functional in (False, True):
                    with self.subTest(
                        memory_format=memory_format_name,
                        shape=shape,
                        functional=functional,
                    ):
                        self.assert_error_matches(
                            lambda functional=functional: self.clone(
                                torch,
                                actual,
                                functional,
                                memory_format_name,
                            ),
                            lambda functional=functional: self.clone(
                                reference_torch,
                                expected,
                                functional,
                                memory_format_name,
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
