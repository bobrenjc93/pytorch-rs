import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class CloneChannelsLastReferenceTests(unittest.TestCase):
    @staticmethod
    def special_values(shape):
        bits = np.array(
            [
                0x00000000,
                0x80000000,
                0x7FC12345,
                0x7F800000,
                0xFF800000,
                0x3F800000,
            ],
            dtype=np.uint32,
        )
        elements = int(np.prod(shape, dtype=np.int64))
        return np.resize(bits, elements).view(np.float32).reshape(shape)

    def make_case(self, module, case):
        if case == "contiguous":
            values = self.special_values((2, 3, 2, 2))
            return module.tensor(values.tolist(), dtype=module.float32, device="cpu")
        if case == "offset":
            values = self.special_values((3, 2, 3, 2, 2))
            return module.tensor(values.tolist(), dtype=module.float32, device="cpu")[1]
        if case == "strided":
            values = self.special_values((2, 3, 2, 4))
            return module.tensor(
                values.tolist(), dtype=module.float32, device="cpu"
            ).transpose(2, 3)
        if case == "channels_last":
            values = self.special_values((2, 3, 2, 2))
            return module.tensor(
                values.tolist(), dtype=module.float32, device="cpu"
            ).contiguous(memory_format=module.channels_last)

        empty_shapes = {
            "empty_channels": (2, 0, 4, 5),
            "empty_height": (2, 3, 0, 5),
            "empty_width": (2, 3, 4, 0),
            "empty_batch": (0, 3, 4, 5),
        }
        return module.zeros(empty_shapes[case], dtype=module.float32, device="cpu")

    @staticmethod
    def clone(module, tensor, operation):
        if operation == "method":
            return tensor.clone(memory_format=module.channels_last)
        return module.clone(tensor, memory_format=module.channels_last)

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

    def test_rank_four_layout_values_independence_and_lifetime_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        cases = (
            "contiguous",
            "offset",
            "strided",
            "channels_last",
            "empty_channels",
            "empty_height",
            "empty_width",
            "empty_batch",
        )
        for case in cases:
            for operation in ("method", "function"):
                with self.subTest(case=case, operation=operation):
                    actual_source = self.make_case(torch, case)
                    expected_source = self.make_case(reference_torch, case)
                    actual = self.clone(torch, actual_source, operation)
                    expected = self.clone(reference_torch, expected_source, operation)

                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(actual.storage_offset(), expected.storage_offset())
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertEqual(
                        actual.is_contiguous(memory_format=torch.channels_last),
                        expected.is_contiguous(
                            memory_format=reference_torch.channels_last
                        ),
                    )
                    self.assertIs(actual.dtype, actual_source.dtype)
                    self.assertIs(expected.dtype, expected_source.dtype)
                    self.assertEqual(str(actual.dtype), str(expected.dtype))
                    self.assertEqual(actual.device, actual_source.device)
                    self.assertEqual(expected.device, expected_source.device)
                    self.assertEqual(str(actual.device), str(expected.device))
                    self.assertFalse(actual.is_set_to(actual_source))
                    self.assertFalse(expected.is_set_to(expected_source))

                    actual_bits = np.asarray(actual).view(np.uint32).copy()
                    expected_bits = expected.numpy().view(np.uint32).copy()
                    np.testing.assert_array_equal(actual_bits, expected_bits)

                    if actual.numel() != 0:
                        self.assertNotEqual(actual.data_ptr(), actual_source.data_ptr())
                        self.assertNotEqual(expected.data_ptr(), expected_source.data_ptr())
                        first = (0,) * len(actual.shape)
                        np.asarray(actual_source)[first] = 42.25
                        expected_source.numpy()[first] = 42.25
                        np.testing.assert_array_equal(
                            np.asarray(actual).view(np.uint32), actual_bits
                        )
                        np.testing.assert_array_equal(
                            expected.numpy().view(np.uint32), expected_bits
                        )

                    del actual_source, expected_source
                    np.testing.assert_array_equal(
                        np.asarray(actual).view(np.uint32), actual_bits
                    )
                    np.testing.assert_array_equal(
                        expected.numpy().view(np.uint32), expected_bits
                    )

    def run_autograd_case(self, module, operation):
        values = np.arange(48, dtype=np.float32).reshape(2, 3, 2, 4)
        leaf = module.tensor(values.tolist(), requires_grad=True)
        source = leaf.transpose(2, 3)
        cloned = self.clone(module, source, operation)
        metadata = (
            cloned.shape,
            cloned.stride(),
            cloned.storage_offset(),
            cloned.requires_grad,
            cloned.is_leaf,
            str(cloned.dtype),
            str(cloned.device),
        )
        del source

        weights = np.arange(1, 49, dtype=np.float32).reshape(2, 3, 4, 2)
        (cloned * module.tensor(weights.tolist())).sum().backward()
        gradient = np.asarray(leaf.grad).copy()

        with module.no_grad():
            untracked = self.clone(module, leaf, operation)
        no_grad_metadata = (
            untracked.shape,
            untracked.stride(),
            untracked.storage_offset(),
            untracked.requires_grad,
            untracked.is_leaf,
        )
        return metadata, gradient, no_grad_metadata

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for operation in ("method", "function"):
            with self.subTest(operation=operation):
                actual = self.run_autograd_case(torch, operation)
                expected = self.run_autograd_case(reference_torch, operation)
                self.assertEqual(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
                self.assertEqual(actual[2], expected[2])

    def test_rank_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        for rank in range(7):
            if rank == 4:
                continue
            actual = torch.zeros((2,) * rank)
            expected = reference_torch.zeros((2,) * rank)
            for operation in ("method", "function"):
                with self.subTest(rank=rank, operation=operation):
                    self.assert_error_matches(
                        lambda: self.clone(torch, actual, operation),
                        lambda: self.clone(reference_torch, expected, operation),
                    )
