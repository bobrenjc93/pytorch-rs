import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorCopyConstructionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "tensor copy-construction differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_bits(tensor):
        values = np.ascontiguousarray(np.asarray(tensor.detach()))
        return values.reshape(-1).view(np.uint32)

    def make_sources(self, module):
        patterns = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
                0x3F80_0000,
            ),
            dtype=np.uint32,
        )
        bits = np.resize(patterns, 96)
        contiguous = module.tensor(
            memoryview(bits[:48].view(np.float32)), dtype=module.float32
        ).reshape((2, 3, 2, 4))
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("contiguous", contiguous),
            ("offset", contiguous[1]),
            ("transposed", contiguous.transpose(0, 3)),
            (
                "channels-last",
                contiguous.contiguous(memory_format=module.channels_last),
            ),
        )

    def copy_outcome(self, module, source, requires_grad):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copied = module.tensor(
                source,
                dtype=module.float32,
                device=module.device("cpu"),
                requires_grad=requires_grad,
            )
        metadata = (
            tuple(copied.shape),
            copied.stride(),
            copied.storage_offset(),
            copied.requires_grad,
            copied.is_leaf,
            copied.grad is None,
            copied.is_set_to(source),
            copied.data_ptr() == source.data_ptr(),
            str(copied.dtype).replace("torch_rs", "torch"),
            str(copied.device),
        )
        warning_metadata = tuple(
            (item.category.__name__, str(item.message)) for item in caught
        )
        return copied, metadata, warning_metadata, self.tensor_bits(copied).copy()

    def test_values_layouts_offsets_warnings_and_leaf_metadata_match_pytorch_2_13(self):
        actual_sources = self.make_sources(torch)
        expected_sources = self.make_sources(reference_torch)
        for (actual_case, actual_source), (expected_case, expected_source) in zip(
            actual_sources, expected_sources, strict=True
        ):
            self.assertEqual(actual_case, expected_case)
            for requires_grad in (False, True):
                with self.subTest(case=actual_case, requires_grad=requires_grad):
                    actual, actual_metadata, actual_warnings, actual_bits = (
                        self.copy_outcome(torch, actual_source, requires_grad)
                    )
                    expected, expected_metadata, expected_warnings, expected_bits = (
                        self.copy_outcome(
                            reference_torch, expected_source, requires_grad
                        )
                    )
                    self.assertEqual(actual_metadata, expected_metadata)
                    self.assertEqual(actual_warnings, expected_warnings)
                    self.assertEqual(actual.storage_offset(), 0)
                    self.assertEqual(expected.storage_offset(), 0)
                    self.assertFalse(actual.is_set_to(actual_source))
                    self.assertFalse(expected.is_set_to(expected_source))
                    np.testing.assert_array_equal(actual_bits, expected_bits)

    def graph_outcome(self, module):
        leaf = module.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = (leaf * 3.0).transpose(0, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copied = module.tensor(source, requires_grad=True)
        copied.sum().backward()
        return (
            (
                copied.stride(),
                copied.storage_offset(),
                copied.requires_grad,
                copied.is_leaf,
                copied.is_set_to(source),
                leaf.grad is None,
            ),
            self.tensor_bits(copied.grad).copy(),
            tuple((item.category.__name__, str(item.message)) for item in caught),
        )

    def test_detached_leaf_autograd_boundary_matches_pytorch_2_13(self):
        actual_metadata, actual_grad, actual_warnings = self.graph_outcome(torch)
        expected_metadata, expected_grad, expected_warnings = self.graph_outcome(
            reference_torch
        )
        self.assertEqual(actual_metadata, expected_metadata)
        self.assertEqual(actual_warnings, expected_warnings)
        np.testing.assert_array_equal(actual_grad, expected_grad)

    def gradient_layout_outcome(self, module, source):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            copied = module.tensor(source, requires_grad=True)
        if copied.numel() == 0:
            copied.sum().backward()
        else:
            weights = module.tensor(
                np.arange(1, copied.numel() + 1, dtype=np.float32)
                .reshape(tuple(copied.shape))
                .tolist(),
                dtype=module.float32,
            )
            (copied * weights).sum().backward()
        first_gradient = copied.grad
        copied.sum().backward()
        return (
            copied.stride(),
            first_gradient.shape,
            first_gradient.stride(),
            first_gradient.storage_offset(),
            copied.grad is first_gradient,
            self.tensor_bits(first_gradient).copy(),
            self.tensor_bits(copied.grad).copy(),
        )

    def test_noncontiguous_and_empty_gradient_layouts_match_pytorch_2_13(self):
        actual_sources = dict(self.make_sources(torch))
        expected_sources = dict(self.make_sources(reference_torch))
        for case in ("transposed", "channels-last", "empty-offset"):
            with self.subTest(case=case):
                actual = self.gradient_layout_outcome(torch, actual_sources[case])
                expected = self.gradient_layout_outcome(
                    reference_torch, expected_sources[case]
                )
                self.assertEqual(actual[:-2], expected[:-2])
                np.testing.assert_array_equal(actual[-2], expected[-2])
                np.testing.assert_array_equal(actual[-1], expected[-1])

    def test_existing_non_tensor_paths_remain_warning_free(self):
        actual_cases = (
            -0.0,
            [[1.0, 2.0], [3.0, 4.0]],
            memoryview(np.asarray([1.25, -2.5], dtype=np.float32)),
        )
        expected_cases = (
            -0.0,
            [[1.0, 2.0], [3.0, 4.0]],
            memoryview(np.asarray([1.25, -2.5], dtype=np.float32)),
        )
        for case, (actual_data, expected_data) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                with warnings.catch_warnings(record=True) as actual_warnings:
                    warnings.simplefilter("always")
                    actual = torch.tensor(actual_data, dtype=torch.float32)
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter("always")
                    expected = reference_torch.tensor(
                        expected_data, dtype=reference_torch.float32
                    )
                self.assertEqual(actual_warnings, [])
                self.assertEqual(expected_warnings, [])
                self.assertEqual(actual.shape, tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                np.testing.assert_array_equal(
                    self.tensor_bits(actual), self.tensor_bits(expected)
                )


if __name__ == "__main__":
    unittest.main()
