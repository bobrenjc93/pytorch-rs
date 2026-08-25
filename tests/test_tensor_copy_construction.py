import gc
import inspect
import unittest
import warnings

import numpy as np
import torch_rs as torch


COPY_CONSTRUCTION_WARNING = (
    "To copy construct from a tensor, it is recommended to use "
    "sourceTensor.detach().clone() or "
    "sourceTensor.detach().clone().requires_grad_(True), rather than "
    "torch.tensor(sourceTensor)."
)


class TensorCopyConstructionTests(unittest.TestCase):
    @staticmethod
    def tensor_bits(tensor):
        values = np.ascontiguousarray(np.asarray(tensor.detach()))
        return values.reshape(-1).view(np.uint32)

    def make_sources(self):
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
        contiguous = torch.tensor(
            memoryview(bits[:48].view(np.float32)), dtype=torch.float32
        ).reshape((2, 3, 2, 4))
        return (
            ("scalar", torch.tensor(-0.0, dtype=torch.float32)),
            (
                "empty-offset",
                torch.zeros((2, 0, 3), dtype=torch.float32).transpose(0, 2)[1],
            ),
            ("contiguous", contiguous),
            ("offset", contiguous[1]),
            ("transposed", contiguous.transpose(0, 3)),
            (
                "channels-last",
                contiguous.contiguous(memory_format=torch.channels_last),
            ),
        )

    def assert_copy(self, source, requires_grad):
        expected_bits = self.tensor_bits(source).copy()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warning_line = inspect.currentframe().f_lineno + 1
            copied = torch.tensor(
                source,
                dtype=torch.float32,
                device=torch.device("cpu"),
                requires_grad=requires_grad,
            )

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, UserWarning)
        self.assertEqual(str(caught[0].message), COPY_CONSTRUCTION_WARNING)
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(caught[0].lineno, warning_line)
        self.assertIsNot(copied, source)
        self.assertFalse(copied.is_set_to(source))
        if source.numel() != 0:
            self.assertNotEqual(copied.data_ptr(), source.data_ptr())
        self.assertEqual(copied.shape, source.shape)
        self.assertEqual(copied.stride(), source.stride())
        self.assertEqual(copied.storage_offset(), 0)
        self.assertIs(copied.dtype, source.dtype)
        self.assertEqual(copied.device, source.device)
        self.assertIs(copied.requires_grad, requires_grad)
        self.assertTrue(copied.is_leaf)
        self.assertIsNone(copied.grad)
        np.testing.assert_array_equal(self.tensor_bits(copied), expected_bits)
        return copied

    def test_native_tensor_inputs_are_independent_layout_preserving_copies(self):
        retained = []
        for case, source in self.make_sources():
            for requires_grad in (False, True):
                with self.subTest(case=case, requires_grad=requires_grad):
                    retained.append(self.assert_copy(source, requires_grad))

        del source
        gc.collect()
        for copied in retained:
            self.tensor_bits(copied)

    def test_copy_construction_severs_the_source_graph_and_honors_no_grad(self):
        leaf = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        source = (leaf * 3.0).transpose(0, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copied = torch.tensor(source, requires_grad=True)

        self.assertEqual([str(item.message) for item in caught], [COPY_CONSTRUCTION_WARNING])
        self.assertTrue(source.requires_grad)
        self.assertFalse(source.is_leaf)
        self.assertTrue(copied.requires_grad)
        self.assertTrue(copied.is_leaf)
        self.assertEqual(copied.stride(), source.stride())
        copied.sum().backward()
        self.assertIsNone(leaf.grad)
        np.testing.assert_array_equal(
            np.asarray(copied.grad), np.ones(source.shape, dtype=np.float32)
        )

        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            no_grad_copy = torch.tensor(source, requires_grad=True)
        self.assertTrue(no_grad_copy.requires_grad)
        self.assertTrue(no_grad_copy.is_leaf)

    def test_warning_as_error_precedes_copy_construction(self):
        source = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        pointer = source.data_ptr()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with self.assertRaisesRegex(UserWarning, "^To copy construct from a tensor"):
                torch.tensor(source)
        self.assertEqual(source.data_ptr(), pointer)
        self.assertTrue(source.requires_grad)
        self.assertIsNone(source.grad)

    def test_scalar_sequence_and_buffer_paths_do_not_emit_copy_warnings(self):
        cases = (
            ("scalar", -0.0, (), -0.0),
            ("sequence", [[1.0, 2.0], [3.0, 4.0]], (2, 2), [[1.0, 2.0], [3.0, 4.0]]),
            (
                "buffer",
                memoryview(np.asarray([1.25, -2.5], dtype=np.float32)),
                (2,),
                [1.25, -2.5],
            ),
        )
        for case, data, shape, values in cases:
            with self.subTest(case=case):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = torch.tensor(data, dtype=torch.float32)
                self.assertEqual(caught, [])
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), values)

    def test_invalid_options_are_rejected_before_the_copy_warning(self):
        source = torch.tensor([1.0])
        cases = (
            (
                lambda: torch.tensor(source, dtype=object()),
                TypeError,
                "argument 'dtype' must be torch.dtype",
            ),
            (
                lambda: torch.tensor(source, device=object()),
                TypeError,
                "argument 'device' must be torch.device",
            ),
            (
                lambda: torch.tensor(source, requires_grad=1),
                TypeError,
                "argument 'requires_grad' must be bool",
            ),
            (
                lambda: torch.tensor(source, device="cuda:0"),
                RuntimeError,
                "device 'cuda:0' is not supported",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    with self.assertRaisesRegex(error_type, message):
                        call()
                self.assertEqual(caught, [])


if __name__ == "__main__":
    unittest.main()
