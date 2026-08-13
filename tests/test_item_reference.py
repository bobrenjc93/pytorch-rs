import struct
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


SPECIAL_BITS = (
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC1_2345,
    0xFFC5_4321,
)


def python_float_bits(value):
    return struct.unpack("=Q", struct.pack("=d", value))[0]


def item_layouts(module, bits, *, requires_grad=False):
    values = np.asarray((0x3F80_0000, bits), dtype=np.uint32).view(np.float32)
    scalar_leaf = module.tensor(
        memoryview(values[1:]),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    scalar = scalar_leaf.reshape(())

    offset_leaf = module.tensor(
        memoryview(values),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    offset = offset_leaf[1]

    strided_leaf = module.tensor(
        memoryview(values),
        dtype=module.float32,
        requires_grad=requires_grad,
    )
    strided = strided_leaf.reshape((1, 2)).transpose(0, 1)[1]
    return (
        ("scalar", scalar, scalar_leaf, [1.0]),
        ("offset", offset, offset_leaf, [0.0, 1.0]),
        ("strided", strided, strided_leaf, [0.0, 1.0]),
    )


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorItemReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("item differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def test_cardinality_errors_match_pytorch_2_13(self):
        actual_cases = (
            torch.zeros((0,)),
            torch.zeros((2,)),
            torch.zeros((2, 0, 3)).transpose(0, 2),
            torch.zeros((2, 3)).transpose(0, 1),
        )
        expected_cases = (
            reference_torch.zeros((0,)),
            reference_torch.zeros((2,)),
            reference_torch.zeros((2, 0, 3)).transpose(0, 2),
            reference_torch.zeros((2, 3)).transpose(0, 1),
        )
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            with self.subTest(case=case):
                self.assertEqual(actual.shape, tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assert_error_matches(actual.item, expected.item)

    def test_scalar_offset_and_strided_values_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            actual_layouts = item_layouts(torch, bits)
            expected_layouts = item_layouts(reference_torch, bits)
            for actual_case, expected_case in zip(
                actual_layouts, expected_layouts, strict=True
            ):
                actual_layout, actual, _, _ = actual_case
                expected_layout, expected, _, _ = expected_case
                with self.subTest(bits=f"{bits:#010x}", layout=actual_layout):
                    self.assertEqual(actual_layout, expected_layout)
                    self.assertEqual(actual.shape, tuple(expected.shape))
                    self.assertEqual(actual.stride(), expected.stride())
                    self.assertEqual(
                        actual.storage_offset(), expected.storage_offset()
                    )
                    self.assertEqual(
                        python_float_bits(actual.item()),
                        python_float_bits(expected.item()),
                    )

    def test_graph_state_and_backward_match_pytorch_2_13(self):
        for bits in SPECIAL_BITS:
            outcomes = []
            for module in (torch, reference_torch):
                module_outcomes = []
                for layout, tensor, leaf, _ in item_layouts(
                    module, bits, requires_grad=True
                ):
                    graph_before = (
                        tensor.requires_grad,
                        tensor.is_leaf,
                        leaf.requires_grad,
                        leaf.is_leaf,
                        leaf.grad is None,
                    )
                    value = tensor.item()
                    graph_after = (
                        tensor.requires_grad,
                        tensor.is_leaf,
                        leaf.requires_grad,
                        leaf.is_leaf,
                        leaf.grad is None,
                    )
                    tensor.backward()
                    module_outcomes.append(
                        (
                            layout,
                            python_float_bits(value),
                            graph_before,
                            graph_after,
                            leaf.grad.tolist(),
                        )
                    )
                outcomes.append(module_outcomes)
            with self.subTest(bits=f"{bits:#010x}"):
                self.assertEqual(outcomes[0], outcomes[1])


if __name__ == "__main__":
    unittest.main()
