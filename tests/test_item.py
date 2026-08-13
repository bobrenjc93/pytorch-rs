import re
import struct
import unittest

import numpy as np
import torch_rs as torch


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


class TensorItemTests(unittest.TestCase):
    def test_cardinality_errors_match_pytorch_2_13(self):
        cases = (
            (torch.zeros((0,)), 0),
            (torch.zeros((2,)), 2),
            (torch.zeros((2, 0, 3)).transpose(0, 2), 0),
            (torch.zeros((2, 3)).transpose(0, 1), 6),
        )
        for tensor, elements in cases:
            message = (
                f"a Tensor with {elements} elements cannot be converted to Scalar"
            )
            with self.subTest(shape=tensor.shape, stride=tensor.stride()):
                with self.assertRaisesRegex(
                    RuntimeError, f"^{re.escape(message)}$"
                ) as raised:
                    tensor.item()
                self.assertIs(type(raised.exception), RuntimeError)
                self.assertEqual(
                    f"{type(raised.exception).__name__}: {raised.exception}",
                    f"RuntimeError: {message}",
                )

    def test_scalar_offset_and_strided_values_are_bit_exact(self):
        for bits in SPECIAL_BITS:
            expected = float(
                np.asarray((bits,), dtype=np.uint32).view(np.float32)[0]
            )
            for layout, tensor, _, _ in item_layouts(torch, bits):
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    if layout == "scalar":
                        self.assertEqual(tensor.shape, ())
                        self.assertEqual(tensor.stride(), ())
                        self.assertEqual(tensor.storage_offset(), 0)
                    elif layout == "offset":
                        self.assertEqual(tensor.shape, ())
                        self.assertEqual(tensor.stride(), ())
                        self.assertEqual(tensor.storage_offset(), 1)
                    else:
                        self.assertEqual(tensor.shape, (1,))
                        self.assertEqual(tensor.stride(), (2,))
                        self.assertEqual(tensor.storage_offset(), 1)
                    self.assertEqual(
                        python_float_bits(tensor.item()),
                        python_float_bits(expected),
                    )

    def test_item_does_not_mutate_one_element_graphs(self):
        for bits in SPECIAL_BITS:
            expected = float(
                np.asarray((bits,), dtype=np.uint32).view(np.float32)[0]
            )
            for layout, tensor, leaf, expected_grad in item_layouts(
                torch, bits, requires_grad=True
            ):
                graph_before = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                actual = tensor.item()
                graph_after = (
                    tensor.requires_grad,
                    tensor.is_leaf,
                    leaf.requires_grad,
                    leaf.is_leaf,
                    leaf.grad,
                )
                with self.subTest(bits=f"{bits:#010x}", layout=layout):
                    self.assertEqual(python_float_bits(actual), python_float_bits(expected))
                    self.assertEqual(graph_after, graph_before)
                    tensor.backward()
                    self.assertEqual(leaf.grad.tolist(), expected_grad)


if __name__ == "__main__":
    unittest.main()
