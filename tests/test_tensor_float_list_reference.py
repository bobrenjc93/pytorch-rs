import struct
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def float32_from_bits(bits):
    return float(np.array(bits, dtype=np.uint32).view(np.float32)[()])


def float64_from_bits(bits):
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def float32_bits(tensor):
    value = tensor.detach() if tensor.requires_grad else tensor
    return np.asarray(value).view(np.uint32).reshape(-1).tolist()


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFloatListReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "tensor float-list differentials require pinned PyTorch 2.13.0"
            )

    def tensor_contract(self, tensor):
        return {
            "shape": tuple(tensor.shape),
            "stride": tensor.stride(),
            "storage_offset": tensor.storage_offset(),
            "numel": tensor.numel(),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "layout": str(tensor.layout),
            "requires_grad": tensor.requires_grad,
            "is_leaf": tensor.is_leaf,
            "bits": float32_bits(tensor),
        }

    def test_exact_float_values_metadata_and_autograd_match_pytorch_2_13(self):
        values = [
            0.0,
            -0.0,
            1.0,
            -2.5,
            float32_from_bits(0x00000001),
            float32_from_bits(0x007FFFFF),
            float32_from_bits(0x00800000),
            float32_from_bits(0x7F7FFFFF),
            float("inf"),
            float("-inf"),
            float32_from_bits(0x7FC12345),
            float32_from_bits(0xFFC12345),
            float32_from_bits(0x7F812345),
            float32_from_bits(0xFF812345),
            float64_from_bits(0x7FF8000000000001),
            float64_from_bits(0xFFF8000000000001),
        ]
        actual = torch.tensor(
            values,
            dtype=torch.float32,
            device="cpu",
            requires_grad=True,
        )
        expected = reference_torch.tensor(
            values,
            dtype=reference_torch.float32,
            device="cpu",
            requires_grad=True,
        )
        self.assertEqual(self.tensor_contract(actual), self.tensor_contract(expected))

        actual.sum().backward()
        expected.sum().backward()
        self.assertEqual(
            self.tensor_contract(actual.grad),
            self.tensor_contract(expected.grad),
        )

    def test_empty_nested_mixed_and_subclassed_inputs_match_pytorch_2_13(self):
        class ListSubclass(list):
            pass

        class FloatSubclass(float):
            def __float__(self):
                return 99.0

        class CustomNumeric:
            def __float__(self):
                return 3.5

        cases = (
            ("empty", []),
            ("nested", [[1.0, 2.0], [3.0, 4.0]]),
            ("mixed", [1.0, 2, 3.0]),
            ("list subclass", ListSubclass([1.0, 2.0])),
            ("float subclass", [1.0, FloatSubclass(2.5)]),
            ("custom numeric", [1.0, CustomNumeric()]),
        )
        for case, values in cases:
            with self.subTest(case=case):
                actual = torch.tensor(values, dtype=torch.float32, device="cpu")
                expected = reference_torch.tensor(
                    values,
                    dtype=reference_torch.float32,
                    device="cpu",
                )
                self.assertEqual(
                    self.tensor_contract(actual),
                    self.tensor_contract(expected),
                )


if __name__ == "__main__":
    unittest.main()
