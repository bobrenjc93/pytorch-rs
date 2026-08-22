import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorFloatListReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "tensor float-list differentials require pinned PyTorch 2.13.0"
            )

    def float_list_contract(self, module, values):
        result = module.tensor(
            values,
            dtype=module.float32,
            device="cpu",
            requires_grad=True,
        )
        result.sum().backward()
        return {
            "shape": tuple(result.shape),
            "stride": result.stride(),
            "storage_offset": result.storage_offset(),
            "dtype": str(result.dtype),
            "device": str(result.device),
            "requires_grad": result.requires_grad,
            "is_leaf": result.is_leaf,
            "bits": np.asarray(result.detach()).view(np.uint32).copy(),
            "gradient": np.asarray(result.grad).copy(),
        }

    def test_flat_builtin_float_values_metadata_and_autograd_match_pytorch_2_13(self):
        source_bits = np.array(
            [
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7FC1_2345,
                0xFFC5_4321,
            ],
            dtype=np.uint32,
        )
        values = source_bits.view(np.float32).astype(np.float64).tolist()

        actual = self.float_list_contract(torch, values)
        expected = self.float_list_contract(reference_torch, values)
        np.testing.assert_array_equal(actual.pop("bits"), expected.pop("bits"))
        np.testing.assert_array_equal(
            actual.pop("gradient"), expected.pop("gradient")
        )
        self.assertEqual(actual, expected)

    def fallback_contract(self, module, values):
        result = module.tensor(values, dtype=module.float32)
        return (
            tuple(result.shape),
            result.stride(),
            str(result.dtype),
            str(result.device),
            np.asarray(result).reshape(-1).view(np.uint32).copy(),
        )

    def test_nested_mixed_and_subclassed_inputs_match_pytorch_2_13(self):
        class FloatSubclass(float):
            pass

        class ListSubclass(list):
            pass

        cases = (
            [[1.0, 2.0], [3.0, 4.0]],
            [1.0, 2, 3.0],
            [FloatSubclass(4.5), FloatSubclass(-5.5)],
            ListSubclass([6.0, 7.0]),
        )

        for values in cases:
            with self.subTest(values=values):
                actual = self.fallback_contract(torch, values)
                expected = self.fallback_contract(reference_torch, values)
                self.assertEqual(actual[:-1], expected[:-1])
                np.testing.assert_array_equal(actual[-1], expected[-1])


if __name__ == "__main__":
    unittest.main()
