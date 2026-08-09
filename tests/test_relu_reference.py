import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ReluReferenceTests(unittest.TestCase):
    def test_stable_nan_and_nonzero_semantics_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        for elements in (5, 13, 37, 67):
            values = np.empty(elements, dtype=np.float32)
            for index in range(elements):
                values[index] = (
                    np.nan
                    if index % 6 == 0 or index + 1 == elements
                    else (-np.inf, -3.5, -0.0, 2.25, np.inf)[(index - 1) % 6]
                )

            native_output = torch.tensor(values.tolist()).relu()
            native = np.asarray(native_output, dtype=np.float32)
            expected_tensor = reference_torch.tensor(
                values, dtype=reference_torch.float32
            ).relu()
            expected = expected_tensor.cpu().numpy()

            with self.subTest(elements=elements):
                self.assertEqual(native_output.shape, expected_tensor.shape)
                self.assertEqual(native_output.stride(), expected_tensor.stride())
                self.assertIs(native_output.dtype, torch.float32)
                self.assertEqual(native_output.device, torch.device("cpu"))
                np.testing.assert_array_equal(np.isnan(native), np.isnan(expected))

                nonzero = ~np.isnan(expected) & (expected != 0.0)
                np.testing.assert_array_equal(native[nonzero], expected[nonzero])
                self.assertTrue(np.all(native[~np.isnan(expected) & ~nonzero] == 0.0))


if __name__ == "__main__":
    unittest.main()
