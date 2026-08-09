import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ExpReferenceTests(unittest.TestCase):
    def test_seeded_random_shapes_and_values_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0xE11E_213)
        smallest_subnormal = np.nextafter(np.float32(0), np.float32(1))

        shapes = [(), (0,), (2, 0, 5), (3, 1, 7)]
        for _ in range(28):
            rank = int(rng.integers(0, 6))
            shapes.append(tuple(int(value) for value in rng.integers(0, 9, size=rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            selector = rng.integers(0, 4, size=elements)
            values = np.empty(elements, dtype=np.float32)
            values[selector == 0] = rng.uniform(-105.0, 89.5, size=np.count_nonzero(selector == 0))
            values[selector == 1] = rng.normal(0.0, 24.0, size=np.count_nonzero(selector == 1))
            values[selector == 2] = rng.uniform(-1.0e-4, 1.0e-4, size=np.count_nonzero(selector == 2))
            values[selector == 3] = rng.choice(
                np.array(
                    [
                        -104.0,
                        -103.5,
                        -103.0,
                        -100.0,
                        -88.0,
                        -smallest_subnormal,
                        -0.0,
                        0.0,
                        smallest_subnormal,
                        1.0,
                        88.0,
                        88.75,
                        89.0,
                    ],
                    dtype=np.float32,
                ),
                size=np.count_nonzero(selector == 3),
            )
            values = values.reshape(shape)

            native_input = (
                torch.zeros(shape)
                if elements == 0
                else torch.tensor(values.item() if shape == () else values.tolist())
            )
            native_output = native_input.exp()
            native = np.asarray(native_output, dtype=np.float32)
            expected_tensor = reference_torch.tensor(values, dtype=reference_torch.float32).exp()
            expected = expected_tensor.cpu().numpy()

            with self.subTest(case=case, shape=shape):
                self.assertEqual(native_output.shape, expected_tensor.shape)
                self.assertEqual(native_output.stride(), expected_tensor.stride())
                self.assertIs(native_output.dtype, torch.float32)
                self.assertEqual(native_output.device, torch.device("cpu"))
                np.testing.assert_allclose(
                    native,
                    expected,
                    rtol=2.0e-6,
                    atol=smallest_subnormal,
                    equal_nan=True,
                )


if __name__ == "__main__":
    unittest.main()
