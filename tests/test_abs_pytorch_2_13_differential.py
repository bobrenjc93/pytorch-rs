import unittest

import numpy as np
import torch as pytorch
import torch_rs


class AbsPyTorch213DifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = pytorch.__version__.split("+", maxsplit=1)[0]
        if version != "2.13.0":
            raise AssertionError(f"differential tests require PyTorch 2.13.0, got {version}")

    def assert_matches_pytorch(self, values, shape, make_view=False):
        flat_values = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
        actual_input = torch_rs.tensor(flat_values).reshape(shape)
        expected_input = pytorch.tensor(flat_values, dtype=pytorch.float32).reshape(shape)

        if make_view and shape:
            actual_input = actual_input[-1]
            expected_input = expected_input[-1]

        for actual in (
            actual_input.abs(),
            torch_rs.abs(actual_input),
            torch_rs.abs(input=actual_input, out=None),
            abs(actual_input),
        ):
            expected = pytorch.abs(expected_input)
            with self.subTest(shape=shape, view=make_view, operation=actual):
                actual_array = np.asarray(actual)
                expected_array = expected.numpy()
                np.testing.assert_array_equal(
                    actual_array.view(np.uint32), expected_array.view(np.uint32)
                )
                self.assertEqual(actual.shape, tuple(expected.shape))
                self.assertEqual(actual.stride(), expected.stride())
                self.assertIs(actual.dtype, torch_rs.float32)
                self.assertEqual(actual.device, torch_rs.device("cpu"))
                self.assertEqual(actual.storage_offset(), 0)

    def test_seeded_held_out_shapes_match_pytorch_exactly(self):
        rng = np.random.default_rng(0xA850_213)
        shapes = [(), (0,), (2, 0, 3)]
        for _ in range(24):
            rank = int(rng.integers(0, 7))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, size=rank)))

        for shape in shapes:
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.standard_normal(elements, dtype=np.float32).reshape(shape)
            self.assert_matches_pytorch(values, shape)

            if shape and shape[0] > 0:
                self.assert_matches_pytorch(values, shape, make_view=True)

    def test_special_float32_values_match_pytorch_exactly(self):
        input_bits = np.array(
            [
                0x00000000,
                0x80000000,
                0x00000001,
                0x80000001,
                0x7F7FFFFF,
                0xFF7FFFFF,
                0x7F800000,
                0xFF800000,
                0x7FC12345,
                0xFFC12345,
            ],
            dtype=np.uint32,
        )
        self.assert_matches_pytorch(input_bits.view(np.float32), (2, 5))


if __name__ == "__main__":
    unittest.main()
