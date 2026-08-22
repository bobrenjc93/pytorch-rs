import unittest

import numpy as np
import torch_rs as torch


class TensorFloatListTests(unittest.TestCase):
    def test_flat_builtin_float_list_preserves_bits_metadata_and_autograd_leaf(self):
        expected_bits = np.array(
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
        values = expected_bits.view(np.float32).astype(np.float64).tolist()

        result = torch.tensor(
            values,
            dtype=torch.float32,
            device="cpu",
            requires_grad=True,
        )

        self.assertEqual(result.shape, (len(values),))
        self.assertEqual(result.stride(), (1,))
        self.assertEqual(result.storage_offset(), 0)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertTrue(result.requires_grad)
        self.assertTrue(result.is_leaf)
        np.testing.assert_array_equal(
            np.asarray(result.detach()).view(np.uint32), expected_bits
        )

        result.sum().backward()
        self.assertEqual(result.grad.tolist(), [1.0] * len(values))

    def test_empty_builtin_list_retains_vector_shape_and_leaf_metadata(self):
        result = torch.tensor([], requires_grad=True)

        self.assertEqual(result.shape, (0,))
        self.assertEqual(result.stride(), (1,))
        self.assertEqual(result.storage_offset(), 0)
        self.assertIs(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertTrue(result.requires_grad)
        self.assertTrue(result.is_leaf)
        self.assertEqual(result.tolist(), [])

    def test_non_exact_float_lists_retain_recursive_parser_behavior(self):
        class FloatSubclass(float):
            def __float__(self):
                raise AssertionError("float subclass coercion should not run")

        class CustomNumeric:
            def __init__(self, value):
                self.value = value
                self.calls = 0

            def __float__(self):
                self.calls += 1
                return self.value

        class ListSubclass(list):
            def __init__(self, values):
                super().__init__(values)
                self.calls = []

            def __len__(self):
                self.calls.append("len")
                return super().__len__()

            def __getitem__(self, index):
                self.calls.append(("getitem", index))
                return super().__getitem__(index)

        custom = CustomNumeric(6.25)
        list_subclass = ListSubclass([7.0, 8.0])
        cases = (
            ("nested", [[1.0, 2.0], [3.0, 4.0]], (2, 2), [[1.0, 2.0], [3.0, 4.0]]),
            ("mixed", [1.0, 2, 3.0], (3,), [1.0, 2.0, 3.0]),
            (
                "float subclass",
                [FloatSubclass(4.5), FloatSubclass(-5.5)],
                (2,),
                [4.5, -5.5],
            ),
            ("custom numeric", [custom], (1,), [6.25]),
            ("list subclass", list_subclass, (2,), [7.0, 8.0]),
        )

        for name, values, shape, expected in cases:
            with self.subTest(name=name):
                result = torch.tensor(values)
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.tolist(), expected)

        self.assertEqual(custom.calls, 1)
        self.assertEqual(
            list_subclass.calls,
            ["len", ("getitem", 0), ("getitem", 1)],
        )

    def test_failed_fast_path_preserves_existing_errors(self):
        cases = (
            (
                [1.0, object()],
                TypeError,
                "tensor data must contain real numbers in a rectangular sequence",
            ),
            (
                [1.0, [2.0]],
                ValueError,
                "expected a rectangular sequence, but nested shapes differ",
            ),
        )

        for values, error_type, message in cases:
            with self.subTest(values=values), self.assertRaisesRegex(
                error_type, f"^{message}$"
            ):
                torch.tensor(values)


if __name__ == "__main__":
    unittest.main()
