import os
import struct
import subprocess
import sys
import textwrap
import unittest

import numpy as np
import torch_rs as torch


def float64_from_bits(bits):
    return struct.unpack("=d", struct.pack("=Q", bits))[0]


class TensorFloatListTests(unittest.TestCase):
    def assert_float32_bits(self, tensor, expected):
        actual_bits = np.asarray(tensor).reshape(-1).view(np.uint32)
        with np.errstate(over="ignore", invalid="ignore"):
            expected_bits = (
                np.asarray(expected, dtype=np.float32).reshape(-1).view(np.uint32)
            )
        np.testing.assert_array_equal(actual_bits, expected_bits)

    def test_exact_builtin_float_list_preserves_values_metadata_and_copy(self):
        values = [
            0.0,
            -0.0,
            float("inf"),
            -float("inf"),
            float64_from_bits(0x7FF8_0000_0000_0001),
            float64_from_bits(0xFFF8_1234_5678_9ABC),
            1.0000000596046448,
            1.0000001788139343,
            3.4028235677973366e38,
            1.0e-50,
        ]
        tensor = torch.tensor(values, dtype=torch.float32, device="cpu")

        self.assertEqual(tensor.shape, (len(values),))
        self.assertEqual(tensor.stride(), (1,))
        self.assertEqual(tensor.storage_offset(), 0)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(tensor.device, torch.device("cpu"))
        self.assertFalse(tensor.requires_grad)
        self.assertTrue(tensor.is_leaf)
        self.assert_float32_bits(tensor, values)

        values[0] = 99.0
        self.assertEqual(tensor[0].item(), 0.0)

    def test_empty_float_list_and_autograd_leaf_metadata(self):
        empty = torch.tensor([])
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.stride(), (1,))
        self.assertEqual(empty.tolist(), [])

        leaf = torch.tensor([1.25, -2.5, 4.0], requires_grad=True)
        self.assertTrue(leaf.requires_grad)
        self.assertTrue(leaf.is_leaf)
        self.assertIsNone(leaf.grad)
        leaf.sum().backward()
        self.assertEqual(leaf.grad.tolist(), [1.0, 1.0, 1.0])

    @unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux RLIMIT_AS")
    def test_large_float_list_avoids_a_linear_pointer_snapshot(self):
        script = textwrap.dedent(
            """\
            import os
            import resource

            import torch_rs as torch

            elements = 5_000_000
            values = [float(index) for index in range(elements)]
            with open("/proc/self/statm", encoding="ascii") as statm:
                virtual_pages = int(statm.read().split()[0])
            current_virtual_size = virtual_pages * os.sysconf("SC_PAGE_SIZE")
            limit = current_virtual_size + 45 * 1024 * 1024
            _, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if hard_limit != resource.RLIM_INFINITY and limit > hard_limit:
                raise SystemExit(77)
            resource.setrlimit(resource.RLIMIT_AS, (limit, hard_limit))

            output = torch.tensor(values)
            assert output.shape == (elements,)
            assert output[0].item() == 0.0
            assert output[-1].item() == float(elements - 1)
            """
        )
        completed = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
        )
        if completed.returncode == 77:
            self.skipTest("process hard address-space limit is too low")
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )

    def test_nested_mixed_and_subclassed_lists_keep_sequence_fallback(self):
        class TrackingList(list):
            def __init__(self, values):
                super().__init__(values)
                self.length_calls = 0
                self.item_calls = []

            def __len__(self):
                self.length_calls += 1
                return super().__len__()

            def __getitem__(self, index):
                self.item_calls.append(index)
                return super().__getitem__(index)

        nested = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(nested.shape, (2, 2))
        self.assertEqual(nested.tolist(), [[1.0, 2.0], [3.0, 4.0]])

        mixed = torch.tensor([1.25, 2, 3.75], dtype=torch.float32)
        self.assertEqual(mixed.tolist(), [1.25, 2.0, 3.75])

        source = TrackingList([5.0, 6.0])
        subclassed = torch.tensor(source)
        self.assertEqual(subclassed.tolist(), [5.0, 6.0])
        self.assertEqual(source.length_calls, 1)
        self.assertEqual(source.item_calls, [0, 1])

    def test_float_subclasses_and_custom_numerics_keep_scalar_fallback(self):
        class FloatSubclass(float):
            float_calls = 0

            def __float__(self):
                type(self).float_calls += 1
                return 99.0

        subclass_values = [FloatSubclass(1.25), FloatSubclass(-2.5)]
        subclassed = torch.tensor(subclass_values, dtype=torch.float32)
        self.assertEqual(subclassed.tolist(), [1.25, -2.5])
        self.assertEqual(FloatSubclass.float_calls, 0)

        class CustomNumeric:
            float_calls = 0

            def __init__(self, value):
                self.value = value

            def __float__(self):
                type(self).float_calls += 1
                return self.value

        custom = torch.tensor(
            [CustomNumeric(1.5), CustomNumeric(-3.25)],
            dtype=torch.float32,
        )
        self.assertEqual(custom.tolist(), [1.5, -3.25])
        self.assertEqual(CustomNumeric.float_calls, 2)

    def test_fallback_conversion_and_shape_errors_are_unchanged(self):
        class FailingNumeric:
            float_calls = 0

            def __float__(self):
                type(self).float_calls += 1
                raise RuntimeError("custom conversion failed")

        with self.assertRaisesRegex(TypeError, "must contain real numbers"):
            torch.tensor([1.0, FailingNumeric()], dtype=torch.float32)
        self.assertEqual(FailingNumeric.float_calls, 1)
        with self.assertRaisesRegex(ValueError, "nested shapes differ"):
            torch.tensor([[1.0], [2.0, 3.0]])
        with self.assertRaisesRegex(TypeError, "must contain real numbers"):
            torch.tensor([1.0, object()], dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
