import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ExpReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.exp differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(
        self, actual, expected, *, case, exact_non_nan_bits=False
    ):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)

        actual_values = np.asarray(actual, dtype=np.float32)
        expected_values = expected.detach().cpu().numpy()
        with self.subTest(case=case, values=True):
            if exact_non_nan_bits:
                np.testing.assert_array_equal(
                    np.isnan(actual_values), np.isnan(expected_values)
                )
                finite = ~np.isnan(expected_values.reshape(-1))
                np.testing.assert_array_equal(
                    actual_values.reshape(-1).view(np.uint32)[finite],
                    expected_values.reshape(-1).view(np.uint32)[finite],
                )
            else:
                np.testing.assert_allclose(
                    actual_values,
                    expected_values,
                    rtol=2.0e-6,
                    atol=np.nextafter(np.float32(0), np.float32(1)),
                    equal_nan=True,
                )

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(1.5, dtype=module.float32, requires_grad=True)
            return leaf, leaf, None
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf, None

        values = np.linspace(-3.0, 3.0, 24, dtype=np.float32).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=True
        )
        if case == "offset":
            input = leaf[1]
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 12, dtype=np.float32)
                .reshape(3, 4)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, input, weights
        if case == "noncontiguous":
            input = leaf.transpose(0, 2)
            weights = module.tensor(
                np.linspace(-2.0, 2.0, 24, dtype=np.float32)
                .reshape(4, 3, 2)
                .tolist(),
                dtype=module.float32,
            )
            return leaf, input, weights
        raise AssertionError(f"unknown Tensor.exp autograd case: {case}")

    def test_seeded_random_shapes_and_values_match_pytorch_2_13(self):
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

    def test_autograd_scalar_empty_offset_and_noncontiguous_match_pytorch_2_13(
        self,
    ):
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            actual_leaf, actual_input, actual_weights = self.autograd_case(torch, case)
            expected_leaf, expected_input, expected_weights = self.autograd_case(
                reference_torch, case
            )
            actual_output = actual_input.exp()
            expected_output = expected_input.exp()
            self.assert_tensor_matches(
                actual_output, expected_output, case=(case, "forward")
            )
            if actual_weights is None:
                actual_loss = actual_output if case == "scalar" else actual_output.sum()
                expected_loss = (
                    expected_output if case == "scalar" else expected_output.sum()
                )
            else:
                actual_loss = (actual_output * actual_weights).sum()
                expected_loss = (expected_output * expected_weights).sum()
            actual_loss.backward()
            expected_loss.backward()
            self.assert_tensor_matches(
                actual_leaf.grad,
                expected_leaf.grad,
                case=(case, "gradient"),
            )

    def test_autograd_overflow_subnormal_infinity_and_nan_bits_match_pytorch_2_13(
        self,
    ):
        input_bits = np.asarray(
            (
                0xFF800000,
                0xC2D00000,
                0xC2CF0000,
                0xC2CE0000,
                0xC2C80000,
                0xC2B00000,
                0xBF800000,
                0x80000000,
                0x00000000,
                0x3F800000,
                0x41200000,
                0x42A00000,
                0x42B00000,
                0x42B18000,
                0x42B20000,
                0x7F800000,
                0x7F812345,
                0xFF812345,
                0x7FC12345,
                0xFFC54321,
            ),
            dtype=np.uint32,
        )
        weight_bits = np.asarray(
            (
                0x7F800000,
                0xBF800000,
                0x3F000000,
                0xBF000000,
                0x00000001,
                0xFF800000,
                0x00000000,
                0x80000000,
                0x3F800000,
                0xBF800000,
                0x00000001,
                0x3E800000,
                0x40000000,
                0xBF800000,
                0x3F800000,
                0x00000000,
                0x3F800000,
                0xBF800000,
                0x7FC01234,
                0xFFC05678,
            ),
            dtype=np.uint32,
        )
        tensors = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                memoryview(input_bits.view(np.float32)), requires_grad=True
            )
            weights = module.tensor(memoryview(weight_bits.view(np.float32)))
            output = leaf.exp()
            (output * weights).sum().backward()
            tensors.append((output, leaf.grad))

        self.assert_tensor_matches(
            tensors[0][0],
            tensors[1][0],
            case="special forward",
            exact_non_nan_bits=True,
        )
        self.assert_tensor_matches(
            tensors[0][1],
            tensors[1][1],
            case="special gradient",
            exact_non_nan_bits=True,
        )

    def test_exp_backward_node_identity_matches_pytorch_2_13(self):
        errors = []
        for module in (torch, reference_torch):
            probability = module.tensor(
                [1.0], dtype=module.float32, requires_grad=True
            ).exp()
            try:
                module.nn.functional.dropout(
                    module.tensor([1.0], dtype=module.float32),
                    p=probability,
                    training=False,
                )
            except Exception as error:
                errors.append((type(error).__name__, str(error)))
            else:
                self.fail("dropout unexpectedly accepted an exponential probability")

        self.assertEqual(errors[0], errors[1])
        self.assertIn("grad_fn=<ExpBackward0>", errors[0][1])

    @staticmethod
    def error(action):
        try:
            action()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("Tensor.exp unexpectedly accepted an invalid operation")

    def test_accumulation_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        snapshots = []
        for module in (torch, reference_torch):
            accumulated = module.tensor(
                [-1.0, 0.0, 1.0, 4.0],
                dtype=module.float32,
                requires_grad=True,
            )
            accumulated.exp().sum().backward()
            first = np.asarray(accumulated.grad, dtype=np.float32).copy()
            accumulated.exp().sum().backward()
            second = np.asarray(accumulated.grad, dtype=np.float32).copy()

            freed = module.tensor(
                [-1.0, 0.0, 1.0], dtype=module.float32, requires_grad=True
            )
            loss = freed.exp().sum()
            loss.backward()
            repeated_backward = self.error(loss.backward)
            snapshots.append((first, second, repeated_backward))

        np.testing.assert_array_equal(snapshots[0][0], snapshots[1][0])
        np.testing.assert_array_equal(snapshots[0][1], snapshots[1][1])
        self.assertEqual(snapshots[0][2], snapshots[1][2])

        actual_leaf = torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, 0.0, 1.0], [2.0, 4.0, 6.0]], requires_grad=True
        )
        with torch.no_grad():
            actual = actual_leaf.transpose(0, 1)[1].exp()
        with reference_torch.no_grad():
            expected = expected_leaf.transpose(0, 1)[1].exp()
        self.assert_tensor_matches(actual, expected, case="no_grad")
        self.assertIsNone(actual_leaf.grad)
        self.assertTrue(actual_leaf.exp().requires_grad)

        self.assert_tensor_matches(
            actual_leaf.detach().exp(),
            expected_leaf.detach().exp(),
            case="detached input",
        )


if __name__ == "__main__":
    unittest.main()
