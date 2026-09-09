import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelStackReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.stack differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_from_array(module, array, *, requires_grad=False):
        kwargs = {"dtype": module.float32, "requires_grad": requires_grad}
        if array.shape == ():
            return module.tensor(float(array.reshape(()).item()), **kwargs)
        if any(dimension == 0 for dimension in array.shape):
            return module.zeros(tuple(array.shape), **kwargs)
        return module.tensor(array.tolist(), **kwargs)

    @staticmethod
    def generated_array(shape, seed):
        if shape == ():
            return np.asarray((seed % 17) - 8.5, dtype=np.float32)
        elements = int(np.prod(shape))
        values = np.arange(elements, dtype=np.float32).reshape(shape)
        return values * np.float32(0.25) + np.float32((seed % 11) - 5)

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
        with self.subTest(case=case, values=True):
            actual_bits = np.asarray(actual).reshape(-1).view(np.uint32)
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    @staticmethod
    def fresh_storage_observation(result, inputs):
        return tuple(
            (not result.is_set_to(input), result.data_ptr() != input.data_ptr())
            for input in inputs
            if input.numel()
        )

    @staticmethod
    def error(call):
        try:
            call()
        except Exception as error:
            return type(error).__name__, str(error)
        raise AssertionError("call unexpectedly succeeded")

    def test_values_metadata_and_fresh_storage_match_pytorch_2_13(self):
        actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_base = reference_torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype=reference_torch.float32,
        )
        actual_view = actual_base.transpose(0, 1)
        expected_view = expected_base.transpose(0, 1)

        cases = (
            (
                "list dim 1",
                [actual_view, torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]])],
                [
                    expected_view,
                    reference_torch.tensor(
                        [[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]],
                        dtype=reference_torch.float32,
                    ),
                ],
                1,
            ),
            (
                "tuple dim -1",
                (torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])),
                (
                    reference_torch.tensor([1.0, 2.0], dtype=reference_torch.float32),
                    reference_torch.tensor([3.0, 4.0], dtype=reference_torch.float32),
                ),
                -1,
            ),
            (
                "scalars",
                [torch.tensor(2.0), torch.tensor(-0.0)],
                [
                    reference_torch.tensor(2.0, dtype=reference_torch.float32),
                    reference_torch.tensor(-0.0, dtype=reference_torch.float32),
                ],
                0,
            ),
            (
                "empty middle",
                [torch.zeros((2, 0, 3)), torch.zeros((2, 0, 3))],
                [
                    reference_torch.zeros((2, 0, 3), dtype=reference_torch.float32),
                    reference_torch.zeros((2, 0, 3), dtype=reference_torch.float32),
                ],
                2,
            ),
            (
                "axis alias",
                [torch.tensor([14.0]), torch.tensor([15.0])],
                [
                    reference_torch.tensor([14.0], dtype=reference_torch.float32),
                    reference_torch.tensor([15.0], dtype=reference_torch.float32),
                ],
                1,
            ),
        )
        for case, actual_inputs, expected_inputs, dimension in cases:
            with self.subTest(case=case):
                if case == "axis alias":
                    actual = torch.stack(actual_inputs, axis=dimension)
                    expected = reference_torch.stack(expected_inputs, axis=dimension)
                else:
                    actual = torch.stack(actual_inputs, dim=dimension)
                    expected = reference_torch.stack(expected_inputs, dim=dimension)
                self.assert_matches(actual, expected, case=case)
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

    def test_generated_same_shape_matrix_matches_pytorch_2_13(self):
        generated_cases = (
            ("rank0_four_inputs", (), 4, (0,)),
            ("rank1_three_inputs", (5,), 3, (0, 1)),
            ("rank2_two_inputs", (2, 3), 2, (0, 1, 2)),
            ("rank3_three_inputs", (2, 1, 3), 3, (0, 2, 3)),
            ("empty_rank3", (1, 0, 2), 2, (0, 1, 3)),
        )
        for case, shape, input_count, dimensions in generated_cases:
            arrays = [
                self.generated_array(shape, 20260908 + index * 17 + len(shape))
                for index in range(input_count)
            ]
            for dimension in dimensions:
                with self.subTest(case=case, dimension=dimension):
                    actual_inputs = [
                        self.tensor_from_array(torch, array) for array in arrays
                    ]
                    expected_inputs = [
                        self.tensor_from_array(reference_torch, array)
                        for array in arrays
                    ]
                    actual = torch.stack(actual_inputs, dim=dimension)
                    expected = reference_torch.stack(expected_inputs, dim=dimension)
                    self.assert_matches(actual, expected, case=(case, dimension))

        actual_base = self.tensor_from_array(
            torch,
            self.generated_array((4, 2, 3), 20261001),
        )
        expected_base = self.tensor_from_array(
            reference_torch,
            self.generated_array((4, 2, 3), 20261001),
        )
        actual_offset_inputs = [actual_base[1], actual_base[2]]
        expected_offset_inputs = [expected_base[1], expected_base[2]]
        actual = torch.stack(actual_offset_inputs, dim=1)
        expected = reference_torch.stack(expected_offset_inputs, dim=1)
        self.assert_matches(actual, expected, case="held-out offset dim 1")

        actual_transposed = [
            self.tensor_from_array(torch, self.generated_array((2, 3), 20261011)).transpose(0, 1),
            self.tensor_from_array(torch, self.generated_array((2, 3), 20261012)).transpose(0, 1),
        ]
        expected_transposed = [
            self.tensor_from_array(
                reference_torch,
                self.generated_array((2, 3), 20261011),
            ).transpose(0, 1),
            self.tensor_from_array(
                reference_torch,
                self.generated_array((2, 3), 20261012),
            ).transpose(0, 1),
        ]
        for dimension in (0, 1, 2):
            with self.subTest(case="held-out transposed", dimension=dimension):
                actual = torch.stack(actual_transposed, dim=dimension)
                expected = reference_torch.stack(expected_transposed, dim=dimension)
                self.assert_matches(actual, expected, case=("held-out transposed", dimension))

        actual_autograd_inputs = [
            self.tensor_from_array(
                torch,
                self.generated_array((3, 4), 20261021 + index),
                requires_grad=True,
            )
            for index in range(3)
        ]
        expected_autograd_inputs = [
            self.tensor_from_array(
                reference_torch,
                self.generated_array((3, 4), 20261021 + index),
                requires_grad=True,
            )
            for index in range(3)
        ]
        actual = torch.stack(actual_autograd_inputs, dim=2)
        expected = reference_torch.stack(expected_autograd_inputs, dim=2)
        self.assert_matches(actual, expected, case="held-out autograd forward dim 2")

    def test_autograd_repeated_inputs_and_no_grad_match_pytorch_2_13(self):
        actual_left = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=True
        )
        actual_right = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, requires_grad=True
        )
        expected_left = reference_torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        expected_right = reference_torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )

        actual = torch.stack([actual_left, actual_right, actual_left], dim=1)
        expected = reference_torch.stack(
            [expected_left, expected_right, expected_left], dim=1
        )
        self.assert_matches(actual, expected, case="autograd output")

        weights = [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ]
        (actual * torch.tensor(weights)).sum().backward()
        (expected * reference_torch.tensor(weights, dtype=reference_torch.float32)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_left.grad),
            expected_left.grad.detach().numpy(),
        )
        np.testing.assert_array_equal(
            np.asarray(actual_right.grad),
            expected_right.grad.detach().numpy(),
        )

        actual_no_grad = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_no_grad = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        with torch.no_grad():
            actual_untracked = torch.stack([actual_no_grad, actual_no_grad], dim=-1)
        with reference_torch.no_grad():
            expected_untracked = reference_torch.stack(
                [expected_no_grad, expected_no_grad], dim=-1
            )
        self.assert_matches(actual_untracked, expected_untracked, case="no_grad")
        self.assertIsNone(actual_no_grad.grad)
        self.assertIsNone(expected_no_grad.grad)

    def test_representative_errors_match_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0])
        expected_tensor = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            ("empty", lambda: torch.stack([]), lambda: reference_torch.stack([])),
            (
                "non sequence",
                lambda: torch.stack(actual_tensor),
                lambda: reference_torch.stack(expected_tensor),
            ),
            (
                "bad element",
                lambda: torch.stack([actual_tensor, 1]),
                lambda: reference_torch.stack([expected_tensor, 1]),
            ),
            (
                "shape mismatch",
                lambda: torch.stack([actual_tensor, torch.tensor([1.0, 2.0])]),
                lambda: reference_torch.stack(
                    [
                        expected_tensor,
                        reference_torch.tensor([1.0, 2.0], dtype=reference_torch.float32),
                    ]
                ),
            ),
            (
                "high dim",
                lambda: torch.stack([actual_tensor], dim=2),
                lambda: reference_torch.stack([expected_tensor], dim=2),
            ),
            (
                "low dim",
                lambda: torch.stack([actual_tensor], dim=-3),
                lambda: reference_torch.stack([expected_tensor], dim=-3),
            ),
            (
                "bool dim",
                lambda: torch.stack([actual_tensor], dim=True),
                lambda: reference_torch.stack([expected_tensor], dim=True),
            ),
            (
                "invalid out",
                lambda: torch.stack([actual_tensor], out=1),
                lambda: reference_torch.stack([expected_tensor], out=1),
            ),
            (
                "axis conflict",
                lambda: torch.stack([actual_tensor], dim=0, axis=0),
                lambda: reference_torch.stack([expected_tensor], dim=0, axis=0),
            ),
            ("missing", lambda: torch.stack(), lambda: reference_torch.stack()),
            (
                "too many",
                lambda: torch.stack([actual_tensor], 0, 1),
                lambda: reference_torch.stack([expected_tensor], 0, 1),
            ),
        )
        for case, actual_call, expected_call in cases:
            with self.subTest(case=case):
                self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_public_callable_surface_matches_pytorch_2_13(self):
        actual_function = torch.stack
        expected_function = reference_torch.stack
        self.assertEqual(actual_function.__name__, expected_function.__name__)
        self.assertEqual(actual_function.__module__, expected_function.__module__)
        self.assertIn("stack", torch.__all__)
        self.assertEqual(torch.__all__.count("stack"), 1)


if __name__ == "__main__":
    unittest.main()
