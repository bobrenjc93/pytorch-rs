import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


VSTACK_ALIASES = ("vstack", "row_stack")


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelVstackReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.vstack differentials require pinned PyTorch 2.13.0"
            )

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
        for name in VSTACK_ALIASES:
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            with self.subTest(function=name, case="scalars"):
                actual_inputs = [torch.tensor(-0.0), torch.tensor(2.5)]
                expected_inputs = [
                    reference_torch.tensor(-0.0, dtype=reference_torch.float32),
                    reference_torch.tensor(2.5, dtype=reference_torch.float32),
                ]
                actual = actual_function(actual_inputs)
                expected = expected_function(expected_inputs)
                self.assert_matches(actual, expected, case=f"{name} scalars")
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

            with self.subTest(function=name, case="vectors"):
                actual_base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                expected_base = reference_torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    dtype=reference_torch.float32,
                )
                actual_inputs = [
                    actual_base.transpose(0, 1)[1],
                    torch.tensor([-0.0, 7.5]),
                ]
                expected_inputs = [
                    expected_base.transpose(0, 1)[1],
                    reference_torch.tensor([-0.0, 7.5], dtype=reference_torch.float32),
                ]
                actual = actual_function(actual_inputs)
                expected = expected_function(expected_inputs)
                self.assert_matches(actual, expected, case=f"{name} vectors")
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

            with self.subTest(function=name, case="matrices"):
                base_values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
                actual_base = torch.tensor(base_values.tolist())
                expected_base = reference_torch.tensor(
                    base_values.tolist(),
                    dtype=reference_torch.float32,
                )
                actual_inputs = (
                    actual_base[1].transpose(0, 1),
                    torch.zeros((0, 3)),
                    torch.tensor([[-0.0, 100.0, 101.0]]),
                )
                expected_inputs = (
                    expected_base[1].transpose(0, 1),
                    reference_torch.zeros((0, 3), dtype=reference_torch.float32),
                    reference_torch.tensor(
                        [[-0.0, 100.0, 101.0]],
                        dtype=reference_torch.float32,
                    ),
                )
                actual = actual_function(actual_inputs, out=None)
                expected = expected_function(expected_inputs, out=None)
                self.assert_matches(actual, expected, case=f"{name} matrices")
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

            with self.subTest(function=name, case="mixed ranks"):
                actual_inputs = [
                    torch.tensor(1.0),
                    torch.tensor([2.0]),
                    torch.tensor([[3.0], [4.0]]),
                ]
                expected_inputs = [
                    reference_torch.tensor(1.0, dtype=reference_torch.float32),
                    reference_torch.tensor([2.0], dtype=reference_torch.float32),
                    reference_torch.tensor(
                        [[3.0], [4.0]], dtype=reference_torch.float32
                    ),
                ]
                actual = actual_function(tensors=actual_inputs)
                expected = expected_function(tensors=expected_inputs)
                self.assert_matches(actual, expected, case=f"{name} mixed ranks")
                self.assertEqual(
                    self.fresh_storage_observation(actual, actual_inputs),
                    self.fresh_storage_observation(expected, expected_inputs),
                )

            with self.subTest(function=name, case="empty vectors"):
                actual_inputs = [torch.tensor([]), torch.tensor([])]
                expected_inputs = [
                    reference_torch.tensor([], dtype=reference_torch.float32),
                    reference_torch.tensor([], dtype=reference_torch.float32),
                ]
                actual = actual_function(actual_inputs)
                expected = expected_function(expected_inputs)
                self.assert_matches(actual, expected, case=f"{name} empty vectors")

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        for name in VSTACK_ALIASES:
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            actual_left = torch.tensor([1.0, 2.0], requires_grad=True)
            actual_right = torch.tensor([3.0, 4.0], requires_grad=True)
            expected_left = reference_torch.tensor(
                [1.0, 2.0],
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            expected_right = reference_torch.tensor(
                [3.0, 4.0],
                dtype=reference_torch.float32,
                requires_grad=True,
            )

            actual = actual_function([actual_left, actual_right, actual_left])
            expected = expected_function(
                [expected_left, expected_right, expected_left]
            )
            self.assert_matches(actual, expected, case=f"{name} autograd output")

            weights = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
            (actual * torch.tensor(weights)).sum().backward()
            (
                expected
                * reference_torch.tensor(weights, dtype=reference_torch.float32)
            ).sum().backward()
            np.testing.assert_array_equal(
                np.asarray(actual_left.grad),
                expected_left.grad.detach().numpy(),
            )
            np.testing.assert_array_equal(
                np.asarray(actual_right.grad),
                expected_right.grad.detach().numpy(),
            )

            actual_scalar = torch.tensor(1.5, requires_grad=True)
            actual_vector = torch.tensor([2.5], requires_grad=True)
            actual_matrix = torch.tensor([[3.5]], requires_grad=True)
            expected_scalar = reference_torch.tensor(
                1.5,
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            expected_vector = reference_torch.tensor(
                [2.5],
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            expected_matrix = reference_torch.tensor(
                [[3.5]],
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            actual = actual_function(
                [actual_scalar, actual_vector, actual_matrix, actual_scalar]
            )
            expected = expected_function(
                [expected_scalar, expected_vector, expected_matrix, expected_scalar]
            )
            self.assert_matches(actual, expected, case=f"{name} mixed autograd output")
            weights = [[1.0], [2.0], [3.0], [4.0]]
            (actual * torch.tensor(weights)).sum().backward()
            (
                expected
                * reference_torch.tensor(weights, dtype=reference_torch.float32)
            ).sum().backward()
            self.assertEqual(actual_scalar.grad.item(), expected_scalar.grad.item())
            np.testing.assert_array_equal(
                np.asarray(actual_vector.grad),
                expected_vector.grad.detach().numpy(),
            )
            np.testing.assert_array_equal(
                np.asarray(actual_matrix.grad),
                expected_matrix.grad.detach().numpy(),
            )

            actual_no_grad = torch.tensor([1.0, 2.0], requires_grad=True)
            expected_no_grad = reference_torch.tensor(
                [1.0, 2.0],
                dtype=reference_torch.float32,
                requires_grad=True,
            )
            with torch.no_grad():
                actual_untracked = actual_function([actual_no_grad, actual_no_grad])
            with reference_torch.no_grad():
                expected_untracked = expected_function(
                    [expected_no_grad, expected_no_grad]
                )
            self.assert_matches(actual_untracked, expected_untracked, case=f"{name} no_grad")
            self.assertIsNone(actual_no_grad.grad)
            self.assertIsNone(expected_no_grad.grad)

    def test_dispatch_surface_matches_pytorch_2_13(self):
        for name in VSTACK_ALIASES:
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)

            def capture_call(module, function):
                left = module.tensor([1.0])
                right = module.tensor([2.0])
                inputs = [left, right]
                marker = object()

                class RecordingMode(module.overrides.TorchFunctionMode):
                    def __init__(self, result=marker):
                        self.calls = []
                        self.result = result

                    def __torch_function__(self, func, types, args=(), kwargs=None):
                        self.calls.append(
                            (
                                func is function,
                                tuple(item.__name__ for item in types),
                                len(args),
                                kwargs,
                            )
                        )
                        return self.result

                mode = RecordingMode()
                with mode:
                    result = function(inputs, out=None)
                return result is marker, tuple(mode.calls)

            with self.subTest(function=name, case="mode"):
                self.assertEqual(
                    capture_call(torch, actual_function),
                    capture_call(reference_torch, expected_function),
                )

            def capture_override(module, function):
                marker = object()
                events = []

                class LeftOverride:
                    @classmethod
                    def __torch_function__(cls, func, types, args=(), kwargs=None):
                        events.append(
                            ("left", func is function, tuple(item.__name__ for item in types), kwargs)
                        )
                        return NotImplemented

                class RightOverride:
                    @classmethod
                    def __torch_function__(cls, func, types, args=(), kwargs=None):
                        events.append(
                            ("right", func is function, tuple(item.__name__ for item in types), kwargs)
                        )
                        return marker

                inputs = [LeftOverride(), module.tensor([]), RightOverride()]
                result = function(inputs)
                return result is marker, tuple(events)

            with self.subTest(function=name, case="sequence overrides"):
                self.assertEqual(
                    capture_override(torch, actual_function),
                    capture_override(reference_torch, expected_function),
                )

    def test_representative_errors_match_pytorch_2_13(self):
        for name in VSTACK_ALIASES:
            actual_function = getattr(torch, name)
            expected_function = getattr(reference_torch, name)
            actual_tensor = torch.tensor([1.0])
            expected_tensor = reference_torch.tensor(
                [1.0], dtype=reference_torch.float32
            )
            cases = (
                ("empty", lambda: actual_function([]), lambda: expected_function([])),
                (
                    "non sequence",
                    lambda: actual_function(actual_tensor),
                    lambda: expected_function(expected_tensor),
                ),
                (
                    "bad element",
                    lambda: actual_function([actual_tensor, 1]),
                    lambda: expected_function([expected_tensor, 1]),
                ),
                (
                    "shape mismatch",
                    lambda: actual_function(
                        [actual_tensor, torch.tensor([1.0, 2.0])]
                    ),
                    lambda: expected_function(
                        [
                            expected_tensor,
                            reference_torch.tensor(
                                [1.0, 2.0],
                                dtype=reference_torch.float32,
                            ),
                        ]
                    ),
                ),
                (
                    "bad out",
                    lambda: actual_function([actual_tensor], out=1),
                    lambda: expected_function([expected_tensor], out=1),
                ),
                (
                    "unexpected dim",
                    lambda: actual_function([actual_tensor], dim=0),
                    lambda: expected_function([expected_tensor], dim=0),
                ),
                (
                    "missing",
                    lambda: actual_function(),
                    lambda: expected_function(),
                ),
                (
                    "too many",
                    lambda: actual_function([actual_tensor], 0),
                    lambda: expected_function([expected_tensor], 0),
                ),
            )
            for case, actual_call, expected_call in cases:
                with self.subTest(function=name, case=case):
                    self.assertEqual(self.error(actual_call), self.error(expected_call))

    def test_public_callable_surface_matches_pytorch_2_13(self):
        for name in VSTACK_ALIASES:
            with self.subTest(function=name):
                actual_function = getattr(torch, name)
                expected_function = getattr(reference_torch, name)
                self.assertEqual(actual_function.__name__, expected_function.__name__)
                self.assertEqual(actual_function.__qualname__, expected_function.__qualname__)
                self.assertEqual(actual_function.__module__, expected_function.__module__)
                self.assertEqual(torch.__all__.count(name), reference_torch.__all__.count(name))
                self.assertEqual(
                    hasattr(torch.functional, name),
                    hasattr(reference_torch.functional, name),
                )
                self.assertIs(getattr(torch._C._VariableFunctionsClass, name), actual_function)
        self.assertEqual(torch.vstack is torch.row_stack, reference_torch.vstack is reference_torch.row_stack)


if __name__ == "__main__":
    unittest.main()
