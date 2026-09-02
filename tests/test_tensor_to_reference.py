import inspect
import sys
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorToReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.to differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(
            type(actual_raised.exception).__name__,
            type(expected_raised.exception).__name__,
        )
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_identity_cases(self, module):
        leaf = module.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=module.float32,
            requires_grad=True,
        )
        tracked = (leaf * 2.0).transpose(0, 1)
        leaf.sum().backward()
        return (
            module.tensor(-0.0, dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            module.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0, 7.0],
                    [8.0, 9.0, 10.0, 11.0],
                ],
                dtype=module.float32,
            ).transpose(0, 1)[1],
            module.zeros((0,), dtype=module.float32)
            .reshape((2, 0, sys.maxsize))
            .transpose(0, 2),
            leaf,
            tracked,
            leaf.grad,
        )

    def test_identity_overloads_match_pytorch_2_13(self):
        actual_cases = self.make_identity_cases(torch)
        expected_cases = self.make_identity_cases(reference_torch)
        actual_other = torch.ones((1,), requires_grad=True)
        expected_other = reference_torch.ones(
            (1,), dtype=reference_torch.float32, requires_grad=True
        )
        call_pairs = (
            (
                lambda tensor: tensor.to(),
                lambda tensor: tensor.to(),
            ),
            (
                lambda tensor: tensor.to(torch.float32),
                lambda tensor: tensor.to(reference_torch.float32),
            ),
            (
                lambda tensor: tensor.to(torch.float),
                lambda tensor: tensor.to(reference_torch.float),
            ),
            (
                lambda tensor: tensor.to(dtype=torch.float32),
                lambda tensor: tensor.to(dtype=reference_torch.float32),
            ),
            (
                lambda tensor: tensor.to(dtype=None),
                lambda tensor: tensor.to(dtype=None),
            ),
            (
                lambda tensor: tensor.to("cpu"),
                lambda tensor: tensor.to("cpu"),
            ),
            (
                lambda tensor: tensor.to(device="cpu"),
                lambda tensor: tensor.to(device="cpu"),
            ),
            (
                lambda tensor: tensor.to(torch.device("cpu")),
                lambda tensor: tensor.to(reference_torch.device("cpu")),
            ),
            (
                lambda tensor: tensor.to(None, torch.float32, False, False),
                lambda tensor: tensor.to(None, reference_torch.float32, False, False),
            ),
            (
                lambda tensor: tensor.to(actual_other),
                lambda tensor: tensor.to(expected_other),
            ),
            (
                lambda tensor: tensor.to(tensor=actual_other),
                lambda tensor: tensor.to(tensor=expected_other),
            ),
            (
                lambda tensor: tensor.to(non_blocking=True),
                lambda tensor: tensor.to(non_blocking=True),
            ),
            (
                lambda tensor: tensor.to(copy=False),
                lambda tensor: tensor.to(copy=False),
            ),
            (
                lambda tensor: tensor.to(memory_format=None),
                lambda tensor: tensor.to(memory_format=None),
            ),
            (
                lambda tensor: tensor.to(memory_format=torch.preserve_format),
                lambda tensor: tensor.to(
                    memory_format=reference_torch.preserve_format
                ),
            ),
            (
                lambda tensor: tensor.to(memory_format=torch.contiguous_format),
                lambda tensor: tensor.to(
                    memory_format=reference_torch.contiguous_format
                ),
            ),
        )

        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            for call_index, (actual_call, expected_call) in enumerate(call_pairs):
                with self.subTest(case=case, call=call_index):
                    actual_result = actual_call(actual)
                    expected_result = expected_call(expected)
                    self.assertEqual(
                        actual_result is actual, expected_result is expected
                    )
                    self.assertEqual(actual_result.shape, tuple(expected_result.shape))
                    self.assertEqual(actual_result.stride(), expected_result.stride())
                    self.assertEqual(
                        actual_result.storage_offset(),
                        expected_result.storage_offset(),
                    )
                    self.assertEqual(actual_result.dtype, torch.float32)
                    self.assertEqual(str(expected_result.dtype), "torch.float32")
                    self.assertEqual(
                        actual_result.requires_grad, expected_result.requires_grad
                    )
                    self.assertEqual(actual_result.is_leaf, expected_result.is_leaf)
                    if 0 not in actual_result.shape:
                        np.testing.assert_array_equal(
                            np.asarray(actual_result),
                            expected_result.detach().numpy(),
                        )

    def test_descriptor_metadata_matches_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "to")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "to")

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual.to, expected.to, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(
                actual_callable.__qualname__, expected_callable.__qualname__
            )
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            with self.assertRaises(ValueError):
                inspect.signature(actual_callable)
            with self.assertRaises(ValueError):
                inspect.signature(expected_callable)

        self.assertEqual(repr(actual_descriptor), repr(expected_descriptor))
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(
            hasattr(actual_descriptor, "__module__"),
            hasattr(expected_descriptor, "__module__"),
        )
        self.assertIs(actual_descriptor(actual), actual)
        self.assertIs(expected_descriptor(expected), expected)

    def test_binding_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_other = torch.tensor([2.0])
        expected_other = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32
        )
        actual_descriptor = inspect.getattr_static(torch.Tensor, "to")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "to")
        call_pairs = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

        unsupported_call_pairs = (
            (
                lambda: actual.to(torch.float32, device="cpu"),
                lambda: expected.to(reference_torch.float32, device="cpu"),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(unsupported_call_pairs):
            with self.subTest(unsupported_case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
