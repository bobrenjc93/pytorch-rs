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
class TensorCpuReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("cpu differentials require pinned PyTorch 2.13.0")

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

    def test_cpu_identity_formats_match_pytorch_2_13(self):
        actual_cases = self.make_identity_cases(torch)
        expected_cases = self.make_identity_cases(reference_torch)
        for case, (actual, expected) in enumerate(
            zip(actual_cases, expected_cases, strict=True)
        ):
            actual_calls = (
                lambda tensor=actual: tensor.cpu(),
                lambda tensor=actual: tensor.cpu(memory_format=None),
                lambda tensor=actual: tensor.cpu(
                    memory_format=torch.preserve_format
                ),
                lambda tensor=actual: tensor.cpu(
                    memory_format=torch.contiguous_format
                ),
            )
            expected_calls = (
                lambda tensor=expected: tensor.cpu(),
                lambda tensor=expected: tensor.cpu(memory_format=None),
                lambda tensor=expected: tensor.cpu(
                    memory_format=reference_torch.preserve_format
                ),
                lambda tensor=expected: tensor.cpu(
                    memory_format=reference_torch.contiguous_format
                ),
            )
            for format_case, (actual_call, expected_call) in enumerate(
                zip(actual_calls, expected_calls, strict=True)
            ):
                with self.subTest(case=case, format_case=format_case):
                    actual_result = actual_call()
                    expected_result = expected_call()
                    self.assertEqual(
                        actual_result is actual, expected_result is expected
                    )
                    self.assertEqual(actual_result.shape, expected_result.shape)
                    self.assertEqual(
                        actual_result.stride(), expected_result.stride()
                    )
                    self.assertEqual(
                        actual_result.storage_offset(),
                        expected_result.storage_offset(),
                    )
                    self.assertEqual(
                        actual_result.requires_grad,
                        expected_result.requires_grad,
                    )
                    self.assertEqual(actual_result.is_leaf, expected_result.is_leaf)

    def test_extreme_empty_copy_validation_order_matches_pytorch_2_13(self):
        actual = torch.zeros((3, 0, 1, sys.maxsize)).transpose(0, 1)
        expected = reference_torch.zeros(
            (3, 0, 1, sys.maxsize), dtype=reference_torch.float32
        ).transpose(0, 1)
        for actual_format, expected_format in (
            (torch.channels_last, reference_torch.channels_last),
            (torch.channels_last_3d, reference_torch.channels_last_3d),
        ):
            with self.subTest(memory_format=actual_format):
                self.assert_error_matches(
                    lambda memory_format=actual_format: actual.cpu(
                        memory_format=memory_format
                    ),
                    lambda memory_format=expected_format: expected.cpu(
                        memory_format=memory_format
                    ),
                )

    def test_channels_last_materialization_and_autograd_match_pytorch_2_13(self):
        cases = (
            (
                (2, 3, 2, 4),
                (0, 3),
                torch.channels_last,
                reference_torch.channels_last,
            ),
            (
                (2, 3, 2, 4, 5),
                (0, 4),
                torch.channels_last_3d,
                reference_torch.channels_last_3d,
            ),
        )
        for shape, dimensions, actual_format, expected_format in cases:
            with self.subTest(memory_format=actual_format):
                actual_leaf = torch.ones(shape, requires_grad=True)
                expected_leaf = reference_torch.ones(
                    shape,
                    dtype=reference_torch.float32,
                    requires_grad=True,
                )
                actual_source = (actual_leaf * 3.0).transpose(*dimensions)
                expected_source = (expected_leaf * 3.0).transpose(*dimensions)
                actual_result = actual_source.cpu(memory_format=actual_format)
                expected_result = expected_source.cpu(memory_format=expected_format)

                self.assertEqual(
                    actual_result is actual_source,
                    expected_result is expected_source,
                )
                self.assertEqual(actual_result.shape, expected_result.shape)
                self.assertEqual(actual_result.stride(), expected_result.stride())
                self.assertEqual(
                    actual_result.storage_offset(), expected_result.storage_offset()
                )
                self.assertEqual(
                    actual_result.requires_grad, expected_result.requires_grad
                )
                self.assertEqual(actual_result.is_leaf, expected_result.is_leaf)
                np.testing.assert_array_equal(
                    np.asarray(actual_result),
                    expected_result.detach().numpy(),
                )
                self.assertEqual(
                    actual_result.cpu(memory_format=actual_format)
                    is actual_result,
                    expected_result.cpu(memory_format=expected_format)
                    is expected_result,
                )

                actual_row_major = actual_result.cpu(
                    memory_format=torch.contiguous_format
                )
                expected_row_major = expected_result.cpu(
                    memory_format=reference_torch.contiguous_format
                )
                self.assertEqual(
                    actual_row_major is actual_result,
                    expected_row_major is expected_result,
                )
                self.assertEqual(
                    actual_row_major.stride(), expected_row_major.stride()
                )
                np.testing.assert_array_equal(
                    np.asarray(actual_row_major),
                    expected_row_major.detach().numpy(),
                )

                actual_result.sum().backward()
                expected_result.sum().backward()
                np.testing.assert_array_equal(
                    np.asarray(actual_leaf.grad), expected_leaf.grad.numpy()
                )

        actual_singleton = torch.zeros((2, 1, 4, 5))
        expected_singleton = reference_torch.zeros(
            (2, 1, 4, 5), dtype=reference_torch.float32
        )
        actual_singleton_result = actual_singleton.cpu(
            memory_format=torch.channels_last
        )
        expected_singleton_result = expected_singleton.cpu(
            memory_format=reference_torch.channels_last
        )
        self.assertEqual(
            actual_singleton_result is actual_singleton,
            expected_singleton_result is expected_singleton,
        )
        self.assertEqual(
            actual_singleton_result.stride(), expected_singleton_result.stride()
        )
        self.assertEqual(
            actual_singleton_result.cpu(memory_format=torch.channels_last)
            is actual_singleton_result,
            expected_singleton_result.cpu(
                memory_format=reference_torch.channels_last
            )
            is expected_singleton_result,
        )

        actual_empty = torch.zeros((0, 1, 4, 5))
        expected_empty = reference_torch.zeros(
            (0, 1, 4, 5), dtype=reference_torch.float32
        )
        for repeat in range(2):
            with self.subTest(empty_repeat=repeat):
                actual_empty_result = actual_empty.cpu(
                    memory_format=torch.channels_last
                )
                expected_empty_result = expected_empty.cpu(
                    memory_format=reference_torch.channels_last
                )
                self.assertEqual(
                    actual_empty_result is actual_empty,
                    expected_empty_result is expected_empty,
                )
                self.assertEqual(
                    actual_empty_result.stride(), expected_empty_result.stride()
                )
                actual_empty = actual_empty_result
                expected_empty = expected_empty_result

    def test_descriptor_documentation_and_binding_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_descriptor = inspect.getattr_static(torch.Tensor, "cpu")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "cpu"
        )

        for actual_callable, expected_callable, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (actual.cpu, expected.cpu, types.BuiltinMethodType),
        ):
            self.assertIs(type(actual_callable), expected_type)
            self.assertIs(type(expected_callable), expected_type)
            self.assertEqual(actual_callable.__name__, expected_callable.__name__)
            self.assertEqual(actual_callable.__doc__, expected_callable.__doc__)
            self.assertEqual(
                actual_callable.__text_signature__,
                expected_callable.__text_signature__,
            )
            with self.assertRaises(ValueError):
                inspect.signature(actual_callable)
            with self.assertRaises(ValueError):
                inspect.signature(expected_callable)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertIs(actual_descriptor(actual), actual)
        self.assertIs(expected_descriptor(expected), expected)

        call_pairs = (
            (
                lambda: actual.cpu(torch.preserve_format),
                lambda: expected.cpu(reference_torch.preserve_format),
            ),
            (
                lambda: actual.cpu(
                    torch.preserve_format, torch.contiguous_format
                ),
                lambda: expected.cpu(
                    reference_torch.preserve_format,
                    reference_torch.contiguous_format,
                ),
            ),
            (
                lambda: actual.cpu(memory_format=1),
                lambda: expected.cpu(memory_format=1),
            ),
            (
                lambda: actual.cpu(unexpected=True),
                lambda: expected.cpu(unexpected=True),
            ),
            (
                lambda: actual.cpu(
                    **{"unexpected": True, "memory_format": 1}
                ),
                lambda: expected.cpu(
                    **{"unexpected": True, "memory_format": 1}
                ),
            ),
            (
                lambda: actual.cpu(memory_format=torch.channels_last),
                lambda: expected.cpu(
                    memory_format=reference_torch.channels_last
                ),
            ),
            (
                lambda: actual.cpu(memory_format=torch.channels_last_3d),
                lambda: expected.cpu(
                    memory_format=reference_torch.channels_last_3d
                ),
            ),
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (lambda: actual_descriptor(1), lambda: expected_descriptor(1)),
        )
        for case, (actual_call, expected_call) in enumerate(call_pairs):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
