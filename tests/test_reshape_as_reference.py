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
class TensorReshapeAsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "reshape_as differentials require pinned PyTorch 2.13.0"
            )

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def make_layout_cases(self, module):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = module.tensor(values.tolist(), dtype=module.float32)
        return (
            (
                "scalar",
                module.tensor(-0.0, dtype=module.float32),
                module.tensor(8.0, dtype=module.float32),
            ),
            (
                "empty-offset",
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
                module.zeros((2, 0), dtype=module.float32),
            ),
            (
                "contiguous-with-strided-other",
                base,
                module.zeros((4, 6), dtype=module.float32).transpose(0, 1),
            ),
            (
                "contiguous-offset",
                base[1],
                module.zeros((2, 6), dtype=module.float32),
            ),
            (
                "transposed-copy",
                base.transpose(0, 2),
                module.zeros((6, 4), dtype=module.float32),
            ),
        )

    def test_positional_keyword_layout_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_layout_cases(torch)
        expected_cases = self.make_layout_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_source, actual_other = actual_case
            expected_name, expected_source, expected_other = expected_case
            self.assertEqual(case, expected_name)
            for keyword in (False, True):
                with self.subTest(case=case, keyword=keyword):
                    if keyword:
                        actual_result = actual_source.reshape_as(other=actual_other)
                        expected_result = expected_source.reshape_as(
                            other=expected_other
                        )
                    else:
                        actual_result = actual_source.reshape_as(actual_other)
                        expected_result = expected_source.reshape_as(expected_other)

                    actual_direct = actual_source.reshape(actual_other.shape)
                    expected_direct = expected_source.reshape(expected_other.shape)
                    self.assertIsNot(actual_result, actual_source)
                    self.assertIsNot(expected_result, expected_source)
                    self.assertEqual(actual_result.shape, tuple(expected_result.shape))
                    self.assertEqual(actual_result.stride(), expected_result.stride())
                    self.assertEqual(
                        actual_result.storage_offset(),
                        expected_result.storage_offset(),
                    )
                    self.assertEqual(
                        actual_result.is_contiguous(),
                        expected_result.is_contiguous(),
                    )
                    self.assertEqual(
                        actual_result.requires_grad, expected_result.requires_grad
                    )
                    self.assertEqual(actual_result.is_leaf, expected_result.is_leaf)
                    np.testing.assert_array_equal(
                        np.asarray(actual_result),
                        expected_result.detach().cpu().numpy(),
                    )
                    self.assertEqual(
                        actual_result.data_ptr() == actual_source.data_ptr(),
                        expected_result.untyped_storage().data_ptr()
                        == expected_source.untyped_storage().data_ptr(),
                    )
                    self.assertEqual(
                        actual_result.is_set_to(actual_direct),
                        expected_result.is_set_to(expected_direct),
                    )

    def test_extreme_empty_and_mismatch_errors_match_pytorch_2_13(self):
        maximum = sys.maxsize
        actual_source = torch.zeros((0,))
        expected_source = reference_torch.zeros((0,))
        actual_other = torch.zeros((0,)).reshape((0, maximum, maximum))
        expected_other = reference_torch.zeros((0,)).reshape(
            (0, maximum, maximum)
        )

        actual = actual_source.reshape_as(actual_other)
        expected = expected_source.reshape_as(expected_other)
        self.assertEqual(actual.shape, tuple(expected.shape))
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.numel(), expected.numel())

        actual_incompatible = torch.zeros((2, 2))
        expected_incompatible = reference_torch.zeros((2, 2))
        self.assert_error_matches(
            lambda: torch.zeros((6,)).reshape_as(actual_incompatible),
            lambda: reference_torch.zeros((6,)).reshape_as(
                expected_incompatible
            ),
        )

    def autograd_outcomes(self, module):
        outcomes = []
        for case, transpose, other_shape, weights in (
            (
                "view",
                False,
                (3, 2),
                [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
            ),
            (
                "copy",
                True,
                (6,),
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            ),
        ):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = leaf.transpose(0, 1) if transpose else leaf
            other = module.zeros(
                other_shape, dtype=module.float32, requires_grad=True
            )
            result = source.reshape_as(other=other)
            direct = source.reshape(other.shape)
            metadata = (
                case,
                tuple(result.shape),
                result.stride(),
                result.storage_offset(),
                result.requires_grad,
                result.is_leaf,
                result.data_ptr() == source.data_ptr(),
                result.is_set_to(direct),
            )
            (result * module.tensor(weights, dtype=module.float32)).sum().backward()
            outcomes.append(
                (
                    metadata,
                    np.asarray(leaf.grad.detach()).copy(),
                    other.grad,
                )
            )
        return outcomes

    def repeated_backward_outcomes(self, module):
        outcomes = []
        for case, transpose, other_shape in (
            ("view", False, (3, 2)),
            ("copy", True, (6,)),
        ):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = leaf.transpose(0, 1) if transpose else leaf
            other = module.zeros(
                other_shape, dtype=module.float32, requires_grad=True
            )
            loss = source.reshape_as(other).sum()
            loss.backward()
            loss.backward()
            outcomes.append(
                (case, np.asarray(leaf.grad.detach()).copy(), other.grad)
            )
        return outcomes

    def no_grad_outcomes(self, module):
        outcomes = []
        for case, transpose in (("view", False), ("copy", True)):
            leaf = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            source = leaf.transpose(0, 1) if transpose else leaf
            other = module.zeros((4,), dtype=module.float32)
            with module.no_grad():
                result = source.reshape_as(other)
                direct = source.reshape(other.shape)
            outcomes.append(
                (
                    case,
                    tuple(result.shape),
                    result.stride(),
                    result.storage_offset(),
                    result.requires_grad,
                    result.is_leaf,
                    result.data_ptr() == source.data_ptr(),
                    result.is_set_to(direct),
                )
            )
        return outcomes

    def test_autograd_repeated_backward_and_no_grad_match_pytorch_2_13(self):
        actual_autograd = self.autograd_outcomes(torch)
        expected_autograd = self.autograd_outcomes(reference_torch)
        for actual, expected in zip(
            actual_autograd, expected_autograd, strict=True
        ):
            self.assertEqual(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertIsNone(actual[2])
            self.assertIsNone(expected[2])

        actual_repeated = self.repeated_backward_outcomes(torch)
        expected_repeated = self.repeated_backward_outcomes(reference_torch)
        for actual, expected in zip(
            actual_repeated, expected_repeated, strict=True
        ):
            self.assertEqual(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])
            self.assertIsNone(actual[2])
            self.assertIsNone(expected[2])

        self.assertEqual(
            self.no_grad_outcomes(torch),
            self.no_grad_outcomes(reference_torch),
        )

    def test_tensorbase_descriptor_documentation_matches_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0, 2.0])
        expected_tensor = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32
        )
        actual_other = torch.zeros((2, 1))
        expected_other = reference_torch.zeros((2, 1))
        actual_descriptor = inspect.getattr_static(torch.Tensor, "reshape_as")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "reshape_as"
        )

        for actual, expected, expected_type in (
            (actual_descriptor, expected_descriptor, types.MethodDescriptorType),
            (
                actual_tensor.reshape_as,
                expected_tensor.reshape_as,
                types.BuiltinMethodType,
            ),
        ):
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        self.assertEqual(
            actual_descriptor(actual_tensor, actual_other).shape,
            tuple(expected_descriptor(expected_tensor, expected_other).shape),
        )
        self.assertEqual(
            actual_descriptor(actual_tensor, other=actual_other).shape,
            tuple(
                expected_descriptor(expected_tensor, other=expected_other).shape
            ),
        )

    def test_binding_and_type_error_precedence_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        actual_other = torch.tensor([2.0])
        expected_other = reference_torch.tensor(
            [2.0], dtype=reference_torch.float32
        )
        actual_descriptor = inspect.getattr_static(torch.Tensor, "reshape_as")
        expected_descriptor = inspect.getattr_static(
            reference_torch.Tensor, "reshape_as"
        )
        array = np.zeros((2, 3), dtype=np.float32)
        cases = (
            (lambda: actual_descriptor(), lambda: expected_descriptor()),
            (
                lambda: actual_descriptor(1, actual_other),
                lambda: expected_descriptor(1, expected_other),
            ),
            (lambda: actual.reshape_as(), lambda: expected.reshape_as()),
            (
                lambda: actual.reshape_as(actual_other, actual_other),
                lambda: expected.reshape_as(expected_other, expected_other),
            ),
            (
                lambda: actual.reshape_as(actual_other, other=actual_other),
                lambda: expected.reshape_as(expected_other, other=expected_other),
            ),
            (
                lambda: actual.reshape_as(foo=actual_other),
                lambda: expected.reshape_as(foo=expected_other),
            ),
            (
                lambda: actual.reshape_as(actual_other, extra=True),
                lambda: expected.reshape_as(expected_other, extra=True),
            ),
            (lambda: actual.reshape_as(1), lambda: expected.reshape_as(1)),
            (lambda: actual.reshape_as(None), lambda: expected.reshape_as(None)),
            (lambda: actual.reshape_as([]), lambda: expected.reshape_as([])),
            (lambda: actual.reshape_as(array), lambda: expected.reshape_as(array)),
            (
                lambda: actual.reshape_as(other=1),
                lambda: expected.reshape_as(other=1),
            ),
            (
                lambda: actual.reshape_as(other=None),
                lambda: expected.reshape_as(other=None),
            ),
            (
                lambda: actual.reshape_as(other=[]),
                lambda: expected.reshape_as(other=[]),
            ),
            (
                lambda: actual.reshape_as(**{"other": 1, "extra": True}),
                lambda: expected.reshape_as(**{"other": 1, "extra": True}),
            ),
            (
                lambda: actual.reshape_as(**{"extra": True, "other": 1}),
                lambda: expected.reshape_as(**{"extra": True, "other": 1}),
            ),
            (
                lambda: actual.reshape_as(1, other=actual_other),
                lambda: expected.reshape_as(1, other=expected_other),
            ),
            (
                lambda: actual.reshape_as(1, extra=True),
                lambda: expected.reshape_as(1, extra=True),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
