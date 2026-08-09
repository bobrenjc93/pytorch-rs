import sys
import unittest

import numpy as np
import torch_rs as torch


class PermuteTests(unittest.TestCase):
    def assert_tensor(self, actual, expected, shape, stride):
        self.assertEqual(actual.shape, shape)
        self.assertEqual(actual.stride(), stride)
        np.testing.assert_allclose(
            np.asarray(actual), np.asarray(expected, dtype=np.float32), equal_nan=True
        )

    def test_variadic_sequence_keyword_and_negative_dimension_forms(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())
        expected = values.transpose(2, 0, 1)

        views = (
            source.permute(2, 0, 1),
            source.permute((2, 0, 1)),
            source.permute([2, 0, 1]),
            source.permute(-1, -3, -2),
            source.permute(dims=(-1, 0, 1)),
            source.permute(dims=[2, 0, 1]),
        )
        for view in views:
            with self.subTest(stride=view.stride()):
                self.assert_tensor(view, expected, (4, 2, 3), (1, 12, 4))
                self.assertEqual(view.storage_offset(), source.storage_offset())
                self.assertIs(view.dtype, source.dtype)
                self.assertEqual(view.device, source.device)
                self.assertFalse(view.is_contiguous())

        self.assertEqual(source.shape, (2, 3, 4))
        self.assertEqual(source.stride(), (12, 4, 1))

    def test_inverse_composition_and_transpose_equivalence(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        source = torch.tensor(values.tolist())

        restored = source.permute(2, 0, 1).permute(1, 2, 0)
        self.assert_tensor(restored, values, source.shape, source.stride())

        composed = source.permute(1, 2, 0).permute(2, 0, 1)
        self.assert_tensor(composed, values, source.shape, source.stride())

        permuted = source.permute(2, 1, 0)
        transposed = source.transpose(0, 2)
        self.assertEqual(permuted.shape, transposed.shape)
        self.assertEqual(permuted.stride(), transposed.stride())
        self.assertEqual(permuted.storage_offset(), transposed.storage_offset())
        np.testing.assert_array_equal(np.asarray(permuted), np.asarray(transposed))

    def test_non_contiguous_and_offset_inputs_reorder_existing_metadata(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        base = torch.tensor(values.tolist()).transpose(0, 2)
        view = base.permute(2, 0, 1)
        expected = values.transpose(2, 1, 0).transpose(2, 0, 1)

        self.assert_tensor(view, expected, (2, 4, 3), (12, 1, 4))
        self.assertFalse(view.is_contiguous())

        indexed = view[1].permute(1, 0)
        self.assertEqual(indexed.shape, (3, 4))
        self.assertEqual(indexed.stride(), (4, 1))
        self.assertEqual(indexed.storage_offset(), 12)
        self.assertEqual(indexed.tolist(), expected[1].T.tolist())

    def test_scalar_empty_singleton_and_high_rank_metadata(self):
        scalar = torch.tensor(2.5)
        for view in (
            scalar.permute(()),
            scalar.permute([]),
            scalar.permute(dims=()),
        ):
            self.assertEqual(view.shape, ())
            self.assertEqual(view.stride(), ())
            self.assertEqual(view.item(), 2.5)

        empty_cases = (
            ((0,), (0,), (1,)),
            ((2, 0, 3), (3, 0, 2), (1, 3, 3)),
            ((1, 0, 1, 2), (2, 1, 0, 1), (1, 2, 2, 2)),
        )
        for shape, expected_shape, expected_stride in empty_cases:
            dimensions = tuple(reversed(range(len(shape))))
            view = torch.zeros(shape).permute(dimensions)
            with self.subTest(shape=shape):
                self.assertEqual(view.shape, expected_shape)
                self.assertEqual(view.stride(), expected_stride)
                self.assertEqual(view.numel(), 0)
                self.assertEqual(view.tolist(), np.zeros(expected_shape).tolist())
                self.assertTrue(view.is_contiguous())

        high_values = np.arange(2, dtype=np.float32).reshape((1,) * 7 + (2,))
        high_rank = torch.tensor(high_values.tolist()).permute(tuple(range(7, -1, -1)))
        self.assert_tensor(
            high_rank,
            high_values.transpose(tuple(range(7, -1, -1))),
            (2,) + (1,) * 7,
            (1,) + (2,) * 7,
        )

        extreme = torch.zeros((sys.maxsize, 0, 2, 2))
        zero_first = extreme.permute(1, 2, 3, 0)
        self.assertEqual(zero_first.shape, (0, 2, 2, sys.maxsize))
        self.assertEqual(zero_first.stride(), (4, 2, 1, 4))
        with self.assertRaisesRegex(
            RuntimeError, "^numel: integer multiplication overflow$"
        ):
            extreme.permute(0, 2, 3, 1)

    def test_invalid_dimensions_match_pytorch_error_classes_and_messages(self):
        tensor = torch.zeros((2, 3, 4))
        cases = (
            (
                lambda: tensor.permute(),
                TypeError,
                'permute() missing 1 required positional arguments: "dims"',
            ),
            (
                lambda: tensor.permute(0, 1),
                RuntimeError,
                "permute(sparse_coo): number of dimensions in the tensor input does not match the length of the desired ordering of dimensions i.e. input.dim() = 3 is not equal to len(dims) = 2",
            ),
            (
                lambda: tensor.permute(0, -3, 2),
                RuntimeError,
                "permute(): duplicate dims are not allowed.",
            ),
            (
                lambda: tensor.permute(0, 1, 3),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got 3)",
            ),
            (
                lambda: tensor.permute(-4, 1, 2),
                IndexError,
                "Dimension out of range (expected to be in range of [-3, 2], but got -4)",
            ),
            (
                lambda: tensor.permute(None, 1, 2, nope=1),
                TypeError,
                "permute() takes 1 positional argument but 3 were given",
            ),
            (
                lambda: tensor.permute((0, 1, 2), 0, dims=(0, 1, 2)),
                TypeError,
                "permute() takes 1 positional argument but 2 were given",
            ),
            (
                lambda: tensor.permute(0, 1.0, 2, dims=(0, 1, 2)),
                TypeError,
                "permute() got multiple values for argument 'dims'",
            ),
            (
                lambda: tensor.permute(0, 1.0, 2, nope=1),
                TypeError,
                "permute() got an unexpected keyword argument 'nope'",
            ),
            (
                lambda: tensor.permute((0, 1.0, 2), dims=(0, 1, 2)),
                TypeError,
                "permute() got multiple values for argument 'dims'",
            ),
            (
                lambda: tensor.permute((0, 1.0, 2), nope=1),
                TypeError,
                "permute() got an unexpected keyword argument 'nope'",
            ),
        )
        for call, error_type, message in cases:
            with self.subTest(message=message):
                with self.assertRaises(error_type) as raised:
                    call()
                self.assertEqual(str(raised.exception), message)

        with self.assertRaisesRegex(RuntimeError, r"input.dim\(\) = 0"):
            torch.tensor(1.0).permute(0)
        with self.assertRaises(TypeError):
            tensor.permute((0, 1, 2), 0)
        with self.assertRaises(TypeError):
            tensor.permute(dims=(0, 1, 2), unexpected=True)
        with self.assertRaises(TypeError):
            tensor.permute(dims=0)

        index_calls = 0

        class IndexProbe:
            def __index__(self):
                nonlocal index_calls
                index_calls += 1
                return 1

        probe = IndexProbe()
        with self.assertRaisesRegex(TypeError, "got multiple values"):
            tensor.permute(0, probe, 2, dims=(0, 1, 2))
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'nope'"):
            tensor.permute((0, probe, 2), nope=1)
        self.assertEqual(index_calls, 0)
        self.assertEqual(tensor.permute(0, probe, 2).shape, tensor.shape)
        self.assertEqual(index_calls, 1)

    def test_integer_protocol_and_argument_type_behavior(self):
        class IntSubclass(int):
            pass

        class IndexOnly:
            def __index__(self):
                return 1

        tensor = torch.zeros((2, 3, 4))
        accepted = (
            tensor.permute(IntSubclass(0), np.int64(1), np.uint32(2)),
            tensor.permute((0, IndexOnly(), 2)),
            tensor.permute((0, True, 2)),
        )
        for view in accepted:
            self.assertEqual(view.shape, tensor.shape)
            self.assertEqual(view.stride(), tensor.stride())

        leading_bool_message = (
            "permute(): argument 'dims' (position 1) must be tuple of ints, "
            "but found element of type bool at pos 0"
        )
        for dimensions in ((True, 0, 2), [True, 0, 2]):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError) as raised:
                    tensor.permute(dimensions)
                self.assertEqual(str(raised.exception), leading_bool_message)

        overflow_prefix = (
            "permute(): argument 'dims' failed to unpack the object at pos 1 "
            'with error "Overflow when unpacking long long'
        )
        for dimensions in ((2**100, 1, 2), [2**100, 1, 2]):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(TypeError) as raised:
                    tensor.permute(dimensions)
                self.assertTrue(str(raised.exception).startswith(overflow_prefix))

        for call in (
            lambda: tensor.permute(0.0, 1, 2),
            lambda: tensor.permute(0, 1.0, 2),
            lambda: tensor.permute((0, "1", 2)),
            lambda: tensor.permute(range(3)),
            lambda: tensor.permute(dims=0),
            lambda: tensor.permute(2**100, 1, 2),
        ):
            with self.subTest(call=call):
                with self.assertRaises(TypeError):
                    call()

    def test_all_stride_aware_consumers_observe_logical_order(self):
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4) - 8.0
        view = torch.tensor(values.tolist()).permute(2, 0, 1)
        expected = values.transpose(2, 0, 1)

        self.assertEqual(view.tolist(), expected.tolist())
        array_copy = np.asarray(view)
        np.testing.assert_array_equal(array_copy, expected)
        array_copy[0, 0, 0] = -999.0
        self.assertEqual(view[0, 0, 0].item(), expected[0, 0, 0])
        self.assertEqual(
            repr(view),
            f"tensor({expected.reshape(-1).tolist()!r}, shape={list(expected.shape)!r})",
        )

        copied = view.clone()
        self.assert_tensor(copied, expected, view.shape, view.stride())
        self.assertEqual(copied.storage_offset(), 0)
        contiguous = view.clone(memory_format=torch.contiguous_format)
        self.assert_tensor(contiguous, expected, view.shape, (6, 3, 1))

        same_shape = view.reshape(view.shape)
        self.assert_tensor(same_shape, expected, view.shape, view.stride())
        flattened = view.reshape(-1)
        self.assert_tensor(flattened, expected.reshape(-1), (24,), (1,))

        operations = (
            (view + 2.0, expected + 2.0),
            (view + view, expected + expected),
            (view - 2.0, expected - 2.0),
            (2.0 - view, 2.0 - expected),
            (view * 2.0, expected * 2.0),
            (view / 2.0, expected / 2.0),
            (view.relu(), np.maximum(expected, 0.0)),
            (view.sin(), np.sin(expected)),
            (view.exp(), np.exp(expected)),
        )
        for actual, operation_expected in operations:
            with self.subTest(actual=actual):
                self.assertEqual(actual.stride(), view.stride())
                np.testing.assert_allclose(
                    np.asarray(actual),
                    operation_expected,
                    rtol=2.0e-6,
                    atol=1.0e-6,
                )
        self.assertEqual(view.sum().item(), np.float32(expected.sum()).item())

        left_values = np.arange(6, dtype=np.float32).reshape(2, 3)
        right_values = np.arange(8, dtype=np.float32).reshape(4, 2)
        left = torch.tensor(left_values.tolist()).permute(1, 0)
        right = torch.tensor(right_values.tolist()).permute(1, 0)
        self.assert_tensor(
            left @ right,
            left_values.T @ right_values.T,
            (3, 4),
            (4, 1),
        )


if __name__ == "__main__":
    unittest.main()
