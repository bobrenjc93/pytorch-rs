import inspect
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorPermuteReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, actual_source, expected_source, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.data_ptr(), actual_source.data_ptr())
            self.assertEqual(expected.data_ptr(), expected_source.data_ptr())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_allclose(
                np.asarray(actual),
                expected.detach().cpu().numpy(),
                rtol=2.0e-6,
                atol=1.0e-6,
                equal_nan=True,
            )

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

    def test_seeded_forms_shapes_strides_offsets_and_aliasing_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x9E2_213)
        shapes = [(), (0,), (2, 0, 3), (1, 3, 2), (2, 3, 4)]
        for _ in range(32):
            rank = int(rng.integers(0, 7))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual_source = torch.zeros(shape)
                expected_source = reference_torch.zeros(
                    shape, dtype=reference_torch.float32
                )
            else:
                actual_source = torch.tensor(
                    values.item() if shape == () else values.tolist()
                )
                expected_source = reference_torch.tensor(
                    values, dtype=reference_torch.float32
                )

            if len(shape) >= 2 and shape[0] > 0 and shape[-1] > 0 and case % 3 == 1:
                actual_source = actual_source.transpose(0, -1)[-1]
                expected_source = expected_source.transpose(0, -1)[-1]

            rank = len(actual_source.shape)
            permutation = list(rng.permutation(rank))
            dimensions = tuple(
                axis - rank if (case + index) % 2 else axis
                for index, axis in enumerate(permutation)
            )
            style = case % 5
            if style == 0 and rank:
                actual = actual_source.permute(*dimensions)
                expected = expected_source.permute(*dimensions)
            elif style == 1:
                actual = actual_source.permute(dimensions)
                expected = expected_source.permute(dimensions)
            elif style == 2:
                actual = actual_source.permute(list(dimensions))
                expected = expected_source.permute(list(dimensions))
            elif style == 3:
                actual = actual_source.permute(dims=dimensions)
                expected = expected_source.permute(dims=dimensions)
            else:
                actual = actual_source.permute(dims=list(dimensions))
                expected = expected_source.permute(dims=list(dimensions))

            self.assert_matches(
                actual,
                expected,
                actual_source=actual_source,
                expected_source=expected_source,
                case=case,
            )

    def test_autograd_and_no_grad_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        weights = np.linspace(-2.0, 3.0, num=24, dtype=np.float32).reshape(4, 2, 3)
        actual_leaf = torch.tensor(values.tolist(), requires_grad=True)
        expected_leaf = reference_torch.tensor(values, requires_grad=True)
        actual = actual_leaf.permute(-1, 0, 1)
        expected = expected_leaf.permute(-1, 0, 1)

        self.assert_matches(
            actual,
            expected,
            actual_source=actual_leaf,
            expected_source=expected_leaf,
            case="tracked",
        )
        self.assertEqual(actual.requires_grad, expected.requires_grad)
        self.assertEqual(actual.is_leaf, expected.is_leaf)
        (actual * torch.tensor(weights.tolist())).sum().backward()
        (expected * reference_torch.tensor(weights)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros((2, 0, 3), requires_grad=True)
        actual_empty.permute(dims=(-1, 0, 1)).sum().backward()
        expected_empty.permute(dims=(-1, 0, 1)).sum().backward()
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.detach().cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.permute(dims=(1, 2, 0))
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.permute(dims=(1, 2, 0))
        self.assertEqual(
            (actual_untracked.requires_grad, actual_untracked.is_leaf),
            (expected_untracked.requires_grad, expected_untracked.is_leaf),
        )

    def test_rank_duplicate_range_type_and_binding_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))
        cases = (
            (lambda: actual.permute(), lambda: expected.permute()),
            (
                lambda: actual.permute(unexpected=None),
                lambda: expected.permute(unexpected=None),
            ),
            (lambda: actual.permute(0, 1), lambda: expected.permute(0, 1)),
            (
                lambda: actual.permute(dims=[0, 1]),
                lambda: expected.permute(dims=[0, 1]),
            ),
            (lambda: actual.permute(0, 1, 1), lambda: expected.permute(0, 1, 1)),
            (
                lambda: actual.permute(0, 1, -2),
                lambda: expected.permute(0, 1, -2),
            ),
            (lambda: actual.permute(0, 1, 3), lambda: expected.permute(0, 1, 3)),
            (
                lambda: actual.permute(0, 1, -4),
                lambda: expected.permute(0, 1, -4),
            ),
            (lambda: actual.permute(1.5), lambda: expected.permute(1.5)),
            (lambda: actual.permute(dims=1), lambda: expected.permute(dims=1)),
            (
                lambda: actual.permute(0, 1.5, 2),
                lambda: expected.permute(0, 1.5, 2),
            ),
            (
                lambda: actual.permute([0, np.bool_(True), 2]),
                lambda: expected.permute([0, np.bool_(True), 2]),
            ),
            (
                lambda: actual.permute([1.5, 0, 2]),
                lambda: expected.permute([1.5, 0, 2]),
            ),
            (
                lambda: actual.permute(dims=[1.5, 0, 2]),
                lambda: expected.permute(dims=[1.5, 0, 2]),
            ),
            (
                lambda: actual.permute((2, 0, 1), (0, 1, 2)),
                lambda: expected.permute((2, 0, 1), (0, 1, 2)),
            ),
            (
                lambda: actual.permute(1.5, 0, 2),
                lambda: expected.permute(1.5, 0, 2),
            ),
            (
                lambda: actual.permute(2, 0, 1, unexpected=None),
                lambda: expected.permute(2, 0, 1, unexpected=None),
            ),
            (
                lambda: actual.permute(2, 0, 1, dims=(2, 0, 1)),
                lambda: expected.permute(2, 0, 1, dims=(2, 0, 1)),
            ),
            (
                lambda: actual.permute([0, 1.5, 2], unexpected=None),
                lambda: expected.permute([0, 1.5, 2], unexpected=None),
            ),
            (
                lambda: actual.permute(dims=1, unexpected=None),
                lambda: expected.permute(dims=1, unexpected=None),
            ),
            (
                lambda: torch.tensor(1.0).permute(-1),
                lambda: reference_torch.tensor(1.0).permute(-1),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_index_conversion_order_matches_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")

        class StatefulIndex:
            def __init__(self, name, calls, value):
                self.name = name
                self.calls = calls
                self.value = value

            def __index__(self):
                self.calls.append(self.name)
                return self.value

        for style in ("variadic", "sequence", "keyword"):
            actual_calls = []
            expected_calls = []
            actual_dimensions = [
                StatefulIndex("first", actual_calls, 2),
                StatefulIndex("second", actual_calls, 0),
                StatefulIndex("third", actual_calls, 1),
            ]
            expected_dimensions = [
                StatefulIndex("first", expected_calls, 2),
                StatefulIndex("second", expected_calls, 0),
                StatefulIndex("third", expected_calls, 1),
            ]
            actual_source = torch.zeros((2, 3, 4))
            expected_source = reference_torch.zeros((2, 3, 4))
            if style == "variadic":
                actual = actual_source.permute(*actual_dimensions)
                expected = expected_source.permute(*expected_dimensions)
            elif style == "sequence":
                actual = actual_source.permute(actual_dimensions)
                expected = expected_source.permute(expected_dimensions)
            else:
                actual = actual_source.permute(dims=actual_dimensions)
                expected = expected_source.permute(dims=expected_dimensions)

            self.assertEqual(actual_calls, expected_calls)
            self.assert_matches(
                actual,
                expected,
                actual_source=actual_source,
                expected_source=expected_source,
                case=style,
            )

    def test_descriptor_metadata_and_unbound_behavior_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        actual_descriptor = inspect.getattr_static(torch.Tensor, "permute")
        expected_descriptor = inspect.getattr_static(reference_torch.Tensor, "permute")
        actual = torch.zeros((2, 3, 4))
        expected = reference_torch.zeros((2, 3, 4))

        self.assertIs(type(actual_descriptor), types.MethodDescriptorType)
        self.assertIs(type(expected_descriptor), types.MethodDescriptorType)
        for attribute in ("__name__", "__qualname__", "__doc__", "__text_signature__"):
            self.assertEqual(
                getattr(actual_descriptor, attribute),
                getattr(expected_descriptor, attribute),
            )
        self.assertEqual(
            actual_descriptor.__objclass__.__name__,
            expected_descriptor.__objclass__.__name__,
        )
        self.assertEqual(
            actual_descriptor.__objclass__.__module__,
            expected_descriptor.__objclass__.__module__,
        )
        for callable_object in (
            actual_descriptor,
            expected_descriptor,
            actual.permute,
            expected.permute,
        ):
            with self.subTest(callable_object=callable_object):
                with self.assertRaises(ValueError):
                    inspect.signature(callable_object)

        self.assert_error_matches(
            lambda: actual_descriptor(),
            lambda: expected_descriptor(),
        )
        self.assert_error_matches(
            lambda: actual_descriptor(1, 2, 0, 1),
            lambda: expected_descriptor(1, 2, 0, 1),
        )
        actual_result = actual_descriptor(actual, 2, 0, 1)
        expected_result = expected_descriptor(expected, 2, 0, 1)
        self.assert_matches(
            actual_result,
            expected_result,
            actual_source=actual,
            expected_source=expected,
            case="descriptor-call",
        )


if __name__ == "__main__":
    unittest.main()
