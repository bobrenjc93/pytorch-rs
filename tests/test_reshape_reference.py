import inspect
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class ReshapeReferenceTests(unittest.TestCase):
    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(np.asarray(actual), expected.detach().cpu().numpy())

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

    def test_seeded_values_strides_offsets_and_consumers_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        rng = np.random.default_rng(0x2E5A9E_213)
        shapes = [(), (0,), (1,), (2, 0, 3), (1, 2, 1, 3), (2, 3, 4)]
        for _ in range(36):
            rank = int(rng.integers(0, 7))
            shapes.append(tuple(int(value) for value in rng.integers(0, 5, rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.normal(size=elements).astype(np.float32).reshape(shape)
            if elements:
                actual = torch.tensor(values.item() if not shape else values.tolist())
                expected = reference_torch.tensor(values, dtype=reference_torch.float32)
            else:
                actual = torch.zeros(shape)
                expected = reference_torch.zeros(shape, dtype=reference_torch.float32)

            if len(shape) >= 2:
                actual = actual.transpose(0, -1)
                expected = expected.transpose(0, -1)
            if actual.shape and actual.shape[0] > 0 and case % 3 == 1:
                actual = actual[-1]
                expected = expected[-1]

            elements = actual.numel()
            if elements == 0:
                target = (2, 0)
            elif elements == 1 and case % 2:
                target = ()
            elif case % 2:
                target = (elements,)
            else:
                target = (1, elements)

            if case % 3 == 0:
                actual_output = torch.reshape(actual, target)
                expected_output = reference_torch.reshape(expected, target)
            elif case % 3 == 1:
                actual_output = torch.reshape(actual, shape=list(target))
                expected_output = reference_torch.reshape(expected, shape=list(target))
            else:
                actual_output = torch.reshape(input=actual, shape=target)
                expected_output = reference_torch.reshape(input=expected, shape=target)

            self.assertIsNot(actual_output, actual)
            self.assertIsNot(expected_output, expected)
            self.assert_matches(actual_output, expected_output, case=case)
            self.assert_matches(
                actual_output.clone(), expected_output.clone(), case=(case, "clone")
            )
            self.assert_matches(
                actual_output + 0.25,
                expected_output + 0.25,
                case=(case, "arithmetic"),
            )
            self.assert_matches(
                actual_output.reshape(-1),
                expected_output.reshape(-1),
                case=(case, "method-reshape"),
            )

    def test_view_copy_and_autograd_paths_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        actual_base = torch.tensor(values.tolist())
        expected_base = reference_torch.tensor(values)
        actual_view_source = actual_base[1]
        expected_view_source = expected_base[1]
        actual_view = torch.reshape(actual_view_source, (2, 6))
        expected_view = reference_torch.reshape(expected_view_source, (2, 6))
        self.assert_matches(actual_view, expected_view, case="offset-view")
        self.assertEqual(
            expected_view.untyped_storage().data_ptr(),
            expected_view_source.untyped_storage().data_ptr(),
        )

        actual_copy_source = actual_base.transpose(0, 1)
        expected_copy_source = expected_base.transpose(0, 1)
        actual_copy = torch.reshape(actual_copy_source, (6, 4))
        expected_copy = reference_torch.reshape(expected_copy_source, (6, 4))
        self.assert_matches(actual_copy, expected_copy, case="non-contiguous-copy")
        self.assertNotEqual(
            expected_copy.untyped_storage().data_ptr(),
            expected_copy_source.untyped_storage().data_ptr(),
        )

        gradients = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
            )
            output = module.reshape(leaf.transpose(0, 1), (6,))
            weights = module.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())
        np.testing.assert_array_equal(gradients[0], gradients[1])

    def test_signature_argument_and_shape_errors_match_pytorch_2_13(self):
        self.assertEqual(reference_torch.__version__.split("+")[0], "2.13.0")
        self.assertIsNone(torch.reshape.__text_signature__)
        self.assertIsNone(reference_torch.reshape.__text_signature__)
        for function in (torch.reshape, reference_torch.reshape):
            with self.assertRaises(ValueError):
                inspect.signature(function)

        actual = torch.zeros((6,))
        expected = reference_torch.zeros((6,))
        cases = (
            (lambda: torch.reshape(), lambda: reference_torch.reshape()),
            (lambda: torch.reshape(actual), lambda: reference_torch.reshape(expected)),
            (
                lambda: torch.reshape(shape=(2, 3)),
                lambda: reference_torch.reshape(shape=(2, 3)),
            ),
            (
                lambda: torch.reshape(actual, 2, 3),
                lambda: reference_torch.reshape(expected, 2, 3),
            ),
            (
                lambda: torch.reshape(actual, (2, 3), input=actual),
                lambda: reference_torch.reshape(expected, (2, 3), input=expected),
            ),
            (
                lambda: torch.reshape(actual, (2, 3), shape=(2, 3)),
                lambda: reference_torch.reshape(expected, (2, 3), shape=(2, 3)),
            ),
            (
                lambda: torch.reshape(actual, (2, 3), extra=True),
                lambda: reference_torch.reshape(expected, (2, 3), extra=True),
            ),
            (
                lambda: torch.reshape(actual, (2**100, 3), input=actual),
                lambda: reference_torch.reshape(
                    expected, (2**100, 3), input=expected
                ),
            ),
            (
                lambda: torch.reshape(actual, (2**100, 3), shape=(2, 3)),
                lambda: reference_torch.reshape(
                    expected, (2**100, 3), shape=(2, 3)
                ),
            ),
            (
                lambda: torch.reshape(actual, (2**100, 3), extra=True),
                lambda: reference_torch.reshape(expected, (2**100, 3), extra=True),
            ),
            (
                lambda: torch.reshape(actual, shape=(2**100, 3), extra=True),
                lambda: reference_torch.reshape(
                    expected, shape=(2**100, 3), extra=True
                ),
            ),
            (lambda: torch.reshape([], (0,)), lambda: reference_torch.reshape([], (0,))),
            (
                lambda: torch.reshape(input=None, shape=()),
                lambda: reference_torch.reshape(input=None, shape=()),
            ),
            (lambda: torch.reshape(actual, 6), lambda: reference_torch.reshape(expected, 6)),
            (
                lambda: torch.reshape(actual, shape=6),
                lambda: reference_torch.reshape(expected, shape=6),
            ),
            (
                lambda: torch.reshape(actual, (2.0, 3)),
                lambda: reference_torch.reshape(expected, (2.0, 3)),
            ),
            (
                lambda: torch.reshape(actual, (2.0, 3), extra=True),
                lambda: reference_torch.reshape(expected, (2.0, 3), extra=True),
            ),
            (
                lambda: torch.reshape(actual, shape=(2.0, 3)),
                lambda: reference_torch.reshape(expected, shape=(2.0, 3)),
            ),
            (
                lambda: torch.reshape(actual, (True, 6)),
                lambda: reference_torch.reshape(expected, (True, 6)),
            ),
            (
                lambda: torch.reshape(actual, ((2, 3),)),
                lambda: reference_torch.reshape(expected, ((2, 3),)),
            ),
            (
                lambda: torch.reshape(actual, (4, 2)),
                lambda: reference_torch.reshape(expected, (4, 2)),
            ),
            (
                lambda: torch.reshape(actual, (-1, -1)),
                lambda: reference_torch.reshape(expected, (-1, -1)),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
