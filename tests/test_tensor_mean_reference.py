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
class TensorMeanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("Tensor.mean differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_scalar_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, value=True):
            self.assertEqual(
                np.asarray(actual).view(np.uint32).item(),
                expected.detach().cpu().numpy().view(np.uint32).item(),
            )

    @staticmethod
    def make_cases(module):
        dense = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = dense.transpose(0, 2)
        return (
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("negative zero", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
            (
                "finite cancellation",
                module.tensor(
                    [0.0, 0.0, 1.0, 3.0, 123456789.0], dtype=module.float32
                ),
            ),
            (
                "positive NaN",
                module.tensor(
                    np.asarray([0x7FC1_2345, 0x3F80_0000], dtype=np.uint32).view(
                        np.float32
                    ),
                    dtype=module.float32,
                ),
            ),
            (
                "negative NaN",
                module.tensor(
                    np.asarray([0xFFC5_4321, 0x3F80_0000], dtype=np.uint32).view(
                        np.float32
                    ),
                    dtype=module.float32,
                ),
            ),
            ("infinity", module.tensor([float("inf"), 1.0], dtype=module.float32)),
            (
                "opposite infinities",
                module.tensor([float("inf"), -float("inf")], dtype=module.float32),
            ),
        )

    @staticmethod
    def call_mean(source, form, module):
        if form == "default":
            return source.mean()
        if form == "dtype none":
            return source.mean(dtype=None)
        if form == "dtype float32":
            return source.mean(dtype=module.float32)
        if form == "dtype float alias":
            return source.mean(dtype=module.float)
        raise AssertionError(f"unknown mean form: {form}")

    def test_values_scalar_empty_offset_noncontiguous_and_nonfinite_match_pytorch_2_13(self):
        forms = ("default", "dtype none", "dtype float32", "dtype float alias")
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            name, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(name, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_mean(actual_input, form, torch),
                    self.call_mean(expected_input, form, reference_torch),
                    case=(name, form),
                )

    def test_autograd_accumulation_empty_and_no_grad_match_pytorch_2_13(self):
        values = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]
        actual_leaf = torch.tensor(values, requires_grad=True)
        expected_leaf = reference_torch.tensor(
            values, dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = actual_leaf.transpose(0, 1).mean(dtype=torch.float32)
        expected_loss = expected_leaf.transpose(0, 1).mean(
            dtype=reference_torch.float32
        )
        self.assert_scalar_matches(actual_loss, expected_loss, case="tracked")
        for _ in range(2):
            actual_loss.backward()
            expected_loss.backward()
        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.cpu().numpy()
        )

        actual_empty = torch.zeros((2, 0, 3), requires_grad=True)
        expected_empty = reference_torch.zeros(
            (2, 0, 3), dtype=reference_torch.float32, requires_grad=True
        )
        actual_empty.transpose(0, 2).mean(dtype=None).backward()
        expected_empty.transpose(0, 2).mean(dtype=None).backward()
        self.assertEqual(actual_empty.grad.shape, tuple(expected_empty.grad.shape))
        np.testing.assert_array_equal(
            np.asarray(actual_empty.grad), expected_empty.grad.cpu().numpy()
        )

        with torch.no_grad():
            actual_untracked = actual_leaf.mean(dtype=torch.float)
        with reference_torch.no_grad():
            expected_untracked = expected_leaf.mean(dtype=reference_torch.float)
        self.assert_scalar_matches(actual_untracked, expected_untracked, case="no_grad")

    def test_descriptor_shape_and_documentation_match_pytorch_2_13(self):
        actual_tensor = torch.tensor([1.0, 2.0])
        expected_tensor = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32
        )
        pairs = (
            (
                inspect.getattr_static(torch.Tensor, "mean"),
                inspect.getattr_static(reference_torch.Tensor, "mean"),
                types.MethodDescriptorType,
            ),
            (actual_tensor.mean, expected_tensor.mean, types.BuiltinMethodType),
        )
        for actual, expected, expected_type in pairs:
            self.assertIs(type(actual), expected_type)
            self.assertIs(type(expected), expected_type)
            self.assertEqual(actual.__name__, expected.__name__)
            self.assertEqual(actual.__doc__, expected.__doc__)
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

    def test_invalid_dtype_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32
        )
        cases = (
            (lambda: actual.mean(dtype=1), lambda: expected.mean(dtype=1)),
            (
                lambda: actual.mean(dtype=object()),
                lambda: expected.mean(dtype=object()),
            ),
            (
                lambda: actual.mean(torch.float32),
                lambda: expected.mean(reference_torch.float32),
            ),
            (lambda: actual.mean(extra=True), lambda: expected.mean(extra=True)),
            (
                lambda: actual.mean(0, False, torch.float32),
                lambda: expected.mean(0, False, reference_torch.float32),
            ),
            (lambda: actual.mean(out=None), lambda: expected.mean(out=None)),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_dimension_out_and_cross_dtype_boundaries_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32
        )
        expected_results = (
            expected.mean(0),
            expected.mean(None),
            expected.mean(dim=0),
            expected.mean(dim=None),
            expected.mean(0, False),
            expected.mean(dim=0, keepdim=True),
        )
        self.assertEqual(
            [tuple(result.shape) for result in expected_results],
            [(3,), (), (3,), (), (3,), (1, 3)],
        )
        actual_calls = (
            lambda: actual.mean(0),
            lambda: actual.mean(None),
            lambda: actual.mean(dim=0),
            lambda: actual.mean(dim=None),
            lambda: actual.mean(0, False),
            lambda: actual.mean(dim=0, keepdim=True),
        )
        for case, call in enumerate(actual_calls):
            with self.subTest(case=case):
                with self.assertRaises(TypeError):
                    call()

        with self.assertRaises(TypeError):
            actual.mean(dtype=reference_torch.float64)
        expected_float64 = expected.mean(dtype=reference_torch.float64)
        self.assertIs(expected_float64.dtype, reference_torch.float64)

        for dtype in (reference_torch.int64, reference_torch.bool):
            with self.subTest(dtype=dtype):
                with self.assertRaises(RuntimeError):
                    reference_torch.ones((2, 3), dtype=dtype).mean()


if __name__ == "__main__":
    unittest.main()
