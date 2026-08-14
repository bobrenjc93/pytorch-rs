import inspect
import types
import unittest
import warnings

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchTReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("torch.t differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertEqual(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, expected.shape)
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            np.testing.assert_array_equal(
                np.asarray(actual), expected.detach().cpu().numpy()
            )

    def test_seeded_positional_keyword_and_legacy_alias_views_match_pytorch_2_13(self):
        rng = np.random.default_rng(0x7_213)
        shapes = [(), (0,), (1,), (4,), (0, 3), (2, 0), (2, 3)]
        for _ in range(24):
            rank = int(rng.integers(0, 3))
            shapes.append(tuple(int(size) for size in rng.integers(0, 6, rank)))

        for case, shape in enumerate(shapes):
            elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
            values = rng.uniform(-3.0, 3.0, elements).astype(np.float32).reshape(shape)
            if elements == 0:
                actual_source = torch.zeros(shape, requires_grad=True)
                expected_source = reference_torch.zeros(
                    shape, dtype=reference_torch.float32, requires_grad=True
                )
            else:
                data = values.item() if shape == () else values.tolist()
                actual_source = torch.tensor(data, requires_grad=True)
                expected_source = reference_torch.tensor(
                    values, dtype=reference_torch.float32, requires_grad=True
                )

            for keyword in (None, "input", "a", "x"):
                with warnings.catch_warnings(record=True) as actual_warnings:
                    warnings.simplefilter("always")
                    actual = (
                        torch.t(actual_source)
                        if keyword is None
                        else torch.t(**{keyword: actual_source})
                    )
                with warnings.catch_warnings(record=True) as expected_warnings:
                    warnings.simplefilter("always")
                    expected = (
                        reference_torch.t(expected_source)
                        if keyword is None
                        else reference_torch.t(**{keyword: expected_source})
                    )
                self.assertEqual(actual_warnings, [])
                self.assertEqual(expected_warnings, [])
                self.assertIsNot(actual, actual_source)
                self.assertIsNot(expected, expected_source)
                self.assertEqual(actual.data_ptr(), actual_source.data_ptr())
                self.assertEqual(
                    expected.untyped_storage().data_ptr(),
                    expected_source.untyped_storage().data_ptr(),
                )
                self.assert_matches(actual, expected, case=(case, keyword))
                self.assert_matches(
                    actual_source.t(),
                    expected_source.t(),
                    case=(case, keyword, "method"),
                )
                self.assertEqual(actual.shape, actual_source.t().shape)
                self.assertEqual(actual.stride(), actual_source.t().stride())

    def test_offset_autograd_no_grad_and_rank_errors_match_pytorch_2_13(self):
        gradients = []
        states = []
        for module in (torch, reference_torch):
            values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            leaf = module.tensor(
                values.tolist(), dtype=module.float32, requires_grad=True
            )
            source = leaf.transpose(0, 2)[1]
            output = module.t(input=source)
            weights = module.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32
            )
            (output * weights).sum().backward()
            gradients.append(np.asarray(leaf.grad).copy())
            states.append(
                (
                    output.shape,
                    output.stride(),
                    output.storage_offset(),
                    output.data_ptr() == source.data_ptr(),
                    output.requires_grad,
                    output.is_leaf,
                )
            )

            no_grad_source = module.tensor(
                [[1.0, 2.0], [3.0, 4.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            with module.no_grad():
                no_grad_output = module.t(input=no_grad_source)
            states.append(
                (
                    no_grad_output.requires_grad,
                    no_grad_output.is_leaf,
                    no_grad_output.data_ptr() == no_grad_source.data_ptr(),
                )
            )

        np.testing.assert_array_equal(gradients[0], gradients[1])
        self.assertEqual(states[0], states[2])
        self.assertEqual(states[1], states[3])

        for rank in (3, 4, 65):
            actual = torch.zeros((0,) + (1,) * (rank - 1))
            expected = reference_torch.zeros((0,) + (1,) * (rank - 1))
            for actual_call, expected_call in (
                (lambda: torch.t(actual), lambda: reference_torch.t(expected)),
                (
                    lambda: torch.t(input=actual),
                    lambda: reference_torch.t(input=expected),
                ),
            ):
                self.assert_error_matches(actual_call, expected_call)

    def test_callable_metadata_and_export_match_pytorch_2_13(self):
        actual = torch.t
        expected = reference_torch.t
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__, expected.__doc__)
        self.assertEqual(actual.__module__, torch.tensor.__module__)
        self.assertEqual(expected.__module__, reference_torch.tensor.__module__)
        self.assertIn("t", torch.__all__)
        self.assertIn("t", reference_torch.__all__)
        for function in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(function)

    def test_binding_and_tensor_type_errors_match_pytorch_2_13(self):
        actual = torch.tensor([1.0])
        expected = reference_torch.tensor([1.0])
        cases = (
            (lambda: torch.t(), lambda: reference_torch.t()),
            (
                lambda: torch.t(actual, actual),
                lambda: reference_torch.t(expected, expected),
            ),
            (
                lambda: torch.t(actual, input=actual),
                lambda: reference_torch.t(expected, input=expected),
            ),
            (
                lambda: torch.t(actual, extra=True, input=actual),
                lambda: reference_torch.t(expected, extra=True, input=expected),
            ),
            (
                lambda: torch.t(actual, input=actual, extra=True),
                lambda: reference_torch.t(expected, input=expected, extra=True),
            ),
            (
                lambda: torch.t(extra=actual),
                lambda: reference_torch.t(extra=expected),
            ),
            (
                lambda: torch.t(1, extra=True),
                lambda: reference_torch.t(1, extra=True),
            ),
            (lambda: torch.t(input=[]), lambda: reference_torch.t(input=[])),
            (lambda: torch.t(a=1), lambda: reference_torch.t(a=1)),
            (lambda: torch.t(x=[]), lambda: reference_torch.t(x=[])),
            (
                lambda: torch.t(np.zeros((2, 3), dtype=np.float32)),
                lambda: reference_torch.t(np.zeros((2, 3), dtype=np.float32)),
            ),
            (
                lambda: torch.t(a=actual, x=actual),
                lambda: reference_torch.t(a=expected, x=expected),
            ),
            (
                lambda: torch.t(x=actual, a=actual),
                lambda: reference_torch.t(x=expected, a=expected),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)


if __name__ == "__main__":
    unittest.main()
