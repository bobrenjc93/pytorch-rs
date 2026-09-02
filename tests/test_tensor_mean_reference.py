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

    def assert_scalar_matches(self, actual, expected, source, *, case):
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
            self.assertFalse(actual.is_set_to(source))
        with self.subTest(case=case, value=True):
            actual_bits = np.asarray(actual).view(np.uint32).item()
            expected_bits = expected.detach().cpu().numpy().view(np.uint32).item()
            if np.isnan(expected.detach().cpu().numpy()).item():
                self.assertTrue(np.isnan(np.asarray(actual)).item())
            else:
                self.assertEqual(actual_bits, expected_bits)

    @staticmethod
    def make_cases(module):
        dense = module.tensor(
            np.arange(24, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        noncontiguous = dense.transpose(0, 2)
        return (
            ("division rounding", module.tensor([1.0, 2.0, 4.0], dtype=module.float32)),
            ("scalar", module.tensor(-3.5, dtype=module.float32)),
            ("negative zero", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("singleton", module.tensor([[[7.0]]], dtype=module.float32)[0]),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
            ("nan", module.tensor([1.0, float("nan"), 2.0], dtype=module.float32)),
            ("positive infinity", module.tensor([1.0, float("inf"), 2.0], dtype=module.float32)),
            ("negative infinity", module.tensor([1.0, float("-inf"), 2.0], dtype=module.float32)),
            ("mixed infinities", module.tensor([float("inf"), float("-inf")], dtype=module.float32)),
        )

    @staticmethod
    def call_method_mean(source, form, module):
        if form == "default":
            return source.mean()
        if form == "positional none dim":
            return source.mean(None)
        if form == "keyword none dim":
            return source.mean(dim=None)
        if form == "none dim keepdim false":
            return source.mean(None, False)
        if form == "dtype none":
            return source.mean(dtype=None)
        if form == "dtype float32":
            return source.mean(dtype=module.float32)
        if form == "dtype float alias":
            return source.mean(dtype=module.float)
        if form == "none dim dtype float32":
            return source.mean(dim=None, keepdim=False, dtype=module.float32)
        if form == "positional none dim keepdim true":
            return source.mean(None, True)
        if form == "keyword none dim keepdim true":
            return source.mean(dim=None, keepdim=True)
        if form == "none dim keepdim true dtype none":
            return source.mean(dim=None, keepdim=True, dtype=None)
        if form == "none dim keepdim true dtype float32":
            return source.mean(dim=None, keepdim=True, dtype=module.float32)
        raise AssertionError(f"unknown mean method form: {form}")

    @staticmethod
    def call_top_level_mean(source, form, module):
        if form == "positional":
            return module.mean(source)
        if form == "positional none dim":
            return module.mean(source, None)
        if form == "keyword none dim":
            return module.mean(source, dim=None)
        if form == "none dim keepdim false":
            return module.mean(source, None, False)
        if form == "none dim out none":
            return module.mean(source, dim=None, out=None)
        if form == "input":
            return module.mean(input=source)
        if form == "dtype none":
            return module.mean(source, dtype=None)
        if form == "dtype float32":
            return module.mean(source, dtype=module.float32)
        if form == "dtype float alias":
            return module.mean(source, dtype=module.float)
        if form == "all keyword defaults":
            return module.mean(
                input=source, dim=None, keepdim=False, dtype=module.float32, out=None
            )
        if form == "positional none dim keepdim true":
            return module.mean(source, None, True)
        if form == "keyword none dim keepdim true":
            return module.mean(source, dim=None, keepdim=True)
        if form == "none dim out none keepdim true":
            return module.mean(source, dim=None, keepdim=True, out=None)
        if form == "none dim keepdim true dtype none":
            return module.mean(source, dim=None, keepdim=True, dtype=None)
        if form == "none dim keepdim true dtype float32":
            return module.mean(source, dim=None, keepdim=True, dtype=module.float32)
        if form == "all keyword keepdim defaults":
            return module.mean(
                input=source, dim=None, keepdim=True, dtype=module.float32, out=None
            )
        raise AssertionError(f"unknown top-level mean form: {form}")

    @staticmethod
    def autograd_case(module, case):
        if case == "scalar":
            leaf = module.tensor(-3.0, dtype=module.float32, requires_grad=True)
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=True
            )
            return leaf, leaf.transpose(0, 2)[1]
        if case == "singleton":
            leaf = module.tensor([[[7.0]]], dtype=module.float32, requires_grad=True)
            return leaf, leaf[0]

        leaf = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown mean autograd case: {case}")

    def test_supported_values_metadata_and_storage_match_pytorch_2_13(self):
        method_forms = (
            "default",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "none dim dtype float32",
        )
        top_level_forms = (
            "positional",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "none dim out none",
            "input",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "all keyword defaults",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            name, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(name, expected_name)
            for form in method_forms:
                self.assert_scalar_matches(
                    self.call_method_mean(actual_input, form, torch),
                    self.call_method_mean(expected_input, form, reference_torch),
                    actual_input,
                    case=(name, "method", form),
                )
            for form in top_level_forms:
                self.assert_scalar_matches(
                    self.call_top_level_mean(actual_input, form, torch),
                    self.call_top_level_mean(expected_input, form, reference_torch),
                    actual_input,
                    case=(name, "top-level", form),
                )

    def test_keepdim_full_reduction_forms_match_pytorch_2_13(self):
        method_forms = (
            "positional none dim keepdim true",
            "keyword none dim keepdim true",
            "none dim keepdim true dtype none",
            "none dim keepdim true dtype float32",
        )
        top_level_forms = (
            "positional none dim keepdim true",
            "keyword none dim keepdim true",
            "none dim out none keepdim true",
            "none dim keepdim true dtype none",
            "none dim keepdim true dtype float32",
            "all keyword keepdim defaults",
        )
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            name, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(name, expected_name)
            for form in method_forms:
                self.assert_scalar_matches(
                    self.call_method_mean(actual_input, form, torch),
                    self.call_method_mean(expected_input, form, reference_torch),
                    actual_input,
                    case=(name, "method", form),
                )
            for form in top_level_forms:
                self.assert_scalar_matches(
                    self.call_top_level_mean(actual_input, form, torch),
                    self.call_top_level_mean(expected_input, form, reference_torch),
                    actual_input,
                    case=(name, "top-level", form),
                )

    def test_first_order_autograd_and_no_grad_match_pytorch_2_13(self):
        outcomes = []
        for module in (torch, reference_torch):
            leaf = module.tensor(
                [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                dtype=module.float32,
                requires_grad=True,
            )
            loss = module.mean(leaf.transpose(0, 1), dim=None, dtype=module.float32)
            loss.backward()
            loss.backward()

            empty = module.zeros((2, 0, 3), dtype=module.float32, requires_grad=True)
            empty_loss = empty.transpose(0, 2)[1].mean(None, False)
            empty_loss.backward()
            empty_loss.backward()

            rounding_leaf = module.tensor(
                [1.0, 2.0, 4.0], dtype=module.float32, requires_grad=True
            )
            (rounding_leaf.mean() * 7.0).backward()

            with module.no_grad():
                untracked = module.mean(leaf, dtype=module.float)
            outcomes.append(
                (
                    np.asarray(leaf.grad).copy(),
                    tuple(empty.grad.shape),
                    empty.grad.tolist(),
                    np.asarray(rounding_leaf.grad).copy(),
                    untracked.requires_grad,
                    untracked.is_leaf,
                )
            )
        np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
        self.assertEqual(outcomes[0][1:3], outcomes[1][1:3])
        np.testing.assert_array_equal(outcomes[0][3], outcomes[1][3])
        self.assertEqual(outcomes[0][4:], outcomes[1][4:])

    def test_keepdim_no_grad_and_final_scalar_backward_match_pytorch_2_13(self):
        form_groups = (
            (
                "method",
                self.call_method_mean,
                (
                    "positional none dim keepdim true",
                    "keyword none dim keepdim true",
                    "none dim keepdim true dtype none",
                    "none dim keepdim true dtype float32",
                ),
            ),
            (
                "top-level",
                self.call_top_level_mean,
                (
                    "positional none dim keepdim true",
                    "keyword none dim keepdim true",
                    "none dim out none keepdim true",
                    "none dim keepdim true dtype none",
                    "none dim keepdim true dtype float32",
                    "all keyword keepdim defaults",
                ),
            ),
        )
        for group, call_mean, forms in form_groups:
            for case in ("scalar", "empty", "singleton", "offset", "noncontiguous"):
                for form in forms:
                    actual_leaf, actual_input = self.autograd_case(torch, case)
                    expected_leaf, expected_input = self.autograd_case(reference_torch, case)
                    actual_output = call_mean(actual_input, form, torch)
                    expected_output = call_mean(expected_input, form, reference_torch)
                    self.assert_scalar_matches(
                        actual_output,
                        expected_output,
                        actual_input,
                        case=(group, case, form, "forward"),
                    )

                    actual_output.sum().backward()
                    expected_output.sum().backward()
                    np.testing.assert_array_equal(
                        np.asarray(actual_leaf.grad),
                        expected_leaf.grad.detach().cpu().numpy(),
                    )

        actual_leaf = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [1.0, -2.0, 3.0],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        with torch.no_grad():
            actual_untracked = torch.mean(
                input=actual_leaf, dim=None, keepdim=True, dtype=torch.float
            )
        with reference_torch.no_grad():
            expected_untracked = reference_torch.mean(
                input=expected_leaf,
                dim=None,
                keepdim=True,
                dtype=reference_torch.float,
            )
        self.assert_scalar_matches(
            actual_untracked, expected_untracked, actual_leaf, case="keepdim no_grad"
        )
        self.assertIsNone(actual_leaf.grad)

    def test_callable_metadata_matches_pytorch_2_13(self):
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
            self.assertEqual(actual.__text_signature__, expected.__text_signature__)
            with self.assertRaises(ValueError):
                inspect.signature(actual)
            with self.assertRaises(ValueError):
                inspect.signature(expected)

        self.assertIs(type(torch.mean), types.BuiltinFunctionType)
        self.assertIs(type(reference_torch.mean), types.BuiltinFunctionType)
        self.assertEqual(torch.mean.__name__, reference_torch.mean.__name__)
        self.assertEqual(torch.__all__.count("mean"), reference_torch.__all__.count("mean"))

    def test_invalid_dtype_and_argument_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        cases = (
            (lambda: actual.mean(dtype=1), lambda: expected.mean(dtype=1)),
            (
                lambda: actual.mean(dtype=object()),
                lambda: expected.mean(dtype=object()),
            ),
            (
                lambda: actual.mean(dim=None, dtype=1),
                lambda: expected.mean(dim=None, dtype=1),
            ),
            (
                lambda: actual.mean(None, False, dtype=object()),
                lambda: expected.mean(None, False, dtype=object()),
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
            (lambda: torch.mean(), lambda: reference_torch.mean()),
            (lambda: torch.mean(1), lambda: reference_torch.mean(1)),
            (
                lambda: torch.mean(actual, dtype=1),
                lambda: reference_torch.mean(expected, dtype=1),
            ),
            (
                lambda: torch.mean(actual, None, dtype=1),
                lambda: reference_torch.mean(expected, None, dtype=1),
            ),
            (
                lambda: torch.mean(actual, torch.float32),
                lambda: reference_torch.mean(expected, reference_torch.float32),
            ),
            (
                lambda: torch.mean(actual, 0, False, torch.float32),
                lambda: reference_torch.mean(expected, 0, False, reference_torch.float32),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_dimension_keepdim_out_and_cross_dtype_reductions_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones((2, 3), dtype=reference_torch.float32)
        cases = (
            (lambda: actual.mean(0), lambda: expected.mean(0)),
            (lambda: actual.mean(dim=0), lambda: expected.mean(dim=0)),
            (lambda: actual.mean((0, 1)), lambda: expected.mean((0, 1))),
            (lambda: actual.mean(dim=[0, 1]), lambda: expected.mean(dim=[0, 1])),
            (
                lambda: actual.mean(dtype=reference_torch.float64),
                lambda: expected.mean(dtype=reference_torch.float64),
            ),
            (
                lambda: actual.mean(dim=None, dtype=reference_torch.float64),
                lambda: expected.mean(dim=None, dtype=reference_torch.float64),
            ),
            (lambda: torch.mean(actual, 0), lambda: reference_torch.mean(expected, 0)),
            (
                lambda: torch.mean(actual, dim=(0, 1)),
                lambda: reference_torch.mean(expected, dim=(0, 1)),
            ),
            (
                lambda: torch.mean(actual, out=torch.tensor(0.0)),
                lambda: reference_torch.mean(expected, out=reference_torch.empty(())),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises((TypeError, NotImplementedError)):
                    actual_call()
                expected_call()


if __name__ == "__main__":
    unittest.main()
