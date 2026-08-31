import inspect
import pickle
import re
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TopLevelMeanReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.mean differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_scalar_matches(self, actual, expected, actual_source, expected_source, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.numel(), expected.numel())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertIs(actual.dtype, torch.float32)
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertFalse(actual.is_set_to(actual_source))
            self.assertFalse(expected.is_set_to(expected_source))
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
            ("singleton", module.tensor([5.0], dtype=module.float32)),
            ("contiguous offset", dense[1]),
            ("offset", noncontiguous[1]),
            ("noncontiguous", noncontiguous),
        )

    @staticmethod
    def call_mean(module, source, form):
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
        if form == "out none":
            return module.mean(source, out=None)
        if form == "dtype none":
            return module.mean(source, dtype=None)
        if form == "dtype float32":
            return module.mean(source, dtype=module.float32)
        if form == "dtype float alias":
            return module.mean(source, dtype=module.float)
        if form == "alias and dtype":
            return module.mean(x=source, dtype=module.float32)
        if form == "none dim dtype out none":
            return module.mean(
                input=source, dim=None, keepdim=False, dtype=module.float32, out=None
            )
        return module.mean(**{form: source})

    @staticmethod
    def rank_one_strided_vector(module, values, *, requires_grad=False):
        rows = len(values)
        columns = 5
        selected_column = 2
        matrix = np.full((rows, columns), np.float32(0.5), dtype=np.float32)
        matrix[:, selected_column] = np.asarray(values, dtype=np.float32)
        source = module.tensor(
            matrix.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        return source, source.transpose(0, 1)[selected_column]

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

        leaf = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
            requires_grad=True,
        )
        if case == "offset":
            return leaf, leaf[1]
        if case == "noncontiguous":
            return leaf, leaf.transpose(0, 2)
        raise AssertionError(f"unknown torch.mean autograd case: {case}")

    def test_supported_values_metadata_and_storage_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "positional",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "none dim out none",
            "out none",
            "input",
            "x",
            "a",
            "x1",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "alias and dtype",
            "none dim dtype out none",
        )
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            case, actual_input = actual_case
            expected_name, expected_input = expected_case
            self.assertEqual(case, expected_name)
            for form in forms:
                self.assert_scalar_matches(
                    self.call_mean(torch, actual_input, form),
                    self.call_mean(reference_torch, expected_input, form),
                    actual_input,
                    expected_input,
                    case=(case, form),
                )

    def test_autograd_empty_and_no_grad_match_pytorch_2_13(self):
        forms = (
            "positional",
            "positional none dim",
            "keyword none dim",
            "none dim keepdim false",
            "none dim out none",
            "out none",
            "dtype none",
            "dtype float32",
            "dtype float alias",
            "alias and dtype",
            "none dim dtype out none",
        )
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input = self.autograd_case(torch, case)
                expected_leaf, expected_input = self.autograd_case(
                    reference_torch, case
                )
                actual_loss = self.call_mean(torch, actual_input, form)
                expected_loss = self.call_mean(reference_torch, expected_input, form)
                self.assert_scalar_matches(
                    actual_loss,
                    expected_loss,
                    actual_input,
                    expected_input,
                    case=(case, form),
                )

                actual_loss.backward()
                expected_loss.backward()
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
            actual = torch.mean(input=actual_leaf, dim=None, dtype=torch.float, out=None)
        with reference_torch.no_grad():
            expected = reference_torch.mean(
                input=expected_leaf, dim=None, dtype=reference_torch.float, out=None
            )
        self.assert_scalar_matches(
            actual, expected, actual_leaf, expected_leaf, case="no_grad"
        )
        self.assertIsNone(actual_leaf.grad)

    def test_repeated_backward_matches_pytorch_2_13(self):
        actual_leaf = torch.tensor([1.0, 2.0], requires_grad=True)
        expected_leaf = reference_torch.tensor(
            [1.0, 2.0], dtype=reference_torch.float32, requires_grad=True
        )
        actual_loss = torch.mean(actual_leaf)
        expected_loss = reference_torch.mean(expected_leaf)

        actual_loss.backward()
        expected_loss.backward()
        actual_loss.backward()
        expected_loss.backward()

        np.testing.assert_array_equal(
            np.asarray(actual_leaf.grad), expected_leaf.grad.detach().cpu().numpy()
        )
        self.assertEqual(actual_leaf.grad.tolist(), [1.0, 1.0])

    def test_rank_one_transpose_selected_offset_mean_edges_match_pytorch_2_13(self):
        cases = (
            ("signed zero", [-0.0, 0.0, -0.0, 0.0]),
            ("nan", [1.0, np.nan, 2.0, -3.0]),
            ("positive infinity", [1.0, np.inf, 2.0, 3.0]),
            ("negative infinity", [1.0, -np.inf, 2.0, 3.0]),
            ("sequential cancellation", [1.0e20, -1.0e20, 3.0, -0.0]),
        )

        for case, values in cases:
            _, actual = self.rank_one_strided_vector(torch, values)
            _, expected = self.rank_one_strided_vector(reference_torch, values)
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertFalse(actual.is_contiguous())
            self.assertFalse(expected.is_contiguous())
            self.assert_scalar_matches(
                torch.mean(actual),
                reference_torch.mean(expected),
                actual,
                expected,
                case=("rank-one offset", case),
            )

    def test_callable_contract_matches_pytorch_2_13(self):
        actual = torch.mean
        expected = reference_torch.mean
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertIs(type(expected), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__qualname__, expected.__qualname__)
        self.assertEqual(actual.__module__, expected.__module__)
        self.assertIn("mean(input, *, dtype=None) -> Tensor", actual.__doc__)
        self.assertIn("mean(input, *, dtype=None) -> Tensor", expected.__doc__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(torch.__all__.count("mean"), reference_torch.__all__.count("mean"))
        self.assertNotIn("_VariableFunctionsClass", torch.__all__)
        self.assertFalse(hasattr(torch, "_VariableFunctionsClass"))

        owner = actual.__reduce__()[1][0]
        self.assertIs(owner, torch._C._VariableFunctionsClass)
        self.assertIs(owner.mean, actual)
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(protocol=protocol):
                self.assertIs(pickle.loads(pickle.dumps(actual, protocol=protocol)), actual)
        self.assertEqual(
            re.sub(r"0x[0-9a-f]+", "0x...", repr(actual)),
            re.sub(r"0x[0-9a-f]+", "0x...", repr(expected)),
        )
        with self.assertRaises(ValueError):
            inspect.signature(actual)
        with self.assertRaises(ValueError):
            inspect.signature(expected)

    def test_invalid_binding_errors_match_pytorch_2_13(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32
        )
        cases = (
            (lambda: torch.mean(), lambda: reference_torch.mean()),
            (lambda: torch.mean(extra=True), lambda: reference_torch.mean(extra=True)),
            (lambda: torch.mean(out=None), lambda: reference_torch.mean(out=None)),
            (lambda: torch.mean(1), lambda: reference_torch.mean(1)),
            (
                lambda: torch.mean(1, dtype=torch.float32),
                lambda: reference_torch.mean(1, dtype=reference_torch.float32),
            ),
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
            (
                lambda: torch.mean(actual, input=actual),
                lambda: reference_torch.mean(expected, input=expected),
            ),
            (
                lambda: torch.mean(actual, keepdim=True),
                lambda: reference_torch.mean(expected, keepdim=True),
            ),
            (
                lambda: torch.mean(actual, actual),
                lambda: reference_torch.mean(expected, expected),
            ),
            (
                lambda: torch.mean(actual, 0, dtype=1),
                lambda: reference_torch.mean(expected, 0, dtype=1),
            ),
            (
                lambda: torch.mean(actual, 0, out=[]),
                lambda: reference_torch.mean(expected, 0, out=[]),
            ),
            (
                lambda: torch.mean(actual, None, out=[]),
                lambda: reference_torch.mean(expected, None, out=[]),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                self.assert_error_matches(actual_call, expected_call)

    def test_dimension_out_and_dtype_conversions_remain_unsupported(self):
        actual = torch.ones((2, 3))
        expected = reference_torch.ones(
            (2, 3), dtype=reference_torch.float32
        )
        actual_out = torch.tensor(0.0)
        expected_out = reference_torch.tensor(0.0, dtype=reference_torch.float32)
        expected_dim_out = reference_torch.empty(3, dtype=reference_torch.float32)
        cases = (
            (lambda: torch.mean(actual, 0), lambda: reference_torch.mean(expected, 0)),
            (
                lambda: torch.mean(input=actual, dim=0),
                lambda: reference_torch.mean(input=expected, dim=0),
            ),
            (
                lambda: torch.mean(actual, (0, 1)),
                lambda: reference_torch.mean(expected, (0, 1)),
            ),
            (
                lambda: torch.mean(actual, [0, 1]),
                lambda: reference_torch.mean(expected, [0, 1]),
            ),
            (
                lambda: torch.mean(actual, 0, keepdim=True),
                lambda: reference_torch.mean(expected, 0, keepdim=True),
            ),
            (
                lambda: torch.mean(actual, None, keepdim=True),
                lambda: reference_torch.mean(expected, None, keepdim=True),
            ),
            (
                lambda: torch.mean(actual, out=actual_out),
                lambda: reference_torch.mean(expected, out=expected_out),
            ),
            (
                lambda: torch.mean(actual, 0, out=actual_out),
                lambda: reference_torch.mean(expected, 0, out=expected_dim_out),
            ),
            (
                lambda: torch.mean(actual, dtype=reference_torch.float64),
                lambda: reference_torch.mean(expected, dtype=reference_torch.float64),
            ),
            (
                lambda: torch.mean(actual, None, dtype=reference_torch.float64),
                lambda: reference_torch.mean(expected, None, dtype=reference_torch.float64),
            ),
        )
        for case, (actual_call, expected_call) in enumerate(cases):
            with self.subTest(case=case):
                with self.assertRaises((TypeError, NotImplementedError)):
                    actual_call()
                expected_call()


if __name__ == "__main__":
    unittest.main()
