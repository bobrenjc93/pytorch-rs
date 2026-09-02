import copy
import inspect
import pickle
import types
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchAllcloseReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        version = reference_torch.__version__.split("+")[0]
        if version != "2.13.0":
            raise AssertionError("torch.allclose differentials require pinned PyTorch 2.13.0")

    def assert_error_matches(self, actual_call, expected_call):
        with self.assertRaises(Exception) as actual_raised:
            actual_call()
        with self.assertRaises(Exception) as expected_raised:
            expected_call()
        self.assertIs(type(actual_raised.exception), type(expected_raised.exception))
        self.assertEqual(str(actual_raised.exception), str(expected_raised.exception))

    def assert_result_matches(self, actual_call, expected_call):
        actual = actual_call()
        expected = expected_call()
        self.assertIs(type(actual), bool)
        self.assertIs(type(expected), bool)
        self.assertIs(actual, expected)

    @staticmethod
    def bits_tensor(module, bits):
        values = np.asarray(bits, dtype=np.uint32).view(np.float32)
        return module.tensor(memoryview(values), dtype=module.float32)

    def make_cases(self, module):
        contiguous_offset = module.tensor(
            [[10.0, 11.0, 12.0], [1.0, 2.0, 3.0]], dtype=module.float32
        )[1]
        strided = module.tensor(
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]], dtype=module.float32
        ).transpose(0, 1)
        offset = module.tensor(
            [[10.0, 20.0], [1.0, 3.0], [2.0, 4.0]], dtype=module.float32
        ).transpose(0, 1)[1]
        strided_empty = module.zeros((2, 0, 3), dtype=module.float32).transpose(
            0, 2
        )
        next_after_one = float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        next_step = float(np.float32(next_after_one - np.float32(1.0)))
        return (
            (
                "same-shape equal",
                module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32),
                module.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=module.float32),
                {},
            ),
            (
                "same-shape unequal",
                module.tensor([1.0, 2.0], dtype=module.float32),
                module.tensor([1.0, 2.0002], dtype=module.float32),
                {},
            ),
            (
                "rank-zero broadcasts",
                module.tensor(3.0, dtype=module.float32),
                module.tensor([3.0, 3.0], dtype=module.float32),
                {},
            ),
            (
                "unequal broadcasted shapes equal",
                module.tensor([[1.0], [1.0]], dtype=module.float32),
                module.tensor([1.0, 1.0], dtype=module.float32),
                {},
            ),
            (
                "unequal broadcasted shapes unequal",
                module.tensor([[1.0], [1.0]], dtype=module.float32),
                module.tensor([1.0, 2.0], dtype=module.float32),
                {},
            ),
            (
                "empty vector broadcasts",
                module.zeros((0,), dtype=module.float32),
                module.ones((1, 0), dtype=module.float32),
                {},
            ),
            (
                "empty middle dimension broadcasts",
                module.zeros((1, 0, 2), dtype=module.float32),
                module.ones((3, 0, 2), dtype=module.float32),
                {},
            ),
            (
                "contiguous offset within atol",
                contiguous_offset,
                module.tensor([1.0, 2.000001, 3.0], dtype=module.float32),
                {"rtol": 0.0, "atol": 2.0e-6},
            ),
            (
                "strided equal",
                strided,
                module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=module.float32),
                {},
            ),
            (
                "strided unequal",
                strided,
                module.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0002]], dtype=module.float32),
                {},
            ),
            (
                "offset equal",
                offset,
                module.tensor([20.0, 3.0, 4.00003], dtype=module.float32),
                {},
            ),
            (
                "offset unequal",
                offset,
                module.tensor([20.0, 3.0, 4.0001], dtype=module.float32),
                {},
            ),
            (
                "empty offset",
                strided_empty[1],
                module.zeros((0, 2), dtype=module.float32),
                {},
            ),
            (
                "signed zero",
                module.tensor([0.0, -0.0], dtype=module.float32),
                module.tensor([-0.0, 0.0], dtype=module.float32),
                {},
            ),
            (
                "same infinities",
                module.tensor([float("inf"), -float("inf")], dtype=module.float32),
                module.tensor([float("inf"), -float("inf")], dtype=module.float32),
                {},
            ),
            (
                "opposite infinities",
                module.tensor([float("inf")], dtype=module.float32),
                module.tensor([-float("inf")], dtype=module.float32),
                {},
            ),
            (
                "infinite left remains unequal with infinite tolerance",
                module.tensor([float("inf")], dtype=module.float32),
                module.tensor([0.0], dtype=module.float32),
                {"atol": float("inf")},
            ),
            (
                "finite values match infinite tolerance",
                module.tensor([1.0], dtype=module.float32),
                module.tensor([2.0], dtype=module.float32),
                {"atol": float("inf")},
            ),
            (
                "relative infinite tolerance times zero is not close",
                module.tensor([1.0], dtype=module.float32),
                module.tensor([0.0], dtype=module.float32),
                {"rtol": float("inf"), "atol": 0.0},
            ),
            (
                "nan unequal by default",
                module.tensor([float("nan"), 1.0], dtype=module.float32),
                module.tensor([float("nan"), 1.0], dtype=module.float32),
                {},
            ),
            (
                "nan unequal explicitly",
                module.tensor([float("nan"), 1.0], dtype=module.float32),
                module.tensor([float("nan"), 1.0], dtype=module.float32),
                {"equal_nan": False},
            ),
            (
                "nan equal when requested",
                module.tensor([float("nan"), 1.0], dtype=module.float32),
                module.tensor([float("nan"), 1.0], dtype=module.float32),
                {"equal_nan": True},
            ),
            (
                "single-sided nan stays unequal",
                module.tensor([float("nan")], dtype=module.float32),
                module.tensor([1.0], dtype=module.float32),
                {"equal_nan": True},
            ),
            (
                "subnormal atol boundary",
                module.tensor([0.0], dtype=module.float32),
                self.bits_tensor(module, [0x0000_0001]),
                {"rtol": 0.0, "atol": 1.0e-45},
            ),
            (
                "subnormal below boundary",
                module.tensor([0.0], dtype=module.float32),
                self.bits_tensor(module, [0x0000_0001]),
                {"rtol": 0.0, "atol": 0.0},
            ),
            (
                "nextafter atol boundary",
                module.tensor([1.0], dtype=module.float32),
                module.tensor([next_after_one], dtype=module.float32),
                {"rtol": 0.0, "atol": next_step},
            ),
            (
                "nextafter below boundary",
                module.tensor([1.0], dtype=module.float32),
                module.tensor([next_after_one], dtype=module.float32),
                {"rtol": 0.0, "atol": next_step / 2.0},
            ),
        )

    @staticmethod
    def call_allclose(module, left, right, form, kwargs):
        if form == "positional":
            return module.allclose(left, right, **kwargs)
        if form == "keywords":
            return module.allclose(input=left, other=right, **kwargs)
        if form == "x/x2 aliases":
            return module.allclose(x=left, x2=right, **kwargs)
        if form == "a/other aliases":
            return module.allclose(a=left, other=right, **kwargs)
        if form == "x1/x2 aliases":
            return module.allclose(x1=left, x2=right, **kwargs)
        raise AssertionError(f"unknown allclose call form: {form}")

    def test_supported_results_match_pytorch_2_13(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "positional",
            "keywords",
            "x/x2 aliases",
            "a/other aliases",
            "x1/x2 aliases",
        )
        for actual_case, expected_case in zip(
            actual_cases, expected_cases, strict=True
        ):
            actual_name, actual_left, actual_right, actual_kwargs = actual_case
            expected_name, expected_left, expected_right, expected_kwargs = expected_case
            self.assertEqual(actual_name, expected_name)
            for form in forms:
                with self.subTest(case=actual_name, form=form):
                    self.assert_result_matches(
                        lambda: self.call_allclose(
                            torch, actual_left, actual_right, form, actual_kwargs
                        ),
                        lambda: self.call_allclose(
                            reference_torch,
                            expected_left,
                            expected_right,
                            form,
                            expected_kwargs,
                        ),
                    )

    def test_error_cases_match_pytorch_2_13(self):
        actual = torch.tensor([1.0], dtype=torch.float32)
        expected = reference_torch.tensor([1.0], dtype=reference_torch.float32)
        cases = (
            (
                "missing both",
                lambda module, tensor: module.allclose(),
            ),
            (
                "missing other",
                lambda module, tensor: module.allclose(tensor),
            ),
            (
                "other keyword still missing input",
                lambda module, tensor: module.allclose(other=tensor),
            ),
            (
                "non-tensor input",
                lambda module, tensor: module.allclose(None, tensor),
            ),
            (
                "non-tensor other",
                lambda module, tensor: module.allclose(tensor, None),
            ),
            (
                "too many positional",
                lambda module, tensor: module.allclose(tensor, tensor, 0.0, 0.0, False, False),
            ),
            (
                "bad rtol",
                lambda module, tensor: module.allclose(tensor, tensor, rtol=object()),
            ),
            (
                "bad atol",
                lambda module, tensor: module.allclose(tensor, tensor, 0.0, object()),
            ),
            (
                "bad equal_nan keyword",
                lambda module, tensor: module.allclose(tensor, tensor, equal_nan=1),
            ),
            (
                "bad equal_nan positional",
                lambda module, tensor: module.allclose(tensor, tensor, 0.0, 0.0, 1),
            ),
            (
                "nan rtol",
                lambda module, tensor: module.allclose(tensor, tensor, rtol=float("nan")),
            ),
            (
                "negative rtol",
                lambda module, tensor: module.allclose(tensor, tensor, rtol=-1.0e-5),
            ),
            (
                "unexpected out",
                lambda module, tensor: module.allclose(tensor, tensor, out=None),
            ),
            (
                "duplicate canonical keyword",
                lambda module, tensor: module.allclose(tensor, tensor, 0.0, rtol=0.0),
            ),
            (
                "duplicate alias keyword",
                lambda module, tensor: module.allclose(input=tensor, x=tensor, other=tensor),
            ),
            (
                "unbroadcastable empty shapes",
                lambda module, tensor: module.allclose(
                    module.zeros((2, 0, 3), dtype=module.float32),
                    module.zeros((2, 0, 4), dtype=module.float32),
                ),
            ),
        )
        for name, call in cases:
            with self.subTest(name=name):
                self.assert_error_matches(
                    lambda call=call: call(torch, actual),
                    lambda call=call: call(reference_torch, expected),
                )

    def test_callable_metadata_copy_pickle_and_exports_match_pytorch_2_13(self):
        actual = torch.allclose
        expected = reference_torch.allclose
        self.assertIs(type(actual), type(expected))
        self.assertIs(type(actual), types.BuiltinFunctionType)
        self.assertEqual(actual.__name__, expected.__name__)
        self.assertEqual(actual.__text_signature__, expected.__text_signature__)
        self.assertEqual(actual.__doc__.splitlines()[1], expected.__doc__.splitlines()[1])
        for callable_object in (actual, expected):
            with self.assertRaises(ValueError):
                inspect.signature(callable_object)
        self.assertIs(copy.copy(actual), actual)
        self.assertIs(copy.deepcopy(actual), actual)
        self.assertIs(pickle.loads(pickle.dumps(actual)), actual)
        self.assertIn("allclose", torch.__all__)
        self.assertNotIn("isclose", torch.__all__)

    def test_unsupported_native_boundaries_are_explicit(self):
        self.assertFalse(hasattr(torch, "float64"))
        self.assertTrue(hasattr(reference_torch, "float64"))
        self.assertTrue(
            reference_torch.allclose(
                reference_torch.tensor([1.0], dtype=reference_torch.float64),
                reference_torch.tensor([1.0], dtype=reference_torch.float64),
            )
        )
        with self.assertRaisesRegex(
            TypeError,
            r"^allclose\(\): argument 'input' \(position 1\) must be Tensor, not Tensor$",
        ):
            torch.allclose(
                reference_torch.tensor([1.0], dtype=reference_torch.float64),
                torch.tensor([1.0]),
            )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda")
        with self.assertRaisesRegex(
            TypeError,
            r"^type 'torch_rs\.Tensor' is not an acceptable base type$",
        ):
            type("TensorSubclass", (torch.Tensor,), {})


if __name__ == "__main__":
    unittest.main()
