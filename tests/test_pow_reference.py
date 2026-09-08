import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TensorPowReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError("pow differentials require pinned PyTorch 2.13.0")

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(tuple(actual.shape), tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(str(actual.dtype), str(expected.dtype))
            self.assertEqual(str(actual.device), str(expected.device))
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
        with self.subTest(case=case, values=True):
            np.testing.assert_array_equal(
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    @staticmethod
    def tensor_cases(module):
        base = module.tensor(
            np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4).tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3F80_0000,
                0xBF80_0000,
                0x7F7F_FFFF,
                0xFF7F_FFFF,
                0x7F80_0000,
                0xFF80_0000,
                0x7F81_2345,
                0xFF81_2345,
                0x7FC1_2345,
                0xFFC5_4321,
            ),
            dtype=np.uint32,
        )
        return (
            ("scalar", module.tensor(-0.0, dtype=module.float32)),
            (
                "empty",
                module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1],
            ),
            ("offset", strided[1]),
            ("noncontiguous", strided),
            (
                "signed zero and non-finites",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

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
        raise AssertionError(f"unknown pow autograd case: {case}")

    @staticmethod
    def supported_calls(module, source):
        return (
            ("method int", lambda: source.pow(2)),
            ("method float", lambda: source.pow(2.0)),
            ("method keyword", lambda: source.pow(exponent=2)),
            ("dunder", lambda: source.__pow__(2)),
            ("operator", lambda: source**2),
            ("top positional int", lambda: module.pow(source, 2)),
            ("top positional float", lambda: module.pow(source, 2.0)),
            ("top keyword exponent", lambda: module.pow(source, exponent=2)),
            ("top all keywords", lambda: module.pow(input=source, exponent=2)),
            ("top out none", lambda: module.pow(source, 2, out=None)),
        )

    def test_scalar_empty_offset_noncontiguous_and_ieee_values_match_pytorch_2_13(
        self,
    ):
        actual_cases = self.tensor_cases(torch)
        expected_cases = self.tensor_cases(reference_torch)
        for (case, actual), (_, expected) in zip(
            actual_cases, expected_cases, strict=True
        ):
            for (form, actual_call), (_, expected_call) in zip(
                self.supported_calls(torch, actual),
                self.supported_calls(reference_torch, expected),
                strict=True,
            ):
                actual_output = actual_call()
                expected_output = expected_call()
                self.assert_tensor_matches(
                    actual_output, expected_output, case=(case, form)
                )
                self.assertFalse(actual_output.is_set_to(actual))
                self.assertFalse(expected_output.is_set_to(expected))

    def test_backward_through_sum_matches_pytorch_2_13(self):
        forms = tuple(form for form, _ in self.supported_calls(torch, torch.tensor(1.0)))
        for case in ("scalar", "empty", "offset", "noncontiguous"):
            for form in forms:
                actual_leaf, actual_input = self.autograd_case(torch, case)
                expected_leaf, expected_input = self.autograd_case(reference_torch, case)
                actual_output = dict(self.supported_calls(torch, actual_input))[form]()
                expected_output = dict(
                    self.supported_calls(reference_torch, expected_input)
                )[form]()

                self.assert_tensor_matches(
                    actual_output, expected_output, case=(case, form, "forward")
                )
                actual_output.sum().backward()
                expected_output.sum().backward()
                self.assert_tensor_matches(
                    actual_leaf.grad,
                    expected_leaf.grad,
                    case=(case, form, "gradient"),
                )


if __name__ == "__main__":
    unittest.main()
