import unittest

import numpy as np
import torch_rs as torch
import torch_rs.nn.functional as functional

try:
    import torch as reference_torch
    import torch.nn.functional as reference_functional
except ImportError:
    reference_torch = None
    reference_functional = None


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class SoftsignAliasReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "softsign alias differentials require pinned PyTorch 2.13.0"
            )

    @staticmethod
    def tensor_values(tensor):
        if type(tensor) is torch.Tensor:
            return np.asarray(tensor, dtype=np.float32)
        return tensor.detach().cpu().numpy()

    @classmethod
    def tensor_bits(cls, tensor):
        return cls.tensor_values(tensor).reshape(-1).view(np.uint32)

    @staticmethod
    def make_cases(module):
        base = module.tensor(
            np.linspace(-3.0, 3.0, 24, dtype=np.float32)
            .reshape(2, 3, 4)
            .tolist(),
            dtype=module.float32,
        )
        strided = base.transpose(0, 2)
        rank_two = module.tensor(
            [[-4.0, -0.0, 0.5], [1.0, 2.0, 8.0]],
            dtype=module.float32,
        )
        special_bits = np.asarray(
            (
                0x0000_0000,
                0x8000_0000,
                0x0000_0001,
                0x8000_0001,
                0x0080_0000,
                0x8080_0000,
                0x3EAA_AAAB,
                0xBEAA_AAAB,
                0x3F80_0000,
                0xBF80_0000,
                0x4000_0000,
                0xC000_0000,
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
                module.zeros((2, 0, 3), dtype=module.float32)
                .transpose(0, 2)[1],
            ),
            ("contiguous", base),
            ("offset", base[1]),
            ("noncontiguous", strided[1]),
            ("rank2", rank_two),
            (
                "numerical_edges",
                module.tensor(memoryview(special_bits.view(np.float32))),
            ),
        )

    @staticmethod
    def call_top_level(module, tensor, form):
        if form == "positional":
            return module.softsign(tensor)
        if form == "out none":
            return module.softsign(tensor, out=None)
        if form == "alias and out none":
            return module.softsign(x=tensor, out=None)
        return module.softsign(**{form: tensor})

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
        np.testing.assert_array_equal(self.tensor_bits(actual), self.tensor_bits(expected))

    def test_alias_values_metadata_and_storage_match_functional_softsign(self):
        actual_cases = self.make_cases(torch)
        expected_cases = self.make_cases(reference_torch)
        forms = (
            "method",
            "positional",
            "input",
            "x",
            "a",
            "x1",
            "out none",
            "alias and out none",
        )
        for (case, actual_input), (expected_case, expected_input) in zip(
            actual_cases, expected_cases, strict=True
        ):
            self.assertEqual(case, expected_case)
            for form in forms:
                if form == "method":
                    actual = actual_input.softsign()
                else:
                    actual = self.call_top_level(torch, actual_input, form)
                expected = reference_functional.softsign(expected_input)
                self.assert_tensor_matches(actual, expected, case=(case, form))
                self.assertFalse(actual.is_set_to(actual_input))
                self.assertFalse(expected.is_set_to(expected_input))
                if actual_input.numel():
                    self.assertNotEqual(actual.data_ptr(), actual_input.data_ptr())
                    self.assertNotEqual(expected.data_ptr(), expected_input.data_ptr())

    def test_tracked_inputs_match_functional_softsign_inside_no_grad_and_after_detach(self):
        actual_leaf = torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]], requires_grad=True
        )
        expected_leaf = reference_torch.tensor(
            [[-2.0, -0.0, 1.0], [2.0, 4.0, 8.0]],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        actual_input = actual_leaf.transpose(0, 1)[1]
        expected_input = expected_leaf.transpose(0, 1)[1]

        with torch.no_grad():
            actual_method = actual_input.softsign()
            actual_top_level = torch.softsign(actual_input, out=None)
        with reference_torch.no_grad():
            expected = reference_functional.softsign(expected_input)
        self.assert_tensor_matches(actual_method, expected, case="method no_grad")
        self.assert_tensor_matches(actual_top_level, expected, case="top level no_grad")

        actual_detached = actual_input.detach()
        expected_detached = expected_input.detach()
        self.assert_tensor_matches(
            actual_detached.softsign(),
            reference_functional.softsign(expected_detached),
            case="method detached",
        )
        self.assert_tensor_matches(
            torch.softsign(actual_detached),
            reference_functional.softsign(expected_detached),
            case="top level detached",
        )
        self.assertIsNone(actual_leaf.grad)
        self.assertIsNone(expected_leaf.grad)

    def test_boundaries_are_explicit_against_reference_functional_capabilities(self):
        actual = torch.tensor([0.5, -0.5], requires_grad=True)
        expected = reference_torch.tensor(
            [0.5, -0.5],
            dtype=reference_torch.float32,
            requires_grad=True,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            r"^softsign\(\): autograd recording is not supported$",
        ):
            actual.softsign()
        self.assertTrue(reference_functional.softsign(expected).requires_grad)

        destination = torch.tensor([17.0, 19.0])
        with self.assertRaisesRegex(
            RuntimeError,
            r"^softsign\(\): the 'out' argument is not supported$",
        ):
            torch.softsign(actual, out=destination)
        self.assertEqual(destination.tolist(), [17.0, 19.0])

        expected64 = reference_torch.tensor(
            [0.5], dtype=reference_torch.float64
        )
        self.assertEqual(
            str(reference_functional.softsign(expected64).dtype),
            "torch.float64",
        )
        self.assertFalse(hasattr(torch, "float64"))
        with self.assertRaisesRegex(
            TypeError,
            r"^tensor\(\): argument 'dtype' must be torch\.dtype, not object$",
        ):
            torch.tensor([1.0], dtype=object()).softsign()

        with self.assertRaisesRegex(
            RuntimeError,
            r"^tensor\(\): device 'cuda' is not supported; only 'cpu' is implemented$",
        ):
            torch.tensor([1.0], device="cuda").softsign()
        if reference_torch.cuda.is_available():
            expected_cuda = reference_torch.tensor(
                [0.5],
                device="cuda:0",
                dtype=reference_torch.float32,
            )
            self.assertEqual(
                reference_functional.softsign(expected_cuda).device.type,
                "cuda",
            )

        self.assertFalse(hasattr(reference_torch, "softsign"))
        self.assertFalse(hasattr(reference_torch.Tensor, "softsign"))


if __name__ == "__main__":
    unittest.main()
