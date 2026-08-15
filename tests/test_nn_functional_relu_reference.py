import inspect
import types
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
class FunctionalReluReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "nn.functional.relu differentials require pinned PyTorch 2.13.0"
            )

    def make_case(self, module, case, *, requires_grad):
        if case == "scalar":
            leaf = module.tensor(
                2.0, dtype=module.float32, requires_grad=requires_grad
            )
            return leaf, leaf
        if case == "empty":
            leaf = module.zeros(
                (2, 0, 3), dtype=module.float32, requires_grad=requires_grad
            )
            return leaf, leaf.transpose(0, 2)[1]

        values = np.asarray(
            [9.0] * 12
            + [
                -1.0,
                2.0,
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                0.5,
                3.0,
                -4.0,
                5.0,
                -6.0,
                7.0,
            ],
            dtype=np.float32,
        ).reshape(2, 3, 4)
        leaf = module.tensor(
            values.tolist(), dtype=module.float32, requires_grad=requires_grad
        )
        offset = leaf[1]
        if case == "offset":
            return leaf, offset
        return leaf, offset.transpose(0, 1)

    def assert_metadata_matches(self, actual, expected, *, case):
        with self.subTest(case=case):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))

    def assert_values_match(self, actual, expected, *, case):
        with self.subTest(case=case):
            np.testing.assert_array_equal(
                np.asarray(actual.detach()).reshape(-1).view(np.uint32),
                expected.detach().cpu().numpy().reshape(-1).view(np.uint32),
            )

    def test_imports_signature_and_documentation_match_pytorch_2_13(self):
        self.assertIs(torch.nn.functional, functional)
        self.assertIs(type(functional.relu), types.FunctionType)
        self.assertIs(type(reference_functional.relu), types.FunctionType)
        self.assertEqual(functional.__doc__, reference_functional.__doc__)
        self.assertEqual(functional.relu.__name__, reference_functional.relu.__name__)
        self.assertEqual(
            functional.relu.__qualname__, reference_functional.relu.__qualname__
        )
        self.assertEqual(functional.relu.__doc__, reference_functional.relu.__doc__)
        self.assertEqual(
            functional.relu.__defaults__, reference_functional.relu.__defaults__
        )
        self.assertEqual(
            functional.relu.__kwdefaults__, reference_functional.relu.__kwdefaults__
        )

        actual_signature = inspect.signature(functional.relu)
        expected_signature = inspect.signature(reference_functional.relu)
        actual_parameters = tuple(actual_signature.parameters.values())
        expected_parameters = tuple(expected_signature.parameters.values())
        for actual, expected in zip(
            actual_parameters, expected_parameters, strict=True
        ):
            self.assertEqual(actual.name, expected.name)
            self.assertEqual(actual.kind, expected.kind)
            self.assertEqual(actual.default, expected.default)
        self.assertIs(actual_parameters[0].annotation, torch.Tensor)
        self.assertIs(expected_parameters[0].annotation, reference_torch.Tensor)
        self.assertIs(actual_parameters[1].annotation, bool)
        self.assertIs(expected_parameters[1].annotation, bool)
        self.assertIs(actual_signature.return_annotation, torch.Tensor)
        self.assertIs(
            expected_signature.return_annotation, reference_torch.Tensor
        )

    def test_values_layouts_and_out_of_place_calls_match_pytorch_2_13(self):
        for case in ("scalar", "empty", "offset", "strided"):
            _, actual_input = self.make_case(torch, case, requires_grad=False)
            _, expected_input = self.make_case(
                reference_torch, case, requires_grad=False
            )
            self.assert_metadata_matches(actual_input, expected_input, case=(case, "input"))

            calls = (
                (
                    lambda: functional.relu(actual_input),
                    lambda: reference_functional.relu(expected_input),
                ),
                (
                    lambda: functional.relu(actual_input, False),
                    lambda: reference_functional.relu(expected_input, False),
                ),
                (
                    lambda: functional.relu(input=actual_input, inplace=False),
                    lambda: reference_functional.relu(
                        input=expected_input, inplace=False
                    ),
                ),
            )
            for form, (actual_call, expected_call) in enumerate(calls):
                actual = actual_call()
                expected = expected_call()
                invocation = (case, form)
                self.assert_metadata_matches(actual, expected, case=invocation)
                self.assert_values_match(actual, expected, case=invocation)
                with self.subTest(case=invocation):
                    self.assertFalse(actual.is_set_to(actual_input))
                    self.assertFalse(expected.is_set_to(expected_input))

    def test_autograd_matches_for_every_layout_case(self):
        for case in ("scalar", "empty", "offset", "strided"):
            actual_leaf, actual_input = self.make_case(
                torch, case, requires_grad=True
            )
            expected_leaf, expected_input = self.make_case(
                reference_torch, case, requires_grad=True
            )
            actual_output = functional.relu(actual_input, inplace=False)
            expected_output = reference_functional.relu(
                expected_input, inplace=False
            )
            self.assert_metadata_matches(
                actual_output, expected_output, case=(case, "output")
            )
            self.assert_values_match(
                actual_output, expected_output, case=(case, "output")
            )

            actual_output.sum().backward()
            expected_output.sum().backward()
            self.assert_metadata_matches(
                actual_leaf.grad, expected_leaf.grad, case=(case, "gradient")
            )
            self.assert_values_match(
                actual_leaf.grad, expected_leaf.grad, case=(case, "gradient")
            )

    def test_no_grad_matches_for_every_layout_case(self):
        for case in ("scalar", "empty", "offset", "strided"):
            actual_leaf, actual_input = self.make_case(
                torch, case, requires_grad=True
            )
            expected_leaf, expected_input = self.make_case(
                reference_torch, case, requires_grad=True
            )
            with torch.no_grad():
                actual_output = functional.relu(actual_input, inplace=False)
            with reference_torch.no_grad():
                expected_output = reference_functional.relu(
                    expected_input, inplace=False
                )
            self.assert_metadata_matches(
                actual_output, expected_output, case=(case, "no_grad")
            )
            self.assert_values_match(
                actual_output, expected_output, case=(case, "no_grad")
            )
            with self.subTest(case=case):
                self.assertIsNone(actual_leaf.grad)
                self.assertIsNone(expected_leaf.grad)

    def test_inplace_true_is_explicitly_unsupported_and_non_mutating(self):
        leaf = torch.tensor(
            [[9.0, 9.0, 9.0], [-1.0, 2.0, -0.0]], requires_grad=True
        )
        source = leaf[1]
        bits_before = np.asarray(leaf.detach()).copy().view(np.uint32)
        metadata_before = (
            source.shape,
            source.stride(),
            source.storage_offset(),
            source.data_ptr(),
            source.requires_grad,
            source.is_leaf,
        )

        with self.assertRaisesRegex(
            NotImplementedError,
            "^torch_rs\\.nn\\.functional\\.relu does not support inplace=True$",
        ):
            functional.relu(source, inplace=True)

        np.testing.assert_array_equal(
            np.asarray(leaf.detach()).view(np.uint32), bits_before
        )
        self.assertEqual(
            (
                source.shape,
                source.stride(),
                source.storage_offset(),
                source.data_ptr(),
                source.requires_grad,
                source.is_leaf,
            ),
            metadata_before,
        )
        self.assertIsNone(leaf.grad)


if __name__ == "__main__":
    unittest.main()
