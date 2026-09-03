from pathlib import Path
import sys
import subprocess
import unittest

import numpy as np
import torch_rs as torch

try:
    import torch as reference_torch
except ImportError:
    reference_torch = None


def _actual_method_relu(x):
    return x.relu()


def _actual_top_level_relu(x):
    return torch.relu(x)


def _expected_method_relu(x):
    return x.relu()


def _expected_top_level_relu(x):
    return reference_torch.relu(x)


@unittest.skipIf(reference_torch is None, "install the reference dependency group")
class TorchCompileReluReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if reference_torch.__version__.split("+")[0] != "2.13.0":
            raise AssertionError(
                "torch.compile relu differentials require pinned PyTorch 2.13.0"
            )

    def assert_tensor_matches(self, actual, expected, *, case):
        with self.subTest(case=case, metadata=True):
            self.assertEqual(actual.shape, tuple(expected.shape))
            self.assertEqual(actual.stride(), expected.stride())
            self.assertEqual(actual.storage_offset(), expected.storage_offset())
            self.assertEqual(actual.is_contiguous(), expected.is_contiguous())
            self.assertEqual(actual.requires_grad, expected.requires_grad)
            self.assertEqual(actual.is_leaf, expected.is_leaf)
            self.assertIs(actual.dtype, torch.float32)
            self.assertEqual(actual.device, torch.device("cpu"))
            self.assertIs(expected.dtype, reference_torch.float32)
            self.assertEqual(expected.device, reference_torch.device("cpu"))

        with self.subTest(case=case, values=True):
            actual_bits = (
                np.asarray(actual, dtype=np.float32).reshape(-1).view(np.uint32)
            )
            expected_bits = expected.detach().cpu().numpy().reshape(-1).view(np.uint32)
            np.testing.assert_array_equal(actual_bits, expected_bits)

    def make_input(self, module, case):
        if case == "scalar":
            return module.tensor(-1.5, dtype=module.float32)
        if case == "empty":
            return module.zeros((2, 0, 3), dtype=module.float32).transpose(0, 2)[1]
        if case == "contiguous":
            return module.tensor(
                [[-3.0, -0.0, 0.0], [2.0, float("inf"), float("-inf")]],
                dtype=module.float32,
            )
        if case == "offset":
            return module.tensor(
                np.arange(-12, 12, dtype=np.float32).reshape(2, 3, 4).tolist(),
                dtype=module.float32,
            )[1]
        if case == "noncontiguous":
            return module.tensor(
                [[-3.0, 1.0, -2.0], [0.0, 4.0, -5.0]],
                dtype=module.float32,
            ).transpose(0, 1)
        raise AssertionError(f"unknown case {case!r}")

    def compiled_pair(self, actual_model, expected_model, compile_form):
        if compile_form == "direct":
            actual = torch.compile(actual_model, backend="eager")
            expected = reference_torch.compile(expected_model, backend="eager")
        elif compile_form == "decorator":
            actual = torch.compile(backend="eager")(actual_model)
            expected = reference_torch.compile(backend="eager")(expected_model)
        else:
            raise AssertionError(f"unknown compile form {compile_form!r}")
        self.assertEqual(actual._torch_rs_compile_graph, "relu")
        self.assertEqual(actual._torch_rs_compile_execution, "torch_rs")
        return actual, expected

    def test_direct_and_decorator_forms_match_pytorch_2_13(self):
        models = (
            ("method", _actual_method_relu, _expected_method_relu),
            ("top_level", _actual_top_level_relu, _expected_top_level_relu),
        )
        cases = ("scalar", "empty", "contiguous", "offset", "noncontiguous")

        for compile_form in ("direct", "decorator"):
            for model_form, actual_model, expected_model in models:
                actual_compiled, expected_compiled = self.compiled_pair(
                    actual_model,
                    expected_model,
                    compile_form,
                )
                for case in cases:
                    with self.subTest(
                        compile_form=compile_form,
                        model_form=model_form,
                        case=case,
                    ):
                        actual_input = self.make_input(torch, case)
                        expected_input = self.make_input(reference_torch, case)
                        actual_output = actual_compiled(actual_input)
                        expected_output = expected_compiled(expected_input)
                        self.assert_tensor_matches(
                            actual_output,
                            expected_output,
                            case=(compile_form, model_form, case),
                        )

    def test_capture_does_not_import_pytorch_or_execute_on_real_inputs(self):
        probe = Path(__file__).with_name("compile_no_pytorch_import_probe.py")
        completed = subprocess.run(
            [sys.executable, probe],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
