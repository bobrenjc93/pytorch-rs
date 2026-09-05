import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import torch_rs as torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_compile_cpu.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_compile_cpu_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_compile_cpu = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_compile_cpu
spec.loader.exec_module(benchmark_compile_cpu)


class CompileBenchmarkArtifactTests(unittest.TestCase):
    def test_checked_in_raw_artifact_matches_markdown_summary(self):
        benchmark_compile_cpu.validate_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
            benchmark_compile_cpu.DEFAULT_MARKDOWN_REPORT_PATH,
        )

    def test_rendered_summary_counts_inference_and_training_as_supported(self):
        report = benchmark_compile_cpu._load_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
        )
        summary = benchmark_compile_cpu.render_markdown_summary(report)

        self.assertIn("7 inference", summary)
        self.assertIn("7 training-autograd", summary)
        self.assertIn("7 python-control-flow", summary)
        self.assertIn("7 decomposition", summary)
        self.assertIn("7 mutation_aliasing_views", summary)
        self.assertIn("7 dtype-device-transitions", summary)
        self.assertIn(
            "| `inference` | 6 | Supported and timed public cases: "
            "`cpu_float32_inference_relu_no_grad` |",
            summary,
        )
        self.assertIn(
            "| `training_autograd` | 8 | Supported and timed public cases: "
            "`cpu_float32_training_unary_neg_abs_add` |",
            summary,
        )
        self.assertIn(
            "| `python_control_flow` | 8 | Supported and timed public cases: "
            "`cpu_float32_requires_grad_branch_unary` |",
            summary,
        )
        self.assertIn(
            "| `mutation_aliasing_views` | 8 | Supported and timed public cases: "
            "`cpu_float32_detach_alias_view` |",
            summary,
        )
        self.assertIn(
            "| `decompositions` | 6 | Supported and timed public cases: "
            "`cpu_float32_decomposition_square_scalar` |",
            summary,
        )
        self.assertIn(
            "| `dtype_device_transitions` | 4 | Supported and timed public cases: "
            "`cpu_float32_float_identity_view` |",
            summary,
        )
        self.assertNotIn("`training_autograd` | 8 | Zero credit", summary)
        self.assertNotIn("`python_control_flow` | 8 | Zero credit", summary)
        self.assertNotIn("`mutation_aliasing_views` | 8 | Zero credit", summary)
        self.assertNotIn("`decompositions` | 6 | Zero credit", summary)
        self.assertNotIn("`dtype_device_transitions` | 4 | Zero credit", summary)

    def test_raw_artifact_records_backward_validation_cases(self):
        report = benchmark_compile_cpu._load_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
        )
        cases = {
            case["name"]: case
            for case in (
                *report["corpus"]["public_cases"],
                *report["corpus"]["held_out_cases"],
            )
        }

        self.assertIs(
            cases["cpu_float32_training_unary_neg_abs_add"][
                "backward_through_sum"
            ],
            True,
        )
        self.assertIs(
            cases["cpu_float32_heldout_training_broadcast_neg_abs_add"][
                "backward_through_sum"
            ],
            True,
        )
        self.assertIs(
            cases["cpu_float32_inference_relu_no_grad"]["backward_through_sum"],
            False,
        )

    def test_backward_through_sum_check_rejects_gradient_mismatch(self):
        actual_input = torch.tensor([1.0, -2.0], requires_grad=True)
        expected_input = torch.tensor([1.0, -2.0], requires_grad=True)
        actual = actual_input.neg()
        expected = expected_input.abs()

        with self.assertRaisesRegex(
            AssertionError,
            "leaf gradients after backward-through-sum mismatch",
        ):
            benchmark_compile_cpu._assert_backward_through_sum_matches(
                actual,
                (actual_input,),
                expected,
                (expected_input,),
                cell_name="gradient_probe",
            )

    def test_backward_through_sum_check_rejects_input_mutation(self):
        class FakeTensor:
            shape = (1,)
            dtype = "torch.float32"
            device = "cpu"

            def __init__(self, values, *, requires_grad=True):
                self.values = list(values)
                self.requires_grad = requires_grad
                self.grad = None

            def stride(self):
                return (1,)

            def storage_offset(self):
                return 0

            def is_contiguous(self):
                return True

            def retain_grad(self):
                pass

            def tolist(self):
                return list(self.values)

        class FakeOutput:
            requires_grad = True

            def __init__(self, input, *, mutate_input=False):
                self.input = input
                self.mutate_input = mutate_input

            def sum(self):
                return self

            def backward(self):
                self.input.grad = FakeTensor([1.0], requires_grad=False)
                if self.mutate_input:
                    self.input.values[0] = 99.0

        actual_input = FakeTensor([1.0])
        expected_input = FakeTensor([1.0])

        with self.assertRaisesRegex(
            AssertionError,
            "inputs after backward-through-sum mismatch",
        ):
            benchmark_compile_cpu._assert_backward_through_sum_matches(
                FakeOutput(actual_input, mutate_input=True),
                (actual_input,),
                FakeOutput(expected_input),
                (expected_input,),
                cell_name="mutation_probe",
            )

    def test_output_pytree_payloads_include_tensor_observables(self):
        actual = (
            torch.tensor([1.0, -2.0], dtype=torch.float32),
            [torch.tensor([[3.0]], dtype=torch.float32, requires_grad=True)],
        )
        expected = (
            torch.tensor([1.0, -2.0], dtype=torch.float32),
            [torch.tensor([[3.0]], dtype=torch.float32, requires_grad=True)],
        )

        benchmark_compile_cpu._assert_outputs_match(
            actual,
            expected,
            cell_name="pytree_output",
        )
        payload = benchmark_compile_cpu._materialized_output_payload(actual)
        metadata = benchmark_compile_cpu._output_metadata(actual)

        self.assertEqual(payload["container"], "tuple")
        self.assertEqual(payload["elements"][1]["container"], "list")
        self.assertEqual(metadata["container"], "tuple")
        self.assertTrue(benchmark_compile_cpu._metadata_requires_grad(metadata))
        self.assertIn(
            "tuple[",
            benchmark_compile_cpu._format_output_metadata(metadata),
        )
        self.assertEqual(
            benchmark_compile_cpu._checksum_tensor(actual),
            benchmark_compile_cpu._checksum_tensor(expected),
        )

    def test_validator_rejects_previous_corpus_version_artifact(self):
        report = benchmark_compile_cpu._load_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
        )
        stale = copy.deepcopy(report)
        stale["environment"]["corpus_version"] = "torch_compile_corpus_v7"

        with self.assertRaisesRegex(
            AssertionError,
            "environment corpus version mismatch",
        ):
            benchmark_compile_cpu._validate_expected_artifact_shape(stale)

    def test_validator_rejects_stale_pre_inference_artifact_shape(self):
        report = benchmark_compile_cpu._load_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
        )
        stale = copy.deepcopy(report)
        stale["cases"] = [
            row for row in stale["cases"] if row.get("category") != "inference"
        ]

        with self.assertRaisesRegex(AssertionError, "timed cell count mismatch"):
            benchmark_compile_cpu._validate_expected_artifact_shape(stale)

    def test_validator_rejects_training_artifact_without_grad_enabled_cell(self):
        report = benchmark_compile_cpu._load_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
        )
        stale = copy.deepcopy(report)
        for row in stale["cases"]:
            if row.get("case") == "cpu_float32_training_unary_neg_abs_add":
                row["output_metadata"]["requires_grad"] = False

        with self.assertRaisesRegex(
            AssertionError,
            "missing grad-enabled training-autograd timed cells",
        ):
            benchmark_compile_cpu._validate_expected_artifact_shape(stale)


if __name__ == "__main__":
    unittest.main()
