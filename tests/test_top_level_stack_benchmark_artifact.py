import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_top_level_stack.py"
VALIDATOR_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "validate_top_level_stack_benchmark.py"
)

spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_stack_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_top_level_stack = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_top_level_stack
spec.loader.exec_module(benchmark_top_level_stack)

validator_spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_stack_validator_for_tests",
    VALIDATOR_SCRIPT,
)
validate_top_level_stack_benchmark = importlib.util.module_from_spec(validator_spec)
assert validator_spec.loader is not None
sys.modules[validator_spec.name] = validate_top_level_stack_benchmark
validator_spec.loader.exec_module(validate_top_level_stack_benchmark)


def _has_reference_torch_2_13():
    try:
        import torch as reference_torch
    except ImportError:
        return False
    return reference_torch.__version__.split("+", 1)[0] == "2.13.0"


class TopLevelStackBenchmarkArtifactTests(unittest.TestCase):
    def test_checked_in_raw_artifact_matches_markdown_summary(self):
        benchmark_top_level_stack.validate_artifact(
            benchmark_top_level_stack.DEFAULT_ARTIFACT_PATH,
            benchmark_top_level_stack.DEFAULT_MARKDOWN_REPORT_PATH,
        )

    def test_workload_matrix_covers_stack_policy_categories(self):
        categories = {workload.category for workload in benchmark_top_level_stack.WORKLOADS}

        self.assertEqual(
            categories,
            {
                "scalar",
                "vector",
                "matrix",
                "empty",
                "offset",
                "noncontiguous",
                "autograd forward",
                "autograd forward+backward",
            },
        )
        self.assertEqual(len(benchmark_top_level_stack.WORKLOADS), 8)

    def test_boundary_rows_cover_stack_policy_cases(self):
        self.assertEqual(
            {
                cell.name
                for cell in benchmark_top_level_stack.UNSUPPORTED_CELLS
                if cell.credit == benchmark_top_level_stack.CREDIT_ZERO
            },
            {
                "mixed_metadata",
                "concrete_out",
            },
        )
        self.assertEqual(
            {
                cell.name
                for cell in benchmark_top_level_stack.UNSUPPORTED_CELLS
                if cell.credit == benchmark_top_level_stack.CREDIT_ERROR_PARITY
            },
            {
                "empty_input_sequence",
                "mixed_shapes",
            },
        )

    def test_generated_validator_cases_are_held_out_and_deterministic(self):
        first = validate_top_level_stack_benchmark.generate_cases(
            seed=20260908,
            cases_per_category=1,
            max_elements=4096,
        )
        second = validate_top_level_stack_benchmark.generate_cases(
            seed=20260908,
            cases_per_category=1,
            max_elements=4096,
        )
        self.assertEqual(first, second)

        self.assertEqual(
            {case.category for case in first},
            set(validate_top_level_stack_benchmark.REQUIRED_CATEGORIES),
        )
        self.assertEqual(len({case.name for case in first}), len(first))
        for case in first:
            with self.subTest(case=case.name):
                self.assertNotIn(
                    case.shape,
                    validate_top_level_stack_benchmark.PUBLIC_INPUT_SHAPES,
                )
                self.assertLessEqual(
                    validate_top_level_stack_benchmark._product(case.shape),
                    4096,
                )
                self.assertEqual(
                    len(case.input_source_indices),
                    3
                    if case.category == "autograd forward+backward"
                    else len(case.seeds),
                )

        workloads = validate_top_level_stack_benchmark._workloads_for_cases(first)
        self.assertEqual(len(workloads), len(first))
        self.assertEqual(
            {workload.name for workload in workloads},
            {case.name for case in first},
        )
        self.assertTrue(all(callable(workload.make_operands) for workload in workloads))

    @unittest.skipUnless(
        _has_reference_torch_2_13(),
        "requires pinned PyTorch 2.13 reference dependency",
    )
    def test_generated_validator_smoke_artifact_validates(self):
        target_dir = REPOSITORY_ROOT / "target"
        target_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="stack-validator-",
            dir=target_dir,
        ) as temporary_directory:
            artifact_path = Path(temporary_directory) / "stack-validator.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--seed",
                    "20260908",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "0",
                    "--samples",
                    "1",
                    "--output",
                    str(artifact_path),
                ],
                check=False,
                capture_output=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                text=True,
                timeout=120,
            )
            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stdout + completed.stderr,
            )

            report = json.loads(artifact_path.read_text(encoding="utf-8"))
            validate_top_level_stack_benchmark.validate_artifact_dict(report)
            self.assertEqual(report["validator"]["seed"], 20260908)
            self.assertEqual(report["validator"]["cases_per_category"], 1)
            self.assertEqual(
                report["environment"]["benchmark_integrity"]["workload_set"],
                validate_top_level_stack_benchmark.WORKLOAD_SET,
            )
            self.assertEqual(report["aggregates"]["timed_supported_cell_count"], 6)
            self.assertEqual(
                report["aggregates"]["generated_category_counts"],
                {
                    category: 1
                    for category in validate_top_level_stack_benchmark.REQUIRED_CATEGORIES
                },
            )
            for case in report["cases"]:
                with self.subTest(case=case["workload"]):
                    self.assertTrue(case["generated"])
                    self.assertTrue(case["validation"]["held_out_generated_shape"])
                    self.assertTrue(
                        case["validation"]["same_shape_cpu_float32_inputs"]
                    )
                    self.assertNotIn(
                        tuple(case["shape"]),
                        validate_top_level_stack_benchmark.PUBLIC_INPUT_SHAPES,
                    )

            validation_completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(artifact_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                validation_completed.returncode,
                0,
                msg=validation_completed.stdout + validation_completed.stderr,
            )


if __name__ == "__main__":
    unittest.main()
