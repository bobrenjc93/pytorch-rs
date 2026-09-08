import copy
from collections import Counter
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
                    "1",
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
            validate_top_level_stack_benchmark.validate_artifact_dict(
                report,
                expected_seed=20260908,
                expected_cases_per_category=1,
                expected_max_elements=4096,
                expected_warmups=1,
                expected_samples=1,
                expected_threads=1,
                require_clean_git=False,
            )
            current_git = benchmark_top_level_stack._git_provenance()
            report = copy.deepcopy(report)
            report["environment"]["git"] = {
                "head": current_git["head"],
                "status_short": "",
                "diff_stat": "",
            }
            artifact_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validate_top_level_stack_benchmark.validate_artifact_dict(
                report,
                expected_seed=20260908,
                expected_cases_per_category=1,
                expected_max_elements=4096,
                expected_warmups=1,
                expected_samples=1,
                expected_threads=1,
            )
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
            self.assertEqual(
                set(report["aggregates"]["groups"]),
                {
                    "all supported cells",
                    *{
                        f"{category} cells"
                        for category in validate_top_level_stack_benchmark.REQUIRED_CATEGORIES
                    },
                },
            )
            self.assertEqual(
                report["aggregates"]["groups"]["contiguous cells"]["cell_count"],
                1,
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

            def rebuild_aggregates_from_rows(artifact):
                supported = artifact["cases"]
                unsupported = artifact["zero_credit_unsupported_cells"]
                error_parity = artifact["boundary_error_parity_cells"]
                aggregates = (
                    validate_top_level_stack_benchmark._aggregate_generated_rows(
                        supported
                    )
                )
                aggregates["zero_credit_unsupported_cell_count"] = len(unsupported)
                aggregates["boundary_error_parity_cell_count"] = len(error_parity)
                aggregates["combined_capped_with_zero_credit_unsupported"] = (
                    benchmark_top_level_stack._geomean(
                        [
                            min(
                                10.0,
                                max(
                                    0.10,
                                    row["ratios"][
                                        "steady_torch_rs_over_pytorch"
                                    ],
                                ),
                            )
                            for row in supported
                        ]
                        + [10.0] * len(unsupported)
                    )
                )
                aggregates["generated_category_counts"] = dict(
                    sorted(Counter(row["category"] for row in supported).items())
                )
                artifact["aggregates"] = aggregates

            def tamper_ratio_and_rebuild_aggregates(artifact):
                artifact["cases"][0]["ratios"][
                    "steady_torch_rs_over_pytorch"
                ] = 123.0
                rebuild_aggregates_from_rows(artifact)

            def strip_measured_checksums(artifact):
                for row in artifact["cases"]:
                    for implementation in ("torch_rs", "pytorch"):
                        for pass_result in row["implementations"][implementation][
                            "passes"
                        ]:
                            pass_result["steady_checksums"] = []
                            pass_result["warmup_checksums"] = []

            def forge_output_checksums(artifact):
                fake_checksum = "fabricated-checksum"

                def checksum_sink(count):
                    sink = "0"
                    for _ in range(count):
                        sink = benchmark_top_level_stack._roll_checksum(
                            sink,
                            fake_checksum,
                        )
                    return sink

                steady_sink = checksum_sink(artifact["environment"]["samples"])
                warmup_sink = checksum_sink(artifact["environment"]["warmups"])
                for row in artifact["cases"]:
                    row["validation"]["reference_checksum"] = fake_checksum
                    for implementation in ("torch_rs", "pytorch"):
                        implementation_result = row["implementations"][implementation]
                        implementation_result["checksums"] = [fake_checksum]
                        for pass_result in implementation_result["passes"]:
                            pass_result["cold_checksum"] = fake_checksum
                            pass_result["steady_checksums"] = [fake_checksum]
                            pass_result["steady_checksum_sink"] = steady_sink
                            pass_result["warmup_checksums"] = [fake_checksum]
                            pass_result["warmup_checksum_sink"] = warmup_sink

            tamper_cases = (
                (
                    "seed",
                    lambda artifact: (
                        artifact["validator"].__setitem__("seed", 1),
                        artifact["environment"]["validator"].__setitem__("seed", 1),
                    ),
                    "validator generated cases do not match seed/config",
                ),
                (
                    "aggregate",
                    lambda artifact: artifact["aggregates"].__setitem__(
                        "steady_geomean_torch_rs_over_pytorch",
                        123.0,
                    ),
                    "aggregates.steady_geomean_torch_rs_over_pytorch mismatch",
                ),
                (
                    "pass-summary",
                    lambda artifact: artifact["cases"][0]["implementations"][
                        "torch_rs"
                    ]["passes"][0]["steady"].__setitem__(
                        "median_us",
                        123.0,
                    ),
                    "steady.median_us mismatch",
                ),
                (
                    "implementation-median",
                    lambda artifact: artifact["cases"][0]["implementations"][
                        "torch_rs"
                    ].__setitem__(
                        "steady_median_us",
                        123.0,
                    ),
                    "torch_rs.steady_median_us mismatch",
                ),
                (
                    "ratio-with-rebuilt-aggregates",
                    tamper_ratio_and_rebuild_aggregates,
                    "ratios.steady_torch_rs_over_pytorch mismatch",
                ),
                (
                    "row-shape",
                    lambda artifact: artifact["cases"][0].__setitem__(
                        "shape",
                        [999],
                    ),
                    "shape mismatch",
                ),
                (
                    "row-layout",
                    lambda artifact: artifact["cases"][0].__setitem__(
                        "layout",
                        "fabricated",
                    ),
                    "layout mismatch",
                ),
                (
                    "measured-checksums-stripped",
                    strip_measured_checksums,
                    "steady_checksums mismatch",
                ),
                (
                    "output-checksums-forged",
                    forge_output_checksums,
                    "reference checksum mismatch",
                ),
                (
                    "steady-checksum-sink",
                    lambda artifact: artifact["cases"][0]["implementations"][
                        "torch_rs"
                    ]["passes"][0].__setitem__(
                        "steady_checksum_sink",
                        "0",
                    ),
                    "steady_checksum_sink mismatch",
                ),
                (
                    "driver-sha",
                    lambda artifact: artifact["environment"]["driver"].__setitem__(
                        "sha256",
                        "0" * 64,
                    ),
                    "driver SHA-256 does not match the checked-in script",
                ),
                (
                    "git-status",
                    lambda artifact: artifact["environment"]["git"].__setitem__(
                        "status_short",
                        " M fabricated.py",
                    ),
                    "git status_short is not clean",
                ),
                (
                    "git-diff-stat",
                    lambda artifact: artifact["environment"]["git"].__setitem__(
                        "diff_stat",
                        " scripts/validate_top_level_stack_benchmark.py | 1 +",
                    ),
                    "git diff_stat is not clean",
                ),
                (
                    "git-missing-head",
                    lambda artifact: artifact["environment"]["git"].__setitem__(
                        "head",
                        None,
                    ),
                    "git head is not a full commit hash",
                ),
                (
                    "unsupported-credit",
                    lambda artifact: artifact["zero_credit_unsupported_cells"][
                        0
                    ].__setitem__(
                        "credit",
                        benchmark_top_level_stack.CREDIT_ERROR_PARITY,
                    ),
                    "credit mismatch",
                ),
                (
                    "python-executable",
                    lambda artifact: artifact.__getitem__("environment").__setitem__(
                        "python_executable",
                        str(REPOSITORY_ROOT.parent / "stale-venv" / "bin" / "python"),
                    ),
                    "python_executable mismatch",
                ),
                (
                    "numpy-path",
                    lambda artifact: artifact["environment"]["numpy"].__setitem__(
                        "path",
                        str(REPOSITORY_ROOT.parent / "stale" / "numpy.py"),
                    ),
                    "numpy.path mismatch",
                ),
                (
                    "torch-rs-path",
                    lambda artifact: artifact["environment"]["torch_rs"].__setitem__(
                        "path",
                        str(
                            REPOSITORY_ROOT.parent
                            / "stale"
                            / "torch_rs"
                            / "__init__.py"
                        ),
                    ),
                    "torch_rs.path mismatch",
                ),
                (
                    "torch-rs-extension-path",
                    lambda artifact: artifact["environment"]["torch_rs"].__setitem__(
                        "extension_path",
                        str(REPOSITORY_ROOT.parent / "stale" / "torch_rs.abi3.so"),
                    ),
                    "torch_rs.extension_path mismatch",
                ),
            )
            for label, mutate, expected_message in tamper_cases:
                with self.subTest(tamper=label):
                    tampered = copy.deepcopy(report)
                    mutate(tampered)
                    with self.assertRaises(AssertionError) as raised:
                        validate_top_level_stack_benchmark.validate_artifact_dict(
                            tampered,
                            expected_seed=20260908,
                            expected_cases_per_category=1,
                            expected_max_elements=4096,
                            expected_warmups=1,
                            expected_samples=1,
                            expected_threads=1,
                        )
                    self.assertIn(expected_message, str(raised.exception))

            default_validation_completed = subprocess.run(
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
            self.assertNotEqual(default_validation_completed.returncode, 0)
            self.assertIn(
                "environment warmups mismatch",
                default_validation_completed.stdout
                + default_validation_completed.stderr,
            )
            self.assertIn(
                "validator cases_per_category mismatch",
                default_validation_completed.stdout
                + default_validation_completed.stderr,
            )

            strict_validation_completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(artifact_path),
                    "--seed",
                    "20260908",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--threads",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                strict_validation_completed.returncode,
                0,
                msg=(
                    strict_validation_completed.stdout
                    + strict_validation_completed.stderr
                ),
            )

            wrong_seed_validation = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(artifact_path),
                    "--seed",
                    "222",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--threads",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(wrong_seed_validation.returncode, 0)
            self.assertIn(
                "validator seed mismatch",
                wrong_seed_validation.stdout + wrong_seed_validation.stderr,
            )

            tampered_path = Path(temporary_directory) / "tampered-stack-validator.json"
            tampered = copy.deepcopy(report)
            tampered["aggregates"][
                "steady_geomean_capped_0_10_10_0"
            ] = 456.0
            tampered_path.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            tampered_validation = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_SCRIPT),
                    "--validate-artifact",
                    str(tampered_path),
                    "--seed",
                    "20260908",
                    "--cases-per-category",
                    "1",
                    "--max-elements",
                    "4096",
                    "--warmups",
                    "1",
                    "--samples",
                    "1",
                    "--threads",
                    "1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(tampered_validation.returncode, 0)
            self.assertIn(
                "aggregates.steady_geomean_capped_0_10_10_0 mismatch",
                tampered_validation.stdout + tampered_validation.stderr,
            )


if __name__ == "__main__":
    unittest.main()
