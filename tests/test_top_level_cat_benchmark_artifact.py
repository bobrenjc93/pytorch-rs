import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_top_level_cat.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_cat_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_top_level_cat = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_top_level_cat
spec.loader.exec_module(benchmark_top_level_cat)


def _has_reference_torch_2_13():
    try:
        import torch as reference_torch
    except ImportError:
        return False
    return reference_torch.__version__.split("+", 1)[0] == "2.13.0"


class TopLevelCatBenchmarkArtifactTests(unittest.TestCase):
    def test_workload_matrix_covers_representative_cat_cases(self):
        self.assertEqual(
            benchmark_top_level_cat.APIS,
            ("cat", "concat", "concatenate"),
        )
        workloads = benchmark_top_level_cat.WORKLOADS
        self.assertEqual(len(workloads), 14)
        categories = {workload.category for workload in workloads}
        self.assertEqual(
            categories,
            {
                "singleton",
                "multi-input",
                "empty operand",
                "offset",
                "noncontiguous",
                "axis keyword",
                "no_grad",
                "active autograd",
                "rank-2 dim0",
                "rank-2 dim1",
                "rank-2 neutral empty",
                "backward",
            },
        )
        names = {workload.name for workload in workloads}
        for required_name in (
            "singleton_contiguous_8192",
            "multi_input_contiguous_257_263_269",
            "empty_operand_middle_1024_0_511",
            "all_empty_tuple_dim_negative_one",
            "offset_contiguous_views_4096",
            "noncontiguous_stride2_views_4096",
            "tuple_dim_negative_one_513_509",
            "axis_keyword_17_19",
            "no_grad_grad_inputs_257_263",
            "active_autograd_grad_inputs_257_263",
            "rank2_dim0_contiguous_64x32_17x32",
            "rank2_dim1_contiguous_64x16_64x9",
            "rank2_dim1_neutral_empty_48x11_0_48x5",
            "backward_repeated_inputs_257_263",
        ):
            self.assertIn(required_name, names)
        self.assertTrue(
            all(workload.input_seeds for workload in workloads),
            "each public workload should record deterministic input seeds",
        )

    def test_output_path_rejects_burner_managed_artifacts(self):
        for path in (
            REPOSITORY_ROOT / ".burner" / "cat.json",
            REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
            REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
        ):
            with self.subTest(path=path):
                with self.assertRaises(SystemExit):
                    benchmark_top_level_cat._output_path(path)

    @unittest.skipUnless(
        _has_reference_torch_2_13(),
        "requires pinned PyTorch 2.13 reference dependency",
    )
    def test_smoke_report_records_symmetric_metadata_checked_timings(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--warmups",
                "0",
                "--samples",
                "1",
                "--workloads",
                "singleton_contiguous_8192",
                "empty_operand_middle_1024_0_511",
                "noncontiguous_stride2_views_4096",
                "active_autograd_grad_inputs_257_263",
                "rank2_dim0_contiguous_64x32_17x32",
                "rank2_dim1_contiguous_64x16_64x9",
                "backward_repeated_inputs_257_263",
                "--apis",
                "cat",
            ],
            check=False,
            capture_output=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
            text=True,
            timeout=90,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stdout + completed.stderr,
        )

        report = json.loads(completed.stdout)
        self.assertEqual(
            report["environment"]["benchmark_version"],
            benchmark_top_level_cat.BENCHMARK_VERSION,
        )
        self.assertEqual(report["environment"]["cuda_visible_devices"], "")
        self.assertEqual(
            report["environment"]["implementation_orders"],
            [list(order) for order in benchmark_top_level_cat.IMPLEMENTATION_ORDERS],
        )
        self.assertEqual(report["aggregates"]["timed_supported_cell_count"], 7)
        self.assertIs(
            report["aggregates"]["unsupported_cells_in_performance_score"],
            True,
        )
        coverage_adjusted = report["aggregates"]["coverage_adjusted"]
        self.assertEqual(coverage_adjusted["timed_supported_cell_count"], 7)
        self.assertEqual(coverage_adjusted["zero_credit_unsupported_cell_count"], 3)
        self.assertEqual(coverage_adjusted["zero_credit_incorrect_cell_count"], 0)
        self.assertEqual(coverage_adjusted["denominator_cell_count"], 10)
        self.assertEqual(
            coverage_adjusted["zero_credit_capped_ratio"],
            benchmark_top_level_cat.ZERO_CREDIT_CAPPED_RATIO,
        )
        self.assertIs(coverage_adjusted["unsupported_cells_in_denominator"], True)
        self.assertNotIn(
            "combined_capped_with_zero_credit_unsupported",
            report["aggregates"],
        )
        self.assertEqual(
            report["held_out_validation"]["version"],
            benchmark_top_level_cat.HELD_OUT_VALIDATION_VERSION,
        )
        self.assertEqual(
            report["held_out_validation"]["case_count"],
            len(benchmark_top_level_cat.HELD_OUT_VALIDATION_WORKLOADS),
        )
        self.assertTrue(report["held_out_validation"]["metadata_checked"])
        self.assertTrue(report["held_out_validation"]["value_bits_checked"])
        self.assertTrue(
            report["held_out_validation"]["gradient_accumulation_checked"]
        )

        by_name = {case["workload"]: case for case in report["cases"]}
        self.assertEqual(
            set(by_name),
            {
                "singleton_contiguous_8192",
                "empty_operand_middle_1024_0_511",
                "noncontiguous_stride2_views_4096",
                "active_autograd_grad_inputs_257_263",
                "rank2_dim0_contiguous_64x32_17x32",
                "rank2_dim1_contiguous_64x16_64x9",
                "backward_repeated_inputs_257_263",
            },
        )
        for case in report["cases"]:
            self.assertIs(case["validation"]["metadata_checked"], True)
            self.assertIs(case["validation"]["value_bits_checked"], True)
            self.assertIs(case["validation"]["steady_checksums_checked"], True)
            self.assertIs(case["validation"]["operand_nonmutation_checked"], True)
            self.assertEqual(set(case["implementations"]), {"torch_rs", "pytorch"})
            for implementation in ("torch_rs", "pytorch"):
                passes = case["implementations"][implementation]["passes"]
                self.assertEqual(len(passes), 2)
                self.assertEqual(
                    {tuple(pass_result["order"]) for pass_result in passes},
                    set(benchmark_top_level_cat.IMPLEMENTATION_ORDERS),
                )
                self.assertGreater(
                    case["implementations"][implementation]["steady_median_us"],
                    0.0,
                )
                self.assertTrue(
                    case["implementations"][implementation]["checksums"],
                    msg=f"missing timed checksum for {case['workload']}",
                )

        self.assertEqual(
            by_name["singleton_contiguous_8192"]["output_metadata"][0]["shape"],
            [8192],
        )
        self.assertEqual(
            by_name["empty_operand_middle_1024_0_511"]["output_metadata"][0]["shape"],
            [1535],
        )
        self.assertEqual(
            by_name["noncontiguous_stride2_views_4096"]["input_metadata"][0]["stride"],
            [2],
        )
        active_output = by_name["active_autograd_grad_inputs_257_263"][
            "output_metadata"
        ][0]
        self.assertEqual(active_output["shape"], [520])
        self.assertTrue(active_output["requires_grad"])
        self.assertFalse(active_output["is_leaf"])
        self.assertEqual(
            by_name["rank2_dim0_contiguous_64x32_17x32"]["output_metadata"][0][
                "shape"
            ],
            [81, 32],
        )
        self.assertEqual(
            by_name["rank2_dim1_contiguous_64x16_64x9"]["output_metadata"][0][
                "shape"
            ],
            [64, 25],
        )
        backward_metadata = by_name["backward_repeated_inputs_257_263"][
            "output_metadata"
        ]
        self.assertEqual(
            [entry["label"] for entry in backward_metadata],
            ["output", "tensors[0].grad", "tensors[1].grad", "tensors[2].grad"],
        )
        self.assertEqual(backward_metadata[0]["shape"], [777])

    def test_coverage_adjusted_aggregate_keeps_zero_credit_denominator(self):
        report = json.loads(
            benchmark_top_level_cat.DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8")
        )
        aggregates = report["aggregates"]
        unsupported = report["zero_credit_unsupported_cells"]
        self.assertIs(aggregates["unsupported_cells_in_performance_score"], True)
        self.assertIs(
            aggregates["supported_only_speed_geomeans_exclude_zero_credit"],
            True,
        )
        self.assertNotIn("combined_capped_with_zero_credit_unsupported", aggregates)
        self.assertEqual(
            aggregates["zero_credit_unsupported_cell_count"],
            len(unsupported),
        )
        coverage_adjusted = aggregates["coverage_adjusted"]
        self.assertEqual(
            coverage_adjusted["denominator_cell_count"],
            len(report["cases"]) + len(unsupported),
        )
        self.assertEqual(
            coverage_adjusted["timed_supported_cell_count"],
            len(report["cases"]),
        )
        self.assertEqual(
            coverage_adjusted["zero_credit_cell_count"],
            len(unsupported),
        )
        self.assertEqual(
            coverage_adjusted["zero_credit_unsupported_cell_count"],
            len(unsupported),
        )
        self.assertEqual(coverage_adjusted["zero_credit_incorrect_cell_count"], 0)
        self.assertIs(coverage_adjusted["unsupported_cells_in_denominator"], True)
        self.assertIs(coverage_adjusted["incorrect_cells_in_denominator"], True)
        self.assertEqual(
            coverage_adjusted["zero_credit_capped_ratio"],
            benchmark_top_level_cat.ZERO_CREDIT_CAPPED_RATIO,
        )
        self.assertGreater(coverage_adjusted["geomean_capped_0_10_10_0"], 0.0)
        self.assertTrue(unsupported)
        self.assertTrue(all(row["credit"] == "zero" for row in unsupported))
        self.assertEqual(
            aggregates["groups"]["all supported cells"]["cell_count"],
            len(report["cases"]),
        )

    def test_checked_in_raw_artifact_matches_markdown_summary(self):
        markdown = benchmark_top_level_cat.DEFAULT_MARKDOWN_REPORT_PATH.read_text(
            encoding="utf-8"
        )
        self.assertTrue(markdown.startswith("# CPU `torch.cat` Release Timings\n"))
        benchmark_top_level_cat.validate_artifact(
            benchmark_top_level_cat.DEFAULT_ARTIFACT_PATH,
            benchmark_top_level_cat.DEFAULT_MARKDOWN_REPORT_PATH,
        )


if __name__ == "__main__":
    unittest.main()
