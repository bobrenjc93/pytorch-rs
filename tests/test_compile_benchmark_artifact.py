import copy
import importlib.util
import sys
import unittest
from pathlib import Path


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

    def test_rendered_summary_counts_inference_as_supported(self):
        report = benchmark_compile_cpu._load_artifact(
            benchmark_compile_cpu.DEFAULT_ARTIFACT_PATH,
        )
        summary = benchmark_compile_cpu.render_markdown_summary(report)

        self.assertIn("7 inference", summary)
        self.assertIn(
            "| `inference` | 6 | Supported and timed public cases: "
            "`cpu_float32_relu_no_grad_inference` |",
            summary,
        )
        self.assertNotIn(
            "`inference` | 6 | Zero credit",
            summary,
        )

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


if __name__ == "__main__":
    unittest.main()
