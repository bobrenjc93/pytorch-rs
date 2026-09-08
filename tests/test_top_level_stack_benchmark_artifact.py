import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_top_level_stack.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_stack_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_top_level_stack = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_top_level_stack
spec.loader.exec_module(benchmark_top_level_stack)


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

    def test_zero_credit_rows_cover_unsupported_stack_boundaries(self):
        self.assertEqual(
            {cell.name for cell in benchmark_top_level_stack.UNSUPPORTED_CELLS},
            {
                "empty_input_sequence",
                "mixed_shapes",
                "mixed_metadata",
                "concrete_out",
            },
        )


if __name__ == "__main__":
    unittest.main()
