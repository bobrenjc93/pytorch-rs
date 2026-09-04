import importlib.util
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_top_level_subtract.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_top_level_subtract_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_top_level_subtract = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_top_level_subtract
spec.loader.exec_module(benchmark_top_level_subtract)


class TopLevelSubtractBenchmarkArtifactTests(unittest.TestCase):
    def test_checked_in_raw_artifact_matches_markdown_summary(self):
        benchmark_top_level_subtract.validate_artifact(
            benchmark_top_level_subtract.DEFAULT_ARTIFACT_PATH,
            benchmark_top_level_subtract.DEFAULT_MARKDOWN_REPORT_PATH,
        )


if __name__ == "__main__":
    unittest.main()
