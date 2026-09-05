import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_creation_factories.py"

spec = importlib.util.spec_from_file_location(
    "_torch_rs_creation_factory_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_creation_factories = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_creation_factories
spec.loader.exec_module(benchmark_creation_factories)


def _has_reference_torch_2_13():
    try:
        import torch as reference_torch
    except ImportError:
        return False
    return reference_torch.__version__.split("+", 1)[0] == "2.13.0"


class CreationFactoryBenchmarkTests(unittest.TestCase):
    def test_workload_matrix_covers_requested_factories_and_shapes(self):
        apis = {workload.api for workload in benchmark_creation_factories.WORKLOADS}
        shape_labels = {
            workload.shape_label for workload in benchmark_creation_factories.WORKLOADS
        }
        self.assertEqual(apis, {"empty", "zeros", "ones"})
        self.assertEqual(shape_labels, {"scalar", "empty", "small", "large"})
        self.assertEqual(len(benchmark_creation_factories.WORKLOADS), 12)

        names = {workload.name for workload in benchmark_creation_factories.WORKLOADS}
        for api in apis:
            for shape_label in shape_labels:
                self.assertIn(f"{api}_{shape_label}", names)

    def test_output_path_rejects_burner_managed_artifacts(self):
        for path in (
            REPOSITORY_ROOT / ".burner" / "factory.json",
            REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
            REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
        ):
            with self.subTest(path=path):
                with self.assertRaises(SystemExit):
                    benchmark_creation_factories._output_path(path)

    @unittest.skipUnless(
        _has_reference_torch_2_13(),
        "requires pinned PyTorch 2.13 reference dependency",
    )
    def test_smoke_report_records_symmetric_metadata_checked_eager_timings(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(BENCHMARK_SCRIPT),
                "--warmups",
                "0",
                "--samples",
                "1",
                "--workloads",
                "empty_scalar",
                "zeros_empty",
                "ones_small",
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
            benchmark_creation_factories.BENCHMARK_VERSION,
        )
        self.assertEqual(report["environment"]["cuda_visible_devices"], "")
        self.assertIs(report["environment"]["private_cuda_roundtrip_included"], False)
        self.assertEqual(
            report["environment"]["implementation_orders"],
            [list(order) for order in benchmark_creation_factories.IMPLEMENTATION_ORDERS],
        )
        self.assertEqual(report["aggregates"]["timed_cell_count"], 3)

        by_name = {case["name"]: case for case in report["cases"]}
        self.assertEqual(set(by_name), {"empty_scalar", "zeros_empty", "ones_small"})
        for case in report["cases"]:
            self.assertIs(case["validation"]["metadata_checked"], True)
            self.assertIs(
                case["validation"]["metadata_materialized_inside_timed_loop"],
                True,
            )
            self.assertIs(case["validation"]["private_cuda_roundtrip_excluded"], True)
            self.assertEqual(
                set(case["implementations"]),
                {"torch_rs", "pytorch"},
            )
            for implementation in ("torch_rs", "pytorch"):
                passes = case["implementations"][implementation]["passes"]
                self.assertEqual(len(passes), 2)
                self.assertEqual(
                    {tuple(pass_result["order"]) for pass_result in passes},
                    set(benchmark_creation_factories.IMPLEMENTATION_ORDERS),
                )
                self.assertGreater(
                    case["implementations"][implementation]["steady_median_us"],
                    0.0,
                )

        self.assertIs(
            by_name["empty_scalar"]["validation"][
                "empty_values_unchecked_unspecified"
            ],
            True,
        )
        self.assertEqual(by_name["zeros_empty"]["metadata"]["numel"], 0)
        self.assertEqual(by_name["ones_small"]["metadata"]["shape"], [32, 32])
        self.assertEqual(
            by_name["ones_small"]["implementations"]["torch_rs"]["passes"][0][
                "value_check_sum"
            ],
            1024.0,
        )


if __name__ == "__main__":
    unittest.main()
