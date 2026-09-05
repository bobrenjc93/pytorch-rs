import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_creation_factories.py"
VALIDATOR_SCRIPT = (
    REPOSITORY_ROOT / "scripts" / "validate_creation_factory_benchmark.py"
)

spec = importlib.util.spec_from_file_location(
    "_torch_rs_creation_factory_benchmark_for_tests",
    BENCHMARK_SCRIPT,
)
benchmark_creation_factories = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_creation_factories
spec.loader.exec_module(benchmark_creation_factories)

validator_spec = importlib.util.spec_from_file_location(
    "_torch_rs_creation_factory_validator_for_tests",
    VALIDATOR_SCRIPT,
)
validate_creation_factory_benchmark = importlib.util.module_from_spec(validator_spec)
assert validator_spec.loader is not None
sys.modules[validator_spec.name] = validate_creation_factory_benchmark
validator_spec.loader.exec_module(validate_creation_factory_benchmark)


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
            if case["api"] == "empty":
                self.assertIs(
                    case["validation"][
                        "filled_value_checksum_materialized_inside_timed_loop"
                    ],
                    False,
                )
            else:
                self.assertIs(
                    case["validation"][
                        "filled_value_checksum_materialized_inside_timed_loop"
                    ],
                    True,
                )
                self.assertEqual(
                    case["validation"]["filled_value_checksum_scope"],
                    "final_output_sum_once_per_timed_block",
                )
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
                if case["api"] != "empty":
                    self.assertTrue(
                        passes[0]["steady_value_checksums"],
                        msg=f"missing timed checksum for {case['name']}",
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

    def test_generated_validator_shapes_are_held_out_and_deterministic(self):
        first = validate_creation_factory_benchmark.generate_shapes(
            seed=20260905,
            count=5,
            max_elements=4096,
        )
        second = validate_creation_factory_benchmark.generate_shapes(
            seed=20260905,
            count=5,
            max_elements=4096,
        )
        self.assertEqual(first, second)

        public_shapes = {
            tuple(shape)
            for _, shape, _, _ in benchmark_creation_factories.SHAPE_CASES
        }
        self.assertFalse(public_shapes.intersection(first))
        self.assertEqual(len(set(first)), len(first))

        workloads = validate_creation_factory_benchmark._workloads_for_shapes(first)
        self.assertEqual(len(workloads), 15)
        self.assertEqual(
            {workload.api for workload in workloads},
            {"empty", "zeros", "ones"},
        )
        self.assertTrue(all(workload.generated for workload in workloads))

    @unittest.skipUnless(
        _has_reference_torch_2_13(),
        "requires pinned PyTorch 2.13 reference dependency",
    )
    def test_generated_validator_smoke_report_uses_held_out_shapes(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR_SCRIPT),
                "--seed",
                "20260905",
                "--shape-count",
                "2",
                "--max-elements",
                "2048",
                "--warmups",
                "0",
                "--samples",
                "1",
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
        self.assertEqual(report["validator"]["seed"], 20260905)
        self.assertEqual(report["validator"]["shape_count"], 2)
        self.assertIs(report["validator"]["fixed_public_shapes_excluded"], True)
        self.assertEqual(
            report["environment"]["benchmark_integrity"]["workload_set"],
            "generated_heldout_validator",
        )
        self.assertEqual(report["aggregates"]["timed_cell_count"], 6)

        public_shapes = {
            tuple(shape)
            for _, shape, _, _ in benchmark_creation_factories.SHAPE_CASES
        }
        for case in report["cases"]:
            self.assertTrue(case["generated"])
            self.assertTrue(case["validation"]["held_out_generated_shape"])
            self.assertNotIn(tuple(case["shape"]), public_shapes)
            if case["api"] != "empty":
                for implementation in ("torch_rs", "pytorch"):
                    for pass_result in case["implementations"][implementation][
                        "passes"
                    ]:
                        self.assertTrue(pass_result["steady_value_checksums"])


if __name__ == "__main__":
    unittest.main()
