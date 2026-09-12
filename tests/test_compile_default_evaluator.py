"""Portable contract/anti-shortcut tests; live reference/CUDA runs are separate."""

import ast
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
evaluator = importlib.import_module("evaluate_torch_compile_default")
corpus = importlib.import_module("torch_compile_default_corpus")


def observation(value=1.0):
    return {
        "output": {
            "tensor": 0,
            "shape": [1],
            "stride": [1],
            "dtype": "float32",
            "device": "cuda:0",
            "requires_grad": False,
            "values": [value],
        }
    }


def worker_result():
    return {
        "device": "cuda",
        "cases": {
            case.name: {
                "status": "passed",
                "variants": [
                    {
                        "variant": variant,
                        "median_ms": 2.0,
                        "observed": observation(),
                        "changed_observed": observation(2.0),
                    }
                    for variant in corpus.VARIANTS
                ],
            }
            for case in corpus.CASES
        },
    }


class DefaultCompileEvaluatorTests(unittest.TestCase):
    def test_frozen_manifest_and_weights(self):
        self.assertEqual(corpus.VERSION, "public-default-compile-v2")
        self.assertEqual(len(corpus.CASES), 28)
        self.assertEqual(len({case.name for case in corpus.CASES}), 28)
        self.assertEqual(sum(corpus.CATEGORY_WEIGHTS.values()), 100)
        self.assertEqual(len(corpus.CATEGORY_WEIGHTS), 14)
        for category in corpus.CATEGORY_WEIGHTS:
            self.assertEqual(sum(case.category == category for case in corpus.CASES), 2)
        self.assertEqual(corpus.VARIANTS, (0, 1))

    def test_only_public_default_compile_call(self):
        source = (ROOT / "scripts/evaluate_torch_compile_default.py").read_text()
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "compile"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 1)
        self.assertEqual(calls[0].keywords, [])
        for forbidden in (
            "CudaBenchmarkTensor",
            "_torch_rs_cuda_compile_workload",
            "h100_cuda_pointwise_reduce_float32",
            "graph_module.forward",
        ):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(
                forbidden,
                (ROOT / "scripts/torch_compile_default_corpus.py").read_text(),
            )

    def test_eager_identity_is_rejected(self):
        def original():
            return 3

        with self.assertRaisesRegex(AssertionError, "original Python callable"):
            evaluator.reject_eager_wrapper(original, original, original)

    def test_wrapped_eager_execution_is_rejected_and_profile_restored(self):
        def original():
            return 3

        def wrapper():
            return original()

        previous = sys.getprofile()
        with self.assertRaisesRegex(AssertionError, "original Python body"):
            evaluator.reject_eager_wrapper(original, wrapper, wrapper)
        self.assertIs(sys.getprofile(), previous)

    def test_a_non_replaying_executor_passes_the_body_guard(self):
        def original():
            return 3

        def executor():
            return 3

        self.assertEqual(
            evaluator.reject_eager_wrapper(original, executor, executor), 3
        )

    def test_reference_imports_are_blocked_but_native_imports_are_not(self):
        blocker = evaluator.BlockReferenceImport()
        for name in ("torch", "torch._inductor", "torch.nn"):
            with self.assertRaises(ImportError):
                blocker.find_spec(name)
        self.assertIsNone(blocker.find_spec("torch_rs"))

    def test_reference_failure_invalidates_run_instead_of_shrinking_denominator(self):
        reference = worker_result()
        reference["cases"][corpus.CASES[0].name] = {
            "status": "failed",
            "error": "Inductor failed",
        }
        with self.assertRaisesRegex(
            evaluator.InvalidMeasurement, "reference case.*failed"
        ):
            evaluator.compare_workers(reference, worker_result())

    def test_missing_reference_variant_invalidates_run(self):
        reference = worker_result()
        reference["cases"][corpus.CASES[0].name]["variants"].pop()
        with self.assertRaisesRegex(evaluator.InvalidMeasurement, "missing/reordered"):
            evaluator.compare_workers(reference, worker_result())

    def test_candidate_failure_stays_as_two_zero_cells(self):
        candidate = worker_result()
        candidate["cases"][corpus.CASES[0].name] = {
            "status": "failed",
            "error": "NotImplementedError",
        }
        cells = evaluator.compare_workers(worker_result(), candidate)
        self.assertEqual(len(cells), 56)
        self.assertEqual(sum(row["passed"] for row in cells), 54)
        self.assertTrue(all(row["ratio"] == 0 for row in cells[:2]))
        score = evaluator.aggregate(cells, "coverage")
        self.assertEqual(score["eligible"], 56)
        self.assertAlmostEqual(score["score"], 94)

    def test_incorrect_values_or_gradients_have_no_performance_credit(self):
        for key in ("output", "gradients"):
            reference, candidate = worker_result(), worker_result()
            if key == "gradients":
                for report in (reference, candidate):
                    report["cases"][corpus.CASES[0].name]["variants"][0]["observed"][
                        key
                    ] = deepcopy(observation()["output"])
            candidate["cases"][corpus.CASES[0].name]["variants"][0]["observed"][key][
                "values"
            ] = [2.0]
            cells = evaluator.compare_workers(reference, candidate)
            self.assertFalse(cells[0]["passed"])
            self.assertEqual(cells[0]["ratio"], 0)

    def test_device_dtype_stride_and_aliases_are_exact_observables(self):
        for field, value in (
            ("device", "cpu"),
            ("dtype", "float16"),
            ("stride", [2]),
            ("tensor", 1),
        ):
            actual = observation()
            actual["output"][field] = value
            with self.assertRaises(AssertionError):
                evaluator.compare(actual, observation())
        with self.assertRaises(AssertionError):
            evaluator.compare({"alias_of": 1}, {"alias_of": 0})

    def test_cached_output_on_same_shape_new_values_fails(self):
        candidate = worker_result()
        candidate["cases"][corpus.CASES[0].name]["variants"][0]["changed_observed"] = (
            observation()
        )
        cells = evaluator.compare_workers(worker_result(), candidate)
        self.assertFalse(cells[0]["passed"])
        self.assertEqual(cells[0]["ratio"], 0)

    def test_bfloat16_tolerance_is_symmetric_without_relaxing_float32_cases(self):
        candidate = worker_result()
        for name in ("bfloat16_roundtrip", "affine_relu"):
            candidate["cases"][name]["variants"][0]["observed"]["output"]["values"] = [
                1.004
            ]
        cells = evaluator.compare_workers(worker_result(), candidate)
        bf = next(cell for cell in cells if cell["case"] == "bfloat16_roundtrip")
        fp = next(cell for cell in cells if cell["case"] == "affine_relu")
        self.assertTrue(
            bf["passed"],
            "candidate receives the same bfloat16 allowance as the reference",
        )
        self.assertFalse(fp["passed"], "ordinary float32 correctness remains strict")
        self.assertEqual(evaluator.RTOL_OVERRIDES, {"bfloat16_roundtrip": 8e-3})

    def test_no_candidate_success_means_zero_and_no_common_success_ratio(self):
        candidate = worker_result()
        for name in candidate["cases"]:
            candidate["cases"][name] = {"status": "failed", "error": "unsupported"}
        cells = evaluator.compare_workers(worker_result(), candidate)
        for metric in ("coverage", "cuda-perf"):
            result = evaluator.aggregate(cells, metric)
            self.assertEqual(result["score"], 0)
            self.assertEqual(result["eligible"], 56)
            self.assertIsNone(result["common_success_geomean_ratio"])

    def test_capping_precedes_geometric_aggregation(self):
        cells = evaluator.compare_workers(worker_result(), worker_result())
        cells[0]["ratio"] = 10.0
        cells[1]["ratio"] = 0.5
        result = evaluator.aggregate(cells, "cuda-perf")
        self.assertAlmostEqual(result["score"], 88 + 12 * 0.5**0.25)
        self.assertLess(result["score"], 100)

    def test_missing_category_cannot_produce_a_score(self):
        cells = evaluator.compare_workers(worker_result(), worker_result())[4:]
        with self.assertRaises(evaluator.InvalidMeasurement):
            evaluator.aggregate(cells, "coverage")

    def test_reversed_order_failure_is_not_averaged_away(self):
        first = evaluator.compare_workers(worker_result(), worker_result())
        second = deepcopy(first)
        second[0].update(passed=False, ratio=0.0, error="failed")
        merged = evaluator.merge_rounds([first, second])
        self.assertFalse(merged[0]["passed"])
        self.assertEqual(merged[0]["ratio"], 0)
        self.assertEqual(len(merged), 56)

    def test_invalid_timing_samples_do_not_score(self):
        for samples in (
            [0.0] * evaluator.SAMPLES,
            [1.0],
            [float("nan")] * evaluator.SAMPLES,
        ):
            with self.assertRaises(evaluator.InvalidMeasurement):
                evaluator.latency_summary(samples)

    def test_setup_receipt_is_bound_to_measured_source(self):
        provenance = {
            "commit": "abc",
            "source_sha256": "123",
            "working_tree_changes": "",
        }
        receipt = evaluator.setup_identity(
            json.dumps(provenance), "1,2,3,4", provenance
        )
        self.assertAlmostEqual(receipt["total_seconds"], 3e-9)
        with self.assertRaisesRegex(evaluator.InvalidMeasurement, "checkout changed"):
            evaluator.setup_identity(
                json.dumps({**provenance, "commit": "other"}), "1,2,3,4", provenance
            )
        for timestamps in ("1,4,3,2", "0,1,2,3", "1,2"):
            with self.assertRaises(evaluator.InvalidMeasurement):
                evaluator.setup_identity(json.dumps(provenance), timestamps, provenance)

    def test_both_burner_definitions_are_versioned_executable_gates(self):
        definitions = json.loads((ROOT / ".burner/evaluations.json").read_text())[
            "evaluations"
        ]
        by_id = {item["id"]: item for item in definitions}
        for identity, version, metric in (
            ("eval_a61c0e71", "evaldef_repo_a61c0e71_v4", "coverage"),
            ("eval_6f98c42d", "evaldef_repo_6f98c42d_v3", "cuda-perf"),
        ):
            self.assertEqual(by_id[identity]["definitionVersion"], version)
            self.assertEqual(
                by_id[identity]["command"],
                f"bash scripts/evaluate_torch_compile_default.sh --metric {metric}",
            )
            self.assertNotIn("screeningCommand", by_id[identity])


if __name__ == "__main__":
    unittest.main()
