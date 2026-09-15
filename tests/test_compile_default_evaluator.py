"""Portable contract/anti-shortcut tests; live reference/CUDA runs are separate."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import gzip
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

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


class DefaultCompileExportTests(unittest.TestCase):
    """Exercise the real parent and CLI, replacing only machine/worker seams."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="default-compile-export-")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name).resolve()
        self.root = self.temporary / "worktree"
        self.root.mkdir()
        self.sink = self.temporary / "export"
        self.sink.mkdir()
        self.identity = {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "source_sha256": "c" * 64,
            "working_tree_changes": "",
            "files": {"scripts/evaluate_torch_compile_default.py": "d" * 64},
        }
        self.worker_hook = None
        self.worker_calls = []
        self.worker_bytes = {}
        self.invocations = 0
        patches = (
            mock.patch.object(evaluator, "ROOT", self.root),
            mock.patch.object(
                evaluator,
                "source_identity",
                side_effect=lambda: deepcopy(self.identity),
            ),
            mock.patch.object(evaluator, "git", return_value=""),
            mock.patch.object(
                evaluator,
                "check_environment",
                return_value={"cuda_visible_devices": "portable-fixture"},
            ),
            mock.patch.object(
                evaluator,
                "wheel_identity",
                return_value={
                    "path": "fixture.whl", "sha256": "e" * 64, "profile": "release"
                },
            ),
            mock.patch.object(
                evaluator, "gpu_snapshot", return_value="fixture GPU metadata"
            ),
            mock.patch.object(
                evaluator.subprocess, "check_output", side_effect=self.rustc_version
            ),
            mock.patch.object(
                evaluator.subprocess,
                "Popen",
                side_effect=AssertionError("portable tests must not start workers"),
            ),
            mock.patch.object(evaluator, "run_worker", side_effect=self.run_worker),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def rustc_version(self, command, **kwargs):
        self.assertEqual(command, ["rustc", "--version"])
        self.assertEqual(kwargs, {"text": True})
        return "rustc portable-fixture\n"

    def run_worker(self, implementation, device, round_number, directory):
        self.worker_calls.append((implementation, device, round_number))
        label = f"{device}-{round_number}-{implementation}"
        result = worker_result()
        result.update(
            implementation=implementation,
            device=device,
            environment={"threads": 1, "framework_version": "fixture"},
        )
        median = 2.0 if implementation == "reference" else 4.0
        for case in result["cases"].values():
            case["factory_ms"] = 1.25
            for variant in case["variants"]:
                variant.update(evaluator.latency_summary([median] * 17))
                variant["cold_call_ms"] = 7.5
                for key in ("observed", "changed_observed"):
                    variant[key]["output"]["device"] = (
                        "cuda:0" if device == "cuda" else "cpu"
                    )
        output = directory / f"{label}.json.gz"
        log = directory / f"{label}.log"
        output.write_bytes(gzip.compress(json.dumps(result).encode(), mtime=0))
        log.write_bytes(f"{label}: fixture compiler log\n".encode())
        for suffix in ("inductor-cache", "triton-cache"):
            cache = directory / f"{label}-{suffix}"
            cache.mkdir()
            (cache / "not-evidence.bin").write_bytes(b"private compiler cache")
        result["artifacts"] = {
            "output": str(output),
            "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "log": str(log),
            "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "wall_seconds": 3.25,
            "cache_state": "fixture empty process-local caches at worker start",
        }
        self.worker_bytes.update(
            {output.name: output.read_bytes(), log.name: log.read_bytes()}
        )
        if self.worker_hook:
            self.worker_hook(directory, output, log)
        return result

    def invoke(
        self, metric="cuda-perf", *, export=True, sink=None,
        diagnostic=False, output=False,
    ):
        self.invocations += 1
        self.worker_calls = []
        self.worker_bytes = {}
        run_root = self.root / "target/default-compile-eval"
        before = set(run_root.glob("run-*"))
        self.output = self.root / f"published-{self.invocations}.json"
        argv = [
            "evaluate_torch_compile_default.py",
            "--metric", metric,
            "--wheel", "fixture.whl",
            "--build-identity",
            json.dumps(
                {
                    key: self.identity[key]
                    for key in ("commit", "source_sha256", "working_tree_changes")
                }
            ),
            "--setup-timestamps",
            "1000000000,2000000000,4000000000,5000000000",
        ]
        if diagnostic:
            argv.append("--diagnostic")
        if output:
            argv.extend(["--output", str(self.output)])
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            mock.patch.dict(os.environ),
            mock.patch.object(sys, "argv", argv),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            os.environ.pop("BURNER_EVALUATION_ARTIFACT_DIR", None)
            if export:
                os.environ["BURNER_EVALUATION_ARTIFACT_DIR"] = str(
                    self.sink if sink is None else sink
                )
            try:
                return evaluator.main()
            finally:
                self.stdout, self.stderr = stdout.getvalue(), stderr.getvalue()
                created = set(run_root.glob("run-*")) - before
                self.assertEqual(len(created), 1)
                self.directory = created.pop()

    def report(self):
        return json.loads((self.directory / "report.json").read_text())

    def assert_export_matches(self, expected_count):
        originals = {
            path.name: path for path in self.directory.iterdir() if path.is_file()
        }
        self.assertEqual(len(originals), expected_count)
        self.assertEqual({path.name for path in self.sink.iterdir()}, set(originals))
        for name, source in originals.items():
            copied = self.sink / name
            self.assertEqual(copied.read_bytes(), source.read_bytes(), name)
            self.assertNotEqual(
                (copied.stat().st_dev, copied.stat().st_ino),
                (source.stat().st_dev, source.stat().st_ino),
                name,
            )
        for worker in self.report()["workers"]:
            for kind in ("output", "log"):
                artifacts = worker["artifacts"]
                copied = self.sink / Path(artifacts[kind]).name
                self.assertEqual(
                    hashlib.sha256(copied.read_bytes()).hexdigest(),
                    artifacts[f"{kind}_sha256"],
                )

    def test_parent_exports_both_schedules_with_unchanged_scores_and_metadata(self):
        cuda = [
            ("reference", "cuda", 0),
            ("candidate", "cuda", 0),
            ("candidate", "cuda", 1),
            ("reference", "cuda", 1),
        ]
        for metric, schedule, score in (
            (
                "coverage",
                [("reference", "cpu", 0), ("candidate", "cpu", 0)] + cuda,
                100,
            ),
            ("cuda-perf", cuda, 50),
        ):
            with self.subTest(metric=metric):
                self.sink = self.temporary / metric
                self.sink.mkdir()
                opened = []
                original_open = Path.open

                def observe_open(path, mode="r", *args, **kwargs):
                    if mode == "xb" and path.parent == self.sink:
                        opened.append(path.name)
                    return original_open(path, mode, *args, **kwargs)

                with mock.patch.object(Path, "open", observe_open):
                    self.assertEqual(self.invoke(metric), 0)
                self.assertEqual(self.stderr, "")
                self.assertEqual(self.worker_calls, schedule)
                self.assertEqual(json.loads(self.stdout)["score"], score)
                self.assertEqual(opened[-1], "report.json")
                self.assertEqual(len(opened), 2 * len(schedule) + 1)
                self.assert_export_matches(2 * len(schedule) + 1)
                report = self.report()
                self.assertTrue(report["valid"])
                self.assertFalse(report["diagnostic"])
                self.assertNotIn("error", report)
                self.assertEqual(report["provenance"], self.identity)
                self.assertEqual(report["compile_call"], "framework.compile(program)")
                self.assertEqual((report["warmups"], report["samples"]), (5, 17))
                self.assertEqual((report["rtol"], report["atol"]), (1e-4, 1e-4))
                self.assertEqual(report["rtol_overrides"], {"bfloat16_roundtrip": 8e-3})
                self.assertEqual(report["setup"]["total_seconds"], 4)
                self.assertEqual(report["setup"]["native_build_seconds"], 2)
                self.assertEqual(report["scores"]["cuda-perf"]["score"], 50)
                self.assertEqual(report["scores"]["cuda-perf"]["eligible"], 56)
                eligible = 112 if metric == "coverage" else 56
                self.assertEqual(report["scores"][metric]["eligible"], eligible)
                self.assertEqual(len(report["cells"]), eligible)
                self.assertTrue(all(cell["ratio"] == 0.5 for cell in report["cells"]))
                for worker in report["workers"]:
                    self.assertNotIn("cases", worker)
                    copied = self.sink / Path(worker["artifacts"]["output"]).name
                    raw = json.loads(gzip.decompress(copied.read_bytes()))
                    raw_case = raw["cases"][corpus.CASES[0].name]
                    variant = raw_case["variants"][0]
                    self.assertEqual(len(variant["samples_ms"]), 17)
                    self.assertEqual(raw_case["factory_ms"], 1.25)
                    self.assertEqual(variant["cold_call_ms"], 7.5)
                    self.assertIn("changed_observed", variant)

    def test_extra_evidence_survives_cleanup_and_caches_are_skipped(self):
        extra = b"\x00additional evidence without a recognized suffix\xff"

        def add_extra(directory, output, log):
            (directory / ".extra-evidence").write_bytes(extra)
            (directory / "z-extra-evidence").write_bytes(extra)

        self.worker_hook = add_extra
        opened = []
        original_open = Path.open

        def observe_open(path, mode="r", *args, **kwargs):
            if mode == "xb" and path.parent == self.sink:
                opened.append(path.name)
            return original_open(path, mode, *args, **kwargs)

        with mock.patch.object(Path, "open", observe_open):
            self.assertEqual(self.invoke(), 0)
        self.assertEqual(opened[-1], "report.json")
        self.assertLess(opened.index("z-extra-evidence"), opened.index("report.json"))
        self.assert_export_matches(11)
        self.assertEqual((self.sink / ".extra-evidence").read_bytes(), extra)
        self.assertEqual((self.sink / "z-extra-evidence").read_bytes(), extra)
        retained = {path.name: path.read_bytes() for path in self.sink.iterdir()}
        self.assertTrue(any(path.is_dir() for path in self.directory.iterdir()))
        shutil.rmtree(self.root)
        self.assertEqual(
            {path.name: path.read_bytes() for path in self.sink.iterdir()}, retained
        )
        self.assertTrue(json.loads(retained["report.json"])["valid"])

    def test_absent_export_variable_preserves_standalone_score_and_output(self):
        self.assertEqual(self.invoke(export=False, output=True), 0)
        self.assertEqual(self.stderr, "")
        self.assertEqual(json.loads(self.stdout)["score"], 50)
        self.assertEqual(list(self.sink.iterdir()), [])
        self.assertEqual(
            self.output.read_bytes(), (self.directory / "report.json").read_bytes()
        )

    def test_diagnostic_and_both_keep_summary_and_output_contract(self):
        for metric, diagnostic in (
            ("both", False), ("both", True), ("coverage", True), ("cuda-perf", True)
        ):
            with self.subTest(metric=metric, diagnostic=diagnostic):
                self.sink = self.temporary / f"summary-{metric}-{diagnostic}"
                self.sink.mkdir()
                self.assertEqual(
                    self.invoke(metric, diagnostic=diagnostic, output=True), 0
                )
                summary = json.loads(self.stdout)
                self.assertNotIn("score", summary)
                self.assertEqual(summary["valid"], not diagnostic)
                self.assertEqual(summary["diagnostic"], diagnostic)
                if metric != "cuda-perf":
                    self.assertEqual(summary["scores"]["coverage"]["score"], 100)
                self.assertEqual(summary["scores"]["cuda-perf"]["score"], 50)
                self.assert_export_matches(9 if metric == "cuda-perf" else 13)
                self.assertEqual(
                    self.output.read_bytes(), (self.sink / "report.json").read_bytes()
                )

    def test_empty_missing_and_non_directory_sinks_suppress_all_success_output(self):
        occupied = self.temporary / "not-a-directory"
        occupied.write_bytes(b"preserve existing file")
        for sink in ("", self.temporary / "missing", occupied):
            for metric, diagnostic in (
                ("cuda-perf", False), ("both", False), ("coverage", True)
            ):
                with self.subTest(sink=str(sink), metric=metric, diagnostic=diagnostic):
                    self.assertEqual(
                        self.invoke(
                            metric, sink=sink, diagnostic=diagnostic, output=True
                        ),
                        2,
                    )
                    self.assertEqual(self.stdout, "")
                    self.assertIn(
                        "evaluation artifact export failed; no score:", self.stderr
                    )
                    self.assertNotIn("invalid measurement", self.stderr)
                    self.assertFalse(self.output.exists())
                    report = self.report()
                    self.assertEqual(report["valid"], not diagnostic)
                    self.assertNotIn("error", report)
                    self.assertEqual(report["scores"]["cuda-perf"]["score"], 50)
        self.assertEqual(occupied.read_bytes(), b"preserve existing file")
        self.assertFalse((self.temporary / "missing").exists())

    def test_destination_collisions_never_overwrite_or_reuse_existing_bytes(self):
        for collision in ("different", "identical", "report", "symlink"):
            with self.subTest(collision=collision):
                self.sink = self.temporary / f"collision-{collision}"
                self.sink.mkdir()
                name = (
                    "report.json" if collision == "report" else "cuda-0-reference.log"
                )
                existing = self.sink / name
                expected = b"preexisting evidence"
                self.worker_hook = None
                if collision == "identical":
                    expected = b"cuda-0-reference: fixture compiler log\n"
                if collision == "symlink":
                    target = self.temporary / "collision-target"
                    target.write_bytes(expected)
                    existing.symlink_to(target)
                else:
                    existing.write_bytes(expected)
                self.assertEqual(self.invoke(), 2)
                self.assertEqual(self.stdout, "")
                self.assertIn("evaluation artifact export failed", self.stderr)
                self.assertTrue(self.report()["valid"])
                self.assertEqual(existing.read_bytes(), expected)
                if collision == "symlink":
                    self.assertTrue(existing.is_symlink())
                copied = self.sink / "cuda-0-candidate.json.gz"
                self.assertEqual(copied.read_bytes(), self.worker_bytes[copied.name])
                if collision != "report":
                    self.assertFalse((self.sink / "report.json").exists())

    def test_source_symlinks_and_nonregular_entries_fail_without_following_them(self):
        for kind in ("file-link", "directory-link", "dangling-link", "fifo"):
            with self.subTest(kind=kind):
                if kind == "fifo" and not hasattr(os, "mkfifo"):
                    self.skipTest("FIFO entries are unavailable on this platform")
                self.sink = self.temporary / f"reject-{kind}"
                self.sink.mkdir()
                target = self.temporary / f"target-{kind}"
                if kind == "directory-link":
                    target.mkdir()
                    (target / "outside.bin").write_bytes(b"outside bytes")
                elif kind == "file-link":
                    target.write_bytes(b"outside bytes")

                def add_nonregular(directory, output, log):
                    if len(self.worker_calls) != 1:
                        return
                    entry = directory / "unsafe-entry"
                    if kind == "fifo":
                        os.mkfifo(entry)
                    else:
                        entry.symlink_to(
                            target, target_is_directory=kind == "directory-link"
                        )

                self.worker_hook = add_nonregular
                self.assertEqual(self.invoke(), 2)
                self.assertEqual(self.stdout, "")
                self.assertIn("nonregular evaluation artifact", self.stderr)
                self.assertTrue(self.report()["valid"])
                self.assertFalse((self.sink / "unsafe-entry").exists())
                self.assertFalse((self.sink / "report.json").exists())
                if kind == "file-link":
                    self.assertEqual(target.read_bytes(), b"outside bytes")
                elif kind == "directory-link":
                    self.assertEqual(
                        (target / "outside.bin").read_bytes(), b"outside bytes"
                    )

    def test_read_write_and_close_faults_preserve_sources_and_partial_exports(self):
        for fault, name in (
            ("read", "cuda-0-candidate.json.gz"),
            ("write", "cuda-0-candidate.json.gz"),
            ("close", "cuda-0-candidate.json.gz"),
            ("write", "report.json"),
            ("close", "report.json"),
        ):
            with self.subTest(fault=fault, name=name):
                self.sink = self.temporary / f"fault-{fault}-{name}"
                self.sink.mkdir()
                original_open = Path.open
                read_sizes = []

                class FaultyStream:
                    def __init__(stream_self, stream):
                        stream_self.stream = stream

                    def __enter__(stream_self):
                        return stream_self

                    def __exit__(stream_self, *exception):
                        stream_self.stream.close()
                        if fault == "close":
                            raise OSError("injected export close failure")

                    def read(stream_self, size):
                        read_sizes.append(size)
                        if len(read_sizes) == 1:
                            return stream_self.stream.read(7)
                        raise OSError("injected export read failure")

                    def write(stream_self, data):
                        if fault == "write":
                            stream_self.stream.write(data[:7])
                            raise OSError("injected export write failure")
                        return stream_self.stream.write(data)

                def faulty_open(path, mode="r", *args, **kwargs):
                    stream = original_open(path, mode, *args, **kwargs)
                    if path.name == name and (
                        (
                            fault == "read"
                            and mode == "rb"
                            and path.parent.name.startswith("run-")
                            and (path.parent / "report.json").is_file()
                        )
                        or (
                            fault != "read"
                            and mode == "xb"
                            and path.parent == self.sink
                        )
                    ):
                        return FaultyStream(stream)
                    return stream

                with mock.patch.object(Path, "open", faulty_open):
                    self.assertEqual(self.invoke(output=True), 2)
                self.assertEqual(self.stdout, "")
                self.assertIn(f"injected export {fault} failure", self.stderr)
                self.assertFalse(self.output.exists())
                self.assertTrue(self.report()["valid"])
                self.assertNotIn("error", self.report())
                for original_name, data in self.worker_bytes.items():
                    self.assertEqual(
                        (self.directory / original_name).read_bytes(), data
                    )
                source_bytes = (self.directory / name).read_bytes()
                copied_bytes = (self.sink / name).read_bytes()
                self.assertEqual(
                    copied_bytes, source_bytes if fault == "close" else source_bytes[:7]
                )
                if name != "report.json":
                    self.assertFalse((self.sink / "report.json").exists())
                else:
                    self.assertEqual(len(list(self.sink.iterdir())), 9)
                if fault == "read":
                    self.assertEqual(len(read_sizes), 2)
                    self.assertTrue(all(0 < size <= 1024 * 1024 for size in read_sizes))

    def test_failed_worker_partials_are_exported_even_without_a_worker_report_row(self):
        def fail_worker(directory, output, log):
            output.write_bytes(b"\x1f\x8bpartial worker output")
            raise evaluator.InvalidMeasurement("fixture worker failed")

        self.worker_hook = fail_worker
        self.assertEqual(self.invoke(), 2)
        self.assertEqual(self.stdout, "")
        self.assertIn(
            "invalid measurement; no score: InvalidMeasurement: fixture worker failed",
            self.stderr,
        )
        self.assertNotIn("evaluation artifact export failed", self.stderr)
        report = self.report()
        self.assertFalse(report["valid"])
        self.assertEqual(report["workers"], [])
        self.assertEqual(report["error"], "InvalidMeasurement: fixture worker failed")
        self.assert_export_matches(3)
        self.assertEqual(
            (self.sink / "cuda-0-reference.json.gz").read_bytes(),
            b"\x1f\x8bpartial worker output",
        )

    def test_measurement_and_export_errors_are_both_visible_and_report_unchanged(self):
        def fail_worker(directory, output, log):
            raise evaluator.InvalidMeasurement("original worker failure")

        self.worker_hook = fail_worker
        self.assertEqual(self.invoke(sink="", output=True), 2)
        self.assertEqual(self.stdout, "")
        self.assertFalse(self.output.exists())
        self.assertLess(
            self.stderr.index("original worker failure"),
            self.stderr.index("evaluation artifact export failed"),
        )
        self.assertEqual(
            self.report()["error"], "InvalidMeasurement: original worker failure"
        )
        self.assertFalse(self.report()["valid"])
        self.assertTrue((self.directory / "cuda-0-reference.json.gz").exists())
        self.assertEqual(list(self.sink.iterdir()), [])

    def test_local_report_write_failure_is_not_mislabeled_as_transfer_failure(self):
        original_open = Path.open

        def fail_report_write(path, mode="r", *args, **kwargs):
            if path.name == "report.json" and mode == "w":
                raise OSError("local report write failed")
            return original_open(path, mode, *args, **kwargs)

        with mock.patch.object(Path, "open", fail_report_write):
            with self.assertRaisesRegex(OSError, "local report write failed"):
                self.invoke()
        self.assertEqual(self.stdout, "")
        self.assertNotIn("evaluation artifact export failed", self.stderr)
        self.assertEqual(list(self.sink.iterdir()), [])

    def test_non_parent_cli_modes_do_not_export(self):
        for arguments in (
            ["--source-identity"],
            [
                "--worker", "candidate", "--device", "cpu",
                "--worker-output", str(self.root / "worker.json.gz"),
            ],
        ):
            with (
                self.subTest(arguments=arguments),
                mock.patch.dict(os.environ, {"BURNER_EVALUATION_ARTIFACT_DIR": ""}),
                mock.patch.object(sys, "argv", ["evaluator", *arguments]),
                mock.patch.object(evaluator, "worker") as worker,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(evaluator.main(), 0)
                self.assertEqual(worker.call_count, int("--worker" in arguments))
                self.assertFalse((self.root / "target/default-compile-eval").exists())
                self.assertEqual(list(self.sink.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
