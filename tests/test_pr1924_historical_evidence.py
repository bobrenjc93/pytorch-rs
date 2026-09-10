"""Offline preservation audit; never imports or executes archived benchmark code."""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from statistics import median
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "docs/diagnostics/compile-cuda-neg"
INITIAL = HISTORY / "initial-integration-1924/integration-checks"
PAIRED = HISTORY / "paired-validation-1924"
INITIAL_COMMIT = "764f159b0ae08e3a9be4793bc7c528844b322380"
COMMITS = {
    "baseline": "e4cddf0ecb45c23c683953fa2f8c45904b170a80",
    "candidate": "c41a850446c0cd94e70027b13d197a9c8413ff15",
}
ORDER = ["baseline", "candidate", "candidate", "baseline", "baseline", "candidate"]
SHAPES = {
    "square_256x256": [256, 256],
    "square_1024x1024": [1024, 1024],
    "tall_4096x256": [4096, 256],
    "wide_256x4096": [256, 4096],
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipts(name):
    return {row["name"]: row for row in read_json(INITIAL / name)}


class HistoricalEvidenceTests(unittest.TestCase):
    def test_complete_bundle_hashes(self):
        bundles = [
            (INITIAL.parent, "initial-integration-checks.tar.gz", 109,
             "aec0e3b5ec51c443c80160f2a42d25dfdc3ec73149a7ff5db9e3284efe77fa2f"),
            (PAIRED, "paired-validation-reports.tar.gz", 42,
             "7de4d24e5b0143d0f323115c8ac4847047bdc0980873c5cddf1b4776e1c47689"),
        ]
        for directory, archive, count, digest in bundles:
            with self.subTest(bundle=directory.name):
                manifest = read_json(directory / "artifact-sha256.json")
                self.assertEqual(manifest["archive_name"], archive)
                self.assertEqual(manifest["archive_sha256"], digest)
                files = manifest["files"]
                self.assertEqual(len(files), count)
                paths = list(directory.rglob("*"))
                self.assertFalse(any(p.is_symlink() for p in paths))
                self.assertEqual(
                    {p.relative_to(directory).as_posix() for p in paths if p.is_file()},
                    set(files) | {"artifact-sha256.json"},
                )
                for name, expected in files.items():
                    with self.subTest(file=name):
                        relative = PurePosixPath(name)
                        self.assertFalse(relative.is_absolute())
                        self.assertNotIn("..", relative.parts)
                        self.assertNotIn("\\", name)
                        self.assertEqual(sha256(directory / name), expected)

    def assert_suite_summary(self, name, count, result):
        # Read the terminal unittest summary, not expected error/traceback logging.
        text = (INITIAL / f"{name}.log").read_text()
        self.assertRegex(text, rf"Ran {count} tests in [\d.]+s\s+{re.escape(result)}\s*\Z")

    def test_initial_identity_and_full_suites(self):
        build = read_json(INITIAL / "build-record.json")
        start = read_json(INITIAL / "build-start.json")
        final = read_json(INITIAL / "final-audit.json")
        self.assertEqual(build["commit"], INITIAL_COMMIT)
        self.assertEqual(start["commit"], INITIAL_COMMIT)
        self.assertEqual(start["status"], "")
        self.assertEqual(final["source"]["commit"], INITIAL_COMMIT)
        self.assertEqual(final["source"]["production_diff_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(final["source"]["source_sha256"], build["source_sha256"])
        self.assertEqual(final["installed_native_sha256"], build["extension_sha256"])
        self.assertIn("?? scripts/diagnose_compile_cuda_neg_add.py", final["git_status"])
        self.assertIn(" M docs/compile-cuda-neg-validation.md", final["git_status"])
        self.assertTrue(build["python"].startswith("3.12.14+meta "))
        self.assertEqual(build["executable"], start["root"] + "/.venv/bin/python")
        baseline = read_json(INITIAL / "baseline-build-record.json")
        self.assertEqual(baseline["source_revision"], COMMITS["baseline"])
        for name, version, native in [
            ("managed312", "3.12.14 ", build["extension_sha256"]),
            ("baseline-managed312", "3.12.14 ", baseline["native_sha256"]),
            ("python314", "3.14.7 ", build["extension_sha256"]),
        ]:
            env = read_json(INITIAL / f"{name}-environment.log")
            self.assertTrue(env["python"].startswith(version))
            self.assertEqual(env["native_sha256"], native)
            self.assertTrue(env["executable"].startswith(start["root"] + "/"))
        for name, record in [("python-full", "checks-record.json"),
                             ("python314-full", "cross-interpreter-record.json")]:
            self.assert_suite_summary(name, 5331, "OK (skipped=11)")
            self.assertEqual(receipts(record)[name]["exit_code"], 0)

    def test_boolean_buffer_and_factory_reproductions(self):
        cross = receipts("cross-interpreter-record.json")
        for name in ("managed312-known-failures", "baseline-managed312-known-failures"):
            self.assert_suite_summary(name, 13, "FAILED (failures=2)")
            self.assertEqual(cross[name]["exit_code"], 1)
            failures = re.findall(r"^FAIL: (.+)$", (INITIAL / f"{name}.log").read_text(), re.M)
            self.assertEqual(len(failures), 2)
            for failure, case in zip(failures, ("bool including noncanonical bytes", "native-prefixed bool")):
                self.assertIn("test_numeric_buffers_match_pytorch_2_13", failure)
                self.assertIn(f"case='{case}'", failure)
        seeds = receipts("factory-seed-record.json")
        self.assertEqual(set(seeds), {f"factory-{v}-seed{s}" for v in COMMITS for s in range(10)})
        for name, record in seeds.items():
            self.assertEqual(record["PYTHONHASHSEED"], name[-1])
            self.assertEqual(record["exit_code"], 0)
            self.assert_suite_summary(name, 5, "OK")
        cache = receipts("factory-cache-record.json")
        self.assertEqual(len(cache), 4)
        for variant in COMMITS:
            warm = cache[f"factory-{variant}-cache-warm-seed0"]
            test = cache[f"factory-{variant}-cache-test-seed6"]
            self.assertEqual((warm["PYTHONHASHSEED"], test["PYTHONHASHSEED"]), ("0", "6"))
            self.assertEqual((warm["exit_code"], test["exit_code"]), (0, 1))
            self.assertEqual(warm["PYTHONPYCACHEPREFIX"], test["PYTHONPYCACHEPREFIX"])
            self.assertIsNone(warm["PYTHONDONTWRITEBYTECODE"])
            self.assertEqual(test["PYTHONDONTWRITEBYTECODE"], "1")
            self.assert_suite_summary(test["name"], 5, "FAILED (failures=3)")

    def assert_timing_accounting(self, report, expected_score):
        aggregate = report["aggregates"]
        env = report["environment"]
        self.assertEqual((env["warmups"], env["samples"], env["repeats"]), (5, 17, 3))
        self.assertEqual(env["cuda_visible_devices"], "0")
        self.assertEqual(env["pytorch"]["version"], "2.13.0+cu130")
        self.assertEqual(env["pytorch"]["cuda_runtime"], "13.0")
        self.assertEqual(env["gpu"]["torch_cuda_device_name"], "NVIDIA H100")
        self.assertEqual(env["gpu"]["nvidia_smi"][0]["driver_version"], "580.82.07")
        self.assertIn("V12.6.85", env["gpu"]["nvcc_version"])
        self.assertEqual(aggregate["workload_count"], 4)
        self.assertEqual(aggregate["common_success_shape_count"], 4)
        self.assertEqual(aggregate["denominator_weight"], 1)
        self.assertEqual(aggregate["zero_credit_cell_count"], 0)
        self.assertEqual(aggregate["coverage_adjusted_overall_percent"], expected_score)
        raw = report["workload_results"]
        rows = aggregate["per_workload"]
        self.assertEqual([w["workload"]["name"] for w in raw], list(SHAPES))
        self.assertEqual([w["name"] for w in rows], list(SHAPES))
        contributions = []
        for result, row in zip(raw, rows):
            workload = result["workload"]
            self.assertEqual(workload["shape"], SHAPES[row["name"]])
            self.assertEqual(workload["weight"], 0.25)
            candidate, reference = result["candidate"], result["reference_workload"]
            self.assertEqual((candidate["status"], reference["status"]), ("ok", "ok"))
            self.assertTrue(candidate["eligibility"]["eligible_cuda_compile_evidence"])
            self.assertTrue(candidate["correctness"]["exact_checksum_match"])
            self.assertTrue(reference["correctness"]["assert_close"])
            self.assertEqual(candidate["steady_checksums"], [reference["cold_checksum"]])
            native_us = candidate["steady"]["median_us"]
            reference_us = reference["steady"]["median_us"]
            self.assertEqual(row["torch_rs_steady_median_us"], native_us)
            self.assertEqual(row["pytorch_steady_median_us"], reference_us)
            self.assertEqual(row["candidate_status"], "ok")
            self.assertTrue(row["eligible_cuda_compile_evidence"])
            contribution = 25 * min(1, reference_us / native_us)
            self.assertAlmostEqual(row["weighted_score_contribution_percent"], contribution)
            self.assertAlmostEqual(result["score"]["weighted_score_contribution_percent"], contribution)
            contributions.append(contribution)
        self.assertAlmostEqual(sum(contributions), expected_score)

    def test_all_four_initial_timings(self):
        expected = {"fresh": 100, "warm": 92.10364622089932,
                    "confirm-1": 100, "confirm-2": 79.30190144231703}
        commands = receipts("checks-record.json") | receipts("final-gpu-record.json")
        self.assertEqual({p.stem for p in INITIAL.glob("cuda-performance-*.json")},
                         {f"cuda-performance-{s}" for s in expected})
        for suffix, score in expected.items():
            name = f"cuda-performance-{suffix}"
            with self.subTest(run=name):
                report = read_json(INITIAL / f"{name}.json")
                self.assert_timing_accounting(report, score)
                env = report["environment"]
                self.assertEqual(env["git"]["head"], INITIAL_COMMIT)
                self.assertTrue(env["git"]["status_short"])
                self.assertTrue(env["python"].startswith("3.12.14+meta "))
                self.assertEqual(commands[name]["exit_code"], 0)

    def test_initial_compile_math_and_diagnostic_accounting(self):
        compile_report = read_json(INITIAL / "compile-evaluation.json")
        self.assertEqual(compile_report["score"], 100)
        self.assertIn("38/38 reference-eligible full corpus cases", compile_report["summary"])
        case_list = next(s for s in compile_report["evidence"] if s.startswith("Public cases:"))
        self.assertEqual(len(set(re.findall(r"cpu_float32_\w+", case_list))), 38)
        math = read_json(INITIAL / "cuda-math.json")
        self.assertEqual(math["seeds"], [9173, 260909, 903217])
        accounting = math["accounting"]
        self.assertEqual((accounting["denominator"], accounting["passed"]), (6, 2))
        supported = {"cuda_f32_add_same_shape", "cuda_f32_neg"}
        self.assertEqual(len(accounting["cases"]), 6)
        trials = {(t["case_id"], t["seed"]): t for t in math["trials"]}
        self.assertEqual(len(trials), 18)
        for case in accounting["cases"]:
            credit = int(case["case_id"] in supported)
            self.assertEqual(case["credit"], credit)
            self.assertEqual([t["seed"] for t in case["trials"]], math["seeds"])
            for trial in case["trials"]:
                self.assertEqual(trial["credit"], credit)
                raw = trials[(case["case_id"], trial["seed"])]
                self.assertEqual(raw["reference"]["status"], "passed")
                self.assertEqual(raw["candidate"]["status"], "passed" if credit else "failed")
        for suffix, passes, unsupported, eligible in [("single", 128, 40, 164), ("multi", 8, 4, 8)]:
            report = read_json(INITIAL / f"diagnostic-{suffix}.json")
            cases = report["cases"]
            self.assertEqual(len(cases), passes + unsupported)
            self.assertEqual(Counter(c["native_outcome"] for c in cases),
                             {"pass": passes, "unsupported": unsupported})
            self.assertTrue(all(c["expectation_met"] for c in cases))
            self.assertEqual(sum(c["reference_eligible"] for c in cases), eligible)
            self.assertEqual(report["summary"], dict(cases=len(cases), reference_eligible=eligible,
                             native_pass=passes, native_unsupported=unsupported, expectation_failures=0))

    def test_paired_receipts_order_scores_and_ratios(self):
        plan = read_json(PAIRED / "plan.json")
        summary = read_json(PAIRED / "summary.json")
        results = read_json(PAIRED / "results.json")
        self.assertEqual(plan["order"], ORDER)
        self.assertEqual(plan["commits"], COMMITS)
        self.assertEqual((plan["runs"], plan["cpu"], plan["gpu"]), (6, 24, 0))
        self.assertTrue(plan["no_retry_or_selection"])
        self.assertEqual([r["variant"] for r in results], ORDER)
        builds = {v: read_json(PAIRED / v / "target/paired/reports/build-record.json") for v in COMMITS}
        for key in ("benchmark_sha256", "lock_sha256"):
            self.assertEqual(builds["baseline"][key], builds["candidate"][key])
        native_times = {v: {name: [] for name in SHAPES} for v in COMMITS}
        for number, row in enumerate(results, 1):
            variant = row["variant"]
            directory = PAIRED / variant / "target/paired/reports"
            build = builds[variant]
            report_path = directory / f"run-{number}.json"
            report = read_json(report_path)
            receipt = read_json(directory / f"run-{number}.receipt.json")
            self.assertEqual(row["number"], number)
            self.assertEqual(row["receipt"], receipt)
            self.assertEqual(row["report_sha256"], sha256(report_path))
            self.assertEqual(receipt["log_sha256"], sha256(directory / f"run-{number}.log"))
            self.assertEqual(receipt["exit_code"], 0)
            self.assertEqual(receipt["command"][:3], ["taskset", "-c", "24"])
            self.assertEqual(row["aggregates"], report["aggregates"])
            self.assertEqual(row["environment"], report["environment"])
            self.assertEqual(row["native_sha256"], build["native_sha256"])
            for key, value in row["source"].items():
                self.assertEqual(build[key], value)
            self.assertEqual(build["commit"], COMMITS[variant])
            self.assertEqual(build["git_status"], "")
            versions = build["versions"]
            self.assertTrue(versions["python"].startswith("3.12.12 "))
            self.assertEqual(versions["numpy"], "2.5.1")
            self.assertEqual(versions["torch"], "2.13.0+cu130")
            self.assertEqual(receipt["command"][3], versions["executable"])
            self.assertEqual(report["environment"]["python"], versions["python"])
            self.assertEqual(report["environment"]["git"]["head"], COMMITS[variant])
            self.assertEqual(report["environment"]["git"]["status_short"], "")
            self.assert_timing_accounting(report, 100)
            for workload in report["workload_results"]:
                native_times[variant][workload["workload"]["name"]].append(workload["candidate"]["steady"]["median_us"])
        self.assertEqual(summary["order"], ORDER)
        self.assertEqual(summary["scores"], [100] * 6)
        self.assertEqual(summary["runs"], 6)
        self.assertEqual(summary["reference_eligible_native_passes"], 24)
        self.assertEqual([w["name"] for w in summary["workloads"]], list(SHAPES))
        ratios = [0.9430325515, 0.9650679390, 0.9739054414, 1.0039280991]
        for row, expected in zip(summary["workloads"], ratios):
            baseline = median(native_times["baseline"][row["name"]])
            candidate = median(native_times["candidate"][row["name"]])
            self.assertEqual(row["baseline_native_us"], baseline)
            self.assertEqual(row["candidate_native_us"], candidate)
            self.assertEqual(row["candidate_over_baseline"], candidate / baseline)
            self.assertAlmostEqual(candidate / baseline, expected, places=10)
            self.assertAlmostEqual(row["percent_change"], 100 * (candidate / baseline - 1))

    def test_documentation_links(self):
        owner = ROOT / "docs/compile-cuda-neg-validation.md"
        guide = HISTORY / "pr1924-history.md"
        self.assertIn("(diagnostics/compile-cuda-neg/pr1924-history.md)", owner.read_text())
        for document in (owner, guide):
            links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", document.read_text())
            self.assertTrue(links)
            for link in links:
                with self.subTest(document=document.name, link=link):
                    parsed = urlsplit(link)
                    if parsed.scheme or parsed.netloc:
                        continue
                    target = (document.parent / unquote(parsed.path)).resolve()
                    self.assertTrue(target.is_relative_to(ROOT))
                    self.assertTrue(target.is_file(), link)
                    if parsed.fragment:
                        headings = re.findall(r"^#+\s+(.+)$", target.read_text(), re.M)
                        anchors = {re.sub(r"[^\w\- ]", "", h.lower()).replace(" ", "-") for h in headings}
                        self.assertIn(unquote(parsed.fragment), anchors)


if __name__ == "__main__":
    unittest.main()
