"""Portable setup-disclosure checks; never run benchmark workloads."""

import copy
import datetime
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts import validate_benchmark_setup as setup


class BenchmarkSetupEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((setup.ROOT / setup.EVIDENCE).read_text())

    def test_checked_in_disclosure_and_all_report_references(self):
        setup.validate(self.record)

    def test_rejects_incomplete_or_misattributed_setup(self):
        mutations = {
            "wrong schema": lambda r: r.update(schema_version=2),
            "boolean schema": lambda r: r.update(schema_version=True),
            "original run claim": lambda r: r.update(measurement_kind="original-run"),
            "invented original time": lambda r: r.update(original_setup_timings=0),
            "missing compile": lambda r: r["steps"].pop(4),
            "missing dependencies": lambda r: r["steps"].pop(2),
            "duplicate stage": lambda r: r["steps"].append(r["steps"][0]),
            "failed install": lambda r: r["steps"][2].update(exit_status=1),
            "boolean status": lambda r: r["steps"][2].update(exit_status=False),
            "missing command": lambda r: r["steps"][4].update(command=[]),
            "missing cache state": lambda r: r["steps"][4].pop("cache_state"),
            "missing environment": lambda r: r["environment"].pop("CARGO_TARGET_DIR"),
            "missing wheel": lambda r: r.pop("wheel"),
            "missing lock hash": lambda r: r["source"]["files_sha256"].pop("uv.lock"),
            "modified source": lambda r: r["source"].update(files_unchanged_after_setup=False),
            "missing report": lambda r: r["reports"].pop(next(iter(setup.REPORTS))),
            "wrong artifact": lambda r: r["reports"][next(iter(setup.REPORTS))].update(artifact="other.json"),
            "tampered log hash": lambda r: r["steps"][2].update(log_sha256="0" * 64),
            "missing log": lambda r: r["steps"][2].update(log="logs/missing.txt"),
            "escaping log": lambda r: r["steps"][2].update(log="../setup.json"),
            "absolute log": lambda r: r["steps"][2].update(log=str(setup.ROOT / setup.EVIDENCE)),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                record = copy.deepcopy(self.record)
                mutate(record)
                with self.assertRaises(ValueError):
                    setup.validate(record)

    def test_rejects_invalid_durations_and_timestamps(self):
        for seconds in (None, True, "45.1", -1, 0, float("nan"), float("inf"), 9999):
            with self.subTest(seconds=seconds):
                record = copy.deepcopy(self.record)
                record["steps"][4]["wall_seconds"] = seconds
                with self.assertRaises(ValueError):
                    setup.validate(record)
        for timestamp in ("bad", "2026-09-09T17:30:03", "2020-01-01T00:00:00+00:00"):
            with self.subTest(timestamp=timestamp):
                record = copy.deepcopy(self.record)
                record["steps"][4]["started_at"] = timestamp
                with self.assertRaises(ValueError):
                    setup.validate(record)

    def test_rerun_cannot_be_backdated_to_original_capture(self):
        record = copy.deepcopy(self.record)
        for step in record["steps"]:
            for name in ("started_at", "ended_at"):
                timestamp = datetime.datetime.fromisoformat(step[name])
                step[name] = (timestamp - datetime.timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(ValueError, "must postdate"):
            setup.validate(record)

    def test_source_commit_must_match_historical_artifacts(self):
        record = copy.deepcopy(self.record)
        record["source"]["commit"] = "a" * 40
        record["source"]["archive_command"][-1] = "a" * 40
        with self.assertRaisesRegex(ValueError, "setup source differs"):
            setup.validate(record)

    def test_report_links_and_historical_measurements_are_checked(self):
        # Fixtures stay inside the worktree, including on hosts with external TMPDIR.
        target = setup.ROOT / "target"
        target.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target, prefix="setup-schema-") as directory:
            root = Path(directory)
            shutil.copytree(setup.ROOT / setup.EVIDENCE.parent, root / setup.EVIDENCE.parent)
            for report, artifact in setup.REPORTS.items():
                for name in (report, artifact):
                    destination = root / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(setup.ROOT / name, destination)
            setup.validate(self.record, root)
            for report in setup.REPORTS:
                path = root / report
                original = path.read_text()
                with self.subTest(report=report):
                    path.write_text(original.replace(
                        setup.EVIDENCE.relative_to("docs").as_posix(), "missing.json"
                    ))
                    with self.assertRaisesRegex(ValueError, "missing labeled setup reference"):
                        setup.validate(self.record, root)
                    path.write_text(original + "\nAltered historical results.\n")
                    with self.assertRaisesRegex(ValueError, "historical results changed"):
                        setup.validate(self.record, root)
                    path.write_text(original)
            for artifact in setup.REPORTS.values():
                path = root / artifact
                original = path.read_bytes()
                with self.subTest(artifact=artifact):
                    path.write_bytes(original + b"\n")
                    with self.assertRaisesRegex(ValueError, "historical workload artifact changed"):
                        setup.validate(self.record, root)
                    path.write_bytes(original)
            log = root / setup.EVIDENCE.parent / self.record["steps"][2]["log"]
            log.write_text("Replaced dependency log.\n")
            with self.assertRaisesRegex(ValueError, "log hash mismatch"):
                setup.validate(self.record, root)


if __name__ == "__main__":
    unittest.main()
