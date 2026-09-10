"""Portable integrity audit of the original PR #1930 evidence; no timing reruns."""

import hashlib
import itertools
import json
from pathlib import Path, PurePosixPath
import statistics
import unittest


BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "docs/diagnostics/boolean-buffer/source-pr-1930"
)
BASE_NATIVE = "2666e69aefda6dea8024972d5ef444d94162a83398cc3c30c1f0ad8c678dc8bf"
FIXED_NATIVE = "326644903ebfc0408dfe6ab55a1468d52c5b2aaf0649322ad5f61603844818aa"


class BooleanBufferEvidenceTests(unittest.TestCase):
    def test_original_payload_checksums_and_membership(self):
        recorded = {}
        for line in (BUNDLE / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            path = PurePosixPath(name)
            self.assertFalse(path.is_absolute())
            self.assertNotIn("..", path.parts)
            self.assertEqual(path.parts[0], "target")
            self.assertNotIn(name, recorded)
            self.assertIn(path.suffix, (".py", ".json", ".log"))
            payload = BUNDLE / name
            self.assertFalse(payload.is_symlink())
            self.assertEqual(hashlib.sha256(payload.read_bytes()).hexdigest(), digest)
            recorded[name] = digest
        self.assertEqual(len(recorded), 26)
        self.assertEqual(
            set(recorded),
            {p.relative_to(BUNDLE).as_posix()
             for p in (BUNDLE / "target").rglob("*") if p.is_file()},
        )

    def test_complete_historical_matrix_and_build_identity(self):
        expected = set(itertools.product(
            (16, 4096, 1048576),
            ("canonical", "sparse", "dense"),
            ("contiguous", "reversed", "strided"),
        ))
        measurements = sorted((BUNDLE / "target").glob("*.json"))
        self.assertEqual(len(measurements), 9)
        for path in measurements:
            with self.subTest(file=path.name):
                data = json.loads(path.read_text())
                repaired = path.name.startswith("after-")
                self.assertEqual(data["native_sha256"], FIXED_NATIVE if repaired else BASE_NATIVE)
                rows = data["rows"]
                self.assertEqual(len(rows), 27)
                self.assertEqual(
                    {(r["size"], r["pattern"], r["layout"]) for r in rows}, expected,
                )
                wrong = 17 if ".venv" in path.name and not repaired else 0
                self.assertEqual(sum(not r["correct"]["native"] for r in rows), wrong)
                for row in rows:
                    self.assertTrue(row["correct"]["torch"])
                    for implementation in ("native", "torch"):
                        timing = row["timings"][implementation]
                        samples = timing["samples_us"]
                        self.assertEqual(len(samples), 10)
                        self.assertTrue(all(x > 0 for x in samples))
                        self.assertEqual(timing["median_us"], statistics.median(samples))
                        self.assertEqual(timing["min_us"], min(samples))
                        self.assertEqual(timing["max_us"], max(samples))
        verification = json.loads(
            (BUNDLE / "target/post-commit-evidence/verification.json").read_text()
        )
        self.assertEqual(verification["verified_commit"], "ed078e1a2dec6ad7c01f6c74c36cce52165d6fc3")
        self.assertEqual(verification["native_sha256"], FIXED_NATIVE)


if __name__ == "__main__":
    unittest.main()
