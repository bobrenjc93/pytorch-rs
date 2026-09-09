"""Current diagnostic accounting; frozen addition-only expectations stay intact."""
import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tests.test_cuda_add import available, torch

if torch is not None:
    from scripts import diagnose_compile_cuda_neg_add as diagnostic
else:
    diagnostic = None

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/diagnose_compile_cuda_neg_add.py"


def documented_commands(name):
    text = (ROOT / "docs" / name).read_text().replace("\\\n", " ")
    return [shlex.split(line) for line in text.splitlines()
            if SCRIPT in line and ".venv/bin/python" in line]


@unittest.skipUnless(diagnostic is not None, "requires optional PyTorch reference dependency")
class DiagnosticAccountingTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "target").mkdir(exist_ok=True)

    def test_documented_commands_preserve_raw_results_and_exit_status(self):
        outputs = [{"sha256": "first"}, {"sha256": "second"}]
        supported = {"program": "negate", "expected_supported": True,
                     "reference_eligible": True, "native_outcome": "pass",
                     "reference_outputs": outputs, "native_outputs": outputs}
        unsupported = {"program": "reject_detach", "expected_supported": False,
                       "reference_eligible": True, "native_outcome": "unsupported",
                       "native_error": "NotImplementedError: CUDA unary"}
        scenarios = [
            (supported, 0), (unsupported, 0),
            ({**unsupported, "reference_eligible": False}, 1),
            ({**unsupported, "reference_eligible": False, "kind": "mixed_ordinals", "shape": [19]}, 0),
            ({**unsupported, "reference_eligible": False, "kind": "mixed_cpu", "shape": []}, 1),
            ({**supported, "native_outcome": "unsupported"}, 1),
            ({**supported, "native_outputs": [{"sha256": "wrong"}]}, 1),
            ({**supported, "reference_eligible": False}, 1),
            ({**supported, "native_outputs": [], "reference_outputs": []}, 1),
            ({**unsupported, "native_outcome": "pass"}, 1),
            ({**unsupported, "native_outcome": "error", "native_error": "launch failed"}, 1),
            ({**supported, "native_outcome": "mismatch"}, 1),
        ]
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            output = Path(directory) / "results.json"
            for guide in ("troubleshooting.md", "compile-cuda-add.md", "compile-cuda-neg-validation.md"):
                commands = documented_commands(guide)
                self.assertTrue(commands, guide)
                for command in commands:
                    argv = command[command.index(SCRIPT) + 1:]
                    self.assertEqual(argv[argv.index("--case-set") + 1], "neg_add_v1")
                    argv[argv.index("--output") + 1] = str(output)
                    for record, expected_exit in scenarios:
                        with self.subTest(guide=guide, record=record), \
                             patch.object(diagnostic, "environment", return_value={}), \
                             patch.object(diagnostic, "cases", return_value=[(diagnostic.negate, (3,), 1, "offset", True)]), \
                             patch.object(diagnostic.frozen, "run_case", side_effect=lambda *args: dict(record)), \
                             contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(diagnostic.main(argv), expected_exit)
                            report = json.loads(output.read_text())
                            self.assertEqual(report["summary"]["cases"], 2)
                            self.assertEqual(report["summary"]["expectation_failures"], 2 * expected_exit)
                            for raw in report["cases"]:
                                self.assertEqual({k: v for k, v in raw.items() if k != "expectation_met"}, record)

    def test_empty_matrix_and_provenance_errors_cannot_succeed(self):
        with patch.object(diagnostic, "environment", side_effect=RuntimeError("source mismatch")):
            with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                diagnostic.main(["--case-set", "neg_add_v1", "--output", "unused.json"])
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory, \
             patch.object(diagnostic, "environment", return_value={}), \
             patch.object(diagnostic, "cases", return_value=[]), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(diagnostic.main(["--case-set", "neg_add_v1", "--output",
                                              str(Path(directory) / "empty.json")]), 1)

    def test_versioned_matrix_retains_addition_and_unsupported_guards(self):
        matrix = diagnostic.cases("0")
        positives = {p.__name__ for p, _, _, _, supported in matrix if supported}
        self.assertEqual(positives, {"silver", "orchard", "estuary", "with_global",
                                     "negate", "neg_method", "negative_method", "compose"})
        negatives = {(p.__name__, kind) for p, _, _, kind, supported in matrix if not supported}
        for name in ("reject_detach", "reject_float", "reject_multiply", "reject_reduce",
                     "reject_scalar", "reject_keyword"):
            self.assertIn((name, "plain"), negatives)
        for kind in ("transpose", "broadcast", "gradient", "float64", "mixed_cpu"):
            self.assertIn(("silver", kind), negatives)
            self.assertIn(("compose", kind), negatives)
        self.assertIn((diagnostic.compose, (19,), 2, "mixed_ordinals", False), diagnostic.cases("0,1"))

    @unittest.skipUnless(available("0"), "requires real CUDA with CUDA_VISIBLE_DEVICES=0")
    def test_documented_command_on_real_cuda(self):
        command, = documented_commands("troubleshooting.md")
        # Run the documented interpreter, keeping its provenance guard intact.
        command = command[1:]  # CUDA_VISIBLE_DEVICES is passed in the environment.
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            output = Path(directory) / "gpu.json"
            command[command.index("--output") + 1] = str(output)
            result = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report["case_set"], "neg_add_v1")
            self.assertEqual(report["summary"]["expectation_failures"], 0)
            self.assertTrue(all(r["expectation_met"] for r in report["cases"]))


class DiagnosticDocumentationTests(unittest.TestCase):
    def test_troubleshooting_supported_boundary_and_report_navigation(self):
        text = (ROOT / "docs/troubleshooting.md").read_text()
        self.assertNotIn("compiled CUDA negation", text)
        self.assertIn("Bounded eager native neg/add capture", text)
        self.assertIn("[capture guide](compile-cuda-add.md)", text)
        for boundary in ("Noncontiguous CUDA negation", "other CUDA math", "CUDA autograd",
                         "asynchronous transfers", "dtype changes", "unindexed CUDA targets",
                         "nondefault\nstreams", "general CUDA runtime management"):
            self.assertIn(boundary, text)
        self.assertIn("(compile-cuda-neg-validation.md)", (ROOT / "docs/README.md").read_text())


if __name__ == "__main__":
    unittest.main()
