"""Hardware-free checks for retention of failed companion captures."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


class CaptureFailureRetention(unittest.TestCase):
    def test_provenance_failures_do_not_discard_primary_failure(self):
        spec = importlib.util.spec_from_file_location(
            "integration_capture", ROOT / "docs/diagnostics/gelu-program-integration/capture.py")
        capture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(capture)
        consumer = types.SimpleNamespace(
            gpu_identity=Mock(side_effect=RuntimeError("injected final GPU query failure")),
            libraries=Mock(side_effect=RuntimeError("injected final library query failure")))
        fake_spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target") as directory:
            output = Path(directory) / "partial.json"
            with (patch.object(capture, "packages", side_effect=RuntimeError("injected package failure")),
                  patch.object(capture.importlib.util, "spec_from_file_location", return_value=fake_spec),
                  patch.object(capture.importlib.util, "module_from_spec", return_value=consumer),
                  patch.object(sys, "argv", ["capture.py", str(ROOT / "unused.whl"), str(ROOT / "unused.tar.gz"), str(output)]),
                  patch.dict(sys.modules)):
                # The isolated capture itself forbids importing the reference.
                sys.modules.pop("torch", None)
                with self.assertRaisesRegex(RuntimeError, "injected package failure"):
                    capture.main()
            record = json.loads(output.read_text())
            self.assertFalse(record["passed"])
            self.assertIn("injected package failure", record["failure"])
            self.assertIn("injected final GPU query failure", record["gpuAfterFailure"])
            self.assertIn("injected final library query failure", record["librariesFailure"])


if __name__ == "__main__":
    unittest.main()
