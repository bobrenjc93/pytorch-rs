"""Portable integrity checks for the checked-in historical compiler evidence."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


EVIDENCE = Path(__file__).resolve().parents[1] / "docs/diagnostics/compile-pointwise-jit"
SPEC = importlib.util.spec_from_file_location("verify_compile_evidence", EVIDENCE / "verify_archive.py")
ARCHIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ARCHIVE)


class EvidenceArchive(unittest.TestCase):
    def fixture(self, directory, members=("capture/report.json",), listed=("capture/report.json",)):
        payload = b'{"valid":false,"original_failure":"preserved"}\n'
        path = directory / "history.tar.gz"
        with tarfile.open(path, "w:gz") as bundle:
            for name in members:
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
        data = path.read_bytes()
        manifest = {
            "schema": 1, "source_commit": "test-fixture",
            "archive": {"path": path.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()},
            "files": {name: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
                      for name in listed},
        }
        (directory / "history-manifest.json").write_text(json.dumps(manifest))
        return manifest

    def test_committed_archive(self):
        manifest = ARCHIVE.verify(EVIDENCE)
        self.assertEqual(manifest["source_commit"], "ed6ad9eaaea17ee500f5464183c00984115e7fb3")
        self.assertEqual(len(manifest["files"]), 56)

    def test_committed_later_archive(self):
        manifest = ARCHIVE.verify(EVIDENCE / "history-later")
        self.assertEqual(manifest["source_commit"], "91fd9524fb15a8d46e7c165dd3c87de763826921")
        self.assertEqual(len(manifest["files"]), 56)

    def test_default_command_verifies_both_archives(self):
        result = subprocess.run([sys.executable, str(EVIDENCE / "verify_archive.py")],
                                capture_output=True, text=True, check=True)
        self.assertEqual(len(result.stdout.splitlines()), 2)
        self.assertIn("56 unchanged evidence files from ed6ad9eaaea17ee500f5464183c00984115e7fb3", result.stdout)
        self.assertIn("56 unchanged evidence files from 91fd9524fb15a8d46e7c165dd3c87de763826921", result.stdout)

    def test_collection_requires_later_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with self.assertRaises(FileNotFoundError):
                ARCHIVE.verify_all(directory)

    def test_archive_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory)
            with (directory / "history.tar.gz").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "archive hash or size mismatch"):
                ARCHIVE.verify(directory)

    def test_member_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = self.fixture(directory)
            manifest["files"]["capture/report.json"]["sha256"] = "0" * 64
            (directory / "history-manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "member hash or size mismatch"):
                ARCHIVE.verify(directory)

    def test_missing_duplicate_and_unsafe_members_rejected(self):
        for members, listed in (
                ((), ("capture/report.json",)),
                (("capture/report.json",) * 2, ("capture/report.json",)),
                (("unexpected",), ("capture/report.json",)),
                (("../escape",), ("../escape",)),
                (("/absolute",), ("/absolute",))):
            with self.subTest(members=members), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                self.fixture(directory, members, listed)
                with self.assertRaises(ValueError):
                    ARCHIVE.verify(directory)


TENSOR_MADD = EVIDENCE.parent / "compile-pointwise-tensor-madd"
MADD_SOURCE = "48fb3588c692db5b2a4f69033b5bf8a2bebb2019"
MADD_INVENTORY_SHA256 = "08da41c49c979077c67d2305341e149f41bccb28d984e4a045b714584b96905a"
MADD_REPORTS = (
    "postcommit-bf908578/fixed/candidate/run-20260913T173018Z-d4056067/report.json",
    "postcommit-bf908578/fixed/main/run-20260913T173516Z-6151e3a3/report.json",
)
MADD_WHEELS = (
    "postcommit-bf908578/build/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl",
    "postcommit-bf908578/fixed/candidate/wheel/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl",
    "postcommit-bf908578/fixed/main/wheel/torch_rs-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl",
)


class TensorMaddArchive(unittest.TestCase):
    """Mandatory checks use only checked-in bytes, never Git or the network."""

    def setUp(self):
        raw = (TENSOR_MADD / "original-inventory.json").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), MADD_INVENTORY_SHA256)
        self.inventory = json.loads(raw)
        # Use the canonical owner explicitly; its two-collection CLI is unchanged.
        self.manifest = ARCHIVE.verify(TENSOR_MADD)

    def test_exact_original_inventory_partition_and_safe_members(self):
        self.assertEqual(self.inventory["sourceCommit"], MADD_SOURCE)
        self.assertEqual(self.inventory["sourcePrefix"], "docs/diagnostics/compile-pointwise-tensor-madd/")
        self.assertEqual(self.manifest["source_commit"], MADD_SOURCE)
        originals = {f["path"]: f for f in self.inventory["files"]}
        self.assertEqual(len(originals), 137)
        self.assertEqual(sum(f["size"] for f in originals.values()), 9666017)
        self.assertEqual({f["path"] for f in self.inventory["wheels"]}, set(MADD_WHEELS))
        self.assertEqual(set(self.manifest["files"]), set(originals) - set(MADD_WHEELS))
        self.assertEqual(len(self.manifest["files"]), 134)
        for name, member in self.manifest["files"].items():
            self.assertEqual(member, {k: originals[name][k] for k in ("size", "sha256")})
        wheel_hashes = {f["sha256"] for f in self.inventory["wheels"]}
        with tarfile.open(TENSOR_MADD / "history.tar.gz", "r:gz") as archive:
            for member in archive:
                self.assertTrue(member.isfile())
                self.assertFalse(member.name.endswith(".whl"))
                payload = archive.extractfile(member).read()
                self.assertNotIn(hashlib.sha256(payload).hexdigest(), wheel_hashes)
        for current in TENSOR_MADD.rglob("*"):
            if current.is_file():
                self.assertNotEqual(current.suffix, ".whl")
                self.assertNotIn(hashlib.sha256(current.read_bytes()).hexdigest(), wheel_hashes)
        self.assertFalse((TENSOR_MADD / "sha256.json").exists())
        self.assertFalse((TENSOR_MADD / "postcommit-bf908578/sha256.json").exists())

    def test_wheel_provenance_schema_and_original_build_bindings(self):
        provenance = json.loads((TENSOR_MADD / "wheel-provenance.json").read_text())
        self.assertEqual(provenance["schema"], 1)
        self.assertEqual(provenance["source_commit"], MADD_SOURCE)
        self.assertEqual(provenance["inventory_sha256"], MADD_INVENTORY_SHA256)
        self.assertEqual(len(provenance["wheels"]), 3)
        bindings = ("postcommit-bf908578/build/build.json", *MADD_REPORTS)
        with tarfile.open(TENSOR_MADD / "history.tar.gz", "r:gz") as archive:
            for actual, original, binding in zip(provenance["wheels"], self.inventory["wheels"], bindings):
                self.assertEqual(actual, dict(original, source_commit=MADD_SOURCE))
                self.assertRegex(actual["gitBlob"], r"^[0-9a-f]{40}$")
                self.assertRegex(actual["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(actual["size"], 0)
                self.assertEqual(actual["url"], "https://github.com/bobrenjc93/pytorch-rs/blob/"
                                 + MADD_SOURCE + "/docs/diagnostics/compile-pointwise-tensor-madd/" + actual["path"])
                built = json.loads(archive.extractfile(binding).read())["wheel"]
                self.assertEqual(built["sha256"], actual["sha256"])
                self.assertEqual(Path(built["path"]).name, Path(actual["path"]).name)

    def test_visible_capture_and_fixed_summaries_match_archived_originals(self):
        with tarfile.open(TENSOR_MADD / "history.tar.gz", "r:gz") as archive:
            for name in ("capture.py", *MADD_REPORTS):
                payload = (TENSOR_MADD / name).read_bytes()
                self.assertEqual(payload, archive.extractfile(name).read())
                self.assertEqual(hashlib.sha256(payload).hexdigest(), self.manifest["files"][name]["sha256"])
            for name in ("README.md", "postcommit-bf908578/README.md"):
                self.assertTrue((TENSOR_MADD / name).is_file())
                self.assertIn(name, self.manifest["files"])
        for name, commit in zip(MADD_REPORTS, (
                "bf9085787aca29e523a7a38773f20f965167e734",
                "77aa16fc2d0cbd258f0fa309140dc75708cf2ee3")):
            report = json.loads((TENSOR_MADD / name).read_text())
            self.assertEqual(report["provenance"]["commit"], commit)
            self.assertEqual(report["provenance"]["working_tree_changes"], "")
            self.assertTrue(report["valid"])
            self.assertEqual((report["warmups"], report["samples"]), (5, 17))
            self.assertEqual(len(report["cells"]), 112)
            self.assertEqual(sum(c["device"] == "cuda" for c in report["cells"]), 56)
            self.assertEqual(len(report["workers"]), 6)
        for name, digest in (
                ("operator-result.json", "143f9b2c29d5eac47b32f55b9843d7d0cdc85cee43943f0852fe699a755c6807"),
                ("operator-audit.json", "0f982e2b4c8229a7f0436517a93638ed9be245446c25ac26b78d9e49f55a6338")):
            self.assertEqual(hashlib.sha256((TENSOR_MADD / name).read_bytes()).hexdigest(), digest)
        audit = json.loads((TENSOR_MADD / "operator-audit.json").read_text())
        self.assertEqual(audit["producerExit"]["exitCode"], 2)
        self.assertFalse(audit["producerQualified"])
        self.assertEqual([(s["id"], s["score"]) for s in audit["polishSamples"]], [
            ("evalrun_a9316d55", 82), ("evalrun_7f0fe688", 82), ("evalrun_53ed1eb0", 82)])


class TensorMaddGitHistory(unittest.TestCase):
    """Historical blobs are an additional check, explicitly unavailable offline."""

    def require_history(self, root):
        import os
        import shutil
        if not (root / ".git").exists() or shutil.which("git") is None:
            self.skipTest("historical Git bytes unavailable: no .git or Git executable; no fetch attempted")
        def git(*args):
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                  env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
        shallow = git("rev-parse", "--is-shallow-repository")
        self.assertEqual(shallow.returncode, 0, shallow.stderr)
        if shallow.stdout.strip() == b"true":
            self.skipTest("historical Git bytes unavailable in shallow checkout; no fetch attempted")
        if git("cat-file", "-e", MADD_SOURCE + "^{commit}").returncode:
            self.skipTest("historical source commit unavailable locally; no fetch attempted")
        return git

    def test_historical_ancestor_and_three_original_wheel_blobs(self):
        git = self.require_history(EVIDENCE.parents[2])
        self.assertEqual(git("merge-base", "--is-ancestor", MADD_SOURCE, "HEAD").returncode, 0)
        records = json.loads((TENSOR_MADD / "wheel-provenance.json").read_text())["wheels"]
        for wheel in records:
            with self.subTest(path=wheel["path"]):
                path = "docs/diagnostics/compile-pointwise-tensor-madd/" + wheel["path"]
                self.assertEqual(git("rev-parse", MADD_SOURCE + ":" + path).stdout.decode().strip(), wheel["gitBlob"])
                result = git("cat-file", "blob", wheel["gitBlob"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(result.stdout), wheel["size"])
                self.assertEqual(hashlib.sha256(result.stdout).hexdigest(), wheel["sha256"])
                header = b"blob " + str(len(result.stdout)).encode() + b"\0"
                self.assertEqual(hashlib.sha1(header + result.stdout).hexdigest(), wheel["gitBlob"])

    def test_missing_and_shallow_history_explicitly_skip_without_fetch(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(subprocess, "run") as run:
                with self.assertRaisesRegex(unittest.SkipTest, "no .git"):
                    self.require_history(root)
                run.assert_not_called()
            (root / ".git").mkdir()
            result = subprocess.CompletedProcess([], 0, stdout=b"true\n", stderr=b"")
            with patch("shutil.which", return_value="git"), patch.object(subprocess, "run", return_value=result) as run:
                with self.assertRaisesRegex(unittest.SkipTest, "shallow checkout"):
                    self.require_history(root)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][-2:], ["rev-parse", "--is-shallow-repository"])

    def test_mandatory_checks_in_source_archive_without_git(self):
        import os
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "docs/diagnostics"
            shutil.copytree(TENSOR_MADD, evidence / TENSOR_MADD.name)
            owner = evidence / "compile-pointwise-jit"
            owner.mkdir()
            shutil.copyfile(EVIDENCE / "verify_archive.py", owner / "verify_archive.py")
            test = root / "tests/test_compile_evidence_archive.py"
            test.parent.mkdir()
            shutil.copyfile(__file__, test)
            result = subprocess.run([sys.executable, str(test), "TensorMaddArchive", "-v"],
                                    capture_output=True, text=True,
                                    env=dict(os.environ, PATH=""))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Ran 3 tests", result.stderr)
            self.assertNotIn("skipped", result.stderr)
            result = subprocess.run([sys.executable, str(test), "TensorMaddGitHistory.test_historical_ancestor_and_three_original_wheel_blobs", "-v"],
                                    capture_output=True, text=True,
                                    env=dict(os.environ, PATH=""))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("historical Git bytes unavailable", result.stderr)
            self.assertIn("skipped=1", result.stderr)


if __name__ == "__main__":
    unittest.main()
