"""Offline provenance and optional original-Git-byte checks for GELU archives."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "docs/diagnostics/gelu-program-integration"
SOURCE = "34dcc47454b3faaa27ef46a5189b34e81d3802f2"
ENV = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_NO_REPLACE_OBJECTS="1", GIT_OPTIONAL_LOCKS="0")


class GeluArchiveProvenance(unittest.TestCase):
    def test_unchanged_inventory_manifests_and_recorded_remote_proof(self):
        inventory_bytes = (HERE / "archive-preservation-observation.json").read_bytes()
        inventory = json.loads(inventory_bytes)
        observation = json.loads((HERE / "archive-relocation-observation.json").read_text())
        self.assertEqual(hashlib.sha256(inventory_bytes).hexdigest(), observation["inventory_sha256"])
        self.assertEqual(inventory["source_commit"], SOURCE)
        self.assertEqual(len(inventory["files"]), 28)
        self.assertEqual(sum(r["size"] for r in inventory["files"]), 1612770062)
        originals = {r["path"]: r for r in inventory["files"]}
        declared = []
        self.assertEqual(len(inventory["groups"]), 6)
        for group in inventory["groups"]:
            data = (ROOT / group["manifest"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), group["manifest_sha256"])
            manifest = json.loads(data)
            parts = manifest.get("parts", [dict(path=manifest["archive"], size=manifest["size"], sha256=manifest["sha256"])])
            self.assertEqual(sum(p["size"] for p in parts), manifest["size"])
            for part in parts:
                path = (HERE / part["path"]).relative_to(ROOT).as_posix()
                record = originals[path]
                self.assertEqual((part["size"], part["sha256"]), (record["size"], record["sha256"]))
                self.assertEqual(record["source_commit"], SOURCE)
                self.assertEqual(record["url"], f"https://github.com/bobrenjc93/pytorch-rs/blob/{SOURCE}/{path}")
                self.assertRegex(record["gitBlob"], r"^[0-9a-f]{40}$")
                self.assertFalse((ROOT / path).exists(), "historical blob unexpectedly copied into checkout")
                declared.append(path)
        self.assertEqual(len(declared), len(set(declared)))
        self.assertEqual(set(declared), set(originals))
        for path, digest in observation["remote_observation"].items():
            self.assertEqual(hashlib.sha256((HERE / path).read_bytes()).hexdigest(), digest)
        facts = json.loads((HERE / "gelu-archive-remote-head-facts.json").read_text())
        self.assertTrue(any(r["remoteProbeSucceeded"] and r["headRefOid"] == SOURCE for r in facts["observations"]))

    def require_history(self):
        if not (ROOT / ".git").exists() or shutil.which("git") is None:
            self.skipTest("historical Git bytes unavailable: no Git checkout/executable; no fetch attempted")
        def git(*args):
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, env=ENV)
        shallow = git("rev-parse", "--is-shallow-repository")
        self.assertEqual(shallow.returncode, 0, shallow.stderr)
        if shallow.stdout.strip() == b"true":
            self.skipTest("historical Git bytes unavailable in shallow checkout; no fetch attempted")
        inventory = json.loads((HERE / "archive-preservation-observation.json").read_text())
        for obj in [SOURCE + "^{commit}", *(r["gitBlob"] for r in inventory["files"])]:
            if git("cat-file", "-e", obj).returncode:
                self.skipTest("historical Git object unavailable locally; no fetch attempted")
        return git, inventory

    def test_original_ancestor_and_blob_bytes(self):
        git, inventory = self.require_history()
        self.assertEqual(git("merge-base", "--is-ancestor", SOURCE, "HEAD").returncode, 0)
        for record in inventory["files"]:
            with self.subTest(path=record["path"]):
                self.assertEqual(git("ls-tree", SOURCE, "--", record["path"]).stdout.decode().split()[:3],
                                 ["100644", "blob", record["gitBlob"]])
                process = subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "blob", record["gitBlob"]],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV)
                digest, size = hashlib.sha256(), 0
                blob = hashlib.sha1(f"blob {record['size']}\0".encode())
                while chunk := process.stdout.read(1024 * 1024):
                    digest.update(chunk); blob.update(chunk); size += len(chunk)
                stderr = process.stderr.read()
                self.assertEqual(process.wait(), 0, stderr)
                process.stdout.close(); process.stderr.close()
                self.assertEqual((size, digest.hexdigest(), blob.hexdigest()),
                                 (record["size"], record["sha256"], record["gitBlob"]))

    def test_missing_and_shallow_history_skip_without_fetch(self):
        with patch("shutil.which", return_value=None), patch.object(subprocess, "run") as run:
            with self.assertRaisesRegex(unittest.SkipTest, "no Git checkout/executable"):
                self.require_history()
            run.assert_not_called()
        with patch.object(Path, "exists", return_value=True), patch("shutil.which", return_value="git"):
            result = subprocess.CompletedProcess([], 0, b"true\n", b"")
            with patch.object(subprocess, "run", return_value=result) as run:
                with self.assertRaisesRegex(unittest.SkipTest, "shallow checkout"):
                    self.require_history()
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0][-2:], ["rev-parse", "--is-shallow-repository"])
                self.assertEqual(run.call_args.kwargs["env"]["GIT_NO_LAZY_FETCH"], "1")


if __name__ == "__main__":
    unittest.main()
