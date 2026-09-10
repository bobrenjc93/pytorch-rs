"""Accounting fixtures are not hardware evidence; reference tests use real CUDA."""

import copy
import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_cuda_transfers as evaluator


def fixture_observations(case, seed):
    def tensor(shape, strides, device, values, pointer, offset=0):
        return {"shape": shape, "strides": strides, "storage_offset": offset,
                "device": device, "dtype": "float32", "values": values,
                "pointer": pointer if values else 0,
                "pointer_attributes": {"memory_type": 2, "device_ordinal": 0, "is_managed": 0}
                if device == "cuda:0" and values else None}

    observations = []
    for shape in case["shapes"]:
        zero = "zero_roundtrip" in case["operation"]
        values = evaluator.values_for(shape, seed, zero)
        strides = [math.prod(shape[i+1:]) for i in range(len(shape))]
        device = "cpu" if "upload" in case["operation"] else "cuda:0"
        base = tensor(shape, strides, device, values, 10000)
        source = copy.deepcopy(base)
        out_strides = strides
        if "view" in case:
            source = tensor([11, 7], [1, 13], device,
                            [values[r * 13 + c] for c in range(1, 12) for r in range(7)], 10004, 1)
            out_strides = [1, 11]
        obs = {"base": base, "source": source, "base_after": copy.deepcopy(base),
               "source_after": copy.deepcopy(source), "no_copy_identity": True,
               "mutated_base": tensor(shape, strides, device,
                   [ctypes.c_float(v + 8).value for v in values], 10000), "copies": []}
        for i, method in enumerate(case["copies"]):
            output = tensor(source["shape"], out_strides,
                            "cpu" if method in ("cpu", "to_cpu") else "cuda:0",
                            source["values"].copy(), 20000 + 10000 * i)
            obs["copies"].append({"method": method, "distinct_object": True, "output": output})
        obs["copies_after_source_mutation"] = copy.deepcopy([c["output"] for c in obs["copies"]])
        observations.append(obs)
    return observations


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.corpus = evaluator.corpus()
        self.seeds = [9173, 260909]
        self.source = {"commit": "fixture-commit", "source_sha256": "fixture-source"}
        self.build = {**self.source, "extension_sha256": "fixture-extension",
                      "build_command": "fixture-build", "rustc": "fixture-rustc",
                      "cargo": "fixture-cargo", "nvcc": "unused"}
        self.trials = []
        for case in self.corpus["cases"]:
            for seed in self.seeds:
                row = {"status": "passed", "case_id": case["id"], "seed": seed,
                       "observations": fixture_observations(case, seed),
                       "cuda_runtimes": [{"path": "/fixture/libcudart.so", "version": 13000, "status": 0}]}
                self.trials.append({"case_id": case["id"], "seed": seed,
                    "reference": {**copy.deepcopy(row), "role": "reference", "pid": 101,
                                  "version": "2.13.0+cu130", "gpu": {"name": "unit fixture"}},
                    "candidate": {**copy.deepcopy(row), "role": "candidate", "pid": 202,
                                  "blocked_imports": [], "loaded_torch_modules": [],
                                  "extension": {"sha256": "fixture-extension"}}})

    def score(self):
        return evaluator.account(self.corpus, self.seeds, self.trials, self.build, self.source)

    def assertZero(self):
        score = self.score()
        self.assertEqual((score["denominator"], score["passed"], score["fraction"]), (6, 0, 0))
        self.assertEqual(len(score["cases"]), 6)

    def test_fixed_cases_and_existing_weights(self):
        self.assertEqual([(c["id"], c["shapes"], c["copies"]) for c in self.corpus["cases"]], [
            ("cuda_f32_vector_zero_roundtrip", [[17], [0]], ["cpu", "to_cpu"]),
            ("cuda_f32_matrix_zero_roundtrip", [[7, 13]], ["cpu", "to_cpu"]),
            ("cuda_f32_contiguous_cpu_upload", [[7, 13]], ["to_cuda"]),
            ("cuda_f32_strided_cpu_upload", [[7, 13]], ["to_cuda"]),
            ("cuda_f32_view_to_cpu_copy", [[7, 13]], ["cpu", "to_cpu"]),
            ("cuda_f32_same_device_copy", [[17]], ["clone", "to_cuda_copy"]),
        ])
        self.assertEqual(self.corpus["case_aggregation"], "equal_fraction_fixed_denominator")
        for i in (3, 4):
            self.assertEqual(self.corpus["cases"][i]["view"], "[:, 1:12].transpose(0, 1)")
        self.assertIs(self.corpus["cases"][5]["no_copy_identity"], True)
        matrix = json.loads(evaluator.MATRIX.read_text())
        self.assertEqual({c["id"]: c["weight"] for c in matrix["capabilities"]}, {
            "device_tensors_and_transfers": 15, "dtypes_layout_views": 15,
            "math_reductions_linalg": 15, "autograd_training": 15, "nn_optimizers": 15,
            "rng_serialization": 10, "compilation": 10, "multi_device_distributed": 5})
        self.assertEqual(matrix["backend_aggregation"], "equal_arithmetic_mean")
        self.assertEqual(len(matrix["backends"]), 7)

    def test_all_and_partial_credit(self):
        self.assertEqual(self.score()["fraction"], 1)
        self.trials[0]["candidate"]["status"] = "unsupported"
        self.assertEqual(self.score()["fraction"], 5 / 6)
        self.assertTrue(all(c["reference_eligible"] for c in self.score()["cases"]))
        verdict = self.score()["cases"][0]["trials"][0]
        self.assertEqual(verdict["candidate_outcome"], "unsupported")

    def test_reference_eligibility_is_separate_from_candidate_correctness(self):
        self.trials[0]["candidate"]["observations"][0]["copies"][0]["output"]["values"][0] = 123
        verdict = self.score()["cases"][0]["trials"][0]
        self.assertTrue(verdict["reference_eligible"])
        self.assertEqual(verdict["candidate_outcome"], "incorrect_or_invalid")
        self.assertEqual(verdict["credit"], 0)
        for trial in self.trials:
            trial["reference"]["status"] = "skipped"
        self.assertZero()
        self.assertFalse(any(c["reference_eligible"] for c in self.score()["cases"]))

    def test_missing_skipped_unsupported_forwarded_and_incorrect_stay_zero(self):
        original = copy.deepcopy(self.trials)
        for role in ("reference", "candidate"):
            for status in (None, "skipped", "unsupported", "forwarded", "incorrect", "failed"):
                with self.subTest(role=role, status=status):
                    self.trials = copy.deepcopy(original)
                    for trial in self.trials:
                        trial[role] = {"status": status} if status else None
                    self.assertZero()
        self.trials = []
        self.assertZero()

    def test_invalid_results_cannot_receive_credit(self):
        original = copy.deepcopy(self.trials)
        changes = [
            lambda c: c.update(blocked_imports=["torch"]),
            lambda c: c.update(loaded_torch_modules=["torch._C"]),
            lambda c: c.update(pid=101),
            lambda c: c.update(role="reference"),
            lambda c: c.update(cuda_runtimes=None),
            lambda c: c.update(cuda_runtimes=42),
            lambda c: c.update(extension={"sha256": "stale"}),
            lambda c: c.update(observations=[]),
            lambda c: c["observations"][0].update(base_after=None),
            lambda c: c["observations"][0].update(source_after=None),
            lambda c: c["observations"][0].update(copies_after_source_mutation=[]),
            lambda c: c["observations"][0]["copies"][0].update(distinct_object=False),
            lambda c: c["observations"][0]["copies"][0]["output"].update(dtype="float64"),
            lambda c: c["observations"][0]["copies"][0]["output"].update(values=None),
            lambda c: c["observations"][0]["copies"][0]["output"]["values"].__setitem__(0, 999),
            lambda c: c["observations"][0]["copies"][0]["output"]["values"].__setitem__(0, float("nan")),
            lambda c: c["observations"][0]["copies"][0]["output"].update(strides=[1]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                self.trials = copy.deepcopy(original)
                for trial in self.trials:
                    change(trial["candidate"])
                score = self.score()
                # Rank-one stride [1] is valid for the vector/copy slots.
                self.assertEqual(score["passed"], 2 if index == len(changes) - 1 else 0)
                self.assertEqual(score["denominator"], 6)

    def test_empty_trial_and_each_copy_are_required(self):
        self.trials[0]["candidate"]["observations"].pop()
        self.assertEqual(self.score()["passed"], 5)
        self.trials[-1]["candidate"]["observations"][0]["copies"].pop()
        self.assertEqual(self.score()["passed"], 4)

    def test_aliases_fake_cuda_and_unchanged_mutation_are_rejected(self):
        row = self.trials[-1]["candidate"]
        original = copy.deepcopy(row)
        for mutation in ("alias", "identity", "cpu", "managed", "mutation"):
            with self.subTest(mutation=mutation):
                row = copy.deepcopy(original)
                obs = row["observations"][0]
                output = obs["copies"][0]["output"]
                if mutation == "alias":
                    output["pointer"] = obs["source"]["pointer"]
                    obs["copies_after_source_mutation"][0]["pointer"] = output["pointer"]
                elif mutation == "identity":
                    obs["no_copy_identity"] = False
                elif mutation == "cpu":
                    output["device"] = "cpu"
                elif mutation == "managed":
                    output["pointer_attributes"]["is_managed"] = 1
                else:
                    obs["mutated_base"] = obs["base"]
                self.assertFalse(evaluator.valid_execution(row, self.corpus["cases"][-1], self.seeds[-1], "candidate"))

    def test_build_binding_and_seed_rules(self):
        original = self.build.copy()
        for key in original:
            with self.subTest(key=key):
                self.build = original.copy()
                self.build.pop(key)
                self.assertZero()
        self.build = original
        for seeds in ([], [9173], [9173, 9173], [9173, -9173]):
            self.seeds = seeds
            self.assertZero()

    def test_duplicate_unknown_and_reference_failures_retain_six(self):
        self.trials += copy.deepcopy(self.trials)
        self.assertZero()
        for row in self.trials:
            row["case_id"] = "unknown"
        self.assertZero()


class IsolationTests(unittest.TestCase):
    def test_launch_fails_closed(self):
        case = evaluator.corpus()["cases"][0]
        for error in (FileNotFoundError(), subprocess.TimeoutExpired("worker", 1)):
            with patch.object(evaluator.subprocess, "run", side_effect=error):
                self.assertEqual(evaluator.launch("candidate", case, 1, sys.executable, 1, os.environ)["status"], "failed")
        for output, code in (("invalid", 0), ("[]", 0), ('{"status":"passed"}', 1)):
            with patch.object(evaluator.subprocess, "run", return_value=subprocess.CompletedProcess([], code, output, "")):
                self.assertEqual(evaluator.launch("candidate", case, 1, sys.executable, 1, os.environ)["status"], "failed")

    def test_swallowed_forwarding_is_zero(self):
        work = ROOT / "target/cuda-transfers-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            package = Path(temp) / "python/torch_rs"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("try:\n import torch\nexcept RuntimeError:\n pass\n__version__ = 'fixture'\n")
            code = f"""
import sys, json
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import evaluate_cuda_transfers as e
e.ROOT = Path({temp!r})
print(json.dumps(e.worker('candidate', {{'case': e.corpus()['cases'][0], 'seed': 9173}})))
"""
            completed = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            row = json.loads(completed.stdout)
            self.assertEqual(row["status"], "forwarded")
            self.assertEqual(row["blocked_imports"], ["torch"])

    def test_cli_enforces_device_and_seed_contract(self):
        for mask, args in (("", []), ("0", ["--seed", "1"]), ("0", ["--seed", "1", "--seed", "-1"])):
            completed = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/evaluate_cuda_transfers.py"), *args],
                                       env={**os.environ, "CUDA_VISIBLE_DEVICES": mask}, capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, "")


@unittest.skipUnless(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "requires CUDA_VISIBLE_DEVICES=0")
class HardwareTests(unittest.TestCase):
    def test_all_reference_cases_on_real_cuda(self):
        work = ROOT / "target/cuda-transfers-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            env = {**os.environ, "TMPDIR": temp, "CUDA_CACHE_PATH": temp,
                   "XDG_CACHE_HOME": temp, "PYTHONDONTWRITEBYTECODE": "1"}
            for case in evaluator.corpus()["cases"]:
                row = evaluator.launch("reference", case, 9173, sys.executable, 60, env)
                if row["status"] == "skipped" or "No module named 'torch'" in row.get("error", ""):
                    self.skipTest("NVIDIA CUDA PyTorch reference unavailable")
                with self.subTest(case=case["id"]):
                    self.assertTrue(evaluator.valid_execution(row, case, 9173, "reference"), row)
                    self.assertNotEqual(row["pid"], os.getpid())
                    self.assertTrue(row["gpu"]["uuid"])
                    self.assertTrue(row["cuda_runtimes"])


if __name__ == "__main__":
    unittest.main()
