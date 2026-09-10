"""Synthetic accounting is not hardware evidence; CUDA checks use real workers."""

import copy
import importlib.util
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
SPEC = importlib.util.spec_from_file_location("cuda_compilation_evaluator", ROOT / "scripts/evaluate_cuda_compilation.py")
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.corpus = e.corpus()
        self.seeds = [1927, 83719]
        self.source = dict(commit="fixture", source_sha256="source", production_diff_sha256="diff")
        self.build = dict(self.source, extension_sha256="extension", build_command="synthetic only",
                          rustc="fixture", cargo="fixture", nvcc="unused")
        self.trials = []
        for case in self.corpus["cases"]:
            for seed in self.seeds:
                def tensor(shape, values):
                    return dict(shape=shape, dtype="float32", device="cuda:0", values=values,
                                strides=[math.prod(shape[i + 1:]) for i in range(len(shape))],
                                storage_offset=0, requires_grad=False,
                                pointer_attributes=dict(memory_type=2, device_ordinal=0, is_managed=0))
                executions = []
                for index, data_seed in enumerate((seed, seed ^ e.CHANGE_MASK)):
                    inputs = [tensor(s, e.common.flatten(v)) for s, v in zip(
                        case["input_shapes"], e.common.inputs_for(case, data_seed))]
                    executions.append(dict(seed=data_seed, inputs=inputs, inputs_after=copy.deepcopy(inputs),
                                           output=tensor(case["output_shape"], [0.25 + index] * math.prod(case["output_shape"])),
                                           public_materialization_matches=True))
                row = dict(status="passed", case_id=case["id"], seed=seed, executions=executions,
                           compile_options=e.OPTIONS.copy(),
                           cuda_runtimes=[dict(path="fixture/libcudart.so", version=13000, status=0)])
                ref = dict(copy.deepcopy(row), role="reference", pid=10, version="2.13.0+cu130", gpu=dict(name="fixture"))
                cand = dict(copy.deepcopy(row), role="candidate", pid=20, blocked_imports=[], loaded_torch_modules=[],
                            extension=dict(sha256="extension"), original_body_attempts=0,
                            compile_evidence=[dict(phase=phase, lowered_targets=targets, executor_calls=1,
                                                   native_returns=1, hook=e.native_hook(case))
                                              for phase, targets in (("initial", [[case["operation"]]]), ("changed", []))])
                self.trials.append(dict(case_id=case["id"], seed=seed, reference=ref, candidate=cand))

    def score(self):
        return e.account(self.corpus, self.seeds, self.trials, self.build, self.source)

    def assertZero(self):
        score = self.score()
        self.assertEqual((score["denominator"], score["passed"], score["fraction"]), (6, 0, 0))
        self.assertEqual(len(score["cases"]), 6)

    def test_manifest_configuration_and_weights(self):
        self.assertEqual(self.corpus["version"], "cuda_float32_inference_compilation_v1")
        self.assertEqual(self.corpus["compile_options"], dict(backend="eager", fullgraph=True, dynamic=False))
        self.assertEqual([(c["id"], c["operation"], c["input_shapes"], c["output_shape"])
                          for c in self.corpus["cases"]], [
            ("cuda_f32_compile_add_same_shape", "add", [[7, 13], [7, 13]], [7, 13]),
            ("cuda_f32_compile_add_trailing_vector", "add", [[7, 13], [13]], [7, 13]),
            ("cuda_f32_compile_neg", "neg", [[7, 13]], [7, 13]),
            ("cuda_f32_compile_mul_scalar", "mul_scalar", [[7, 13]], [7, 13]),
            ("cuda_f32_compile_sum_rows", "sum", [[7, 13]], [7]),
            ("cuda_f32_compile_matmul", "matmul", [[7, 13], [13, 5]], [7, 5])])
        self.assertEqual(self.corpus["cases"][3]["scalar"], -1.75)
        self.assertEqual((self.corpus["cases"][4]["dim"], self.corpus["cases"][4]["keepdim"]), (1, False))
        self.assertEqual((self.corpus["rtol"], self.corpus["atol"]), (1e-5, 1e-6))
        matrix = json.loads(e.MATRIX.read_text())
        self.assertEqual([c["weight"] for c in matrix["capabilities"]], [15, 15, 15, 15, 15, 10, 10, 5])
        self.assertEqual(len(matrix["backends"]), 7)
        self.assertEqual(matrix["backend_aggregation"], "equal_arithmetic_mean")
        self.assertEqual(set(e.PROGRAMS), {c["id"] for c in self.corpus["cases"]})

    def test_full_partial_and_absent_credit(self):
        self.assertEqual(self.score()["passed"], 6)
        self.trials[0]["candidate"]["status"] = "unsupported"
        self.assertEqual(self.score()["fraction"], 5 / 6)
        self.trials = []
        self.assertZero()

    def test_missing_malformed_duplicate_unknown_trials(self):
        original = copy.deepcopy(self.trials)
        for trials in (None, [None, [], "bad"], original + original,
                       [dict(t, case_id="unknown") for t in original]):
            with self.subTest(trials=str(trials)[:40]):
                self.trials = trials
                self.assertZero()
        for role in ("reference", "candidate"):
            for invalid in (None, [], "bad", {}, dict(status="skipped"), dict(status="unsupported")):
                with self.subTest(role=role, invalid=invalid):
                    self.trials = copy.deepcopy(original)
                    for trial in self.trials:
                        trial[role] = invalid
                    self.assertZero()

    def test_bad_metadata_forwarding_mutation_and_compilation_proof(self):
        original = copy.deepcopy(self.trials)
        changes = [
            lambda c: c.update(blocked_imports=["torch"]),
            lambda c: c.update(loaded_torch_modules=["torch._C"]),
            lambda c: c.update(pid=10),
            lambda c: c.update(extension=None),
            lambda c: c.update(cuda_runtimes=[]),
            lambda c: c.update(compile_options=dict(backend="inductor", fullgraph=True, dynamic=False)),
            lambda c: c["compile_options"].update(fullgraph=False),
            lambda c: c["compile_options"].update(dynamic=True),
            lambda c: c.update(original_body_attempts=1),
            lambda c: c.update(compile_evidence=None),
            lambda c: c["compile_evidence"][0].update(lowered_targets=[]),
            lambda c: c["compile_evidence"][1].update(lowered_targets=[["add"]]),
            lambda c: c["compile_evidence"][1].update(executor_calls=0),
            lambda c: c["compile_evidence"][1].update(native_returns=0),
            lambda c: c["compile_evidence"][1].update(hook="eager_fallback"),
            lambda c: c.update(executions=c["executions"][:1]),
            lambda c: c["executions"].__setitem__(1, copy.deepcopy(c["executions"][0])),
        ]
        for field, value in (("seed", -1), ("inputs", None), ("inputs_after", []),
                             ("output", None), ("public_materialization_matches", False)):
            changes.append(lambda c, f=field, v=value: c["executions"][1].update({f: v}))
        for field, value in (("values", []), ("device", "cpu"), ("dtype", "float64"),
                             ("shape", [91]), ("strides", [-1]), ("storage_offset", 1),
                             ("requires_grad", True), ("pointer_attributes", None)):
            changes.append(lambda c, f=field, v=value: c["executions"][1]["output"].update({f: v}))
        changes.append(lambda c: c["executions"][1]["inputs_after"][0]["values"].__setitem__(0, 100))
        for index, change in enumerate(changes):
            with self.subTest(change=index):
                self.trials = copy.deepcopy(original)
                for trial in self.trials:
                    change(trial["candidate"])
                self.assertZero()

    def test_both_executions_require_correct_finite_outputs(self):
        original = copy.deepcopy(self.trials)
        for index in (0, 1):
            for invalid in (900, float("nan"), float("inf"), "0.25", True):
                with self.subTest(index=index, invalid=invalid):
                    self.trials = copy.deepcopy(original)
                    for trial in self.trials:
                        trial["candidate"]["executions"][index]["output"]["values"][0] = invalid
                    self.assertZero()
        self.trials = original
        for trial in self.trials:
            trial["candidate"]["executions"][1]["output"]["values"][0] += 1e-7
        self.assertEqual(self.score()["passed"], 6)

    def test_missing_stale_or_malformed_build_zero(self):
        original = self.build.copy()
        for key in original:
            self.build = original.copy()
            del self.build[key]
            with self.subTest(key=key):
                self.assertZero()
        for invalid in (None, [], {}, "bad", dict(original, source_sha256="stale")):
            self.build = invalid
            self.assertZero()

    def test_seeds_cannot_reduce_denominator(self):
        for seeds in ([], [1927], [1927, 1927], [-1927, 1927]):
            self.seeds = seeds
            self.assertZero()

    def test_cli_missing_malformed_receipts_and_changed_source_keep_six_slots(self):
        work = ROOT / "target/cuda-compilation-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            output, receipt = Path(temp) / "result.json", Path(temp) / "build.json"
            def launch(role, case, seed, *args):
                trial = next(t for t in self.trials if t["case_id"] == case["id"] and t["seed"] == seed)
                return copy.deepcopy(trial[role])
            argv = [str(ROOT / "scripts/evaluate_cuda_compilation.py"),
                    "--seed", "1927", "--seed", "83719", "--output", str(output)]
            for scenario in ("valid", "missing", "malformed", "changed", "reference_failed"):
                receipt.write_text("not json" if scenario == "malformed" else json.dumps(self.build))
                options = [] if scenario == "missing" else ["--build-record", str(receipt)]
                after = dict(self.source, source_sha256="changed") if scenario == "changed" else self.source
                if scenario == "reference_failed":
                    for trial in self.trials:
                        trial["reference"]["status"] = "skipped"
                with self.subTest(scenario=scenario), patch.object(sys, "argv", argv + options), \
                        patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"), \
                        patch.object(e, "launch", side_effect=launch), \
                        patch.object(e.common, "source_provenance", side_effect=[self.source, after]), \
                        patch.object(e.subprocess, "check_output", return_value="fixture inventory"):
                    self.assertEqual(e.main(), 2 if scenario == "reference_failed" else 0)
                    report = json.loads(output.read_text())
                    self.assertEqual(report["accounting"]["denominator"], 6)
                    self.assertEqual(report["accounting"]["passed"], 6 if scenario == "valid" else 0)
                    self.assertEqual(len(report["trials"]), 12)


class IsolationTests(unittest.TestCase):
    def test_cli_rejects_bad_seeds_and_output_escape(self):
        for options in (("--seed", "17"), ("--seed", "17", "--seed", "-17"),
                        ("--seed", "17", "--seed", "17"),
                        ("--seed", "17", "--seed", "18", "--output", "../escape.json"),
                        ("--seed", "17", "--seed", "18", "--output", ".burner/escape.json")):
            result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "scripts/evaluate_cuda_compilation.py"), *options],
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(result.stdout, "")

    def test_original_body_blocked_and_attempt_retained(self):
        def lower():
            pass
        def executor():
            pass
        observer = e.CompileObserver(e.negation, lower, executor, abs)
        with observer.phase("initial"):
            with self.assertRaisesRegex(RuntimeError, "original Python body"):
                e.negation(3)
        self.assertEqual(observer.body_attempts, 1)
        with observer.phase("changed"):
            with self.assertRaisesRegex(RuntimeError, "lower again"):
                lower()

    def test_python_executor_cannot_impersonate_native(self):
        with self.assertRaisesRegex(RuntimeError, "extension builtin"):
            e.CompileObserver(e.negation, e.negation, e.negation, lambda x: x)

    def test_crash_timeout_invalid_json_and_missing_python(self):
        case = e.corpus()["cases"][0]
        for error in (FileNotFoundError("missing"), subprocess.TimeoutExpired("worker", 1)):
            with patch.object(e.subprocess, "run", side_effect=error):
                self.assertEqual(e.launch("candidate", case, 1, sys.executable, 1, os.environ)["status"], "failed")
        for output, code in (("bad", 0), ("[]", 0), ('{"status":"passed"}', 1)):
            with patch.object(e.subprocess, "run", return_value=subprocess.CompletedProcess([], code, output, "crash")):
                self.assertEqual(e.launch("candidate", case, 1, sys.executable, 1, os.environ)["status"], "failed")

    def test_worker_blocks_swallowed_torch_import(self):
        work = ROOT / "target/cuda-compilation-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            package = Path(temp) / "python/torch_rs"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("try:\n import torch\nexcept RuntimeError:\n pass\n__version__='fixture'\n")
            code = f"""
import sys,json
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import evaluate_cuda_compilation as e
case=e.corpus()['cases'][0]
e.ROOT=Path({temp!r})
print(json.dumps(e.worker('candidate', {{'case':case,'seed':17}})))
"""
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(result.stdout)
            self.assertEqual(row["status"], "forwarded")
            self.assertEqual(row["blocked_imports"], ["torch"])


@unittest.skipUnless(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "requires CUDA_VISIBLE_DEVICES=0")
class HardwareTests(unittest.TestCase):
    def test_all_reference_and_candidate_cases_at_two_seeds(self):
        work = ROOT / "target/cuda-compilation-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            env = dict(os.environ, TMPDIR=temp, XDG_CACHE_HOME=temp, CUDA_CACHE_PATH=temp,
                       TORCHINDUCTOR_CACHE_DIR=temp, TRITON_CACHE_DIR=temp, PYTHONDONTWRITEBYTECODE="1")
            probe = subprocess.run([sys.executable, "-I", "-B", "-c",
                                    "import torch; raise SystemExit(0 if torch.cuda.is_available() and not torch.version.hip else 77)"],
                                   capture_output=True, text=True, env=env, timeout=60)
            if probe.returncode == 77 or "No module named 'torch'" in probe.stderr:
                self.skipTest("NVIDIA CUDA PyTorch unavailable")
            self.assertEqual(probe.returncode, 0, probe.stderr)
            if not list((ROOT / "python/torch_rs").glob("torch_rs*.so")):
                self.skipTest("requires a fresh local native extension")
            for case in e.corpus()["cases"]:
                for seed in (1927, 83719):
                    with self.subTest(case=case["id"], seed=seed):
                        ref = e.launch("reference", case, seed, sys.executable, 60, env)
                        cand = e.launch("candidate", case, seed, sys.executable, 60, env)
                        self.assertTrue(e.valid_execution(ref, case, seed, "reference"), ref)
                        self.assertNotEqual(ref["pid"], cand.get("pid"))
                        if case["operation"] == "sum":
                            self.assertFalse(e.valid_execution(cand, case, seed, "candidate"))
                            self.assertIn("not support", cand.get("error", ""))
                        else:
                            self.assertTrue(e.valid_execution(cand, case, seed, "candidate"), cand)
                            for r, c in zip(ref["executions"], cand["executions"]):
                                for expected, actual in zip(r["output"]["values"], c["output"]["values"]):
                                    self.assertLessEqual(abs(expected - actual), 1e-6 + 1e-5 * abs(expected))
