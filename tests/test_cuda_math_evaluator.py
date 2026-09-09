"""Accounting fixtures are synthetic; only the subprocess CUDA test is evidence."""

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
SPEC = importlib.util.spec_from_file_location("cuda_math_evaluator", ROOT / "scripts/evaluate_cuda_math.py")
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


class FixedDenominatorTests(unittest.TestCase):
    def setUp(self):
        self.corpus = evaluator.corpus()
        self.seeds = [1459, 9173]
        self.source = {"commit": "test-commit", "source_sha256": "test-source"}
        self.build = {**self.source, "extension_sha256": "test-extension",
                      "build_command": "synthetic unit fixture, not evidence",
                      "rustc": "test-rustc", "cargo": "test-cargo"}
        self.trials = []
        for case in self.corpus["cases"]:
            for seed in self.seeds:
                def tensor(shape, values):
                    return {"shape": shape, "dtype": "float32", "device": "cuda:0",
                            "pointer_attributes": {"memory_type": 2, "device_ordinal": 0, "is_managed": 0},
                            "values": values}

                # Synthetic equal outputs exercise accounting, not GPU support.
                row = {"status": "passed", "case_id": case["id"], "seed": seed,
                       "inputs": [tensor(s, evaluator.flatten(v)) for s, v in zip(
                           case["input_shapes"], evaluator.inputs_for(case, seed))],
                       "output": tensor(case["output_shape"], [0.25] * math.prod(case["output_shape"])),
                       "cuda_runtimes": [{"path": "/fixture/libcudart.so", "version": 13000, "status": 0}]}
                row["inputs_after"] = copy.deepcopy(row["inputs"])
                reference = {**copy.deepcopy(row), "role": "reference", "pid": 101,
                             "version": "2.13.0+cu130", "gpu": {"name": "unit fixture"}}
                candidate = {**copy.deepcopy(row), "role": "candidate", "pid": 202,
                             "blocked_imports": [], "loaded_torch_modules": [],
                             "extension": {"sha256": "test-extension"}}
                self.trials.append({"case_id": case["id"], "seed": seed,
                                    "reference": reference, "candidate": candidate})

    def score(self):
        return evaluator.account(self.corpus, self.seeds, self.trials, self.build, self.source)

    def assertZero(self):
        result = self.score()
        self.assertEqual(result["denominator"], 6)
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["fraction"], 0)
        self.assertEqual(len(result["cases"]), 6)

    def test_fixed_cases_and_unchanged_weights(self):
        cases = self.corpus["cases"]
        self.assertEqual([(c["id"], c["operation"], c["input_shapes"], c["output_shape"])
                          for c in cases], [
            ("cuda_f32_add_same_shape", "add", [[7, 13], [7, 13]], [7, 13]),
            ("cuda_f32_add_trailing_vector", "add", [[7, 13], [13]], [7, 13]),
            ("cuda_f32_neg", "neg", [[7, 13]], [7, 13]),
            ("cuda_f32_mul_scalar", "mul_scalar", [[7, 13]], [7, 13]),
            ("cuda_f32_sum_axis", "sum", [[7, 13]], [7]),
            ("cuda_f32_matmul", "matmul", [[7, 13], [13, 5]], [7, 5]),
        ])
        self.assertEqual(cases[3]["scalar"], -1.75)
        self.assertEqual((cases[4]["dim"], cases[4]["keepdim"]), (1, False))
        self.assertEqual((self.corpus["rtol"], self.corpus["atol"]), (1e-5, 1e-6))
        matrix = json.loads(evaluator.MATRIX.read_text())
        self.assertEqual({c["id"]: c["weight"] for c in matrix["capabilities"]}, {
            "device_tensors_and_transfers": 15, "dtypes_layout_views": 15,
            "math_reductions_linalg": 15, "autograd_training": 15,
            "nn_optimizers": 15, "rng_serialization": 10,
            "compilation": 10, "multi_device_distributed": 5})
        self.assertEqual(len(matrix["backends"]), 7)
        self.assertEqual(matrix["backend_aggregation"], "equal_arithmetic_mean")

    def test_complete_and_partial_credit_use_six(self):
        self.assertEqual(self.score()["fraction"], 1)
        self.trials[0]["candidate"]["status"] = "failed"
        self.assertEqual(self.score()["fraction"], 5 / 6)
        self.assertEqual(self.score()["cases"][0]["credit"], 0)

    def test_missing_failed_skipped_forwarded_unsupported_are_zero(self):
        original = copy.deepcopy(self.trials)
        for status in (None, "failed", "skipped", "forwarded", "unsupported"):
            with self.subTest(status=status):
                self.trials = copy.deepcopy(original)
                for trial in self.trials:
                    if status is None:
                        del trial["candidate"]
                    else:
                        trial["candidate"]["status"] = status
                self.assertZero()
        self.trials = []
        self.assertZero()

    def test_incorrect_and_nonfinite_outputs_are_zero(self):
        for value in (19, float("nan"), float("inf"), "0.25", True):
            with self.subTest(value=value):
                for trial in self.trials:
                    trial["candidate"]["output"]["values"][0] = value
                self.assertZero()

    def test_tolerance_is_not_exact_equality(self):
        for trial in self.trials:
            trial["candidate"]["output"]["values"][0] += 1e-7
        self.assertEqual(self.score()["passed"], 6)

    def test_stale_or_missing_build_receipt_is_zero(self):
        original = self.build.copy()
        for key in ("commit", "source_sha256", "extension_sha256", "build_command", "rustc", "cargo"):
            with self.subTest(key=key):
                self.build = original.copy()
                self.build.pop(key)
                self.assertZero()
        self.build = None
        self.assertZero()

    def test_forwarding_cpu_fallback_wrong_device_and_malformed_results_are_zero(self):
        original = copy.deepcopy(self.trials)
        changes = [
            lambda c: c.update(blocked_imports=["torch"]),
            lambda c: c.update(loaded_torch_modules=["torch._C"]),
            lambda c: c.update(pid=101),
            lambda c: c.update(role="reference"),
            lambda c: c.update(seed=-1),
            lambda c: c.update(inputs=None),
            lambda c: c.update(output=None),
            lambda c: c.update(extension=None),
            lambda c: c.update(cuda_runtimes=[]),
            lambda c: c["output"].update(device="cpu"),
            lambda c: c["output"].update(dtype="float64"),
            lambda c: c["output"].update(shape=[91]),
            lambda c: c["output"].update(values=[]),
            lambda c: c["output"]["pointer_attributes"].update(memory_type=1),
            lambda c: c["output"]["pointer_attributes"].update(device_ordinal=1),
            lambda c: c["output"]["pointer_attributes"].update(is_managed=1),
            lambda c: c["inputs"][0]["values"].__setitem__(0, 1000),
        ]
        for i, change in enumerate(changes):
            with self.subTest(change=i):
                self.trials = copy.deepcopy(original)
                for trial in self.trials:
                    change(trial["candidate"])
                self.assertZero()

    def test_reference_failure_never_removes_denominator(self):
        for trial in self.trials:
            trial["reference"]["status"] = "skipped"
        self.assertZero()

    def test_duplicate_and_unknown_rows_cannot_inflate_credit(self):
        self.trials += copy.deepcopy(self.trials)
        self.assertZero()
        self.trials = [{**t, "case_id": "unknown"} for t in self.trials]
        self.assertZero()

    def test_insufficient_seed_trials_cannot_earn_credit(self):
        self.seeds = self.seeds[:1]
        self.assertZero()

    def test_opposite_signed_seeds_cannot_earn_credit_for_duplicate_workloads(self):
        # Keep otherwise valid rows for both signed seeds so the seed guard,
        # rather than a missing trial or mismatched input, must reject credit.
        positive = [t for t in self.trials if t["seed"] == 9173]
        negative = copy.deepcopy(positive)
        for trial in negative:
            trial["seed"] = -9173
            for role in ("reference", "candidate"):
                trial[role]["seed"] = -9173
        self.seeds = [9173, -9173]
        self.trials = positive + negative
        for case in self.corpus["cases"]:
            self.assertEqual(evaluator.inputs_for(case, 9173),
                             evaluator.inputs_for(case, -9173))
        self.assertZero()

    def test_missing_or_mutated_post_operation_inputs_cannot_earn_credit(self):
        original = copy.deepcopy(self.trials)
        for mutation in ("missing", "first_operand", "last_operand"):
            with self.subTest(mutation=mutation):
                self.trials = copy.deepcopy(original)
                for trial in self.trials:
                    candidate = trial["candidate"]
                    if mutation == "missing":
                        del candidate["inputs_after"]
                    else:
                        index = 0 if mutation == "first_operand" else -1
                        candidate["inputs_after"][index]["values"][0] += 1
                self.assertZero()

    def test_inputs_are_evaluator_selected_and_reproducible(self):
        case = self.corpus["cases"][0]
        a = evaluator.inputs_for(case, 17)
        self.assertEqual(a, evaluator.inputs_for(case, 17))
        self.assertNotEqual(a, evaluator.inputs_for(case, 18))
        self.assertNotEqual(a[0], a[1])
        self.assertTrue(any(v < 0 for v in evaluator.flatten(a)))
        self.assertTrue(any(v > 0 for v in evaluator.flatten(a)))


class IsolationTests(unittest.TestCase):
    def test_cli_rejects_negative_and_opposite_signed_seeds(self):
        for seeds in ((9173, -9173), (-9173, 1459)):
            with self.subTest(seeds=seeds):
                result = subprocess.run(
                    [sys.executable, "-I", "-B", str(ROOT / "scripts/evaluate_cuda_math.py"),
                     "--seed", str(seeds[0]), "--seed", str(seeds[1])],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}, cwd=ROOT,
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("nonnegative", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_candidate_worker_rejects_a_forwarding_package(self):
        work = ROOT / "target/cuda-math-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            package = Path(temp) / "python/torch_rs"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(
                "try:\n    import torch\nexcept RuntimeError:\n    pass\n__version__ = 'fixture'\n")
            code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import evaluate_cuda_math as e
case = e.corpus()['cases'][0]
e.ROOT = Path({temp!r})
print(json.dumps(e.worker('candidate', {{'case': case, 'seed': 9173}})))
"""
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                    capture_output=True, text=True, cwd=ROOT, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(result.stdout)
            self.assertEqual(row["status"], "forwarded")
            self.assertEqual(row["blocked_imports"], ["torch"])
            self.assertEqual(row["loaded_torch_modules"], [])

    def test_real_import_blocker_rejects_even_swallowed_forwarding(self):
        code = f"""
import sys
sys.path.insert(0, {str(ROOT / 'scripts')!r})
from evaluate_cuda_math import BlockTorch, ForwardingError
blocker = BlockTorch()
blocker.check()
sys.meta_path.insert(0, blocker)
for name in ('torch', 'torch.nn'):
    try:
        __import__(name)
    except ForwardingError:
        pass
    else:
        raise AssertionError('import was not blocked')
try:
    blocker.check()
except ForwardingError:
    pass
else:
    raise AssertionError('swallowed forwarding attempt was not retained')
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                capture_output=True, text=True, cwd=ROOT, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_crash_timeout_invalid_json_and_missing_interpreter_fail_closed(self):
        case = evaluator.corpus()["cases"][0]
        for error in (FileNotFoundError("missing interpreter"),
                      subprocess.TimeoutExpired("worker", 0.01)):
            with self.subTest(error=error), patch.object(evaluator.subprocess, "run", side_effect=error):
                result = evaluator.launch("candidate", case, 1, sys.executable, 1, os.environ)
                self.assertEqual(result["status"], "failed")
        for output, returncode in (("not json", 0), ("[]", 0), ('{"status":"passed"}', 1)):
            with self.subTest(output=output), patch.object(evaluator.subprocess, "run", return_value=
                    subprocess.CompletedProcess([], returncode, output, "crashed")):
                self.assertEqual(evaluator.launch("candidate", case, 1, sys.executable, 1,
                                                  os.environ)["status"], "failed")


@unittest.skipUnless(os.environ.get("CUDA_VISIBLE_DEVICES") == "0",
                     "CUDA math hardware checks require CUDA_VISIBLE_DEVICES=0")
class CudaReferenceTests(unittest.TestCase):
    def test_candidate_worker_rejects_operand_mutation_on_cuda(self):
        if not any((ROOT / "python/torch_rs").glob("torch_rs*.so")):
            self.skipTest("requires this checkout's locally built native extension")
        work = ROOT / "target/cuda-math-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            env = {**os.environ, "TMPDIR": temp, "CUDA_CACHE_PATH": temp,
                   "XDG_CACHE_HOME": temp, "PYTHONDONTWRITEBYTECODE": "1"}
            case = evaluator.corpus()["cases"][0]
            reference = evaluator.launch("reference", case, 9173, sys.executable, 60, env)
            if (reference["status"] == "skipped"
                    or "No module named 'torch'" in reference.get("error", "")):
                self.skipTest("NVIDIA CUDA PyTorch reference unavailable")
            self.assertTrue(evaluator.valid_execution(reference, case, 9173, "reference"), reference)
            baseline = evaluator.launch("candidate", case, 9173, sys.executable, 60, env)
            self.assertTrue(evaluator.valid_execution(baseline, case, 9173, "candidate"), baseline)
            self.assertEqual(baseline["output"]["values"], reference["output"]["values"])
            for operand, return_operand in ((0, True), (1, False)):
                with self.subTest(operand=operand, return_operand=return_operand):
                    code = f"""
import ctypes, json, sys
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import evaluate_cuda_math as e
original = e.operation
def mutate(module, case, inputs):
    output = original(module, case, inputs)
    inspector = e.CudaInspector()
    inspector.call('cuCtxSynchronize')
    copy = inspector.driver.cuMemcpyDtoD_v2
    copy.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
    copy.restype = ctypes.c_int
    inspector.call('cuMemcpyDtoD_v2', inputs[{operand}].data_ptr(), output.data_ptr(), 7 * 13 * 4)
    return inputs[{operand}] if {return_operand!r} else output
e.operation = mutate
print(json.dumps(e.worker('candidate', {{'case': e.corpus()['cases'][0], 'seed': 9173}})))
"""
                    completed = subprocess.run([sys.executable, "-I", "-B", "-c", code],
                                               cwd=ROOT, env=env, capture_output=True,
                                               text=True, timeout=60)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    result = json.loads(completed.stdout)
                    self.assertEqual(result["output"]["values"], reference["output"]["values"])
                    self.assertEqual(result["status"], "failed", result)
                    self.assertIn(f"operation mutated input {operand}", result["error"])
                    self.assertNotEqual(result["inputs_after"][operand], result["inputs"][operand])
                    self.assertEqual(result["inputs_after"][1 - operand], result["inputs"][1 - operand])
                    self.assertEqual(result["blocked_imports"], [])
                    self.assertFalse(evaluator.valid_execution(result, case, 9173, "candidate"))

    def test_six_reference_cases_run_in_isolated_processes_on_real_cuda(self):
        work = ROOT / "target/cuda-math-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            env = {**os.environ, "TMPDIR": temp, "CUDA_CACHE_PATH": temp,
                   "XDG_CACHE_HOME": temp, "PYTHONDONTWRITEBYTECODE": "1"}
            probe = subprocess.run([sys.executable, "-I", "-B", "-c",
                                    "import torch; raise SystemExit(0 if torch.cuda.is_available() "
                                    "and not torch.version.hip else 77)"],
                                   capture_output=True, text=True, env=env, timeout=60)
            if probe.returncode == 77 or "No module named 'torch'" in probe.stderr:
                self.skipTest("NVIDIA CUDA PyTorch reference unavailable")
            self.assertEqual(probe.returncode, 0, probe.stderr)
            for case in evaluator.corpus()["cases"]:
                with self.subTest(case=case["id"]):
                    result = evaluator.launch("reference", case, 9173, sys.executable, 60, env)
                    self.assertTrue(evaluator.valid_execution(result, case, 9173, "reference"), result)
                    self.assertNotEqual(result["pid"], os.getpid())
                    self.assertTrue(result["gpu"]["name"])
                    self.assertTrue(result["cuda_runtimes"])
                    # Independent scalar oracle catches accidental changes to
                    # the shared operation dispatcher, not just GPU metadata.
                    inputs = evaluator.inputs_for(case, 9173)
                    a = inputs[0]
                    if case["id"] == "cuda_f32_add_same_shape":
                        expected = [x + y for ar, br in zip(a, inputs[1]) for x, y in zip(ar, br)]
                    elif case["id"] == "cuda_f32_add_trailing_vector":
                        expected = [x + y for ar in a for x, y in zip(ar, inputs[1])]
                    elif case["id"] == "cuda_f32_neg":
                        expected = [-x for ar in a for x in ar]
                    elif case["id"] == "cuda_f32_mul_scalar":
                        expected = [-1.75 * x for ar in a for x in ar]
                    elif case["id"] == "cuda_f32_sum_axis":
                        expected = [sum(ar) for ar in a]
                    elif case["id"] == "cuda_f32_matmul":
                        expected = [sum(x * y for x, y in zip(ar, column))
                                    for ar in a for column in zip(*inputs[1])]
                    else:
                        self.fail("unrecognized reference case")
                    actual = result["output"]["values"]
                    self.assertEqual(len(actual), len(expected))
                    for got, want in zip(actual, expected):
                        self.assertLessEqual(abs(got - want), 1e-6 + 1e-5 * abs(want))


if __name__ == "__main__":
    unittest.main()
