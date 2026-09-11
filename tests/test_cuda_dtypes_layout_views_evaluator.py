"""Synthetic accounting tests are not accelerator evidence."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_cuda_dtypes_layout_views as evaluator


def fixture_observations(case, seed):
    dtype, op = case["dtype"], case["operation"]
    values = evaluator.values_for(case, seed)
    base_device = "cpu" if op == "device_roundtrip" else "cuda:0"
    transpose = op.startswith("transposed_")
    alias = op.endswith("_alias")
    slice_view = op == "offset_slice_alias"

    def tensor(shape, strides, device, data, pointer, offset=0):
        return dict(shape=shape, strides=strides, device=device, values=data.copy(),
                    pointer=pointer, storage_offset=offset, dtype=dtype,
                    pointer_attributes=dict(memory_type=2, device_ordinal=0, is_managed=0)
                    if device == "cuda:0" else None)

    def base(data):
        return tensor([7, 13], [13, 1], base_device, data, 10000)

    def trans(data):
        matrix = [data[i:i+13] for i in range(0, 91, 13)]
        return [v for col in zip(*matrix) for v in col]

    def source(data):
        return tensor([13, 7], [1, 13], base_device, trans(data), 10000) if transpose else base(data)

    def output(data):
        if slice_view:
            return tensor([7, 11], [13, 1], "cuda:0",
                          [v for i in range(0, 91, 13) for v in data[i+1:i+12]], 10004, 1)
        if op == "transpose_alias":
            return tensor([13, 7], [1, 13], "cuda:0", trans(data), 10000)
        if op == "transposed_contiguous":
            return tensor([13, 7], [7, 1], "cuda:0", trans(data), 30000)
        if op == "transposed_reshape_copy":
            return tensor([91], [1], "cuda:0", trans(data), 30000)
        return tensor([7, 13], [13, 1], "cuda:0", data, 30000)

    downloads = [tensor([7, 13], [13, 1], "cpu", values, ptr) for ptr in (40000, 50000)] if op == "device_roundtrip" else []
    mutated = values.copy()
    if op == "device_roundtrip":
        mutated[0] = evaluator.CTYPES[dtype](23).value
    else:
        mutated = [ctypes.c_float(v + 8).value for v in values]
    output_mutated = output(mutated if alias else values)
    final_output = copy.deepcopy(output_mutated)
    final_output["values"][0] = evaluator.CTYPES[dtype](-17).value
    final_base = mutated.copy()
    if alias:
        final_base[1 if slice_view else 0] = -17.0
    return [dict(base=base(values), source=source(values), output=output(values),
                 distinct_object=True, base_after=base(values), source_after=source(values),
                 mutated_base=base(mutated), source_after_base_mutation=source(mutated),
                 output_after_base_mutation=output_mutated,
                 base_after_output_mutation=base(final_base), source_after_output_mutation=source(final_base),
                 output_after_output_mutation=final_output, downloads=downloads,
                 downloads_after_base_mutation=copy.deepcopy(downloads),
                 downloads_after_output_mutation=copy.deepcopy(downloads))]


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.corpus = evaluator.corpus()
        self.seeds = [9173, 260910]
        self.source = dict(commit="fixture-commit", source_sha256="fixture-source")
        self.build = dict(**self.source, extension_sha256="fixture-extension",
                          build_command="fixture-build", rustc="fixture-rustc", cargo="fixture-cargo", nvcc="unused")
        self.trials = []
        for case in self.corpus["cases"]:
            for seed in self.seeds:
                row = dict(status="passed", case_id=case["id"], seed=seed,
                           observations=fixture_observations(case, seed),
                           cuda_runtimes=[dict(path="/fixture/libcudart.so", version=13000, status=0)])
                self.trials.append(dict(case_id=case["id"], seed=seed,
                    reference=dict(**copy.deepcopy(row), role="reference", pid=101,
                                   version="2.13.0+cu130", gpu=dict(name="fixture")),
                    candidate=dict(**copy.deepcopy(row), role="candidate", pid=202,
                                   extension=dict(sha256="fixture-extension"), blocked_imports=[], loaded_torch_modules=[])))

    def score(self):
        return evaluator.account(self.corpus, self.seeds, self.trials, self.build, self.source)

    def test_fixed_six_cases_and_previous_matrix_preserved(self):
        self.assertEqual([(c["id"], c["operation"], c["dtype"], c["shape"]) for c in self.corpus["cases"]], [
            ("cuda_f64_device_roundtrip", "device_roundtrip", "float64", [7, 13]),
            ("cuda_i64_device_roundtrip", "device_roundtrip", "int64", [7, 13]),
            ("cuda_f32_transpose_alias", "transpose_alias", "float32", [7, 13]),
            ("cuda_f32_offset_slice_alias", "offset_slice_alias", "float32", [7, 13]),
            ("cuda_f32_transposed_contiguous", "transposed_contiguous", "float32", [7, 13]),
            ("cuda_f32_transposed_reshape_copy", "transposed_reshape_copy", "float32", [7, 13]),
        ])
        self.assertEqual(self.corpus["case_aggregation"], "equal_fraction_fixed_denominator")
        matrix = json.loads(evaluator.MATRIX.read_text())
        next(c for c in matrix["capabilities"] if c["id"] == evaluator.CAPABILITY).pop("case_sets")
        # Locks all prior backend/capability weights, case sets and policy together.
        digest = hashlib.sha256(json.dumps(matrix, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, "675c71c4840f84d2b205f0a97c3f8c81f6c032a2657d9ff8508dc5b02479f4a3")

    def test_equal_weight_and_all_seeds_required(self):
        self.assertEqual(self.score()["fraction"], 1)
        for trial in self.trials:
            if trial["case_id"] not in ("cuda_f32_transpose_alias", "cuda_f32_offset_slice_alias"):
                trial["candidate"]["status"] = "unsupported"
        self.assertEqual((self.score()["denominator"], self.score()["passed"]), (6, 2))
        self.assertEqual(15 * self.score()["fraction"], 5)
        self.trials[4]["candidate"]["status"] = "failed"
        self.assertEqual(self.score()["passed"], 1)
        self.assertTrue(all(c["reference_eligible"] for c in self.score()["cases"]))

    def test_missing_failed_unsupported_forwarded_malformed_stay_zero(self):
        original = copy.deepcopy(self.trials)
        for role in ("reference", "candidate"):
            for replacement in (None, [], 42, {}, *({"status": s} for s in ("skipped", "unsupported", "forwarded", "failed"))):
                with self.subTest(role=role, replacement=replacement):
                    self.trials = copy.deepcopy(original)
                    for t in self.trials:
                        t[role] = replacement
                    self.assertEqual((self.score()["denominator"], self.score()["passed"]), (6, 0))
        for trials in ([], original + original, [None, 42, {}]):
            self.trials = trials
            self.assertEqual((self.score()["denominator"], self.score()["passed"]), (6, 0))

    def test_every_observation_required_on_both_sides(self):
        for trial in self.trials:
            case = next(c for c in self.corpus["cases"] if c["id"] == trial["case_id"])
            for role in ("reference", "candidate"):
                original = trial[role]
                for key in original["observations"][0]:
                    with self.subTest(case=case["id"], role=role, key=key):
                        row = copy.deepcopy(original)
                        del row["observations"][0][key]
                        self.assertFalse(evaluator.valid_execution(row, case, trial["seed"], role))
                for field, bad in (("dtype", "float16"), ("shape", [123]), ("strides", [123]),
                                   ("storage_offset", 2), ("pointer", 0), ("device", "cpu"),
                                   ("values", None), ("values", [float("nan")] * 91),
                                   ("pointer_attributes", dict(memory_type=2, device_ordinal=1, is_managed=0))):
                    row = copy.deepcopy(original)
                    row["observations"][0]["output"][field] = bad
                    self.assertFalse(evaluator.valid_execution(row, case, trial["seed"], role))

    def test_alias_and_copy_mutation_failures(self):
        for trial in self.trials:
            case = next(c for c in self.corpus["cases"] if c["id"] == trial["case_id"])
            original = trial["candidate"]
            for key in ("output_after_base_mutation", "base_after_output_mutation", "source_after_output_mutation"):
                row = copy.deepcopy(original)
                row["observations"][0][key]["values"][0] += 1
                self.assertFalse(evaluator.valid_execution(row, case, trial["seed"], "candidate"))
            # Change pointers consistently across snapshots to bypass preservation checks.
            row = copy.deepcopy(original)
            obs = row["observations"][0]
            for key in ("output", "output_after_base_mutation", "output_after_output_mutation"):
                obs[key]["pointer"] = 10000 if case["operation"].startswith("transposed_") else 10008
            if case["operation"] != "device_roundtrip":
                self.assertFalse(evaluator.valid_execution(row, case, trial["seed"], "candidate"))

    def test_dtype_precision_is_not_satisfied_by_float32(self):
        for trial in self.trials[:4]:
            case = next(c for c in self.corpus["cases"] if c["id"] == trial["case_id"])
            row = copy.deepcopy(trial["candidate"])
            values = row["observations"][0]["output"]["values"]
            self.assertTrue(any(ctypes.c_float(v).value != v for v in values))
            row["observations"][0]["output"]["values"] = [ctypes.c_float(v).value for v in values]
            self.assertFalse(evaluator.valid_execution(row, case, trial["seed"], "candidate"))

    def test_binding_forwarding_and_runtime_fail_closed(self):
        original = copy.deepcopy(self.trials)
        changes = [dict(extension={"sha256": "stale"}), dict(pid=101), dict(seed=True),
                   dict(blocked_imports=["torch"]), dict(loaded_torch_modules=["torch._C"]),
                   dict(cuda_runtimes=[]), dict(cuda_runtimes=42), dict(observations=42)]
        for change in changes:
            self.trials = copy.deepcopy(original)
            for trial in self.trials:
                trial["candidate"].update(change)
            self.assertEqual(self.score()["passed"], 0)
        self.trials = original
        original_build = self.build.copy()
        for key in original_build:
            self.build = {k: v for k, v in original_build.items() if k != key}
            self.assertEqual(self.score()["passed"], 0)
        for build in (None, [], 42, {}, {**original_build, "nvcc": []}):
            self.build = build
            self.assertEqual(self.score()["passed"], 0)
        self.build = original_build
        for seeds in ([], [9173], [9173, 9173], [9173, -9173]):
            self.seeds = seeds
            self.assertEqual(self.score()["passed"], 0)


class IsolationTests(unittest.TestCase):
    def test_malformed_build_receipt_still_emits_six_slots(self):
        work = ROOT / "target/dtypes-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            receipt, output = Path(temp) / "receipt.json", Path(temp) / "result.json"
            receipt.write_text('{"invalid": NaN}')
            argv = [evaluator.__file__, "--seed", "1", "--seed", "2",
                    "--build-record", str(receipt), "--output", str(output)]
            with (patch.object(sys, "argv", argv),
                  patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"),
                  patch.object(evaluator, "launch", return_value={"status": "failed"}) as launch,
                  patch.object(evaluator.subprocess, "check_output", return_value="fixture inventory"),
                  patch.object(evaluator.common, "source_provenance", return_value={"commit": "fixture"})):
                self.assertEqual(evaluator.main(), 2)
            result = json.loads(output.read_text())
            self.assertEqual(launch.call_count, 24)
            self.assertEqual((result["accounting"]["denominator"], result["accounting"]["passed"]), (6, 0))
            self.assertIn("non-finite", result["build_record_error"])

    def test_worker_launch_rejects_malformed_and_nonfinite_json(self):
        case = evaluator.corpus()["cases"][0]
        for output in ("invalid", "[]", '{"status":"passed", "x":NaN}', '{"x":1e999}'):
            with patch.object(evaluator.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, output, "")):
                self.assertEqual(evaluator.launch("candidate", case, 1, sys.executable, 1, os.environ)["status"], "failed")
        with patch.object(evaluator.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)):
            self.assertEqual(evaluator.launch("candidate", case, 1, sys.executable, 1, os.environ)["status"], "failed")

    def test_cli_device_seed_and_output_contract(self):
        for mask, args in (("", []), ("0", ["--seed", "1"]),
                           ("0", ["--seed", "1", "--seed", "2", "--output", "/tmp/forbidden.json"]),
                           ("0", ["--seed", "1", "--seed", "2", "--output", str(ROOT / ".burner/forbidden.json")])):
            result = subprocess.run([sys.executable, "-I", "-B", evaluator.__file__, *args],
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": mask}, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stderr)

    def test_swallowed_forwarding_is_zero_even_without_runtime_provenance(self):
        work = ROOT / "target/dtypes-tests"
        work.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            package = Path(temp) / "python/torch_rs"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("try:\n import torch\nexcept RuntimeError:\n pass\n__version__ = 'fixture'\n")
            code = f"""
import sys, json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, {str(ROOT / 'scripts')!r})
import evaluate_cuda_dtypes_layout_views as e
e.ROOT = Path({temp!r})
with patch.object(e.common, 'runtime_provenance', side_effect=FileNotFoundError('maps')):
    print(json.dumps(e.worker('candidate', {{'case': e.corpus()['cases'][0], 'seed': 1}})))
"""
            result = subprocess.run([sys.executable, "-I", "-B", "-c", code], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            row = json.loads(result.stdout)
            self.assertEqual(row["status"], "forwarded")
            self.assertEqual(row["blocked_imports"], ["torch"])


@unittest.skipUnless(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "requires CUDA_VISIBLE_DEVICES=0")
class HardwareTests(unittest.TestCase):
    def test_reference_cases(self):
        work = ROOT / "target/dtypes-tests"
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


if __name__ == "__main__":
    unittest.main()
