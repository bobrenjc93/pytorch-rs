#!/usr/bin/env python3
"""Fixed CUDA inference compilation correctness; no performance or overall score."""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import inspect
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone

# -I workers deliberately omit both PYTHONPATH and the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_cuda_math as common

ROOT, MATRIX = common.ROOT, common.MATRIX
VERSION = "cuda_float32_inference_compilation_v1"
OPTIONS = {"backend": "eager", "fullgraph": True, "dynamic": False}
CHANGE_MASK = 0x9E3779B97F4A7C15


def corpus():
    matrix = json.loads(MATRIX.read_text())
    capability = next(c for c in matrix["capabilities"] if c["id"] == "compilation")
    return next(c for c in capability["case_sets"] if c["version"] == VERSION)


# Evaluator-owned programs: no diagnostic, performance, or compiler-corpus imports.
def same_shape_add(a, b):
    return a + b


def trailing_vector_add(a, b):
    return a + b


def negation(a):
    return -a


def literal_multiply(a):
    return a * -1.75


def row_sum(a):
    return a.sum(dim=1, keepdim=False)


def matrix_product(a, b):
    return a @ b


PROGRAMS = dict(zip((
    "cuda_f32_compile_add_same_shape", "cuda_f32_compile_add_trailing_vector",
    "cuda_f32_compile_neg", "cuda_f32_compile_mul_scalar",
    "cuda_f32_compile_sum_rows", "cuda_f32_compile_matmul",
), (same_shape_add, trailing_vector_add, negation, literal_multiply, row_sum, matrix_product)))


def native_hook(case):
    return "_compile_trace_" + {
        "add": "binary", "neg": "unary", "mul_scalar": "scalar",
        "sum": "unary", "matmul": "binary",
    }[case["operation"]]


class CompileObserver:
    """Observe real calls without substituting the lowerer or native executor."""

    def __init__(self, program, lowerer, executor, hook):
        if not inspect.isbuiltin(hook):
            raise RuntimeError("native executor hook is not an extension builtin")
        self.body = program.__code__
        self.lowerer = lowerer.__code__
        self.executor = executor.__code__
        self.hook = hook
        self.body_attempts = 0
        self.phases = []

    @contextmanager
    def phase(self, name):
        evidence = {"phase": name, "lowered_targets": [], "executor_calls": 0,
                    "native_returns": 0, "hook": self.hook.__name__}
        self.phases.append(evidence)

        def observe(frame, event, arg):
            if event == "call" and frame.f_code == self.body:
                self.body_attempts += 1
                raise RuntimeError("original Python body execution blocked")
            if event == "call" and frame.f_code == self.lowerer and name == "changed":
                raise RuntimeError("changed-input execution attempted to lower again")
            if event == "return" and frame.f_code == self.lowerer and arg is not None:
                evidence["lowered_targets"].append([op.target for op in arg.operations])
            if event == "call" and frame.f_code == self.executor:
                evidence["executor_calls"] += 1
            if event == "c_return" and arg is self.hook:
                evidence["native_returns"] += 1

        previous = sys.getprofile()
        try:
            sys.setprofile(observe)
            yield
        finally:
            sys.setprofile(previous)


class Inspector(common.CudaInspector):
    def materialize(self, tensor, shape, tensor_type):
        result = super().materialize(tensor, shape, tensor_type)
        result.update(strides=list(tensor.stride()), storage_offset=tensor.storage_offset(),
                      requires_grad=tensor.requires_grad)
        return result


def execute(module, compiled, case, seed, inspector):
    values = common.inputs_for(case, seed)
    inputs = [module.tensor(v, dtype=module.float32).to("cuda:0") for v in values]
    before = [inspector.materialize(t, s, module.Tensor)
              for t, s in zip(inputs, case["input_shapes"])]
    output = compiled(*inputs)
    materialized = inspector.materialize(output, case["output_shape"], module.Tensor)
    public = output.cpu()
    public_matches = (str(public.device) == "cpu"
                      and common.flatten(public.tolist()) == materialized["values"])
    after = [inspector.materialize(t, s, module.Tensor)
             for t, s in zip(inputs, case["input_shapes"])]
    return {"seed": seed, "inputs": before, "inputs_after": after,
            "output": materialized, "public_materialization_matches": public_matches}


def worker(role, request):
    case, seed = request["case"], request["seed"]
    result = {"status": "failed", "role": role, "case_id": case["id"], "seed": seed,
              "pid": os.getpid(), "python": sys.version, "executable": sys.executable,
              "compile_options": OPTIONS.copy(), "executions": []}
    blocker = common.BlockTorch() if role == "candidate" else None
    observer = None
    try:
        if blocker:
            blocker.check()
            sys.meta_path.insert(0, blocker)
            sys.path.insert(0, str(ROOT / "python"))
        module = importlib.import_module("torch_rs" if blocker else "torch")
        result.update(package=str(Path(module.__file__).resolve()), version=str(module.__version__))
        if blocker:
            extension = importlib.import_module("torch_rs.torch_rs")
            path = Path(extension.__file__).resolve()
            if (path.parent != ROOT / "python/torch_rs" or not isinstance(
                    extension.__spec__.loader, importlib.machinery.ExtensionFileLoader)):
                raise RuntimeError("candidate extension is not local native code")
            result["extension"] = {"path": str(path), "sha256": common.sha256(path)}
            lower = importlib.import_module("torch_rs._compile_bytecode")
            trace = importlib.import_module("torch_rs._compile_trace")
            observer = CompileObserver(PROGRAMS[case["id"]], lower.lower_compile_graph,
                                       trace.execute_compile_trace_graph,
                                       getattr(extension, native_hook(case)))
            blocker.check()
        else:
            if module.__version__.split("+")[0] != "2.13.0" or module.version.hip:
                raise RuntimeError("reference requires NVIDIA PyTorch 2.13.0")
            if not module.cuda.is_available():
                result.update(status="skipped", error="reference NVIDIA CUDA hardware unavailable")
                return result
            module.backends.cuda.matmul.allow_tf32 = False
            result["pytorch_cuda"] = module.version.cuda
            properties = module.cuda.get_device_properties(0)
            result["gpu"] = {"name": properties.name,
                             "compute_capability": [properties.major, properties.minor],
                             "total_memory": properties.total_memory, "uuid": str(properties.uuid)}
            result["memory_free_total"] = list(module.cuda.mem_get_info(0))
        inspector = Inspector()
        # The same wrapper must execute both datasets. Candidate profiling also
        # covers compile() itself, so eager body execution there cannot pass.
        compiled = None
        for phase, data_seed in (("initial", seed), ("changed", seed ^ CHANGE_MASK)):
            def run():
                nonlocal compiled
                if compiled is None:
                    compiled = module.compile(PROGRAMS[case["id"]], **OPTIONS)
                result["executions"].append(execute(module, compiled, case, data_seed, inspector))
            if observer:
                with observer.phase(phase):
                    run()
            else:
                run()
        if blocker:
            blocker.check()
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "forwarded" if isinstance(error, common.ForwardingError) else "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if observer:
            result["compile_evidence"] = observer.phases
            result["original_body_attempts"] = observer.body_attempts
        if blocker:
            result["blocked_imports"] = blocker.attempts
            result["loaded_torch_modules"] = [
                n for n in sys.modules if n == "torch" or n.startswith("torch.")]
            if result["blocked_imports"] or result["loaded_torch_modules"]:
                result["status"] = "forwarded"
        try:
            result.update(common.runtime_provenance())
            result["mapped_cuda_libraries"] = sorted({
                line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                if any(name in line for name in ("/libcuda.so", "/libcublas", "/libnvidia-ptxjitcompiler"))})
            result["TORCH_RS_CUBLAS"] = os.environ.get("TORCH_RS_CUBLAS")
        except Exception as error:
            result.update(status="failed", runtime_error=str(error))
    # Preserve observations even for malformed/mutating execution.
    if result["status"] == "passed" and not valid_execution(result, case, seed, role):
        result.update(status="failed", error="invalid execution or compilation evidence")
    return result


def valid_execution(row, case, seed, role):
    if (type(seed) is not int or seed < 0 or not isinstance(row, dict)
            or row.get("compile_options") != OPTIONS):
        return False
    if (row["compile_options"].get("fullgraph") is not True
            or row["compile_options"].get("dynamic") is not False):
        return False
    if type(row.get("seed")) is not int or row["seed"] != seed:
        return False
    executions = row.get("executions")
    if not isinstance(executions, list) or len(executions) != 2:
        return False
    for execution, data_seed in zip(executions, (seed, seed ^ CHANGE_MASK)):
        if (not isinstance(execution, dict) or type(execution.get("seed")) is not int
                or execution["seed"] != data_seed
                or execution.get("public_materialization_matches") is not True):
            return False
        observed = {**row, "seed": data_seed, **{key: execution.get(key)
                    for key in ("inputs", "inputs_after", "output")}}
        try:
            valid = common.valid_execution(observed, case, data_seed, role)
        except OverflowError:
            # JSON integers are unbounded; the shared validator's isfinite()
            # conversion can overflow. Reject this row without losing other slots.
            return False
        if not valid:
            return False
        for tensor in execution["inputs"] + [execution["output"]]:
            shape = tensor["shape"]
            strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
            if (tensor.get("strides") != strides or tensor.get("storage_offset") != 0
                    or type(tensor.get("storage_offset")) is not int
                    or any(type(v) is not int for v in tensor["strides"] + shape)
                    or tensor.get("requires_grad") is not False):
                return False
    if executions[0]["inputs"] == executions[1]["inputs"]:
        return False
    if role == "candidate":
        if type(row.get("original_body_attempts")) is not int or row["original_body_attempts"] != 0:
            return False
        expected = [{"phase": phase, "lowered_targets": targets, "executor_calls": 1,
                     "native_returns": 1, "hook": native_hook(case)}
                    for phase, targets in (("initial", [[case["operation"]]]), ("changed", []))]
        if row.get("compile_evidence") != expected:
            return False
        if any(type(phase[key]) is not int for phase in row["compile_evidence"]
               for key in ("executor_calls", "native_returns")):
            return False
    return True


def account(case_set, seeds, trials, build_record, source):
    """Fixed six slots; neither malformed rows nor reference failures shrink them."""
    rows = []
    trials = trials if isinstance(trials, list) else []
    for case in case_set["cases"]:
        verdicts = []
        for seed in seeds:
            matches = [t for t in trials if isinstance(t, dict)
                       and t.get("case_id") == case["id"] and t.get("seed") == seed]
            verdict = {"seed": seed, "credit": 0, "reference_eligible": False,
                       "reason": "missing_or_duplicate_trial"}
            if len(matches) == 1:
                ref, cand = matches[0].get("reference"), matches[0].get("candidate")
                eligible = valid_execution(ref, case, seed, "reference")
                verdict["reference_eligible"] = eligible
                if not eligible:
                    verdict["reason"] = "reference_unavailable_or_failed"
                elif not valid_execution(cand, case, seed, "candidate"):
                    verdict["reason"] = "candidate_missing_failed_unsupported_or_invalid"
                elif (not isinstance(build_record, dict)
                      or any(not source.get(k) or build_record.get(k) != source[k]
                             for k in ("commit", "source_sha256", "production_diff_sha256"))
                      or any(not isinstance(build_record.get(k), str) or not build_record[k].strip()
                             for k in ("build_command", "rustc", "cargo", "nvcc"))
                      or build_record.get("extension_sha256") != cand["extension"]["sha256"]
                      or ref["pid"] == cand["pid"]):
                    verdict["reason"] = "unbound_candidate_build_or_process"
                elif not common.valid_seeds(seeds):
                    verdict["reason"] = "invalid_or_insufficient_distinct_seeds"
                else:
                    correct = all(
                        abs(a - b) <= case_set["atol"] + case_set["rtol"] * abs(b)
                        for r, c in zip(ref["executions"], cand["executions"])
                        for a, b in zip(c["output"]["values"], r["output"]["values"]))
                    verdict.update(credit=int(correct), reason="passed" if correct else "incorrect")
            verdicts.append(verdict)
        rows.append({"case_id": case["id"], "credit": int(bool(verdicts) and all(
            v["credit"] for v in verdicts)), "trials": verdicts})
    passed = sum(row["credit"] for row in rows)
    return {"denominator": len(case_set["cases"]), "passed": passed,
            "fraction": passed / len(case_set["cases"]), "cases": rows}


def finite_json_float(token):
    """Reject non-finite constants and exponents before retaining JSON evidence."""
    value = float(token)
    if not math.isfinite(value):
        raise ValueError(f"non-finite JSON number: {token}")
    return value


def launch(role, case, seed, python, timeout, env):
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", str(Path(__file__).resolve()), "--worker", role],
            input=json.dumps({"case": case, "seed": seed}), capture_output=True,
            text=True, cwd=ROOT, env=env, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(f"worker exit {completed.returncode}: {completed.stderr[-4000:]}")
        result = json.loads(completed.stdout, parse_float=finite_json_float,
                            parse_constant=finite_json_float)
        if not isinstance(result, dict):
            raise ValueError("worker result must be an object")
        result["stderr"] = completed.stderr[-4000:]
        return result
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--build-record", type=Path)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("reference", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, json.load(sys.stdin)), allow_nan=False))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("run with CUDA_VISIBLE_DEVICES=0")
    seeds = args.seed if args.seed is not None else [secrets.randbits(63) for _ in range(2)]
    if not common.valid_seeds(seeds):
        parser.error("at least two distinct nonnegative integer seeds are required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.output and (not args.output.resolve().is_relative_to(ROOT)
                        or ".burner" in args.output.resolve().relative_to(ROOT).parts):
        parser.error("output must stay inside this worktree and outside .burner")
    case_set, source = corpus(), common.source_provenance()
    started_at = datetime.now(timezone.utc).isoformat()
    tracked = [Path(__file__), Path(common.__file__), MATRIX,
               *sorted((ROOT / "python/torch_rs").glob("torch_rs*.so"))]
    hashes = {str(p.relative_to(ROOT)): common.sha256(p) for p in tracked}
    build_record, build_error = None, None
    try:
        if args.build_record:
            build_record = json.loads(args.build_record.read_text(), parse_float=finite_json_float,
                                      parse_constant=finite_json_float)
    except (OSError, ValueError) as error:
        build_error = str(error)  # An invalid receipt must still emit all six slots.
    try:
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        inventory = str(error)
    work = ROOT / "target/cuda-compilation"
    work.mkdir(parents=True, exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(dir=work) as temp:
        env = {**os.environ, "TMPDIR": temp, "CUDA_CACHE_PATH": temp,
               "XDG_CACHE_HOME": temp, "TORCHINDUCTOR_CACHE_DIR": temp,
               "TRITON_CACHE_DIR": temp, "PYTHONDONTWRITEBYTECODE": "1"}
        for case in case_set["cases"]:
            for seed in seeds:
                trials.append({"case_id": case["id"], "seed": seed,
                               "reference": launch("reference", case, seed, args.reference_python, args.timeout, env),
                               "candidate": launch("candidate", case, seed, args.candidate_python, args.timeout, env)})
    stable = (source == common.source_provenance()
              and all(p.is_file() and hashes[str(p.relative_to(ROOT))] == common.sha256(p)
                      for p in tracked))
    result = {"evaluator_version": "cuda_inference_compilation_correctness_v1",
              "case_set_version": VERSION, "backend": "cuda_nvidia", "capability": "compilation",
              "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
              "scope": "six-case inference correctness only; no performance, overall score, or implementation gain",
              "source": source, "source_unchanged_during_run": stable, "file_sha256": hashes,
              "build_record": build_record, "build_record_error": build_error, "seeds": seeds,
              "compile_options": OPTIONS, "CUDA_VISIBLE_DEVICES": "0", "gpu_inventory": inventory,
              "trials": trials,
              "accounting": account(case_set, seeds, trials, build_record if stable else None, source)}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if all(valid_execution(t["reference"], c, t["seed"], "reference")
                    for c in case_set["cases"] for t in trials if t["case_id"] == c["id"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
