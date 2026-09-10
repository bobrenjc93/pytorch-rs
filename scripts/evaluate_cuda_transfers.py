#!/usr/bin/env python3
"""Differential correctness for CUDA's six fixed device/transfer cases only."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.machinery
import itertools
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

# Isolated (-I) workers do not otherwise include the script directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_cuda_math as common

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/hardware-heterogeneity-matrix-v1.json"
VERSION = "cuda_float32_transfers_v1"
CAPABILITY = "device_tensors_and_transfers"


def corpus():
    matrix = json.loads(MATRIX.read_text())
    capability = next(c for c in matrix["capabilities"] if c["id"] == CAPABILITY)
    return next(c for c in capability["case_sets"] if c["version"] == VERSION)


def values_for(shape, seed, zeros=False):
    if zeros:
        return [0.0] * math.prod(shape)
    return common.flatten(common.inputs_for({"input_shapes": [shape]}, seed)[0])


class Inspector(common.CudaInspector):
    """Read logical elements directly, including offset and non-dense views."""

    def snapshot(self, tensor, tensor_type):
        if type(tensor) is not tensor_type or str(tensor.dtype) != "torch.float32":
            raise RuntimeError("expected public native float32 Tensor")
        shape, strides = list(tensor.shape), list(tensor.stride())
        device = str(tensor.device)
        if device not in ("cpu", "cuda:0") or bool(tensor.is_cuda) != (device == "cuda:0"):
            raise RuntimeError("unexpected tensor device")
        pointer = tensor.data_ptr()
        attrs = None
        if math.prod(shape):
            if not pointer:
                raise RuntimeError("nonempty tensor has null storage")
            if device == "cuda:0":
                attrs = {}
                for name, attribute in (("memory_type", 2), ("device_ordinal", 9), ("is_managed", 8)):
                    value = ctypes.c_int()
                    self.call("cuPointerGetAttribute", ctypes.byref(value), attribute, pointer)
                    attrs[name] = value.value
                if attrs != {"memory_type": 2, "device_ordinal": 0, "is_managed": 0}:
                    raise RuntimeError("not native CUDA storage")
                self.call("cuCtxSynchronize")
        elif pointer != 0:
            raise RuntimeError("empty allocation must have null data pointer")
        values = []
        for index in itertools.product(*(range(n) for n in shape)):
            address = pointer + 4 * sum(i * s for i, s in zip(index, strides))
            if device == "cuda:0":
                value = ctypes.c_float()
                self.call("cuMemcpyDtoH_v2", ctypes.byref(value), address, 4)
                values.append(value.value)
            else:
                values.append(ctypes.c_float.from_address(address).value)
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("nonfinite storage")
        if device == "cpu" and common.flatten(tensor.tolist()) != values:
            raise RuntimeError("public CPU values disagree with storage")
        return {"shape": shape, "strides": strides, "storage_offset": tensor.storage_offset(),
                "dtype": "float32", "device": device, "pointer": pointer,
                "pointer_attributes": attrs, "values": values}

    def initialize(self, tensor, values):
        # Fixture setup ONLY: this must never implement an evaluated transfer.
        host = (ctypes.c_float * len(values))(*values)
        if str(tensor.device) == "cuda:0":
            fn = self.driver.cuMemcpyHtoD_v2
            fn.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
            fn.restype = ctypes.c_int
            self.call("cuMemcpyHtoD_v2", tensor.data_ptr(), host, ctypes.sizeof(host))
            self.call("cuCtxSynchronize")
        else:
            ctypes.memmove(tensor.data_ptr(), host, ctypes.sizeof(host))


def execute(module, case, seed, inspector, observations):
    op = case["operation"]
    for shape in case["shapes"]:
        zero = op in ("vector_zero_roundtrip", "matrix_zero_roundtrip")
        values = values_for(shape, seed, zero)
        if zero:
            base = module.zeros(shape, dtype=module.float32, device="cuda:0")
        elif op in ("contiguous_cpu_upload", "strided_cpu_upload"):
            base = module.tensor(values, dtype=module.float32).reshape(shape)
        else:
            # Rank-one public allocation avoids coupling view/copy eligibility
            # to matrix allocation or public CPU upload support.
            base = module.zeros((math.prod(shape),), dtype=module.float32, device="cuda:0")
            inspector.initialize(base, values)
            base = base.reshape(shape)
        source = base[:, 1:12].transpose(0, 1) if "view" in case else base
        before = inspector.snapshot(base, module.Tensor)
        source_before = inspector.snapshot(source, module.Tensor)
        if before["values"] != values:
            raise RuntimeError("fixture initialization mismatch")
        observation = {"base": before, "source": source_before, "copies": []}
        observations.append(observation)
        if case.get("no_copy_identity"):
            observation["no_copy_identity"] = source.to("cuda:0") is source
            if not observation["no_copy_identity"]:
                raise RuntimeError("same-device no-copy to must preserve identity")
        outputs = []
        for method in case["copies"]:
            if method == "cpu":
                output = source.cpu()
            elif method == "to_cpu":
                output = source.to("cpu")
            elif method == "to_cuda":
                output = source.to("cuda:0")
            elif method == "clone":
                output = source.clone()
            elif method == "to_cuda_copy":
                output = source.to("cuda:0", copy=True)
            else:
                raise ValueError("unknown fixed copy: " + method)
            outputs.append(output)  # Keep allocations alive to test independence.
            observation["copies"].append({"method": method,
                "distinct_object": output is not source,
                "output": inspector.snapshot(output, module.Tensor)})
        observation["base_after"] = inspector.snapshot(base, module.Tensor)
        observation["source_after"] = inspector.snapshot(source, module.Tensor)
        # A copy must survive mutation of its source. Use an evaluator-owned
        # write on both frameworks, without requiring candidate arithmetic.
        if values:
            inspector.initialize(base, [v + 8.0 for v in values])
        observation["mutated_base"] = inspector.snapshot(base, module.Tensor)
        observation["copies_after_source_mutation"] = [
            inspector.snapshot(output, module.Tensor) for output in outputs]


def worker(role, request):
    result = {"status": "failed", "role": role, "case_id": request["case"]["id"],
              "seed": request["seed"], "pid": os.getpid(), "python": sys.version,
              "executable": sys.executable, "observations": []}
    blocker = common.BlockTorch() if role == "candidate" else None
    try:
        if blocker:
            blocker.check()
            sys.meta_path.insert(0, blocker)
            sys.path.insert(0, str(ROOT / "python"))
        module = importlib.import_module("torch_rs" if blocker else "torch")
        result.update(package=str(Path(module.__file__).resolve()), version=str(module.__version__))
        if blocker:
            blocker.check()
            extension = importlib.import_module("torch_rs.torch_rs")
            path = Path(extension.__file__).resolve()
            if path.parent != ROOT / "python/torch_rs" or not isinstance(
                    extension.__spec__.loader, importlib.machinery.ExtensionFileLoader):
                raise RuntimeError("candidate extension is not local native code")
            result["extension"] = {"path": str(path), "sha256": common.sha256(path)}
        else:
            if module.__version__.split("+")[0] != "2.13.0" or module.version.hip:
                raise RuntimeError("reference requires NVIDIA PyTorch 2.13.0")
            if not module.cuda.is_available():
                result.update(status="skipped", error="reference NVIDIA CUDA hardware unavailable")
                return result
            props = module.cuda.get_device_properties(0)
            result.update(pytorch_cuda=module.version.cuda,
                          pytorch_build_config=module.__config__.show(),
                          gpu={"name": props.name, "uuid": str(props.uuid),
                               "compute_capability": [props.major, props.minor],
                               "total_memory": props.total_memory},
                          memory_free_total=list(module.cuda.mem_get_info(0)))
        execute(module, request["case"], request["seed"], Inspector(), result["observations"])
        if blocker:
            blocker.check()
        result["status"] = "passed"
    except Exception as error:
        status = "unsupported" if isinstance(error, NotImplementedError) else "failed"
        result.update(status=status, error=f"{type(error).__name__}: {error}")
    finally:
        if blocker:
            result["blocked_imports"] = blocker.attempts
            result["loaded_torch_modules"] = [n for n in sys.modules if n == "torch" or n.startswith("torch.")]
            if result["blocked_imports"] or result["loaded_torch_modules"]:
                result["status"] = "forwarded"
        try:
            result.update(common.runtime_provenance())
        except Exception as error:
            result["runtime_error"] = str(error)
            # Missing provenance invalidates success, but must not erase the
            # original zero-credit verdict (including a forwarding attempt).
            if result["status"] == "passed":
                result["status"] = "failed"
    return result


def launch(role, case, seed, python, timeout, env):
    try:
        process = subprocess.run([str(python), "-I", "-B", str(Path(__file__).resolve()), "--worker", role],
                                 input=json.dumps({"case": case, "seed": seed}), capture_output=True,
                                 text=True, cwd=ROOT, env=env, timeout=timeout)
        row = json.loads(process.stdout)
        if process.returncode or not isinstance(row, dict):
            raise ValueError(f"invalid worker result (exit {process.returncode}): {process.stderr[-4000:]}")
        row["stderr"] = process.stderr[-4000:]
        return row
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def tensor_valid(tensor, shape, strides, offset, device, values):
    if not isinstance(tensor, dict):
        return False
    expected = {"shape": shape, "strides": strides, "storage_offset": offset,
                "dtype": "float32", "device": device, "values": values,
                "pointer_attributes": {"memory_type": 2, "device_ordinal": 0, "is_managed": 0}
                if device == "cuda:0" and values else None}
    return (all(tensor.get(k) == v for k, v in expected.items())
            and type(tensor.get("pointer")) is int
            and (tensor["pointer"] > 0 if values else tensor["pointer"] == 0)
            and all(type(v) in (int, float) and math.isfinite(v) for v in tensor["values"]))


def valid_execution(row, case, seed, role):
    try:
        return _valid_execution(row, case, seed, role)
    except (TypeError, KeyError, ValueError, OverflowError):
        # Corrupt or incomplete worker JSON is a zero, never a smaller corpus.
        return False


def _valid_execution(row, case, seed, role):
    """Validate expected semantics independently of either library's claims."""
    if not isinstance(row, dict) or row.get("status") != "passed":
        return False
    if (row.get("case_id"), row.get("seed"), row.get("role")) != (case["id"], seed, role):
        return False
    if (type(row.get("pid")) is not int or not isinstance(row.get("cuda_runtimes"), list)
            or not row["cuda_runtimes"]):
        return False
    if any(not isinstance(r, dict) or r.get("status") != 0 or not r.get("path")
           or type(r.get("version")) is not int or r["version"] <= 0 for r in row["cuda_runtimes"]):
        return False
    if role == "reference":
        if not row.get("gpu") or str(row.get("version")).split("+")[0] != "2.13.0":
            return False
    elif (row.get("blocked_imports") != [] or row.get("loaded_torch_modules") != []
          or not isinstance(row.get("extension"), dict) or not row["extension"].get("sha256")):
        return False
    observations = row.get("observations")
    if not isinstance(observations, list) or len(observations) != len(case["shapes"]):
        return False
    for obs, shape in zip(observations, case["shapes"]):
        if not isinstance(obs, dict):
            return False
        zero = case["operation"] in ("vector_zero_roundtrip", "matrix_zero_roundtrip")
        values = values_for(shape, seed, zero)
        strides = [math.prod(shape[i+1:]) for i in range(len(shape))]
        device = "cpu" if case["operation"] in ("contiguous_cpu_upload", "strided_cpu_upload") else "cuda:0"
        if not tensor_valid(obs.get("base"), shape, strides, 0, device, values):
            return False
        src_shape, src_strides, src_offset, src_values = shape, strides, 0, values
        if "view" in case:
            src_shape, src_strides, src_offset = [11, 7], [1, 13], 1
            src_values = [values[r * 13 + c] for c in range(1, 12) for r in range(7)]
        if not tensor_valid(obs.get("source"), src_shape, src_strides, src_offset, device, src_values):
            return False
        if obs.get("base_after") != obs["base"] or obs.get("source_after") != obs["source"]:
            return False
        if obs["source"]["pointer"] != obs["base"]["pointer"] + 4 * src_offset:
            return False
        mutated = [ctypes.c_float(v + 8.0).value for v in values]
        if not tensor_valid(obs.get("mutated_base"), shape, strides, 0, device, mutated):
            return False
        if obs["mutated_base"]["pointer"] != obs["base"]["pointer"]:
            return False
        if case.get("no_copy_identity") and obs.get("no_copy_identity") is not True:
            return False
        copies = obs.get("copies")
        if not isinstance(copies, list) or len(copies) != len(case["copies"]):
            return False
        pointers = []
        for copy, method in zip(copies, case["copies"]):
            if not isinstance(copy, dict) or copy.get("method") != method or copy.get("distinct_object") is not True:
                return False
            output_device = "cpu" if method in ("cpu", "to_cpu") else "cuda:0"
            output_strides = [1, 11] if "view" in case else strides
            output = copy.get("output")
            if not tensor_valid(output, src_shape, output_strides, 0, output_device, src_values):
                return False
            if values:
                start, end = output["pointer"], output["pointer"] + len(src_values) * 4
                if any(start < other_end and other_start < end for other_start, other_end in pointers):
                    return False
                pointers.append((start, end))
                if device == output_device and start < obs["base"]["pointer"] + len(values) * 4 and obs["base"]["pointer"] < end:
                    return False
        if obs.get("copies_after_source_mutation") != [c["output"] for c in copies]:
            return False
    return True


def account(case_set, seeds, trials, build_record, source):
    rows = []
    for case in case_set["cases"]:
        verdicts = []
        for seed in seeds:
            matches = [t for t in trials if isinstance(t, dict) and t.get("case_id") == case["id"] and t.get("seed") == seed]
            verdict = {"seed": seed, "reference_eligible": False, "candidate_outcome": "missing_or_duplicate_trial",
                       "credit": 0, "reason": "missing_or_duplicate_trial"}
            if len(matches) == 1:
                ref, cand = matches[0].get("reference"), matches[0].get("candidate")
                eligible = valid_execution(ref, case, seed, "reference")
                verdict["reference_eligible"] = eligible
                candidate_valid = valid_execution(cand, case, seed, "candidate")
                verdict["candidate_outcome"] = (
                    "passed" if candidate_valid else
                    "missing" if not isinstance(cand, dict) else
                    "incorrect_or_invalid" if cand.get("status") == "passed" else
                    cand.get("status", "invalid"))
                if not eligible:
                    verdict["reason"] = "reference_unavailable_or_failed"
                elif not candidate_valid:
                    verdict["reason"] = "candidate_missing_unsupported_skipped_forwarded_incorrect_or_invalid"
                elif (not isinstance(build_record, dict)
                      or any(not source.get(k) or build_record.get(k) != source[k] for k in ("commit", "source_sha256"))
                      or any(not build_record.get(k) for k in ("build_command", "rustc", "cargo", "nvcc"))
                      or build_record.get("extension_sha256") != cand["extension"]["sha256"]
                      or ref["pid"] == cand["pid"]):
                    verdict["reason"] = "unbound_candidate_build_or_process"
                elif not common.valid_seeds(seeds):
                    verdict["reason"] = "invalid_or_insufficient_distinct_seeds"
                else:
                    # Both sides must satisfy the same exact independent oracle.
                    verdict.update(credit=1, reason="passed")
            verdicts.append(verdict)
        rows.append({"case_id": case["id"], "reference_eligible": bool(verdicts) and all(v["reference_eligible"] for v in verdicts),
                     "credit": int(bool(verdicts) and all(v["credit"] for v in verdicts)), "trials": verdicts})
    passed = sum(r["credit"] for r in rows)
    return {"denominator": len(case_set["cases"]), "passed": passed,
            "fraction": passed / len(case_set["cases"]), "cases": rows}


def evaluator_hashes():
    return {Path(p).name: common.sha256(p) for p in (__file__, common.__file__, MATRIX)}


def extension_hashes():
    return {str(p.relative_to(ROOT)): common.sha256(p)
            for p in (ROOT / "python/torch_rs").glob("torch_rs*.so")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--build-record", type=Path, help="evaluator's native build receipt; required for credit")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("reference", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("run with CUDA_VISIBLE_DEVICES=0")
    if args.worker:
        print(json.dumps(worker(args.worker, json.load(sys.stdin)), allow_nan=False))
        return 0
    seeds = args.seed if args.seed is not None else [secrets.randbits(63) for _ in range(3)]
    if not common.valid_seeds(seeds):
        parser.error("at least two distinct nonnegative integer seeds are required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.output and (not args.output.resolve().is_relative_to(ROOT)
                       or ".burner" in args.output.resolve().relative_to(ROOT).parts):
        parser.error("output must stay inside this worktree and outside .burner")
    case_set, source, hashes = corpus(), common.source_provenance(), evaluator_hashes()
    extensions = extension_hashes()
    build_record = json.loads(args.build_record.read_text()) if args.build_record else None
    try:
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap", "--format=csv"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        inventory = str(error)
    work = ROOT / "target/cuda-transfers"
    work.mkdir(parents=True, exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(dir=work) as temp:
        env = {**os.environ, "TMPDIR": temp, "CUDA_CACHE_PATH": temp,
               "XDG_CACHE_HOME": temp, "PYTHONDONTWRITEBYTECODE": "1"}
        for case in case_set["cases"]:
            for seed in seeds:
                trial = {"case_id": case["id"], "seed": seed}
                for role, python in (("reference", args.reference_python), ("candidate", args.candidate_python)):
                    trial[role] = launch(role, case, seed, python, args.timeout, env)
                trials.append(trial)
    stable = (source == common.source_provenance() and hashes == evaluator_hashes()
              and extensions == extension_hashes())
    accounting = account(case_set, seeds, trials, build_record if stable else None, source)
    result = {"schema_version": "cuda_transfers_correctness_v1", "case_set": VERSION,
              "backend": "cuda_nvidia", "capability": CAPABILITY,
              "scope": "this six-case capability only; no other capability, overall, breadth, or performance score",
              "source": source, "source_unchanged_during_run": stable, "evaluator_hashes": hashes,
              "native_extension_hashes": extensions,
              "build_record": build_record, "seeds": seeds, "CUDA_VISIBLE_DEVICES": "0",
              "gpu_inventory": inventory, "trials": trials, "accounting": accounting}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if all(c["reference_eligible"] for c in accounting["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
