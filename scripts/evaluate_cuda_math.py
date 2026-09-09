#!/usr/bin/env python3
"""Correctness evidence for the fixed CUDA math cell; never a performance score."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import random
import secrets
import struct
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs/hardware-heterogeneity-matrix-v1.json"
VERSION = "cuda_float32_math_v1"
EVALUATOR_VERSION = "cuda_math_correctness_v1"


def corpus():
    matrix = json.loads(MATRIX.read_text())
    capability = next(c for c in matrix["capabilities"]
                      if c["id"] == "math_reductions_linalg")
    return next(c for c in capability["case_sets"] if c["version"] == VERSION)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args])

    # Exclude evaluator files and merge-managed evidence. Bind the production
    # sources even for a dirty worktree, without representing it as clean HEAD.
    paths = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                "src", "python", "Cargo.toml", "Cargo.lock",
                "pyproject.toml", "rust-toolchain.toml").decode().split("\0")
    hashes = {p: sha256(ROOT / p) if (ROOT / p).is_file() else None
              for p in paths if p}
    diff = git("diff", "HEAD", "--", *hashes)
    return {"commit": git("rev-parse", "HEAD").decode().strip(),
            "source_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "production_diff_sha256": hashlib.sha256(diff).hexdigest()}


class ForwardingError(RuntimeError):
    pass


class BlockTorch:
    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            self.attempts.append(fullname)
            raise ForwardingError("candidate PyTorch import blocked: " + fullname)

    def check(self):
        loaded = [n for n in sys.modules if n == "torch" or n.startswith("torch.")]
        if loaded or self.attempts:
            raise ForwardingError(f"PyTorch forwarding: {loaded or self.attempts}")


def inputs_for(case, seed):
    rng = random.Random(seed)

    def values(shape):
        if not shape:
            return struct.unpack("f", struct.pack("f", rng.uniform(-2, 2)))[0]
        return [values(shape[1:]) for _ in range(shape[0])]

    return [values(shape) for shape in case["input_shapes"]]


def flatten(value):
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def operation(module, case, inputs):
    a = inputs[0]
    op = case["operation"]
    if op == "add":
        return a + inputs[1]
    if op == "neg":
        return -a
    if op == "mul_scalar":
        return a * case["scalar"]
    if op == "sum":
        return module.sum(a, dim=case["dim"], keepdim=case["keepdim"])
    if op == "matmul":
        return module.matmul(a, inputs[1])
    raise ValueError("unknown fixed operation: " + op)


class CudaInspector:
    """Evaluator-owned driver queries, independent of either tensor library."""

    def __init__(self):
        self.driver = ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuInit": [ctypes.c_uint],
            "cuCtxSynchronize": [],
            "cuPointerGetAttribute": [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64],
            "cuMemcpyDtoH_v2": [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t],
        }
        for name, args in signatures.items():
            fn = getattr(self.driver, name)
            fn.argtypes, fn.restype = args, ctypes.c_int
        self.call("cuInit", 0)

    def call(self, name, *args):
        result = getattr(self.driver, name)(*args)
        if result != 0:
            raise RuntimeError(f"{name} failed with CUDA status {result}")

    def materialize(self, tensor, shape, tensor_type):
        if type(tensor) is not tensor_type:
            raise RuntimeError("output is not the public native Tensor type")
        if str(tensor.device) != "cuda:0" or not tensor.is_cuda:
            raise RuntimeError("tensor is not on cuda:0")
        if str(tensor.dtype) != "torch.float32" or list(tensor.shape) != shape:
            raise RuntimeError("tensor dtype or shape mismatch")
        if not tensor.is_contiguous():
            raise RuntimeError("fixed cases require contiguous tensors")
        pointer = tensor.data_ptr()
        attrs = {}
        for name, attribute in (("memory_type", 2), ("device_ordinal", 9),
                                ("is_managed", 8)):
            value = ctypes.c_int()
            self.call("cuPointerGetAttribute", ctypes.byref(value), attribute, pointer)
            attrs[name] = value.value
        if attrs != {"memory_type": 2, "device_ordinal": 0, "is_managed": 0}:
            raise RuntimeError(f"not native CUDA device storage: {attrs}")
        self.call("cuCtxSynchronize")
        host = (ctypes.c_float * math.prod(shape))()
        self.call("cuMemcpyDtoH_v2", host, pointer, ctypes.sizeof(host))
        actual = list(host)
        if not all(math.isfinite(x) for x in actual):
            raise RuntimeError("nonfinite materialized output")
        return {"shape": shape, "dtype": "float32", "device": "cuda:0",
                "pointer_attributes": attrs, "values": actual}


def runtime_provenance():
    # Only libraries already mapped by this worker count as selected runtimes.
    paths = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "/libcudart.so" in line})
    runtimes = []
    for path in paths:
        lib = ctypes.CDLL(path)
        version = ctypes.c_int()
        lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.cudaRuntimeGetVersion.restype = ctypes.c_int
        status = lib.cudaRuntimeGetVersion(ctypes.byref(version))
        runtimes.append({"path": path, "version": version.value, "status": status})
    return {"cuda_runtimes": runtimes, "nvcc": "unused (no CUDA compilation by evaluator)",
            "TORCH_RS_CUDART": os.environ.get("TORCH_RS_CUDART")}


def worker(role, request):
    result = {"status": "failed", "role": role, "case_id": request["case"]["id"],
              "seed": request["seed"], "pid": os.getpid(),
              "python": sys.version, "executable": sys.executable}
    blocker = BlockTorch() if role == "candidate" else None
    try:
        if blocker:
            blocker.check()
            sys.meta_path.insert(0, blocker)
            # Use this checkout's package and locally built extension only.
            sys.path.insert(0, str(ROOT / "python"))
        module = importlib.import_module("torch_rs" if blocker else "torch")
        result["package"] = str(Path(module.__file__).resolve())
        result["version"] = str(module.__version__)
        if blocker:
            extension = importlib.import_module("torch_rs.torch_rs")
            path = Path(extension.__file__).resolve()
            if path.parent != ROOT / "python/torch_rs" or not isinstance(
                extension.__spec__.loader, importlib.machinery.ExtensionFileLoader
            ):
                raise RuntimeError("candidate extension is not local native code")
            result["extension"] = {"path": str(path), "sha256": sha256(path)}
            blocker.check()
        else:
            if module.__version__.split("+")[0] != "2.13.0" or module.version.hip:
                raise RuntimeError("reference requires NVIDIA PyTorch 2.13.0")
            if not module.cuda.is_available():
                result["status"] = "skipped"
                result["error"] = "reference NVIDIA CUDA hardware unavailable"
                return result
            module.backends.cuda.matmul.allow_tf32 = False
            result["pytorch_cuda"] = module.version.cuda
            properties = module.cuda.get_device_properties(0)
            result["gpu"] = {"name": properties.name,
                             "compute_capability": [properties.major, properties.minor],
                             "total_memory": properties.total_memory,
                             "uuid": str(properties.uuid)}
            result["memory_free_total"] = list(module.cuda.mem_get_info(0))

        inspector = CudaInspector()
        case = request["case"]
        values = inputs_for(case, request["seed"])
        inputs = [module.tensor(v, dtype=module.float32).to("cuda:0") for v in values]
        result["inputs"] = []
        for tensor, shape, expected in zip(inputs, case["input_shapes"], values):
            materialized = inspector.materialize(tensor, shape, module.Tensor)
            if materialized["values"] != flatten(expected):
                raise RuntimeError("input initialization mismatch")
            result["inputs"].append(materialized)
        output = operation(module, case, inputs)
        result["output"] = inspector.materialize(output, case["output_shape"], module.Tensor)
        # Validate the public observation too; the driver copy above does not
        # depend on candidate tolist()/cpu() or a candidate synchronization shim.
        public = output.cpu()
        if str(public.device) != "cpu" or flatten(public.tolist()) != result["output"]["values"]:
            raise RuntimeError("public materialization disagrees with CUDA storage")
        if blocker:
            blocker.check()
        result["status"] = "passed"
    except Exception as error:
        result["status"] = "forwarded" if isinstance(error, ForwardingError) else "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if blocker:
            result["blocked_imports"] = blocker.attempts
            result["loaded_torch_modules"] = [
                n for n in sys.modules if n == "torch" or n.startswith("torch.")]
            if result["blocked_imports"] or result["loaded_torch_modules"]:
                result["status"] = "forwarded"
        try:
            result.update(runtime_provenance())
        except Exception as error:
            result["runtime_error"] = str(error)
            if result["status"] == "passed":
                result["status"] = "failed"
    return result


def launch(role, case, seed, python, timeout, env):
    request = {"case": case, "seed": seed}
    try:
        completed = subprocess.run(
            [str(python), "-I", "-B", str(Path(__file__).resolve()), "--worker", role],
            input=json.dumps(request), capture_output=True, text=True,
            cwd=ROOT, env=env, timeout=timeout)
        if completed.returncode:
            raise RuntimeError(f"worker exit {completed.returncode}: {completed.stderr[-4000:]}")
        result = json.loads(completed.stdout)
        if not isinstance(result, dict):
            raise ValueError("worker result must be an object")
        result["stderr"] = completed.stderr[-4000:]
        return result
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


def valid_execution(row, case, seed, role):
    if not isinstance(row, dict) or row.get("status") != "passed":
        return False
    if (row.get("case_id"), row.get("seed"), row.get("role")) != (case["id"], seed, role):
        return False
    runtimes = row.get("cuda_runtimes")
    if type(row.get("pid")) is not int or not isinstance(runtimes, list) or not runtimes:
        return False
    if any(not isinstance(r, dict) or r.get("status") != 0
           or type(r.get("version")) is not int or r["version"] <= 0
           or not r.get("path") for r in runtimes):
        return False
    if role == "candidate":
        if row.get("blocked_imports") != [] or row.get("loaded_torch_modules") != []:
            return False
        if not isinstance(row.get("extension"), dict) or not row["extension"].get("sha256"):
            return False
    elif not row.get("gpu") or str(row.get("version")).split("+")[0] != "2.13.0":
        return False
    if not isinstance(row.get("inputs"), list):
        return False
    tensors = row["inputs"] + [row.get("output")]
    shapes = case["input_shapes"] + [case["output_shape"]]
    if len(tensors) != len(shapes):
        return False
    for tensor, shape in zip(tensors, shapes):
        if (not isinstance(tensor, dict)
                or tensor.get("shape") != shape or tensor.get("dtype") != "float32"
                or tensor.get("device") != "cuda:0"
                or tensor.get("pointer_attributes") != {
                    "memory_type": 2, "device_ordinal": 0, "is_managed": 0}):
            return False
        values = tensor.get("values")
        if not isinstance(values, list) or len(values) != math.prod(shape):
            return False
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            return False
    expected = inputs_for(case, seed)
    return all(t["values"] == flatten(v) for t, v in zip(row["inputs"], expected))


def account(case_set, seeds, trials, build_record, source):
    """Iterate the manifest, never the successful rows. All trials must pass."""
    rows = []
    for case in case_set["cases"]:
        verdicts = []
        for seed in seeds:
            matches = [t for t in trials if t.get("case_id") == case["id"] and t.get("seed") == seed]
            verdict = {"seed": seed, "credit": 0, "reason": "missing_or_duplicate_trial"}
            if len(matches) == 1:
                ref, cand = matches[0].get("reference"), matches[0].get("candidate")
                if not valid_execution(ref, case, seed, "reference"):
                    verdict["reason"] = "reference_unavailable_or_failed"
                elif not valid_execution(cand, case, seed, "candidate"):
                    verdict["reason"] = "candidate_missing_failed_skipped_forwarded_or_invalid"
                elif (not isinstance(build_record, dict) or any(build_record.get(k) != source[k]
                                              for k in ("commit", "source_sha256"))
                      or not build_record.get("build_command")
                      or not build_record.get("rustc")
                      or not build_record.get("cargo")
                      or build_record.get("extension_sha256") != cand.get("extension", {}).get("sha256")
                      or not cand.get("extension", {}).get("sha256")
                      or ref["pid"] == cand["pid"]):
                    verdict["reason"] = "unbound_candidate_build_or_process"
                elif len(seeds) < 2 or len(set(seeds)) != len(seeds):
                    verdict["reason"] = "insufficient_distinct_seeds"
                else:
                    expected, actual = ref["output"]["values"], cand["output"]["values"]
                    correct = all(abs(a - b) <= case_set["atol"] + case_set["rtol"] * abs(b)
                                  for a, b in zip(actual, expected))
                    verdict.update(credit=int(correct), reason="passed" if correct else "incorrect")
            verdicts.append(verdict)
        rows.append({"case_id": case["id"], "credit": int(bool(verdicts) and all(
            v["credit"] for v in verdicts)), "trials": verdicts})
    passed = sum(r["credit"] for r in rows)
    return {"denominator": len(case_set["cases"]), "passed": passed,
            "fraction": passed / len(case_set["cases"]), "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", help="repeat for at least two distinct evaluator seeds")
    parser.add_argument("--reference-python", default=sys.executable)
    parser.add_argument("--candidate-python", default=sys.executable)
    parser.add_argument("--build-record", type=Path, help="evaluator's fresh native build receipt (required for credit)")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", choices=("reference", "candidate"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, json.load(sys.stdin)), allow_nan=False))
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        parser.error("run with CUDA_VISIBLE_DEVICES=0")
    seeds = args.seed if args.seed is not None else [secrets.randbits(63) for _ in range(3)]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("at least two distinct seeds are required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.output and (not args.output.resolve().is_relative_to(ROOT)
                        or ".burner" in args.output.resolve().relative_to(ROOT).parts):
        parser.error("output must stay inside this worktree and outside .burner")
    case_set, source = corpus(), source_provenance()
    evaluator_hash, matrix_hash = sha256(__file__), sha256(MATRIX)
    build_record = json.loads(args.build_record.read_text()) if args.build_record else None
    try:
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,driver_version,compute_cap",
            "--format=csv"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        inventory = str(error)
    work = ROOT / "target/cuda-math"
    work.mkdir(parents=True, exist_ok=True)
    trials = []
    with tempfile.TemporaryDirectory(dir=work) as temp:
        env = {**os.environ, "TMPDIR": temp, "CUDA_CACHE_PATH": temp,
               "XDG_CACHE_HOME": temp, "PYTHONDONTWRITEBYTECODE": "1"}
        for case in case_set["cases"]:
            for seed in seeds:
                trials.append({"case_id": case["id"], "seed": seed,
                               "reference": launch("reference", case, seed, args.reference_python, args.timeout, env),
                               "candidate": launch("candidate", case, seed, args.candidate_python, args.timeout, env)})
    stable = (source == source_provenance() and evaluator_hash == sha256(__file__)
              and matrix_hash == sha256(MATRIX))
    result = {"evaluator_version": EVALUATOR_VERSION, "case_set_version": VERSION,
              "backend": "cuda_nvidia", "capability": "math_reductions_linalg",
              "scope": "correctness evidence for this six-case cell only; no overall score or transfer credit",
              "source": source, "source_unchanged_during_run": stable,
              "evaluator_sha256": evaluator_hash, "matrix_sha256": matrix_hash,
              "build_record": build_record, "seeds": seeds,
              "CUDA_VISIBLE_DEVICES": "0", "gpu_inventory": inventory,
              "trials": trials,
              "accounting": account(case_set, seeds, trials, build_record if stable else None, source)}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    # Unsupported candidates are legitimate zero results. A failed reference
    # is an incomplete hardware run, still with all six denominator slots.
    return 0 if all(valid_execution(t["reference"], c, t["seed"], "reference")
                    for c in case_set["cases"] for t in trials if t["case_id"] == c["id"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
