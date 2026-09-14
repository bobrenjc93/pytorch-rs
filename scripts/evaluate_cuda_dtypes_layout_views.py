#!/usr/bin/env python3
"""Six fixed CUDA dtype/layout/view cases; correctness only, no overall score."""
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
import random
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_cuda_math as common
import evaluate_cuda_transfers as transfers

ROOT, MATRIX = common.ROOT, common.MATRIX
VERSION = "cuda_dtypes_layout_views_v1"
CAPABILITY = "dtypes_layout_views"
CTYPES = {"float32": ctypes.c_float, "float64": ctypes.c_double, "int64": ctypes.c_int64}


def corpus():
    matrix = json.loads(MATRIX.read_text())
    capability = next(c for c in matrix["capabilities"] if c["id"] == CAPABILITY)
    return next(c for c in capability["case_sets"] if c["version"] == VERSION)


def finite_json_float(token):
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("non-finite JSON number: " + token)
    return value


def values_for(case, seed):
    if case["dtype"] == "float32":
        return transfers.values_for(case["shape"], seed)
    rng = random.Random(seed)
    if case["dtype"] == "float64":
        return [rng.uniform(-2, 2) for _ in range(math.prod(case["shape"]))]
    return [rng.choice((-1, 1)) * rng.randrange(2**53, 2**60)
            for _ in range(math.prod(case["shape"]))]


class Inspector(transfers.Inspector):
    """Reuse float32 inspection; extend driver readback to the two fixed dtypes."""

    def snapshot(self, tensor, tensor_type):
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype == "float32":
            return super().snapshot(tensor, tensor_type)
        if type(tensor) is not tensor_type or dtype not in CTYPES:
            raise RuntimeError("expected public native Tensor with fixed dtype")
        shape, strides = list(tensor.shape), list(tensor.stride())
        device, pointer = str(tensor.device), tensor.data_ptr()
        if device not in ("cpu", "cuda:0") or bool(tensor.is_cuda) != (device == "cuda:0"):
            raise RuntimeError("unexpected tensor device")
        if not pointer or not math.prod(shape):
            raise RuntimeError("fixed cases require nonempty storage")
        attrs = None
        if device == "cuda:0":
            attrs = {}
            for name, attribute in (("memory_type", 2), ("device_ordinal", 9), ("is_managed", 8)):
                value = ctypes.c_int()
                self.call("cuPointerGetAttribute", ctypes.byref(value), attribute, pointer)
                attrs[name] = value.value
            if attrs != {"memory_type": 2, "device_ordinal": 0, "is_managed": 0}:
                raise RuntimeError("not native CUDA storage")
            self.call("cuCtxSynchronize")
        scalar = CTYPES[dtype]
        values = []
        for index in itertools.product(*(range(n) for n in shape)):
            address = pointer + ctypes.sizeof(scalar) * sum(i * s for i, s in zip(index, strides))
            if device == "cuda:0":
                value = scalar()
                self.call("cuMemcpyDtoH_v2", ctypes.byref(value), address, ctypes.sizeof(value))
                values.append(value.value)
            else:
                values.append(scalar.from_address(address).value)
        if not all(math.isfinite(v) for v in values):
            raise RuntimeError("nonfinite storage")
        if device == "cpu" and common.flatten(tensor.tolist()) != values:
            raise RuntimeError("public CPU values disagree with storage")
        return {"shape": shape, "strides": strides, "storage_offset": tensor.storage_offset(),
                "dtype": dtype, "device": device, "pointer": pointer,
                "pointer_attributes": attrs, "values": values}

    def write_first(self, tensor, value):
        # Evaluator-owned mutation probe ONLY, never an evaluated operation.
        scalar = CTYPES[str(tensor.dtype).removeprefix("torch.")](value)
        if str(tensor.device) == "cuda:0":
            fn = self.driver.cuMemcpyHtoD_v2
            fn.argtypes, fn.restype = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int
            self.call("cuMemcpyHtoD_v2", tensor.data_ptr(), ctypes.byref(scalar), ctypes.sizeof(scalar))
            self.call("cuCtxSynchronize")
        else:
            ctypes.memmove(tensor.data_ptr(), ctypes.byref(scalar), ctypes.sizeof(scalar))


def execute(module, case, seed, inspector, observations):
    op, shape = case["operation"], case["shape"]
    values = values_for(case, seed)
    if op == "device_roundtrip":
        base = module.tensor(values, dtype=getattr(module, case["dtype"])).reshape(shape)
    else:
        # Fixture-only driver initialization avoids testing upload in view slots.
        base = module.zeros((math.prod(shape),), dtype=module.float32, device="cuda:0")
        inspector.initialize(base, values)
        base = base.reshape(shape)
    source = base.transpose(0, 1) if op.startswith("transposed_") else base
    snapshot = lambda tensor: inspector.snapshot(tensor, module.Tensor)
    obs = {"base": snapshot(base), "source": snapshot(source)}
    observations.append(obs)
    if op == "device_roundtrip":
        output = source.to("cuda:0")
    elif op == "transpose_alias":
        output = source.transpose(0, 1)
    elif op == "offset_slice_alias":
        output = source[:, 1:12]
    elif op == "transposed_contiguous":
        output = source.contiguous()
    elif op == "transposed_reshape_copy":
        output = source.reshape(math.prod(shape))
    else:
        raise ValueError("unknown fixed operation: " + op)
    downloads = [output.cpu(), output.to("cpu")] if op == "device_roundtrip" else []
    obs.update(output=snapshot(output), downloads=[snapshot(t) for t in downloads],
               distinct_object=output is not source, base_after=snapshot(base), source_after=snapshot(source))
    # Mutate both directions. Base element 0 belongs to transpose but lies outside
    # the offset slice; writing element 1 as well verifies slice propagation.
    if op == "device_roundtrip":
        inspector.write_first(base, 23)
    else:
        inspector.initialize(base, [ctypes.c_float(v + 8).value for v in values])
    obs.update(mutated_base=snapshot(base), source_after_base_mutation=snapshot(source),
               output_after_base_mutation=snapshot(output),
               downloads_after_base_mutation=[snapshot(t) for t in downloads])
    inspector.write_first(output, -17)
    obs.update(output_after_output_mutation=snapshot(output), base_after_output_mutation=snapshot(base),
               source_after_output_mutation=snapshot(source),
               downloads_after_output_mutation=[snapshot(t) for t in downloads])


def tensor_valid(tensor, shape, strides, offset, device, dtype, values):
    if not isinstance(tensor, dict):
        return False
    expected = {"shape": shape, "strides": strides, "storage_offset": offset,
                "device": device, "dtype": dtype, "values": values,
                "pointer_attributes": {"memory_type": 2, "device_ordinal": 0, "is_managed": 0}
                if device == "cuda:0" else None}
    return (all(tensor.get(k) == v for k, v in expected.items())
            and all(type(n) is int for n in tensor["shape"] + tensor["strides"])
            and type(tensor["storage_offset"]) is int
            and type(tensor.get("pointer")) is int and tensor["pointer"] > 0
            and all(type(v) is (int if dtype == "int64" else float) and math.isfinite(v)
                    for v in tensor["values"]))


def valid_execution(row, case, seed, role):
    try:
        return _valid_execution(row, case, seed, role)
    except (TypeError, KeyError, ValueError, OverflowError, IndexError):
        return False


def _valid_execution(row, case, seed, role):
    if not isinstance(row, dict) or row.get("status") != "passed":
        return False
    if (row.get("case_id"), row.get("seed"), row.get("role")) != (case["id"], seed, role):
        return False
    if type(row.get("seed")) is not int or type(row.get("pid")) is not int or row["pid"] <= 0:
        return False
    runtimes = row.get("cuda_runtimes")
    if not isinstance(runtimes, list) or not runtimes or any(
            not isinstance(r, dict) or r.get("status") != 0 or not r.get("path")
            or type(r.get("version")) is not int or r["version"] <= 0 for r in runtimes):
        return False
    if role == "reference":
        if not row.get("gpu") or str(row.get("version")).split("+")[0] != "2.13.0":
            return False
    elif (row.get("blocked_imports") != [] or row.get("loaded_torch_modules") != []
          or not isinstance(row.get("extension"), dict) or not row["extension"].get("sha256")):
        return False
    observations = row.get("observations")
    if not isinstance(observations, list) or len(observations) != 1:
        return False
    obs = observations[0]
    if not isinstance(obs, dict) or obs.get("distinct_object") is not True:
        return False
    op, shape, dtype = case["operation"], case["shape"], case["dtype"]
    values = values_for(case, seed)
    rows, cols = shape
    transposed_indices = [r * cols + c for c in range(cols) for r in range(rows)]
    source_indices = transposed_indices if op.startswith("transposed_") else list(range(rows * cols))
    source_shape = [cols, rows] if op.startswith("transposed_") else shape
    source_strides = [1, cols] if op.startswith("transposed_") else [cols, 1]
    output_indices, output_shape, output_strides, offset = source_indices, source_shape, [cols, 1], 0
    if op == "transpose_alias":
        output_indices, output_shape, output_strides = transposed_indices, [cols, rows], [1, cols]
    elif op == "offset_slice_alias":
        output_indices = [r * cols + c for r in range(rows) for c in range(1, 12)]
        output_shape, output_strides, offset = [rows, 11], [cols, 1], 1
    elif op == "transposed_contiguous":
        output_strides = [rows, 1]
    elif op == "transposed_reshape_copy":
        output_shape, output_strides = [rows * cols], [1]
    device = "cpu" if op == "device_roundtrip" else "cuda:0"
    aliases = op in ("transpose_alias", "offset_slice_alias")
    base_pointer, output_pointer = obs["base"]["pointer"], obs["output"]["pointer"]
    size = ctypes.sizeof(CTYPES[dtype])
    if aliases:
        if output_pointer != base_pointer + offset * size:
            return False
    elif device == "cuda:0" and abs(output_pointer - base_pointer) < len(values) * size:
        return False

    def check(key, expected_values, which):
        if which == "base":
            sh, st, off, dev, ptr = shape, [cols, 1], 0, device, base_pointer
        elif which == "source":
            sh, st, off, dev, ptr = source_shape, source_strides, 0, device, base_pointer
        else:
            sh, st, off, dev, ptr = output_shape, output_strides, offset, "cuda:0", output_pointer
        return (tensor_valid(obs.get(key), sh, st, off, dev, dtype, expected_values)
                and obs[key]["pointer"] == ptr)

    src_values = [values[i] for i in source_indices]
    out_values = [values[i] for i in output_indices]
    checks = [("base", values, "base"), ("source", src_values, "source"), ("output", out_values, "output"),
              ("base_after", values, "base"), ("source_after", src_values, "source")]
    mutated = values.copy()
    if op == "device_roundtrip":
        mutated[0] = CTYPES[dtype](23).value
    else:
        mutated = [ctypes.c_float(v + 8).value for v in values]
    mutated_out = [mutated[i] for i in output_indices] if aliases else out_values.copy()
    checks += [("mutated_base", mutated, "base"),
               ("source_after_base_mutation", [mutated[i] for i in source_indices], "source"),
               ("output_after_base_mutation", mutated_out, "output")]
    final_base, final_out = mutated.copy(), mutated_out.copy()
    final_out[0] = CTYPES[dtype](-17).value
    if aliases:
        final_base[output_indices[0]] = final_out[0]
    checks += [("output_after_output_mutation", final_out, "output"),
               ("base_after_output_mutation", final_base, "base"),
               ("source_after_output_mutation", [final_base[i] for i in source_indices], "source")]
    if not all(check(*args) for args in checks):
        return False
    downloads = obs.get("downloads")
    if not isinstance(downloads, list) or len(downloads) != (2 if op == "device_roundtrip" else 0):
        return False
    pointers = [base_pointer]
    for tensor in downloads:
        if not tensor_valid(tensor, shape, [cols, 1], 0, "cpu", dtype, values):
            return False
        if any(abs(tensor["pointer"] - other) < len(values) * size for other in pointers):
            return False
        pointers.append(tensor["pointer"])
    return (obs.get("downloads_after_base_mutation") == downloads
            and obs.get("downloads_after_output_mutation") == downloads)


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
        row = json.loads(process.stdout, parse_float=finite_json_float, parse_constant=finite_json_float)
        if process.returncode or not isinstance(row, dict):
            raise ValueError(f"invalid worker result (exit {process.returncode}): {process.stderr[-4000:]}")
        row["stderr"] = process.stderr[-4000:]
        return row
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}


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
                      or any(not isinstance(build_record.get(k), str) or not build_record[k].strip()
                             for k in ("build_command", "rustc", "cargo", "nvcc"))
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
    return {Path(p).name: common.sha256(p) for p in (__file__, common.__file__, transfers.__file__, MATRIX)}


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
    seeds = args.seed if args.seed is not None else [secrets.randbits(63) for _ in range(2)]
    if not common.valid_seeds(seeds):
        parser.error("at least two distinct nonnegative integer seeds are required")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be positive")
    if args.output and (not args.output.resolve().is_relative_to(ROOT)
                       or ".burner" in args.output.resolve().relative_to(ROOT).parts):
        parser.error("output must stay inside this worktree and outside .burner")
    started_at = datetime.now(timezone.utc).isoformat()
    case_set, source, hashes = corpus(), common.source_provenance(), evaluator_hashes()
    extensions = extension_hashes()
    build_record, build_error = None, None
    try:
        if args.build_record:
            build_record = json.loads(args.build_record.read_text(), parse_float=finite_json_float,
                                      parse_constant=finite_json_float)
    except (OSError, ValueError) as error:
        build_error = str(error)  # Retain every slot even with an invalid receipt.
    try:
        inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,driver_version,compute_cap", "--format=csv"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        inventory = str(error)
    work = ROOT / "target/cuda-dtypes-layout-views"
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
    result = {"schema_version": "cuda_dtypes_layout_views_correctness_v1", "case_set": VERSION,
              "backend": "cuda_nvidia", "capability": CAPABILITY,
              "scope": "this six-case capability only; no other capability, overall, breadth, or performance score",
              "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
              "source": source, "source_unchanged_during_run": stable, "evaluator_hashes": hashes,
              "native_extension_hashes": extensions,
              "build_record": build_record, "build_record_error": build_error, "seeds": seeds, "CUDA_VISIBLE_DEVICES": "0",
              "gpu_inventory": inventory, "trials": trials, "accounting": accounting}
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if all(c["reference_eligible"] for c in accounting["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
