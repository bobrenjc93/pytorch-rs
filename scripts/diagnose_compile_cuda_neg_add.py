#!/usr/bin/env python3
"""Versioned CUDA neg/add correctness diagnostic, separate from frozen scoring."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import runpy
import sys

if __package__:
    from . import diagnose_compile_cuda_add as frozen
else:
    import diagnose_compile_cuda_add as frozen

ROOT = Path(__file__).resolve().parents[1]
CASE_SET = "neg_add_v1"


def negate(a):
    return -a


def neg_method(a):
    return a.neg()


def negative_method(a):
    return a.negative()


def compose(a, b):
    return -(a.neg() + b.negative()).add(a)


def cases(mask):
    """Retain the addition matrix and guards; explicitly version new support."""
    if mask == "0,1":
        return [(p, (19,), arity, "ordinal1", True) for p, arity in
                ((frozen.silver, 2), (negate, 1), (compose, 2))] + [
                    (frozen.with_global, (19,), 1, "capture", True),
                    (frozen.silver, (19,), 2, "mixed_ordinals", False),
                    (compose, (19,), 2, "mixed_ordinals", False)]
    shapes = [(), (0,), (3, 0, 2), (257,), (17, 31), (65539,)]
    rng = frozen.np.random.default_rng(frozen.SEED)
    shapes += [tuple(int(n) for n in rng.integers(1, 15, size=3)) for _ in range(3)]
    result = [(p, shape, arity, "offset", True) for p, arity in
              ((frozen.silver, 2), (frozen.orchard, 1), (frozen.estuary, 2),
               (negate, 1), (neg_method, 1), (negative_method, 1), (compose, 2))
              for shape in shapes]
    result += [(frozen.with_global, (17,), 1, "capture", True)]
    result += [(p, (3, 4), 1, "plain", False) for p in
               (frozen.reject_detach, frozen.reject_float, frozen.reject_multiply,
                frozen.reject_reduce, frozen.reject_scalar)]
    result += [(frozen.reject_keyword, (3, 4), 2, "plain", False)]
    result += [(p, (3, 4), arity, kind, False) for p, arity in
               ((frozen.silver, 2), (compose, 2)) for kind in
               ("transpose", "broadcast", "gradient", "float64", "mixed_cpu")]
    result += [(negate, (3, 4), 1, kind, False) for kind in
               ("transpose", "gradient", "float64")]
    result += [(frozen.silver, (), 2, "mixed_cpu", False)]
    return result


def expectation_met(record):
    # Only the declared nonscalar mixed-device programs are allowed to be
    # reference-ineligible. A broken reference on any other guard is a failure.
    reference_ok = record["reference_eligible"] or (
        record.get("kind") in ("mixed_cpu", "mixed_ordinals")
        and bool(record.get("shape")))
    if record["expected_supported"]:
        return (record["reference_eligible"] and record["native_outcome"] == "pass"
                and len(record.get("reference_outputs", [])) == 2
                and record.get("native_outputs") == record["reference_outputs"])
    return reference_ok and record["native_outcome"] == "unsupported"


def environment(mask):
    if mask not in ("0", "0,1") or not frozen.torch.cuda.is_available():
        raise RuntimeError("requires real CUDA and CUDA_VISIBLE_DEVICES=0 or 0,1")
    if mask == "0,1" and frozen.torch.cuda.device_count() < 2:
        raise RuntimeError("two visible CUDA devices required")
    if frozen.torch.__version__.split("+")[0] != "2.13.0":
        raise RuntimeError("reference must be PyTorch 2.13.0")
    # Keep the canonical workspace-venv guard, including the native ABI loader.
    runpy.run_path(str(ROOT / ".github/scripts/verify_native_extension.py"))["verify"]()
    package = Path(frozen.native.__file__).parent
    sources = sorted((ROOT / "python/torch_rs").rglob("*.py"))
    for source in sources:
        installed = package / source.relative_to(ROOT / "python/torch_rs")
        if not installed.is_file() or frozen.sha(source) != frozen.sha(installed):
            raise RuntimeError(f"installed wheel/source mismatch: {source}")
    from torch_rs._cuda_public_storage import _candidate_libraries
    runtime_path = os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0]
    runtime = ctypes.CDLL(runtime_path)
    version = ctypes.c_int()
    if runtime.cudaRuntimeGetVersion(ctypes.byref(version)):
        raise RuntimeError("cudaRuntimeGetVersion failed")
    sources += [ROOT / name for name in
                ("src/python.rs", "src/cuda/add.ptx", "src/cuda/neg.ptx",
                 "src/cuda/pointwise.rs", "scripts/diagnose_compile_cuda_add.py",
                 "scripts/diagnose_compile_cuda_neg_add.py")]
    return {
        "base_commit": frozen.command("git", "rev-parse", "HEAD"),
        "worktree_status": frozen.command("git", "status", "--porcelain"),
        "source_sha256": {str(p.relative_to(ROOT)): frozen.sha(p) for p in sources},
        "native_extension": frozen._compile_trace._native.__file__,
        "native_extension_sha256": frozen.sha(Path(frozen._compile_trace._native.__file__)),
        "environment": {
            "python": platform.python_version(), "executable": sys.executable,
            "package": str(package), "torch": frozen.torch.__version__,
            "torch_cuda": frozen.torch.version.cuda, "cuda_visible_devices": mask,
            "gpus": frozen.command("nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap", "--format=csv,noheader"),
            "native_runtime_library": runtime_path, "native_runtime_version": version.value,
            "native_kernel_compiler": "NVIDIA driver JIT of embedded PTX 6.0/sm_50; nvcc unused",
            "rust": frozen.command("rustc", "--version"),
            "caches": {key: os.environ.get(key) for key in
                       ("CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", choices=[CASE_SET], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    provenance = environment(mask)
    # Reuse the frozen reference/eager differential, output fingerprints and
    # original-function execution guard without changing its old expectations.
    records = [frozen.run_case(p, shape, arity, kind, fullgraph, supported)
               for p, shape, arity, kind, supported in cases(mask)
               for fullgraph in (True, False)]
    for record in records:
        record["expectation_met"] = expectation_met(record)
    summary = {
        "cases": len(records),
        "reference_eligible": sum(r["reference_eligible"] for r in records),
        "native_pass": sum(r["native_outcome"] == "pass" for r in records),
        "native_unsupported": sum(r["native_outcome"] == "unsupported" for r in records),
        "expectation_failures": sum(not r["expectation_met"] for r in records),
    }
    report = {"schema": "marker_free_cuda_neg_add_diagnostic_v1",
              "case_set": args.case_set, "purpose": "correctness only; no coverage or performance score",
              "seed": frozen.SEED, **provenance, "summary": summary, "cases": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary))
    return int(not records or summary["expectation_failures"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
