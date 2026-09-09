#!/usr/bin/env python3
"""Separate correctness diagnostic; does not score or modify any frozen corpus."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
import torch_rs as native
from torch_rs import _compile_trace

ROOT = Path(__file__).resolve().parents[1]
SEED = 19092675
CAPTURE = None


def silver(a, b):
    return b + a


def orchard(a):
    return a.add(a)


def estuary(a, b):
    c = b.add(a)
    return (c + a).add(b)


def with_global(a):
    return a + CAPTURE


def reject_neg(a):
    return -a


def reject_detach(a):
    return a.detach()


def reject_float(a):
    return a.float()


def reject_multiply(a):
    return a * a


def reject_reduce(a):
    return a.sum()


def reject_scalar(a):
    return a + 1


def reject_keyword(a, b):
    return a.add(b, alpha=1)


def fingerprint(value):
    dtype = np.float64 if str(value.dtype) == "torch.float64" else np.float32
    array = np.asarray(value.detach().cpu().tolist(), dtype=dtype)
    return {
        "shape": list(value.shape), "stride": list(value.stride()),
        "dtype": str(value.dtype), "device": str(value.device),
        "storage_offset": value.storage_offset(), "requires_grad": value.requires_grad,
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def inputs(module, shape, arity, kind, iteration):
    rng = np.random.default_rng(SEED + iteration)
    values = [rng.normal(size=int(np.prod(shape)) + 5).astype(np.float32) for _ in range(arity)]
    result = []
    for i, value in enumerate(values):
        dtype = module.float64 if kind == "float64" else module.float32
        tensor = module.tensor(value.tolist(), dtype=dtype,
                               requires_grad=kind == "gradient")
        target = "cuda:1" if kind == "ordinal1" or (kind == "mixed_ordinals" and i == 1) else "cuda:0"
        if kind == "mixed_cpu" and i == 1:
            target = "cpu"
        tensor = tensor.to(target)[5:].reshape(shape)
        if kind == "transpose":
            tensor = tensor.t()
        if kind == "broadcast" and i == 1:
            tensor = tensor[0]
        result.append(tensor)
    return result


def run_case(program, shape, arity, kind, fullgraph, supported):
    global CAPTURE
    record = {"program": program.__name__, "shape": shape, "kind": kind,
              "backend": "eager", "fullgraph": fullgraph, "expected_supported": supported}
    torch._dynamo.reset()
    reference = torch.compile(program, backend="eager", fullgraph=fullgraph)
    expected = []
    try:
        for iteration in (0, 1):
            args = inputs(torch, shape, arity, kind, iteration)
            CAPTURE = args[0] if kind == "capture" else None
            eager = fingerprint(program(*args))
            observed = fingerprint(reference(*args))
            if observed != eager:
                raise AssertionError("reference compile differs from reference eager")
            expected.append(observed)
        record["reference_eligible"] = True
        record["reference_outputs"] = expected
    except Exception as error:
        record.update(reference_eligible=False, reference_error=f"{type(error).__name__}: {str(error).splitlines()[0]}")

    phase = "compile_configuration"
    try:
        compiled = native.compile(program, backend="eager", fullgraph=fullgraph)
        actual = []
        def reject_original(frame, event, arg):
            if event == "call" and frame.f_code is program.__code__:
                raise AssertionError("native compiler called original function")
        for iteration in (0, 1):
            phase = "input_construction"
            args = inputs(native, shape, arity, kind, iteration)
            CAPTURE = args[0] if kind == "capture" else None
            previous = sys.getprofile()
            try:
                sys.setprofile(reject_original)
                phase = "compiled_execution"
                output = compiled(*args)
            finally:
                sys.setprofile(previous)
            actual.append(fingerprint(output))
        record["native_outputs"] = actual
        record["native_outcome"] = "pass" if record["reference_eligible"] and actual == expected else "mismatch"
    except (NotImplementedError, TypeError) as error:
        record.update(native_outcome="unsupported", rejection_phase=phase,
                      native_error=f"{type(error).__name__}: {error}")
    except AttributeError as error:
        record.update(native_outcome="unsupported" if phase == "input_construction" else "error",
                      rejection_phase=phase, native_error=f"{type(error).__name__}: {error}")
    except Exception as error:
        record.update(native_outcome="error", native_error=f"{type(error).__name__}: {error}")
    finally:
        CAPTURE = None
    return record


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask not in ("0", "0,1") or not torch.cuda.is_available():
        parser.error("requires real CUDA and CUDA_VISIBLE_DEVICES=0 (or 0,1 for ordinal checks)")
    if torch.__version__.split("+")[0] != "2.13.0":
        parser.error("reference must be PyTorch 2.13.0")
    # Ensure the installed Python compiler matches this checkout, not a source
    # overlay or an older wheel with the same package version.
    package = Path(native.__file__).parent
    source_paths = ["python/torch_rs/__init__.py", "python/torch_rs/_compile_trace.py",
                    "python/torch_rs/_compile_bytecode.py"]
    for name in source_paths:
        if sha(ROOT / name) != sha(package / Path(name).name):
            raise RuntimeError(f"installed wheel/source mismatch: {name}")
    from torch_rs._cuda_public_storage import _candidate_libraries
    runtime_path = os.environ.get("TORCH_RS_CUDART") or _candidate_libraries()[0]
    runtime = ctypes.CDLL(runtime_path)
    version = ctypes.c_int()
    if runtime.cudaRuntimeGetVersion(ctypes.byref(version)):
        raise RuntimeError("cudaRuntimeGetVersion failed")
    cases = []
    if mask == "0":
        shapes = [(), (0,), (3, 0, 2), (257,), (17, 31), (65539,)]
        rng = np.random.default_rng(SEED)
        shapes += [tuple(int(n) for n in rng.integers(1, 15, size=3)) for _ in range(3)]
        cases += [(p, shape, arity, "offset", True) for p, arity in
                  ((silver, 2), (orchard, 1), (estuary, 2)) for shape in shapes]
        cases += [(with_global, (17,), 1, "capture", True)]
        cases += [(p, (3, 4), 1, "plain", False) for p in
                  (reject_neg, reject_detach, reject_float, reject_multiply, reject_reduce, reject_scalar)]
        cases += [(reject_keyword, (3, 4), 2, "plain", False)]
        cases += [(silver, (3, 4), 2, kind, False) for kind in
                  ("transpose", "broadcast", "gradient", "float64", "mixed_cpu")]
        # PyTorch permits a CPU scalar tensor with a CUDA scalar tensor. The
        # bounded native compiler still rejects all mixed-device graphs.
        cases += [(silver, (), 2, "mixed_cpu", False)]
    else:
        cases += [(silver, (19,), 2, "ordinal1", True),
                  (with_global, (19,), 1, "capture", True),
                  (silver, (19,), 2, "mixed_ordinals", False)]
    records = [run_case(p, shape, arity, kind, fullgraph, supported)
               for p, shape, arity, kind, supported in cases for fullgraph in (True, False)]
    sources = source_paths + ["src/python.rs", "src/cuda/add.ptx", "src/cuda/pointwise.rs",
                             "scripts/diagnose_compile_cuda_add.py", "tests/test_compile_cuda_boundary.py"]
    report = {
        "schema": "marker_free_cuda_add_diagnostic_v1", "purpose": "correctness only; no coverage or performance score",
        "seed": SEED, "base_commit": command("git", "rev-parse", "HEAD"),
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "native_extension_sha256": sha(Path(_compile_trace._native.__file__)),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torch_cuda": torch.version.cuda, "cuda_visible_devices": mask,
                        "gpus": command("nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap", "--format=csv,noheader"),
                        "native_runtime_library": runtime_path, "native_runtime_version": version.value,
                        "nvcc_available": command("nvcc", "--version"),
                        "native_kernel_compiler": "NVIDIA driver JIT of embedded PTX 6.0/sm_50; nvcc unused",
                        "rust": command("rustc", "--version"), "build": "maturin --release --locked; abi3-py310; thin LTO; codegen-units=1"},
        "summary": {"cases": len(records), "reference_eligible": sum(r["reference_eligible"] for r in records),
                    "native_pass": sum(r["native_outcome"] == "pass" for r in records),
                    "native_unsupported": sum(r["native_outcome"] == "unsupported" for r in records)},
        "cases": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
    return int(any((r["expected_supported"] and r["native_outcome"] != "pass") or
                   (not r["expected_supported"] and r["native_outcome"] != "unsupported") for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
