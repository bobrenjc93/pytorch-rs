#!/usr/bin/env python3
"""Public-API CUDA matmul diagnostic, separate from all Burner scoring gates.

Run after the unchanged build/correctness capture with its build-record.json.
Clean code is required unless --allow-dirty explicitly labels development data.
Output stays under this worktree; publication is an artifact-only delivery step.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from evaluate_cuda_math import ROOT, runtime_provenance, sha256, source_provenance


def stamp():
    return datetime.now(timezone.utc).isoformat()


def local(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise RuntimeError(f"path outside current worktree: {path}")
    return path


def summary(samples):
    ordered = sorted(samples)
    return {"median_us": statistics.median(samples), "min_us": min(samples),
            "p10_us": ordered[int(.1 * (len(ordered)-1))],
            "p90_us": ordered[int(.9 * (len(ordered)-1))],
            "max_us": max(samples), "samples_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=532891)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("run from the current worktree root")
    started = stamp()
    output = local(args.output)
    if output.exists():
        raise RuntimeError("refusing to overwrite a previous attempt")
    output.parent.mkdir(parents=True, exist_ok=True)
    record_path = local(args.build_record)
    build = json.loads(record_path.read_text())
    before = source_provenance()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status and not args.allow_dirty:
        raise RuntimeError("commit implementation/tests/harness through Burner before clean capture")
    for key in ("commit", "source_sha256", "production_diff_sha256"):
        if build[key] != before[key]:
            raise RuntimeError(f"stale build receipt: {key}")
    for key in ("TMPDIR", "CUDA_CACHE_PATH", "XDG_CACHE_HOME", "CARGO_HOME", "CARGO_TARGET_DIR",
                "PYO3_PYTHON", "TORCH_RS_CUDART", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        local(os.environ[key])
    local(sys.executable)
    local(sys.base_prefix)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("single-GPU campaign requires CUDA_VISIBLE_DEVICES=0")

    import numpy as np
    import torch
    import torch_rs as native
    extension = local(importlib.import_module("torch_rs.torch_rs").__file__)
    local(native.__file__)
    local(torch.__file__)
    local(np.__file__)
    if sha256(extension) != build["extension_sha256"]:
        raise RuntimeError("loaded extension differs from build receipt")
    assert torch.__version__.split("+")[0] == "2.13.0" and torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    # Fixed common shapes plus generated held-out square/rectangular dimensions.
    # Generation occurs before timing and no shape/result is dropped.
    shapes = [(1, 65539, 1), (64, 64, 64), (128, 128, 128), (512, 512, 512),
              (1024, 1024, 1024), (2048, 2048, 2048), (127, 1025, 65)]
    shapes += [(v, v, v) for v in map(int, rng.integers(96, 1537, size=2))]
    shapes += [tuple(map(int, rng.integers(65, 1201, size=3))) for _ in range(2)]
    lib = ctypes.CDLL(os.environ["TORCH_RS_CUDART"])
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int

    def materialize(value):
        array = np.empty(tuple(value.shape), dtype=np.float32)
        assert lib.cudaMemcpy(array.ctypes.data, value.data_ptr(), array.nbytes, 2) == 0
        return array

    def upload(module, values, offset, shape):
        base = module.tensor(values.tolist(), dtype=module.float32).to("cuda:0")
        return base, base[offset:].reshape(shape)

    report = {"measurement_kind": "uncommitted-integration-validation" if status else "clean-code",
              "command": [sys.executable, *sys.argv], "cwd": str(ROOT), "started_at": started,
              "source": before, "git_status": status, "harness_sha256": sha256(__file__),
              "build_record": str(record_path), "build_record_sha256": sha256(record_path),
              "extension": str(extension), "extension_sha256": sha256(extension),
              "executable": str(local(sys.executable)), "python": sys.version,
              "cache_state": "native build/cache state in linked build receipt; timing uses warm cuBLAS handles/allocations after declared warmups",
              "environment": {key: os.environ.get(key) for key in (
                  "CUDA_VISIBLE_DEVICES", "TORCH_RS_CUDART", "TORCH_RS_CUBLAS",
                  "CARGO_TARGET_DIR", "CARGO_HOME", "PYO3_PYTHON", "TMPDIR", "CUDA_CACHE_PATH",
                  "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")},
              "packages": {m.__name__: str(local(m.__file__)) for m in (torch, native, np)},
              "pytorch": torch.__version__, "pytorch_cuda": torch.version.cuda,
              "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,compute_cap", "--format=csv"], text=True),
              "available_nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
              "compiler": "nvcc unused; native cuBLAS SGEMM; other kernels driver-JIT PTX",
              "seed": args.seed, "shapes": shapes, "offsets": [0, 3],
              "policy": {"warmups_per_order": 10, "blocks_per_order": 31, "calls_per_block": 5,
                         "orders": [["native", "pytorch"], ["pytorch", "native"]],
                         "timing": "public a @ b, precreated equal inputs; device synchronize before and after each call; retain output until complete",
                         "aggregation": "exp(mean(log(min(1, pytorch_median/native_median)))) over every shape/offset, equal weights",
                         "rtol": 1e-5, "atol": 1e-4, "tf32": False}, "cases": []}
    try:
        for m, k, n in shapes:
            for offset in (0, 3):
                av, bv = [rng.normal(size=offset+size).astype(np.float32) for size in (m*k, k*n)]
                aa, a = upload(native, av, offset, (m,k))
                bb, b = upload(native, bv, offset, (k,n))
                taa, ta = upload(torch, av, offset, (m,k))
                tbb, tb = upload(torch, bv, offset, (k,n))
                calls = {"native": lambda: a @ b, "pytorch": lambda: ta @ tb}
                expected = materialize(ta @ tb)
                actual = a @ b
                np.testing.assert_allclose(materialize(actual), expected, rtol=1e-5, atol=1e-4)
                assert actual.data_ptr() not in (a.data_ptr(), b.data_ptr())
                samples = {key: [] for key in calls}
                orders = []
                for order in (("native", "pytorch"), ("pytorch", "native")):
                    for key in order:
                        for _ in range(10):
                            torch.cuda.synchronize()
                            result = calls[key]()
                            torch.cuda.synchronize()
                    order_samples = {key: [] for key in calls}
                    for _ in range(31):
                        for key in order:
                            elapsed = 0
                            for _ in range(5):
                                torch.cuda.synchronize()
                                tick = time.perf_counter_ns()
                                result = calls[key]()
                                torch.cuda.synchronize()
                                elapsed += time.perf_counter_ns() - tick
                            order_samples[key].append(elapsed / 5000)
                    for key in calls:
                        samples[key].extend(order_samples[key])
                        np.testing.assert_allclose(materialize(calls[key]()), expected, rtol=1e-5, atol=1e-4)
                    orders.append({"order": order, "samples": order_samples})
                for value, original in ((aa,av), (bb,bv), (taa,av), (tbb,bv)):
                    np.testing.assert_array_equal(materialize(value).view(np.uint32), original.view(np.uint32))
                medians = {key: summary(values) for key, values in samples.items()}
                parity = min(1., medians["pytorch"]["median_us"] / medians["native"]["median_us"])
                case = {"shape_mkn": [m,k,n], "offset": offset, "timings": medians,
                        "orders": orders, "capped_parity": parity,
                        "output_and_inputs_validated": True}
                report["cases"].append(case)
                print(json.dumps({"shape": [m,k,n], "offset": offset,
                                  "native_us": medians["native"]["median_us"],
                                  "pytorch_us": medians["pytorch"]["median_us"]}), flush=True)
        report["capped_geometric_parity"] = math.exp(statistics.mean(math.log(c["capped_parity"]) for c in report["cases"]))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["ended_at"] = stamp()
        report["runtime"] = runtime_provenance()
        libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if any(name in line for name in ("/libcudart.so", "/libcublas.so", "/libcublasLt.so"))})
        report["loaded_cuda_libraries"] = [{"path": str(local(p)), "sha256": sha256(p)} for p in libraries]
        blas_versions = []
        for path in libraries:
            if "/libcublas.so" in path:
                loaded = ctypes.CDLL(path)
                loaded.cublasGetProperty.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
                loaded.cublasGetProperty.restype = ctypes.c_int
                version = []
                for property_id in range(3):
                    value = ctypes.c_int()
                    assert loaded.cublasGetProperty(property_id, ctypes.byref(value)) == 0
                    version.append(value.value)
                blas_versions.append({"path": path, "version": version})
        report["cublas_versions"] = blas_versions
        report["source_after"] = source_provenance()
        assert report["source_after"] == before, "code changed during timing"
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Capped geometric parity: {report['capped_geometric_parity']:.4%}")


if __name__ == "__main__":
    main()
