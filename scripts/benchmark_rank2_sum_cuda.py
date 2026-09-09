#!/usr/bin/env python3
"""Reproducible integration screening, not a replacement for Burner evaluation.

Run the worktree wheel build/provenance check first, then:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_rank2_sum_cuda.py > target/integration-screen.json
Reports are local diagnostics; do not check in dirty-tree measurements.
"""
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np
import torch
import torch_rs

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_native_extension", ROOT / ".github/scripts/verify_native_extension.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)
verify.verify()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def measure(call, sync=lambda: None, iterations=50):
    for _ in range(12):
        result = call()
    sync()
    timings = []
    for _ in range(7):
        sync()
        start = time.perf_counter_ns()
        for _ in range(iterations):
            result = call()
        sync()
        timings.append((time.perf_counter_ns() - start) / iterations)
    # Native eager outputs are already materialized; consume the final output
    # after timing on both sides. Backward calls return the leaf gradient.
    result.tolist()
    return {"median_ns": statistics.median(timings), "samples_ns": timings, "stdev_ns": statistics.pstdev(timings)}


def pair(actual, expected, sync=lambda: None, iterations=50):
    np.testing.assert_allclose(actual().tolist(), expected().tolist(), atol=1e-4, rtol=1e-4)
    orders = []
    for order in ((0, 1), (1, 0)):
        elapsed = [0, 0]
        for index in order:
            elapsed[index] = measure((actual, expected)[index], sync, iterations)
        orders.append(elapsed)
    np.testing.assert_allclose(actual().tolist(), expected().tolist(), atol=1e-4, rtol=1e-4)
    native, reference = [statistics.median(row[index]["median_ns"] for row in orders) for index in (0, 1)]
    return {"native_ns": native, "pytorch_ns": reference,
            "slowdown": native / reference, "capped_parity": min(1, reference / native),
            "execution_orders": orders, "iterations_per_sample": iterations, "warmups": 12, "samples_per_order": 7}


def backward(x, axis, weights):
    # Rebuild the graph and accumulate into the leaf on both implementations.
    (x.sum(axis) * weights).sum().backward()
    return x.grad


def main():
    rng = np.random.default_rng(2511894)
    cells = []
    # Rectangular shapes and offset views complement the reported square screens.
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for shape in ((32, 32), (256, 256), (1024, 1024), (2048, 2048), (193, 1031), (2053, 67)):
            values = rng.normal(size=shape).astype(np.float32)
            for layout in ("contiguous", "transposed", "offset"):
                a, b = torch_rs.tensor(values.tolist()), torch.tensor(values)
                if layout == "transposed":
                    a, b = a.t(), b.t()
                elif layout == "offset":
                    a, b = a[:, 3:-2], b[:, 3:-2]
                for axis in (0, 1):
                    cells.append({"shape": shape, "layout": layout, "axis": axis, "pytorch_threads": threads, "native_threads": torch_rs.get_num_threads(),
                                  "kind": "sum", **pair(lambda: a.sum(axis), lambda: b.sum(axis))})
        for axis in (0, 1):
            values = rng.normal(size=(131, 259)).astype(np.float32)
            a, b = torch_rs.tensor(values.tolist(), requires_grad=True), torch.tensor(values, requires_grad=True)
            weights = rng.normal(size=values.shape[1 - axis]).astype(np.float32)
            aw, bw = torch_rs.tensor(weights.tolist()), torch.tensor(weights)
            cells.append({"shape": values.shape, "axis": axis, "pytorch_threads": threads, "native_threads": torch_rs.get_num_threads(), "kind": "backward_accumulate",
                          **pair(lambda: backward(a, axis, aw), lambda: backward(b, axis, bw), iterations=10)})
    gpu_cells = []
    if torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            raise RuntimeError("run with CUDA_VISIBLE_DEVICES=0")
        for elements in (0, 1024, 1048576):
            def actual():
                return torch_rs.zeros((elements,), device="cuda:0").cpu()
            def expected():
                return torch.zeros((elements,), device="cuda:0").cpu()
            gpu_cells.append({"elements": elements, **pair(actual, expected, torch.cuda.synchronize, iterations=20)})
    tracked_sources = command("git", "ls-files", "src", "python", "Cargo.toml", "Cargo.lock").splitlines()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tracked_sources}
    for name in ("src/cuda.rs", "src/reduction.rs", "scripts/benchmark_rank2_sum_cuda.py"):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    loaded_cuda = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libcudart" in line}) if sys.platform == "linux" else []
    report = {"seed": 2511894, "git_commit": command("git", "rev-parse", "HEAD"),
              "git_status": command("git", "status", "--porcelain"), "source_sha256": hashes,
              "worktree": str(ROOT), "python": sys.executable, "native_extension": torch_rs.torch_rs.__file__,
              "build": {"profile": "release", "features": "extension-module", "lto": "thin", "codegen_units": 1, "rustflags": os.environ.get("RUSTFLAGS", ""), "cuda_compilation": "none (runtime API only)", "rustc": command("rustc", "--version"),
                        "native_sha256": hashlib.sha256(Path(torch_rs.torch_rs.__file__).read_bytes()).hexdigest()},
              "os": platform.platform(), "python_version": sys.version, "numpy": np.__version__,
              "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown") if sys.platform == "linux" else "unknown",
              "pytorch": torch.__version__, "pytorch_cuda": torch.version.cuda, "loaded_cudart": loaded_cuda,
              "gpu": command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0") if gpu_cells else None,
              "nvcc": command("nvcc", "--version") if gpu_cells else None,
              "cpu_cells": cells, "cuda_cells": gpu_cells,
              "cpu_capped_geomean_parity_by_reference_threads": {
                  str(threads): math.exp(statistics.mean(math.log(cell["capped_parity"]) for cell in cells if cell["pytorch_threads"] == threads))
                  for threads in (1, 4)
              }, "timing_scope": "steady state only; build and dependency installation excluded", "reference_environment_setup_seconds": None}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
