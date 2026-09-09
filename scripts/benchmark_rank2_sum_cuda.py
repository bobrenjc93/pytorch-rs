#!/usr/bin/env python3
"""Reproducible integration screening, not a replacement for Burner evaluation.

Run the worktree wheel build/provenance check first, then:
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/benchmark_rank2_sum_cuda.py > target/integration-screen.json
Reports are local diagnostics; do not check in dirty-tree measurements.
Scoring requires --campaign (an externally owned definition) and --seed.
"""
import argparse
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
from rank2_sum_cuda_campaign import PUBLIC_CAMPAIGN, PUBLIC_SEED, cell_rng_seed, load_campaign

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_native_extension", ROOT / ".github/scripts/verify_native_extension.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def materialize(result):
    return (result.cpu() if result.is_cuda else result).tolist()


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
    materialize(result)
    return {"median_ns": statistics.median(timings), "samples_ns": timings, "stdev_ns": statistics.pstdev(timings)}


def metadata(result):
    return {"dtype": str(result.dtype), "device": str(result.device),
            "shape": list(result.shape), "stride": list(result.stride()),
            "storage_offset": result.storage_offset(),
            "requires_grad": result.requires_grad, "layout": str(result.layout)}


def validate_outputs(actual, expected):
    native, reference = actual(), expected()
    native_metadata, reference_metadata = metadata(native), metadata(reference)
    if native_metadata != reference_metadata:
        raise ValueError(f"output metadata mismatch: {native_metadata} != {reference_metadata}")
    np.testing.assert_allclose(materialize(native), materialize(reference), atol=1e-4, rtol=1e-4)
    return {"native": native_metadata, "pytorch": reference_metadata}


def invalid_cell(error, native_threads, reference_threads):
    matched = native_threads == reference_threads
    return {"valid": False, "validation_error": str(error),
            "native_threads": native_threads, "pytorch_threads": reference_threads,
            "comparison": "matched_threads" if matched else "scaling_diagnostic",
            "capped_parity": 0.0 if matched else None, "credit": 0.0}


def pair(actual, expected, sync=lambda: None, iterations=50):
    native_threads, reference_threads = torch_rs.get_num_threads(), torch.get_num_threads()
    matched = native_threads == reference_threads
    try:
        validated_metadata = validate_outputs(actual, expected)
        orders = []
        for order in ((0, 1), (1, 0)):
            elapsed = [None, None]
            for index in order:
                elapsed[index] = measure((actual, expected)[index], sync, iterations)
            orders.append(elapsed)
        validate_outputs(actual, expected)
        if (native_threads, reference_threads) != (torch_rs.get_num_threads(), torch.get_num_threads()):
            raise ValueError("thread configuration changed during timing")
    except Exception as error:
        return invalid_cell(error, native_threads, reference_threads)
    native, reference = [statistics.median(row[index]["median_ns"] for row in orders) for index in (0, 1)]
    parity = min(1, reference / native) if matched else None
    return {"valid": True, "output_metadata": validated_metadata,
            "native_threads": native_threads, "pytorch_threads": reference_threads,
            "comparison": "matched_threads" if matched else "scaling_diagnostic",
            "native_ns": native, "pytorch_ns": reference,
            "slowdown": native / reference, "capped_parity": parity, "credit": parity if matched else 0.0,
            "execution_orders": orders, "iterations_per_sample": iterations, "warmups": 12, "samples_per_order": 7}


def aggregate(cells):
    eligible = [cell for cell in cells if cell["comparison"] == "matched_threads"]
    credits = [cell["credit"] for cell in eligible]
    return {"matched_cells": len(eligible),
            "invalid_matched_cells": sum(not cell["valid"] for cell in eligible),
            "scaling_diagnostic_cells": len(cells) - len(eligible),
            "capped_geomean_parity": (None if not credits else 0.0 if 0 in credits
                                      else math.exp(statistics.mean(math.log(x) for x in credits)))}


def backward(x, axis, weights):
    # Rebuild the graph and accumulate into the leaf on both implementations.
    (x.sum(axis) * weights).sum().backward()
    return x.grad


def run_cell(cell, seed):
    torch.set_num_threads(cell["pytorch_threads"])
    kind = cell["kind"]
    try:
        if kind in ("sum", "backward_accumulate"):
            rng = np.random.default_rng(cell_rng_seed(seed, cell["id"]))
            values = rng.normal(size=cell["shape"]).astype(np.float32)
            axis, layout = cell["axis"], cell["layout"]
            grad = kind == "backward_accumulate"
            a = torch_rs.tensor(values.tolist(), requires_grad=grad)
            b = torch.tensor(values, requires_grad=grad)
            if layout == "transposed":
                a, b = a.t(), b.t()
            elif layout == "offset":
                a, b = a[:, 3:-2], b[:, 3:-2]
            elif layout == "selected":
                backing = np.stack((values, -values, values), axis=2)
                a = torch_rs.tensor(backing.tolist()).select(2, 1)
                b = torch.tensor(backing).select(2, 1)
            if grad:
                weights = rng.normal(size=values.shape[1 - axis]).astype(np.float32)
                aw, bw = torch_rs.tensor(weights.tolist()), torch.tensor(weights)
                result = pair(lambda: backward(a, axis, aw), lambda: backward(b, axis, bw), iterations=10)
            else:
                result = pair(lambda: a.sum(axis), lambda: b.sum(axis))
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; requested cell retained with zero credit")
            elements = cell["elements"]
            def actual():
                return torch_rs.zeros((elements,), device="cuda:0")
            def expected():
                return torch.zeros((elements,), device="cuda:0")
            if kind == "roundtrip":
                result = pair(lambda: actual().cpu(), lambda: expected().cpu(), torch.cuda.synchronize, iterations=20)
            elif kind == "zeros":
                result = pair(actual, expected, torch.cuda.synchronize, iterations=20)
            else:
                a, b = actual(), expected()
                result = pair(a.cpu, b.cpu, torch.cuda.synchronize, iterations=20)
    except Exception as error:
        result = invalid_cell(error, torch_rs.get_num_threads(), torch.get_num_threads())
    return {**cell, "input_seed": cell_rng_seed(seed, cell["id"]), **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    campaigns = parser.add_mutually_exclusive_group()
    campaigns.add_argument("--campaign", type=Path, help="read-only, reviewer-owned scoring campaign outside this worktree")
    campaigns.add_argument("--diagnostic-campaign", type=Path, default=PUBLIC_CAMPAIGN, help="local campaign fixture; produces diagnostics only")
    parser.add_argument("--seed", type=int, help="reviewer-supplied held-out seed, required for scoring")
    parser.add_argument("--quick", action="store_true", help="run the campaign's strict subset with unchanged cell inputs")
    args = parser.parse_args()
    scoring = args.campaign is not None
    if scoring and args.seed is None:
        parser.error("--campaign requires --seed from the reviewer/evaluation runner")
    seed = args.seed if args.seed is not None else PUBLIC_SEED
    try:
        workloads, campaign = load_campaign(args.campaign or args.diagnostic_campaign,
                                             scoring=scoring, seed=seed, quick=args.quick)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    verify.verify()
    if any(cell["kind"] in ("zeros", "to_cpu", "roundtrip") for cell in workloads):
        if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            parser.error("run with CUDA_VISIBLE_DEVICES=0")
    results = [run_cell(cell, seed) for cell in workloads]
    cells = [cell for cell in results if cell["kind"] in ("sum", "backward_accumulate")]
    gpu_cells = [cell for cell in results if cell["kind"] not in ("sum", "backward_accumulate")]
    tracked_sources = command("git", "ls-files", "src", "python", "Cargo.toml", "Cargo.lock").splitlines()
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tracked_sources}
    for name in ("src/cuda.rs", "src/reduction.rs", "scripts/benchmark_rank2_sum_cuda.py", "scripts/rank2_sum_cuda_campaign.py", "scripts/campaigns/rank2_sum_cuda_diagnostic.json"):
        hashes[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    loaded_cuda = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines() if "libcudart" in line}) if sys.platform == "linux" else []
    report = {"seed": seed, "campaign": campaign, "git_commit": command("git", "rev-parse", "HEAD"),
              "git_status": command("git", "status", "--porcelain"), "source_sha256": hashes,
              "worktree": str(ROOT), "python": sys.executable, "native_extension": torch_rs.torch_rs.__file__,
              "build": {"profile": "release", "features": "extension-module", "lto": "thin", "codegen_units": 1, "rustflags": os.environ.get("RUSTFLAGS", ""), "cuda_compilation": "none (runtime API only)", "rustc": command("rustc", "--version"),
                        "native_sha256": hashlib.sha256(Path(torch_rs.torch_rs.__file__).read_bytes()).hexdigest()},
              "os": platform.platform(), "python_version": sys.version, "numpy": np.__version__,
              "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), "unknown") if sys.platform == "linux" else "unknown",
              "pytorch": torch.__version__, "pytorch_cuda": torch.version.cuda, "loaded_cudart": loaded_cuda,
              "gpu": command("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader", "-i", "0") if gpu_cells and torch.cuda.is_available() else None,
              "nvcc": command("nvcc", "--version") if gpu_cells and torch.cuda.is_available() else None,
              "cpu_cells": cells, "cuda_cells": gpu_cells,
              "parity" if scoring else "diagnostic_parity": {
                  "cpu": aggregate(cells),
                  "cuda_by_operation": {kind: aggregate([cell for cell in gpu_cells if cell["kind"] == kind])
                                        for kind in ("zeros", "to_cpu", "roundtrip")}
              }, "timing_scope": "steady state only; build and dependency installation excluded", "reference_environment_setup_seconds": None}
    print(json.dumps(report, indent=2))
    return 0 if all(cell["valid"] for cell in results) else 1


if __name__ == "__main__":
    sys.exit(main())
