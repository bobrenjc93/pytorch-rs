#!/usr/bin/env python3
"""Benchmark one CUDA ``torch.compile`` reference workload.

The benchmark is intentionally narrow: it measures a single PyTorch 2.13
CUDA/H100 reference workload and records the current ``torch_rs`` CUDA compile
path as explicit zero-credit unsupported evidence. CPU execution, eager
fallback, and forwarding to installed PyTorch are fail-closed and never count as
eligible CUDA compile evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_OUTPUT_PATHS = {
    REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
    REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
}

REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "torch_compile_cuda_h100_reference_benchmark_v1"
WORKLOAD_VERSION = "h100_cuda_pointwise_reduce_float32_v1"
WORKLOAD_SHAPE = (1024, 1024)
WORKLOAD_SEED = 20260904

REFERENCE_COMPILE_CONFIG = {
    "backend": "inductor",
    "fullgraph": True,
    "dynamic": False,
    "mode": None,
    "options": None,
}

DEFAULT_WARMUPS = 5
DEFAULT_SAMPLES = 17
DEFAULT_REPEATS = 3
DEFAULT_REQUIRED_CUDA_VISIBLE_DEVICES = "0"


def h100_cuda_pointwise_reduce_float32(x, bias):
    """One versioned CUDA reference workload for PyTorch 2.13."""
    mixed = (x + bias).sin() * (x - bias).cos()
    return (mixed + x.relu()).sum(dim=1)


def _version_without_local(version):
    return version.split("+", 1)[0]


def _package_version(distribution_name, module):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _run_text(command):
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _git_provenance():
    return {
        "head": _run_text(["git", "rev-parse", "HEAD"]),
        "status_short": _run_text(["git", "status", "--short"]),
        "diff_stat": _run_text(["git", "diff", "HEAD", "--stat"]),
    }


def _nvidia_smi_query():
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
    )
    if output is None:
        return None
    rows = []
    for line in output.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) == 5:
            rows.append(
                {
                    "index": columns[0],
                    "name": columns[1],
                    "memory_total_mib": columns[2],
                    "driver_version": columns[3],
                    "compute_capability": columns[4],
                }
            )
    return rows


def _nvcc_version():
    return _run_text(["nvcc", "--version"])


def _output_path(path):
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        raise SystemExit(
            f"output path must stay inside the worktree: {resolved}"
        ) from None
    if (
        resolved == REPOSITORY_ROOT / ".burner"
        or (REPOSITORY_ROOT / ".burner") in resolved.parents
        or resolved in PROTECTED_OUTPUT_PATHS
    ):
        raise SystemExit(f"refusing to write Burner-managed output path: {resolved}")
    return resolved


def _require_reference_environment(reference_torch, args):
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise SystemExit(
            "CUDA compile benchmark requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        args.required_cuda_visible_devices is not None
        and cuda_visible_devices != args.required_cuda_visible_devices
    ):
        raise SystemExit(
            "CUDA compile benchmark must run with "
            f"CUDA_VISIBLE_DEVICES={args.required_cuda_visible_devices!r}; "
            f"got {cuda_visible_devices!r}"
        )

    if not reference_torch.cuda.is_available():
        raise SystemExit(
            "CUDA compile benchmark requires a CUDA-visible PyTorch runtime"
        )
    if reference_torch.cuda.device_count() < 1:
        raise SystemExit("CUDA compile benchmark requires at least one visible GPU")

    reference_torch.cuda.set_device(0)
    device_name = reference_torch.cuda.get_device_name(0)
    if not args.allow_non_h100 and "H100" not in device_name:
        raise SystemExit(
            "CUDA compile benchmark is calibrated for NVIDIA H100; "
            f"visible device 0 is {device_name!r}"
        )


def _validate_counts(args):
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")


def _make_reference_inputs(reference_torch):
    reference_torch.manual_seed(WORKLOAD_SEED)
    x = reference_torch.randn(
        WORKLOAD_SHAPE,
        device="cuda",
        dtype=reference_torch.float32,
    )
    bias = reference_torch.randn(
        (WORKLOAD_SHAPE[1],),
        device="cuda",
        dtype=reference_torch.float32,
    )
    return x, bias


def _synchronize(reference_torch):
    reference_torch.cuda.synchronize(0)


def _tensor_metadata(tensor):
    device = tensor.device
    return {
        "shape": list(tuple(tensor.shape)),
        "stride": list(tuple(tensor.stride())),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "requires_grad": bool(tensor.requires_grad),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _checksum_tensor(tensor):
    payload = {
        "metadata": _tensor_metadata(tensor),
        "values": tensor.detach().cpu().contiguous().tolist(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _summarize_samples(samples_ns, repeats):
    samples_us = [sample / repeats / 1000.0 for sample in samples_ns]
    median_us = statistics.median(samples_us)
    deviations = [abs(sample - median_us) for sample in samples_us]
    variance_us2 = statistics.pvariance(samples_us) if len(samples_us) > 1 else 0.0
    return {
        "median_us": median_us,
        "mad_us": statistics.median(deviations),
        "variance_us2": variance_us2,
        "sample_count": len(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _time_once(reference_torch, compiled, inputs):
    _synchronize(reference_torch)
    started_ns = time.perf_counter_ns()
    output = compiled(*inputs)
    _synchronize(reference_torch)
    checksum = _checksum_tensor(output)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, checksum, output


def _time_repeated(reference_torch, compiled, inputs, repeats):
    _synchronize(reference_torch)
    started_ns = time.perf_counter_ns()
    output = None
    for _ in range(repeats):
        output = compiled(*inputs)
    _synchronize(reference_torch)
    checksum = _checksum_tensor(output)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, checksum


def _compile_reference(reference_torch):
    kwargs = {
        "backend": REFERENCE_COMPILE_CONFIG["backend"],
        "fullgraph": REFERENCE_COMPILE_CONFIG["fullgraph"],
        "dynamic": REFERENCE_COMPILE_CONFIG["dynamic"],
    }
    if REFERENCE_COMPILE_CONFIG["mode"] is not None:
        kwargs["mode"] = REFERENCE_COMPILE_CONFIG["mode"]
    if REFERENCE_COMPILE_CONFIG["options"] is not None:
        kwargs["options"] = REFERENCE_COMPILE_CONFIG["options"]
    return reference_torch.compile(h100_cuda_pointwise_reduce_float32, **kwargs)


def _run_pytorch_reference(reference_torch, args):
    inputs = _make_reference_inputs(reference_torch)
    expected = h100_cuda_pointwise_reduce_float32(*inputs)
    _synchronize(reference_torch)

    factory_started_ns = time.perf_counter_ns()
    compiled = _compile_reference(reference_torch)
    factory_ns = time.perf_counter_ns() - factory_started_ns

    cold_ns, cold_checksum, cold_output = _time_once(reference_torch, compiled, inputs)
    reference_torch.testing.assert_close(cold_output, expected)

    for _ in range(args.warmups):
        _time_repeated(reference_torch, compiled, inputs, args.repeats)

    sample_ns = []
    sample_checksums = []
    for _ in range(args.samples):
        elapsed_ns, checksum = _time_repeated(
            reference_torch,
            compiled,
            inputs,
            args.repeats,
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)

    expected_checksum = _checksum_tensor(expected)
    checksums = sorted(set([cold_checksum, *sample_checksums]))
    if len(checksums) != 1:
        raise AssertionError(
            "compiled CUDA workload produced unstable checksums: "
            f"{checksums!r}"
        )

    return {
        "implementation": "pytorch",
        "status": "ok",
        "workload_version": WORKLOAD_VERSION,
        "compile_config": dict(REFERENCE_COMPILE_CONFIG),
        "factory_us": factory_ns / 1000.0,
        "cold_first_call_us": cold_ns / 1000.0,
        "cold_checksum": cold_checksum,
        "steady": _summarize_samples(sample_ns, args.repeats),
        "steady_checksums": sorted(set(sample_checksums)),
        "input_metadata": [_tensor_metadata(input) for input in inputs],
        "output_metadata": _tensor_metadata(cold_output),
        "correctness": {
            "eager_reference_checksum": expected_checksum,
            "assert_close": True,
        },
    }


def classify_torch_rs_cuda_compile_evidence(evidence):
    """Return fail-closed CUDA compile eligibility for a torch_rs evidence row."""
    reasons = []

    if evidence.get("implementation") != "torch_rs":
        reasons.append("implementation is not torch_rs")
    if evidence.get("status") != "ok":
        reasons.append("execution status is not ok")
    if evidence.get("workload_version") != WORKLOAD_VERSION:
        reasons.append("workload version does not match the CUDA benchmark")
    if evidence.get("compile_backend") != REFERENCE_COMPILE_CONFIG["backend"]:
        reasons.append(
            "compile backend is not the declared CUDA reference backend "
            f"{REFERENCE_COMPILE_CONFIG['backend']!r}"
        )
    if evidence.get("input_device_type") != "cuda":
        reasons.append("inputs did not execute on CUDA")
    if evidence.get("output_device_type") != "cuda":
        reasons.append("outputs did not materialize on CUDA")
    if evidence.get("native_cuda_compile") is not True:
        reasons.append("native CUDA compile execution was not demonstrated")
    if evidence.get("eager_fallback") is not False:
        reasons.append("eager fallback is not eligible CUDA compile evidence")
    if evidence.get("forwarded_to_pytorch") is not False:
        reasons.append("forwarding to installed PyTorch is not eligible evidence")

    return {
        "eligible_cuda_compile_evidence": not reasons,
        "score_credit": 1.0 if not reasons else 0.0,
        "rejection_reasons": reasons,
    }


def _torch_rs_cuda_probes(torch_rs):
    return {
        "cuda_is_available": bool(torch_rs.cuda.is_available()),
        "cuda_device_count": int(torch_rs.cuda.device_count()),
        "cuda_is_initialized": bool(torch_rs.cuda.is_initialized()),
        "accelerator_is_available": bool(torch_rs.accelerator.is_available()),
        "accelerator_device_count": int(torch_rs.accelerator.device_count()),
    }


def torch_rs_zero_credit_unsupported_row(torch_rs):
    evidence = {
        "implementation": "torch_rs",
        "status": "unsupported",
        "workload_version": WORKLOAD_VERSION,
        "compile_backend": None,
        "input_device_type": None,
        "output_device_type": None,
        "native_cuda_compile": False,
        "eager_fallback": False,
        "forwarded_to_pytorch": False,
    }
    classification = classify_torch_rs_cuda_compile_evidence(evidence)
    return {
        "implementation": "torch_rs",
        "status": "zero_credit_unsupported",
        "workload_version": WORKLOAD_VERSION,
        "score_credit": 0.0,
        "reason": (
            "torch_rs currently has no native CUDA tensor execution or CUDA "
            "torch.compile backend; this benchmark records the unsupported "
            "cell explicitly instead of substituting CPU execution, "
            "backend='eager', eager fallback, or installed-PyTorch forwarding."
        ),
        "cuda_probes": _torch_rs_cuda_probes(torch_rs),
        "rejected_fallbacks": [
            "CPU tensor execution",
            "backend='eager' compile execution",
            "eager fallback",
            "forwarding to installed PyTorch",
            "skipped or missing execution",
        ],
        "eligibility": classification,
    }


def _device_provenance(reference_torch):
    properties = reference_torch.cuda.get_device_properties(0)
    memory_free, memory_total = reference_torch.cuda.mem_get_info(0)
    return {
        "torch_cuda_device_index": 0,
        "torch_cuda_device_name": reference_torch.cuda.get_device_name(0),
        "torch_cuda_device_capability": list(
            reference_torch.cuda.get_device_capability(0)
        ),
        "torch_cuda_total_memory_bytes": int(properties.total_memory),
        "torch_cuda_multiprocessor_count": int(properties.multi_processor_count),
        "torch_cuda_memory_free_bytes": int(memory_free),
        "torch_cuda_memory_total_bytes": int(memory_total),
        "nvidia_smi": _nvidia_smi_query(),
        "nvcc_version": _nvcc_version(),
    }


def _environment(reference_torch, torch_rs, args):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "workload_version": WORKLOAD_VERSION,
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warmups": args.warmups,
        "samples": args.samples,
        "repeats": args.repeats,
        "pytorch": {
            "version": reference_torch.__version__,
            "path": getattr(reference_torch, "__file__", None),
            "cuda_runtime": getattr(
                getattr(reference_torch, "version", None),
                "cuda",
                None,
            ),
            "git_version": getattr(
                getattr(reference_torch, "version", None),
                "git_version",
                None,
            ),
            "cuda_available": bool(reference_torch.cuda.is_available()),
            "cuda_device_count": int(reference_torch.cuda.device_count()),
            "cudnn_version": reference_torch.backends.cudnn.version(),
        },
        "torch_rs": {
            "version": _package_version("torch-rs", torch_rs),
            "path": getattr(torch_rs, "__file__", None),
        },
        "reference_compile_config": dict(REFERENCE_COMPILE_CONFIG),
        "workload": {
            "name": h100_cuda_pointwise_reduce_float32.__name__,
            "shape": list(WORKLOAD_SHAPE),
            "dtype": "torch.float32",
            "seed": WORKLOAD_SEED,
        },
        "gpu": _device_provenance(reference_torch),
        "git": _git_provenance(),
    }


def run_benchmark(args):
    _validate_counts(args)
    import torch as reference_torch

    _require_reference_environment(reference_torch, args)
    import torch_rs

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        pytorch_reference = _run_pytorch_reference(reference_torch, args)
        torch_rs_row = torch_rs_zero_credit_unsupported_row(torch_rs)
    finally:
        if gc_was_enabled:
            gc.enable()

    return {
        "environment": _environment(reference_torch, torch_rs, args),
        "reference_workload": pytorch_reference,
        "candidate": torch_rs_row,
        "aggregates": {
            "common_success_geomean_speed_ratio": None,
            "coverage_adjusted_overall_percent": 0.0,
            "torch_rs_cuda_compile_score_percent": 0.0,
            "zero_credit_unsupported_cell_count": 1,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--required-cuda-visible-devices",
        default=DEFAULT_REQUIRED_CUDA_VISIBLE_DEVICES,
        help=(
            "required CUDA_VISIBLE_DEVICES value; use an empty string to skip "
            "this check for local experimentation"
        ),
    )
    parser.add_argument(
        "--allow-non-h100",
        action="store_true",
        help="allow running the reference benchmark on a non-H100 CUDA device",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.required_cuda_visible_devices == "":
        args.required_cuda_visible_devices = None
    report = run_benchmark(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        output = _output_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
