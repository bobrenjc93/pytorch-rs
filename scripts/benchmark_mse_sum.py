#!/usr/bin/env python3
"""Benchmark supported CPU ``torch.nn.functional.mse_loss(..., reduction="sum")`` cells."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "mse_sum_reduction_benchmark_v2"
DEFAULT_WARMUPS = 15
DEFAULT_SAMPLES = 81
DEFAULT_THREADS = 1
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
IMPLEMENTATION_ORDERS = (
    ("torch_rs", "pytorch"),
    ("pytorch", "torch_rs"),
)
BENCHMARK_CATEGORY_WEIGHTS = {
    "public_fixed_shape": 40,
    "generated_same_shape": 30,
    "held_out_same_shape": 20,
    "unsupported_boundaries": 10,
}


@dataclass(frozen=True)
class Workload:
    name: str
    category: str
    denominator_category: str
    split: str
    description: str
    shape: tuple[int, ...]
    seed: int
    repeats: int
    rtol: float
    atol: float


@dataclass(frozen=True)
class UnsupportedCell:
    name: str
    category: str
    reason: str


WORKLOADS = (
    Workload(
        name="mse_sum_same_contiguous_1024x1024",
        category="same-shape contiguous no-grad sum",
        denominator_category="public_fixed_shape",
        split="public",
        description="row-major contiguous same-shape CPU float32 tensors",
        shape=(1024, 1024),
        seed=20260903,
        repeats=16,
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
    Workload(
        name="mse_sum_generated_vector_65537",
        category="same-shape contiguous no-grad sum",
        denominator_category="generated_same_shape",
        split="generated",
        description="generated rank-1 prime-length CPU float32 tensors",
        shape=(65537,),
        seed=20260904,
        repeats=64,
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
    Workload(
        name="mse_sum_generated_rank3_17x19x23",
        category="same-shape contiguous no-grad sum",
        denominator_category="generated_same_shape",
        split="generated",
        description="generated row-major rank-3 CPU float32 tensors",
        shape=(17, 19, 23),
        seed=20260905,
        repeats=128,
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
    Workload(
        name="mse_sum_heldout_prime_matrix_257x263",
        category="same-shape contiguous no-grad sum",
        denominator_category="held_out_same_shape",
        split="held_out",
        description="held-out row-major prime-shape CPU float32 tensors",
        shape=(257, 263),
        seed=20260906,
        repeats=64,
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
    Workload(
        name="mse_sum_heldout_skinny_matrix_1x8192",
        category="same-shape contiguous no-grad sum",
        denominator_category="held_out_same_shape",
        split="held_out",
        description="held-out skinny row-major CPU float32 tensors",
        shape=(1, 8192),
        seed=20260907,
        repeats=128,
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
)

ZERO_CREDIT_UNSUPPORTED_CELLS = (
    UnsupportedCell(
        name="mse_sum_broadcasted_operands",
        category="unsupported_boundaries",
        reason=(
            "this same-shape contiguous benchmark campaign does not claim "
            "broadcast-reduction coverage"
        ),
    ),
    UnsupportedCell(
        name="mse_sum_noncontiguous_same_shape",
        category="unsupported_boundaries",
        reason=(
            "non-contiguous layouts are validated by correctness tests but are "
            "not timed as supported direct fast-path cells"
        ),
    ),
    UnsupportedCell(
        name="mse_sum_active_autograd",
        category="unsupported_boundaries",
        reason=(
            "active-autograd inputs intentionally fall back to the composed "
            "differentiable path and are not counted as fast-path timing cells"
        ),
    ),
    UnsupportedCell(
        name="mse_sum_dtype_or_device_expansion",
        category="unsupported_boundaries",
        reason=(
            "non-float32 or non-CPU tensors remain outside this native fast path"
        ),
    ),
)


def _version_without_local(version):
    return version.split("+", 1)[0]


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


def _cpu_model_name():
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _affinity():
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _pin_cpu(requested_cpu):
    original_affinity = _affinity()
    if original_affinity is None:
        raise SystemExit("CPU affinity pinning requires os.sched_setaffinity support")
    if not original_affinity:
        raise SystemExit("CPU affinity pinning found no available CPUs")

    cpu = min(original_affinity) if requested_cpu is None else requested_cpu
    if cpu not in original_affinity:
        raise SystemExit(
            f"requested CPU {cpu} is outside the initial affinity {original_affinity!r}"
        )
    os.sched_setaffinity(0, {cpu})
    pinned_affinity = _affinity()
    if pinned_affinity != [cpu]:
        raise SystemExit(
            f"failed to pin benchmark to CPU {cpu}: affinity is {pinned_affinity!r}"
        )
    return {
        "requested_cpu": requested_cpu,
        "selected_cpu": cpu,
        "initial_affinity": original_affinity,
        "pinned_affinity": pinned_affinity,
    }


def _configure_thread_environment(threads, cuda_visible_devices):
    thread_value = str(threads)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = thread_value
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def _package_version(distribution_name, module):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _import_backends():
    import numpy as np
    import torch as reference_torch
    import torch_rs

    return np, torch_rs, reference_torch


def _configure_reference_threads(reference_torch, threads):
    reference_torch.set_num_threads(threads)
    reference_torch.set_num_interop_threads(threads)


def _validate_thread_configuration(torch_rs, reference_torch, threads):
    if reference_torch.get_num_threads() != threads:
        raise SystemExit(
            f"PyTorch intra-op threads are {reference_torch.get_num_threads()}, "
            f"expected {threads}"
        )
    if reference_torch.get_num_interop_threads() != threads:
        raise SystemExit(
            "PyTorch inter-op threads are "
            f"{reference_torch.get_num_interop_threads()}, expected {threads}"
        )
    if torch_rs.get_num_threads() != threads:
        raise SystemExit(
            f"torch_rs threads are {torch_rs.get_num_threads()}, expected {threads}"
        )
    if torch_rs.get_num_interop_threads() != threads:
        raise SystemExit(
            "torch_rs inter-op threads are "
            f"{torch_rs.get_num_interop_threads()}, expected {threads}"
        )


def _make_workload_arrays(np, workload):
    if any(dimension < 0 for dimension in workload.shape):
        raise SystemExit(f"{workload.name} has a negative dimension: {workload.shape!r}")
    rng = np.random.default_rng(workload.seed)
    base_steps = rng.integers(
        -128,
        129,
        size=workload.shape,
        dtype=np.int16,
    )
    delta_steps = rng.integers(
        -2,
        3,
        size=workload.shape,
        dtype=np.int16,
    )
    left = base_steps.astype(np.float32) * np.float32(0.25)
    right = (base_steps - delta_steps).astype(np.float32) * np.float32(0.25)
    return left, right


def _make_operands(module, left_array, right_array):
    return (
        module.tensor(left_array.tolist(), dtype=module.float32),
        module.tensor(right_array.tolist(), dtype=module.float32),
    )


def _synchronize(module):
    cuda = getattr(module, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    synchronize = getattr(cuda, "synchronize", None)
    if is_available is not None and synchronize is not None and is_available():
        synchronize()


def _tensor_metadata(tensor):
    return {
        "shape": list(tuple(tensor.shape)),
        "stride": list(tuple(tensor.stride())),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "is_leaf": bool(tensor.is_leaf),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _tensor_array(np, tensor):
    return np.ascontiguousarray(np.asarray(tensor, dtype=np.float32))


def _tensor_value_bits(np, tensor):
    return _tensor_array(np, tensor).reshape(-1).view(np.uint32).tolist()


def _checksum_tensor(np, tensor):
    payload = {
        "metadata": _tensor_metadata(tensor),
        "value_bits": _tensor_value_bits(np, tensor),
    }
    encoded = json.dumps(
        payload,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _assert_cpu_float32_contiguous_operands(operands, *, implementation, workload):
    for index, tensor in enumerate(operands):
        metadata = _tensor_metadata(tensor)
        expected = {
            "shape": list(workload.shape),
            "stride": _contiguous_stride(workload.shape),
            "storage_offset": 0,
            "dtype": "torch.float32",
            "device": "cpu",
            "requires_grad": False,
            "is_leaf": True,
            "is_contiguous": True,
        }
        if metadata != expected:
            raise AssertionError(
                f"{workload.name}/{implementation} input {index} metadata mismatch:\n"
                f"actual={metadata!r}\nexpected={expected!r}"
            )


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dimension in reversed(shape):
        stride.append(running)
        running *= dimension
    return list(reversed(stride))


def _metadata_without_leaf(metadata):
    return {key: value for key, value in metadata.items() if key != "is_leaf"}


def _assert_output_matches(np, actual, expected, *, workload, implementation):
    actual_metadata = _tensor_metadata(actual)
    expected_metadata = _tensor_metadata(expected)
    if _metadata_without_leaf(actual_metadata) != _metadata_without_leaf(expected_metadata):
        raise AssertionError(
            f"{workload.name}/{implementation} output metadata mismatch:\n"
            f"actual={actual_metadata!r}\nexpected={expected_metadata!r}"
        )
    if not bool(actual_metadata["is_leaf"]) or not bool(expected_metadata["is_leaf"]):
        raise AssertionError(
            f"{workload.name}/{implementation} expected no-grad scalar leaf outputs: "
            f"actual={actual_metadata!r}, expected={expected_metadata!r}"
        )

    actual_values = _tensor_array(np, actual)
    expected_values = _tensor_array(np, expected)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=workload.rtol,
        atol=workload.atol,
        equal_nan=True,
    ):
        raise AssertionError(
            f"{workload.name}/{implementation} output values mismatch:\n"
            f"actual={actual_values.tolist()!r}\nexpected={expected_values.tolist()!r}"
        )
    actual_checksum = _checksum_tensor(np, actual)
    expected_checksum = _checksum_tensor(np, expected)
    if actual_checksum != expected_checksum:
        raise AssertionError(
            f"{workload.name}/{implementation} output checksum mismatch: "
            f"actual={actual_checksum!r} expected={expected_checksum!r}"
        )


def _call_mse_sum(module, input_tensor, target_tensor):
    with module.no_grad():
        return module.nn.functional.mse_loss(input_tensor, target_tensor, reduction="sum")


def _time_block(np, module, input_tensor, target_tensor, repeats):
    started_ns = time.perf_counter_ns()
    output = None
    materialized_scalar_sum = 0.0
    for _ in range(repeats):
        output = _call_mse_sum(module, input_tensor, target_tensor)
        _synchronize(module)
        materialized_scalar_sum += float(output.item())
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, _checksum_tensor(np, output), materialized_scalar_sum, output


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


def _measure_one_pass(np, module, implementation, workload, left_array, right_array, args):
    operands = _make_operands(module, left_array, right_array)
    _assert_cpu_float32_contiguous_operands(
        operands,
        implementation=implementation,
        workload=workload,
    )
    input_checksums_before = [_checksum_tensor(np, operand) for operand in operands]

    cold_ns, cold_checksum, cold_sink, cold_output = _time_block(
        np,
        module,
        operands[0],
        operands[1],
        1,
    )
    for _ in range(args.warmups):
        _time_block(np, module, operands[0], operands[1], workload.repeats)

    sample_ns = []
    sample_checksums = []
    sample_sinks = []
    for _ in range(args.samples):
        elapsed_ns, checksum, sink, _ = _time_block(
            np,
            module,
            operands[0],
            operands[1],
            workload.repeats,
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)
        sample_sinks.append(sink)

    input_checksums_after = [_checksum_tensor(np, operand) for operand in operands]
    if input_checksums_after != input_checksums_before:
        raise AssertionError(
            f"{workload.name}/{implementation} mutated benchmark operands: "
            f"before={input_checksums_before!r} after={input_checksums_after!r}"
        )

    return {
        "cold_first_call_us": cold_ns / 1000.0,
        "cold_checksum": cold_checksum,
        "cold_materialized_scalar_sum": cold_sink,
        "steady": _summarize_samples(sample_ns, workload.repeats),
        "steady_checksums": sorted(set(sample_checksums)),
        "steady_materialized_scalar_sums": sorted(set(sample_sinks)),
        "input_metadata": [_tensor_metadata(operand) for operand in operands],
        "input_checksums": input_checksums_before,
        "output_metadata": _tensor_metadata(cold_output),
        "cold_output": cold_output,
    }


def _geomean(values):
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _split_counts(workloads):
    counts = {}
    for workload in workloads:
        counts[workload.split] = counts.get(workload.split, 0) + 1
    return dict(sorted(counts.items()))


def _coverage_denominator(workloads):
    timed_by_category = {}
    for workload in workloads:
        timed_by_category.setdefault(workload.denominator_category, []).append(
            workload.name
        )
    unsupported_by_category = {}
    for cell in ZERO_CREDIT_UNSUPPORTED_CELLS:
        unsupported_by_category.setdefault(cell.category, []).append(
            {
                "name": cell.name,
                "reason": cell.reason,
            }
        )

    supported_categories = []
    zero_credit_categories = []
    supported_weight = 0
    for category, weight in BENCHMARK_CATEGORY_WEIGHTS.items():
        timed_workloads = timed_by_category.get(category, [])
        if timed_workloads:
            supported_weight += weight
            supported_categories.append(
                {
                    "category": category,
                    "weight": weight,
                    "timed_workloads": timed_workloads,
                }
            )
        else:
            unsupported_cells = unsupported_by_category.get(category, [])
            reason = None
            if not unsupported_cells:
                reason = "no timed workload selected for this category"
            zero_credit_categories.append(
                {
                    "category": category,
                    "weight": weight,
                    "reason": reason,
                    "unsupported_cells": unsupported_cells,
                }
            )

    total_weight = sum(BENCHMARK_CATEGORY_WEIGHTS.values())
    return {
        "category_weights": BENCHMARK_CATEGORY_WEIGHTS,
        "total_weight": total_weight,
        "supported_weight": supported_weight,
        "zero_credit_weight": total_weight - supported_weight,
        "weighted_supported_percent": (supported_weight / total_weight * 100.0)
        if total_weight
        else None,
        "timed_workload_count": len(workloads),
        "timed_workload_split_counts": _split_counts(workloads),
        "zero_credit_unsupported_cell_count": len(ZERO_CREDIT_UNSUPPORTED_CELLS),
        "supported_categories": supported_categories,
        "zero_credit_categories": zero_credit_categories,
    }


def _select_workloads(selected_names):
    if not selected_names:
        return WORKLOADS
    by_name = {workload.name: workload for workload in WORKLOADS}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise SystemExit(
            f"unknown workload: {', '.join(missing)}. Available: {available}"
        )
    return tuple(by_name[name] for name in selected_names)


def _output_path(path):
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        raise SystemExit(f"output path must stay inside the worktree: {resolved}") from None
    protected_paths = {
        REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
        REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
    }
    if (
        resolved == REPOSITORY_ROOT / ".burner"
        or (REPOSITORY_ROOT / ".burner") in resolved.parents
        or resolved in protected_paths
    ):
        raise SystemExit(f"refusing to write Burner-managed output path: {resolved}")
    return resolved


def _threshold_gate(report, args):
    failures = []
    geomean_limit = args.max_steady_geomean_ratio
    cell_limit = args.max_steady_cell_ratio

    if geomean_limit is not None and geomean_limit <= 0.0:
        raise SystemExit("--max-steady-geomean-ratio must be positive")
    if cell_limit is not None and cell_limit <= 0.0:
        raise SystemExit("--max-steady-cell-ratio must be positive")

    geomean = report["aggregates"]["steady_geomean_torch_rs_over_pytorch"]
    if geomean_limit is not None and geomean > geomean_limit:
        failures.append(
            "steady geomean "
            f"{geomean:.6g} exceeded --max-steady-geomean-ratio "
            f"{geomean_limit:.6g}"
        )

    if cell_limit is not None:
        for row in report["cases"]:
            ratio = row["ratios"]["steady_torch_rs_over_pytorch"]
            if ratio > cell_limit:
                failures.append(
                    f"{row['name']} steady ratio {ratio:.6g} exceeded "
                    f"--max-steady-cell-ratio {cell_limit:.6g}"
                )

    report["threshold_gate"] = {
        "max_steady_geomean_ratio": geomean_limit,
        "max_steady_cell_ratio": cell_limit,
        "status": "failed" if failures else "passed",
        "failures": failures,
    }
    if failures:
        message = "benchmark threshold gate failed:\n" + "\n".join(
            f"- {item}" for item in failures
        )
        raise SystemExit(message)


def _environment(torch_rs, reference_torch, np, affinity, args):
    command = [sys.executable, *sys.argv]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "command_argv": command,
        "command_shell": " ".join(shlex.quote(argument) for argument in command),
        "cwd": str(REPOSITORY_ROOT),
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu": _cpu_model_name(),
        "cpu_affinity": affinity,
        "env_threads": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "numpy": {
            "version": np.__version__,
            "path": getattr(np, "__file__", None),
        },
        "pytorch": {
            "version": reference_torch.__version__,
            "path": getattr(reference_torch, "__file__", None),
            "cuda": getattr(getattr(reference_torch, "version", None), "cuda", None),
            "cuda_available": bool(reference_torch.cuda.is_available()),
            "threads": reference_torch.get_num_threads(),
            "interop_threads": reference_torch.get_num_interop_threads(),
        },
        "torch_rs": {
            "version": _package_version("torch-rs", torch_rs),
            "path": getattr(torch_rs, "__file__", None),
            "threads": torch_rs.get_num_threads(),
            "interop_threads": torch_rs.get_num_interop_threads(),
        },
        "rust": {
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "git": _git_provenance(),
        "warmups": args.warmups,
        "samples": args.samples,
        "threads": args.threads,
        "thresholds": {
            "max_steady_geomean_ratio": args.max_steady_geomean_ratio,
            "max_steady_cell_ratio": args.max_steady_cell_ratio,
        },
        "implementation_orders": IMPLEMENTATION_ORDERS,
    }


def _validate_reference_version(reference_torch):
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise SystemExit(
            "MSE sum benchmark requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )


def run_benchmark(args):
    affinity = _pin_cpu(args.cpu)
    _configure_thread_environment(args.threads, args.cuda_visible_devices)
    np, torch_rs, reference_torch = _import_backends()
    _validate_reference_version(reference_torch)
    _configure_reference_threads(reference_torch, args.threads)
    _validate_thread_configuration(torch_rs, reference_torch, args.threads)

    workloads = _select_workloads(args.workloads)
    cases = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for workload in workloads:
            left_array, right_array = _make_workload_arrays(np, workload)
            reference_operands = _make_operands(reference_torch, left_array, right_array)
            with warnings.catch_warnings(), reference_torch.no_grad():
                warnings.simplefilter("error")
                reference_output = reference_torch.nn.functional.mse_loss(
                    reference_operands[0],
                    reference_operands[1],
                    reduction="sum",
                )
            reference_checksum = _checksum_tensor(np, reference_output)
            pass_results = {"torch_rs": [], "pytorch": []}

            for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
                for implementation in order:
                    module = torch_rs if implementation == "torch_rs" else reference_torch
                    measured = _measure_one_pass(
                        np,
                        module,
                        implementation,
                        workload,
                        left_array,
                        right_array,
                        args,
                    )
                    _assert_output_matches(
                        np,
                        measured["cold_output"],
                        reference_output,
                        workload=workload,
                        implementation=implementation,
                    )
                    if measured["steady_checksums"] != [reference_checksum]:
                        raise AssertionError(
                            f"{workload.name}/{implementation} steady checksum mismatch "
                            f"or instability: actual={measured['steady_checksums']!r} "
                            f"expected={[reference_checksum]!r}"
                        )
                    pass_results[implementation].append(
                        {
                            "order_index": order_index,
                            "order": list(order),
                            "cold_first_call_us": measured["cold_first_call_us"],
                            "cold_checksum": measured["cold_checksum"],
                            "cold_materialized_scalar_sum": measured[
                                "cold_materialized_scalar_sum"
                            ],
                            "steady": measured["steady"],
                            "steady_checksums": measured["steady_checksums"],
                            "steady_materialized_scalar_sums": measured[
                                "steady_materialized_scalar_sums"
                            ],
                            "input_metadata": measured["input_metadata"],
                            "input_checksums": measured["input_checksums"],
                            "output_metadata": measured["output_metadata"],
                        }
                    )

            implementations = {}
            for implementation, passes in pass_results.items():
                medians = [item["steady"]["median_us"] for item in passes]
                mads = [item["steady"]["mad_us"] for item in passes]
                variances = [item["steady"]["variance_us2"] for item in passes]
                checksums = sorted(
                    {
                        checksum
                        for item in passes
                        for checksum in (
                            item["steady_checksums"] + [item["cold_checksum"]]
                        )
                    }
                )
                input_checksums = sorted(
                    {
                        checksum
                        for item in passes
                        for checksum in item["input_checksums"]
                    }
                )
                implementations[implementation] = {
                    "steady_median_us": statistics.median(medians),
                    "steady_mad_us": statistics.median(mads),
                    "steady_variance_us2": statistics.median(variances),
                    "steady_sample_count": sum(
                        item["steady"]["sample_count"] for item in passes
                    ),
                    "checksums": checksums,
                    "input_checksums": input_checksums,
                    "passes": passes,
                }

            torch_rs_median = implementations["torch_rs"]["steady_median_us"]
            pytorch_median = implementations["pytorch"]["steady_median_us"]
            cases.append(
                {
                    "name": workload.name,
                    "category": workload.category,
                    "denominator_category": workload.denominator_category,
                    "split": workload.split,
                    "description": workload.description,
                    "shape": list(workload.shape),
                    "seed": workload.seed,
                    "repeats": workload.repeats,
                    "tolerance": {
                        "rtol": workload.rtol,
                        "atol": workload.atol,
                        "equal_nan": True,
                    },
                    "input_metadata": pass_results["torch_rs"][0]["input_metadata"],
                    "output_metadata": pass_results["torch_rs"][0]["output_metadata"],
                    "implementations": implementations,
                    "ratios": {
                        "steady_torch_rs_over_pytorch": torch_rs_median
                        / pytorch_median,
                    },
                    "validation": {
                        "reference_checksum": reference_checksum,
                        "metadata_checked": True,
                        "checksums_checked": True,
                        "operand_nonmutation_checked": True,
                        "warnings_as_errors": True,
                    },
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    ratios = [case["ratios"]["steady_torch_rs_over_pytorch"] for case in cases]
    return {
        "environment": _environment(torch_rs, reference_torch, np, affinity, args),
        "cases": cases,
        "aggregates": {
            "timed_cell_count": len(cases),
            "steady_geomean_torch_rs_over_pytorch": _geomean(ratios),
            "steady_geomean_capped_0_10_10_0": _geomean(
                [min(10.0, max(0.10, ratio)) for ratio in ratios]
            ),
        },
        "coverage_denominator": _coverage_denominator(workloads),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--workloads", nargs="*", default=())
    parser.add_argument(
        "--max-steady-geomean-ratio",
        type=float,
        help="Fail when the steady-state torch_rs/PyTorch geomean exceeds this ratio.",
    )
    parser.add_argument(
        "--max-steady-cell-ratio",
        type=float,
        help="Fail when any steady-state timed cell exceeds this torch_rs/PyTorch ratio.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    output = _output_path(args.output) if args.output is not None else None
    report = run_benchmark(args)
    _threshold_gate(report, args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(encoded)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
