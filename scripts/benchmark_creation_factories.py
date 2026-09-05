#!/usr/bin/env python3
"""Benchmark supported eager CPU tensor creation factories against PyTorch."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
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
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "creation_factories_cpu_eager_benchmark_v1"
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
FACTORY_APIS = ("empty", "zeros", "ones")
SHAPE_CASES = (
    ("scalar", (), 4096, "rank-0 scalar"),
    ("empty", (2, 0, 3), 4096, "rank-3 zero-element shape"),
    ("small", (32, 32), 512, "small row-major rank-2 shape"),
    ("large", (1024, 1024), 8, "large row-major rank-2 shape"),
)
PROTECTED_OUTPUT_PATHS = {
    REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
    REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
}


@dataclass(frozen=True)
class Workload:
    name: str
    api: str
    shape_label: str
    shape: tuple[int, ...]
    repeats: int
    description: str


WORKLOADS = tuple(
    Workload(
        name=f"{api}_{shape_label}",
        api=api,
        shape_label=shape_label,
        shape=shape,
        repeats=repeats,
        description=f"{api} factory for {description}",
    )
    for api in FACTORY_APIS
    for shape_label, shape, repeats, description in SHAPE_CASES
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
    if requested_cpu is None:
        return {
            "requested_cpu": None,
            "selected_cpu": None,
            "initial_affinity": original_affinity,
            "pinned_affinity": original_affinity,
        }
    if original_affinity is None:
        raise SystemExit("CPU affinity pinning requires os.sched_setaffinity")
    if requested_cpu not in original_affinity:
        raise SystemExit(
            f"requested CPU {requested_cpu} is outside the initial affinity "
            f"{original_affinity!r}"
        )
    os.sched_setaffinity(0, {requested_cpu})
    pinned_affinity = _affinity()
    if pinned_affinity != [requested_cpu]:
        raise SystemExit(
            f"failed to pin benchmark to CPU {requested_cpu}: "
            f"affinity is {pinned_affinity!r}"
        )
    return {
        "requested_cpu": requested_cpu,
        "selected_cpu": requested_cpu,
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
    import torch as reference_torch
    import torch_rs

    return torch_rs, reference_torch


def _configure_reference_threads(reference_torch, threads):
    reference_torch.set_num_threads(threads)
    reference_torch.set_num_interop_threads(threads)


def _validate_reference_version(reference_torch):
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise SystemExit(
            "creation factory benchmark requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )


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


def _synchronize(module):
    cuda = getattr(module, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    synchronize = getattr(cuda, "synchronize", None)
    if is_available is not None and synchronize is not None and is_available():
        synchronize()


def _expected_numel(shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _tensor_metadata(tensor):
    return {
        "shape": list(tuple(tensor.shape)),
        "stride": list(tuple(tensor.stride())),
        "storage_offset": int(tensor.storage_offset()),
        "numel": int(tensor.numel()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "requires_grad": bool(tensor.requires_grad),
        "is_leaf": bool(tensor.is_leaf),
        "is_contiguous": bool(tensor.is_contiguous()),
        "data_ptr_nonzero": bool(tensor.data_ptr()),
    }


def _comparable_metadata(metadata):
    return {key: value for key, value in metadata.items() if key != "data_ptr_nonzero"}


def _assert_metadata_matches(actual, expected, *, workload, implementation):
    if _comparable_metadata(actual) != _comparable_metadata(expected):
        raise AssertionError(
            f"{workload.name}/{implementation} metadata mismatch:\n"
            f"actual={actual!r}\nexpected={expected!r}"
        )
    expected_shape = list(workload.shape)
    if actual["shape"] != expected_shape:
        raise AssertionError(
            f"{workload.name}/{implementation} shape changed: {actual!r}"
        )
    if actual["numel"] != _expected_numel(workload.shape):
        raise AssertionError(
            f"{workload.name}/{implementation} numel changed: {actual!r}"
        )


def _create_tensor(module, workload):
    factory = getattr(module, workload.api)
    return factory(workload.shape, dtype=module.float32, device="cpu")


def _validate_factory_values(tensor, workload, implementation):
    if workload.api == "empty":
        return None
    if workload.shape:
        expected_value = (
            0.0 if workload.api == "zeros" else float(_expected_numel(workload.shape))
        )
    else:
        expected_value = 0.0 if workload.api == "zeros" else 1.0
    actual_value = float(tensor.sum().item())
    if actual_value != expected_value:
        raise AssertionError(
            f"{workload.name}/{implementation} value check failed: "
            f"actual={actual_value!r} expected={expected_value!r}"
        )
    return actual_value


def _metadata_sink(metadata):
    sink = int(metadata["storage_offset"])
    sink ^= int(metadata["numel"]) << 1
    sink ^= int(metadata["requires_grad"]) << 2
    sink ^= int(metadata["is_leaf"]) << 3
    sink ^= int(metadata["is_contiguous"]) << 4
    sink ^= int(metadata["data_ptr_nonzero"]) << 5
    for index, dimension in enumerate(metadata["shape"]):
        sink ^= (index + 11) * int(dimension)
    for index, stride in enumerate(metadata["stride"]):
        sink ^= (index + 17) * int(stride)
    return sink


def _time_block(module, workload, repeats):
    started_ns = time.perf_counter_ns()
    output = None
    sink = 0
    metadata = None
    for _ in range(repeats):
        output = _create_tensor(module, workload)
        metadata = _tensor_metadata(output)
        sink ^= _metadata_sink(metadata)
        _synchronize(module)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, metadata, sink, output


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


def _measure_one_pass(module, implementation, workload, reference_metadata, args):
    cold_ns, cold_metadata, cold_sink, cold_output = _time_block(module, workload, 1)
    _assert_metadata_matches(
        cold_metadata,
        reference_metadata,
        workload=workload,
        implementation=implementation,
    )
    value_check = _validate_factory_values(cold_output, workload, implementation)

    for _ in range(args.warmups):
        _time_block(module, workload, workload.repeats)

    sample_ns = []
    sample_sinks = []
    for _ in range(args.samples):
        elapsed_ns, metadata, sink, _ = _time_block(module, workload, workload.repeats)
        _assert_metadata_matches(
            metadata,
            reference_metadata,
            workload=workload,
            implementation=implementation,
        )
        sample_ns.append(elapsed_ns)
        sample_sinks.append(sink)

    return {
        "cold_first_call_us": cold_ns / 1000.0,
        "cold_metadata": cold_metadata,
        "cold_metadata_sink": cold_sink,
        "steady": _summarize_samples(sample_ns, workload.repeats),
        "steady_metadata_sinks": sorted(set(sample_sinks)),
        "value_check_sum": value_check,
    }


def _geomean(values):
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


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
    if (
        resolved == REPOSITORY_ROOT / ".burner"
        or (REPOSITORY_ROOT / ".burner") in resolved.parents
        or resolved in PROTECTED_OUTPUT_PATHS
    ):
        raise SystemExit(f"refusing to write Burner-managed output path: {resolved}")
    return resolved


def _environment(torch_rs, reference_torch, affinity, args):
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
        "private_cuda_roundtrip_included": False,
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
            "extension_path": getattr(getattr(torch_rs, "_C", None), "__file__", None),
            "threads": torch_rs.get_num_threads(),
            "interop_threads": torch_rs.get_num_interop_threads(),
            "empty_storage_initialization": (
                "zero_initialized_cpu_float32_storage_current_implementation_detail"
            ),
        },
        "rust": {
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "git": _git_provenance(),
        "warmups": args.warmups,
        "samples": args.samples,
        "threads": args.threads,
        "implementation_orders": IMPLEMENTATION_ORDERS,
        "factory_apis": list(FACTORY_APIS),
        "shape_cases": [
            {
                "label": label,
                "shape": list(shape),
                "repeats": repeats,
                "description": description,
            }
            for label, shape, repeats, description in SHAPE_CASES
        ],
    }


def run_benchmark(args):
    affinity = _pin_cpu(args.cpu)
    _configure_thread_environment(args.threads, args.cuda_visible_devices)
    torch_rs, reference_torch = _import_backends()
    _validate_reference_version(reference_torch)
    _configure_reference_threads(reference_torch, args.threads)
    _validate_thread_configuration(torch_rs, reference_torch, args.threads)

    workloads = _select_workloads(args.workloads)
    cases = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for workload in workloads:
            reference_tensor = _create_tensor(reference_torch, workload)
            reference_metadata = _tensor_metadata(reference_tensor)
            _validate_factory_values(reference_tensor, workload, "pytorch")
            torch_rs_tensor = _create_tensor(torch_rs, workload)
            torch_rs_metadata = _tensor_metadata(torch_rs_tensor)
            _assert_metadata_matches(
                torch_rs_metadata,
                reference_metadata,
                workload=workload,
                implementation="torch_rs",
            )
            _validate_factory_values(torch_rs_tensor, workload, "torch_rs")

            pass_results = {"torch_rs": [], "pytorch": []}
            for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
                for implementation in order:
                    module = torch_rs if implementation == "torch_rs" else reference_torch
                    measured = _measure_one_pass(
                        module,
                        implementation,
                        workload,
                        reference_metadata,
                        args,
                    )
                    measured.update(
                        {
                            "order_index": order_index,
                            "order": list(order),
                        }
                    )
                    pass_results[implementation].append(measured)

            implementations = {}
            for implementation, passes in pass_results.items():
                medians = [item["steady"]["median_us"] for item in passes]
                mads = [item["steady"]["mad_us"] for item in passes]
                variances = [item["steady"]["variance_us2"] for item in passes]
                metadata_sinks = sorted(
                    {
                        sink
                        for item in passes
                        for sink in (
                            item["steady_metadata_sinks"]
                            + [item["cold_metadata_sink"]]
                        )
                    }
                )
                implementations[implementation] = {
                    "steady_median_us": statistics.median(medians),
                    "steady_mad_us": statistics.median(mads),
                    "steady_variance_us2": statistics.median(variances),
                    "steady_sample_count": sum(
                        item["steady"]["sample_count"] for item in passes
                    ),
                    "metadata_sinks": metadata_sinks,
                    "passes": passes,
                }

            torch_rs_median = implementations["torch_rs"]["steady_median_us"]
            pytorch_median = implementations["pytorch"]["steady_median_us"]
            cases.append(
                {
                    "name": workload.name,
                    "api": workload.api,
                    "shape_label": workload.shape_label,
                    "description": workload.description,
                    "shape": list(workload.shape),
                    "repeats": workload.repeats,
                    "metadata": reference_metadata,
                    "implementations": implementations,
                    "ratios": {
                        "steady_torch_rs_over_pytorch": torch_rs_median
                        / pytorch_median,
                    },
                    "validation": {
                        "metadata_checked": True,
                        "metadata_materialized_inside_timed_loop": True,
                        "filled_values_checked_for_zeros_and_ones": True,
                        "empty_values_unchecked_unspecified": workload.api == "empty",
                        "private_cuda_roundtrip_excluded": True,
                    },
                }
            )
    finally:
        if gc_was_enabled:
            gc.enable()

    ratios = [case["ratios"]["steady_torch_rs_over_pytorch"] for case in cases]
    ratios_by_api = defaultdict(list)
    ratios_by_shape = defaultdict(list)
    for case in cases:
        ratio = case["ratios"]["steady_torch_rs_over_pytorch"]
        ratios_by_api[case["api"]].append(ratio)
        ratios_by_shape[case["shape_label"]].append(ratio)

    return {
        "environment": _environment(torch_rs, reference_torch, affinity, args),
        "cases": cases,
        "aggregates": {
            "timed_cell_count": len(cases),
            "steady_geomean_torch_rs_over_pytorch": _geomean(ratios),
            "steady_geomean_capped_0_10_10_0": _geomean(
                [min(10.0, max(0.10, ratio)) for ratio in ratios]
            ),
            "steady_geomean_by_api": {
                api: _geomean(ratios_by_api[api]) for api in FACTORY_APIS
            },
            "steady_geomean_by_shape": {
                label: _geomean(ratios_by_shape[label])
                for label, _, _, _ in SHAPE_CASES
            },
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--workloads", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")

    output = _output_path(args.output) if args.output is not None else None
    report = run_benchmark(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if output is None:
        print(encoded)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
