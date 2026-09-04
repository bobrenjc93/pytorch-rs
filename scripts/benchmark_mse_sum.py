#!/usr/bin/env python3
"""Benchmark ``torch.nn.functional.mse_loss(..., reduction="sum")`` on CPU."""

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
BENCHMARK_VERSION = "mse_sum_release_benchmark_v1"
DEFAULT_CPU = 24
DEFAULT_THREADS = 1
DEFAULT_WARMUPS = 15
DEFAULT_SAMPLES = 81
IMPLEMENTATION_ORDERS = (
    ("torch_rs", "pytorch"),
    ("pytorch", "torch_rs"),
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class Workload:
    name: str
    category: str
    description: str
    shape: tuple[int, ...]
    repeats: int
    seed: int
    rtol: float
    atol: float


WORKLOADS = (
    Workload(
        name="mse_sum_same_contiguous_1024x1024",
        category="same-shape contiguous no-grad sum",
        description="CPU float32 row-major contiguous (1024, 1024) inputs",
        shape=(1024, 1024),
        repeats=16,
        seed=20260903,
        rtol=1.0e-4,
        atol=1.0e-4,
    ),
)


def _run_text(command: list[str]) -> str | None:
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


def _version_without_local(version: str) -> str:
    return version.split("+", 1)[0]


def _affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def _pin_cpu(requested_cpu: int | None) -> dict[str, object]:
    before = _affinity()
    if before is None:
        raise SystemExit("this benchmark requires os.sched_setaffinity support")
    if not before:
        raise SystemExit("process has no schedulable CPUs")

    selected_cpu = requested_cpu
    if selected_cpu is None:
        selected_cpu = DEFAULT_CPU if DEFAULT_CPU in before else before[0]
    if selected_cpu not in before:
        raise SystemExit(
            f"requested CPU {selected_cpu} is outside the current affinity mask {before}"
        )

    os.sched_setaffinity(0, {selected_cpu})
    after = _affinity()
    if after != [selected_cpu]:
        raise SystemExit(f"failed to pin CPU affinity to {selected_cpu}: {after}")

    return {
        "requested_cpu": requested_cpu,
        "selected_cpu": selected_cpu,
        "before": before,
        "after": after,
    }


def _configure_thread_environment(threads: int, cuda_visible_devices: str) -> None:
    thread_value = str(threads)
    for variable in THREAD_ENVIRONMENT:
        os.environ[variable] = thread_value
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices


def _shell_command(arguments: list[str]) -> str:
    return " ".join(shlex.quote(argument) for argument in arguments)


def _cpu_model_name() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _package_version(distribution_name: str, module: object) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _load_modules(args: argparse.Namespace) -> dict[str, object]:
    import numpy as np
    import torch as reference_torch
    import torch.nn.functional as reference_functional
    import torch_rs
    import torch_rs.nn.functional as torch_rs_functional

    if _version_without_local(reference_torch.__version__) != args.reference_version:
        raise SystemExit(
            f"expected PyTorch {args.reference_version}, got {reference_torch.__version__}"
        )

    reference_torch.set_num_threads(args.threads)
    reference_torch.set_num_interop_threads(args.threads)

    return {
        "np": np,
        "torch_rs": torch_rs,
        "torch_rs_functional": torch_rs_functional,
        "pytorch": reference_torch,
        "pytorch_functional": reference_functional,
    }


def _environment(
    *,
    args: argparse.Namespace,
    modules: dict[str, object],
    affinity_state: dict[str, object],
) -> dict[str, object]:
    torch_rs = modules["torch_rs"]
    reference_torch = modules["pytorch"]
    np = modules["np"]
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "commands": {
            "argv": [sys.executable, *sys.argv],
            "shell": _shell_command([sys.executable, *sys.argv]),
            "cwd": str(Path.cwd()),
        },
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "libc": " ".join(platform.libc_ver()).strip(),
        "cpu": _cpu_model_name(),
        "lscpu": _run_text(["lscpu"]),
        "process_affinity": affinity_state,
        "env_threads": {
            variable: os.environ.get(variable) for variable in THREAD_ENVIRONMENT
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
        "numpy": np.__version__,
        "rust": {
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "maturin": _run_text(["maturin", "--version"]),
        "git": {
            "head": _run_text(["git", "rev-parse", "HEAD"]),
            "status_short": _run_text(["git", "status", "--short"]),
            "diff_stat": _run_text(["git", "diff", "HEAD", "--stat"]),
        },
        "warmups": args.warmups,
        "samples": args.samples,
        "threads": args.threads,
        "implementation_orders": IMPLEMENTATION_ORDERS,
    }


def _numel(shape: tuple[int, ...]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _workload_arrays(np: object, workload: Workload) -> tuple[object, object]:
    rng = np.random.default_rng(workload.seed)
    elements = _numel(workload.shape)
    left = rng.uniform(-3.0, 5.0, size=elements).astype(np.float32)
    right = rng.uniform(-4.0, 2.0, size=elements).astype(np.float32)
    return left, right


def _make_tensor(module: object, flat_values: object, shape: tuple[int, ...]) -> object:
    tensor = module.tensor(memoryview(flat_values), dtype=module.float32)
    if shape:
        return tensor.view(*shape)
    return tensor.view(())


def _make_inputs(
    module: object,
    arrays: tuple[object, object],
    workload: Workload,
) -> tuple[object, object]:
    left, right = arrays
    return (
        _make_tensor(module, left, workload.shape),
        _make_tensor(module, right, workload.shape),
    )


def _synchronize(module: object) -> None:
    cuda = getattr(module, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    synchronize = getattr(cuda, "synchronize", None)
    if is_available is not None and synchronize is not None and is_available():
        synchronize()


def _tensor_array(np: object, tensor: object) -> object:
    return np.asarray(tensor)


def _tensor_metadata(tensor: object) -> dict[str, object]:
    return {
        "shape": list(tuple(tensor.shape)),
        "stride": list(tuple(tensor.stride())),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
        "is_leaf": bool(tensor.is_leaf),
        "is_contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
    }


def _tensor_payload(np: object, tensor: object) -> dict[str, object]:
    array = _tensor_array(np, tensor)
    flat = array.reshape(-1)
    return {
        "metadata": _tensor_metadata(tensor),
        "value_bits": flat.view(np.uint32).tolist(),
    }


def _encode_payload(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _checksum_payload(payload: dict[str, object]) -> str:
    return hashlib.blake2b(_encode_payload(payload), digest_size=8).hexdigest()


def _assert_payloads_match(
    np: object,
    actual_payload: dict[str, object],
    expected_payload: dict[str, object],
    *,
    workload: Workload,
    implementation: str,
) -> None:
    actual_metadata = actual_payload["metadata"]
    expected_metadata = expected_payload["metadata"]
    if actual_metadata != expected_metadata:
        raise AssertionError(
            f"{workload.name}/{implementation} metadata mismatch:\n"
            f"actual={actual_metadata!r}\nexpected={expected_metadata!r}"
        )

    actual_bits = np.asarray(actual_payload["value_bits"], dtype=np.uint32)
    expected_bits = np.asarray(expected_payload["value_bits"], dtype=np.uint32)
    actual_values = actual_bits.view(np.float32)
    expected_values = expected_bits.view(np.float32)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=workload.rtol,
        atol=workload.atol,
        equal_nan=True,
    ):
        raise AssertionError(
            f"{workload.name}/{implementation} value mismatch:\n"
            f"actual_bits={actual_payload['value_bits']!r}\n"
            f"expected_bits={expected_payload['value_bits']!r}"
        )


def _assert_inputs_unchanged(
    before: list[dict[str, object]],
    after: list[dict[str, object]],
    *,
    workload: Workload,
    implementation: str,
) -> None:
    if before != after:
        raise AssertionError(
            f"{workload.name}/{implementation} mutated inputs:\n"
            f"before={before!r}\nafter={after!r}"
        )


def _mse_sum(functional_module: object, inputs: tuple[object, object]) -> object:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = functional_module.mse_loss(inputs[0], inputs[1], reduction="sum")
    if caught:
        rendered = [
            (warning.category.__name__, str(warning.message)) for warning in caught
        ]
        raise AssertionError(f"unexpected warnings from same-shape workload: {rendered!r}")
    return output


def _time_block(
    *,
    np: object,
    module: object,
    functional_module: object,
    inputs: tuple[object, object],
    repeats: int,
) -> dict[str, object]:
    block_digest = hashlib.blake2b(digest_size=8)
    last_payload = None
    started_ns = time.perf_counter_ns()
    for _ in range(repeats):
        output = _mse_sum(functional_module, inputs)
        _synchronize(module)
        payload = _tensor_payload(np, output)
        block_digest.update(_checksum_payload(payload).encode("ascii"))
        last_payload = payload
    elapsed_ns = time.perf_counter_ns() - started_ns
    if last_payload is None:
        raise AssertionError("timed block did not execute")
    return {
        "elapsed_ns": elapsed_ns,
        "per_call_us": elapsed_ns / repeats / 1000.0,
        "block_checksum": block_digest.hexdigest(),
        "last_output_checksum": _checksum_payload(last_payload),
        "last_output": last_payload,
    }


def _summarize_samples(samples: list[dict[str, object]]) -> dict[str, object]:
    per_call_us = [float(sample["per_call_us"]) for sample in samples]
    median_us = statistics.median(per_call_us)
    deviations = [abs(sample - median_us) for sample in per_call_us]
    return {
        "median_us": median_us,
        "mad_us": statistics.median(deviations),
        "variance_us2": statistics.pvariance(per_call_us)
        if len(per_call_us) > 1
        else 0.0,
        "sample_count": len(per_call_us),
        "min_us": min(per_call_us),
        "max_us": max(per_call_us),
        "block_checksums": sorted(
            {str(sample["block_checksum"]) for sample in samples}
        ),
        "last_output_checksums": sorted(
            {str(sample["last_output_checksum"]) for sample in samples}
        ),
    }


def _measure_implementation(
    *,
    np: object,
    module: object,
    functional_module: object,
    implementation: str,
    arrays: tuple[object, object],
    workload: Workload,
    args: argparse.Namespace,
) -> dict[str, object]:
    inputs = _make_inputs(module, arrays, workload)
    input_payloads_before = [_tensor_payload(np, tensor) for tensor in inputs]

    cold = _time_block(
        np=np,
        module=module,
        functional_module=functional_module,
        inputs=inputs,
        repeats=1,
    )
    warmups = [
        _time_block(
            np=np,
            module=module,
            functional_module=functional_module,
            inputs=inputs,
            repeats=workload.repeats,
        )
        for _ in range(args.warmups)
    ]
    samples = [
        _time_block(
            np=np,
            module=module,
            functional_module=functional_module,
            inputs=inputs,
            repeats=workload.repeats,
        )
        for _ in range(args.samples)
    ]

    input_payloads_after = [_tensor_payload(np, tensor) for tensor in inputs]
    _assert_inputs_unchanged(
        input_payloads_before,
        input_payloads_after,
        workload=workload,
        implementation=implementation,
    )

    steady = _summarize_samples(samples)
    last_output_checksums = set(steady["last_output_checksums"])
    last_output_checksums.add(str(cold["last_output_checksum"]))
    last_output_checksums.update(
        str(warmup["last_output_checksum"]) for warmup in warmups
    )
    if len(last_output_checksums) != 1:
        raise AssertionError(
            f"{workload.name}/{implementation} unstable output checksum: "
            f"{sorted(last_output_checksums)!r}"
        )

    block_checksums = set(steady["block_checksums"])
    block_checksums.update(str(warmup["block_checksum"]) for warmup in warmups)
    if len(block_checksums) != 1:
        raise AssertionError(
            f"{workload.name}/{implementation} unstable repeated block checksum: "
            f"{sorted(block_checksums)!r}"
        )

    return {
        "cold": {
            "per_call_us": cold["per_call_us"],
            "last_output_checksum": cold["last_output_checksum"],
            "last_output": cold["last_output"],
        },
        "warmup_count": len(warmups),
        "warmup_block_checksums": sorted(
            {str(warmup["block_checksum"]) for warmup in warmups}
        ),
        "steady": steady,
        "input_metadata": [payload["metadata"] for payload in input_payloads_before],
        "input_checksums": [
            _checksum_payload(payload) for payload in input_payloads_before
        ],
        "output_metadata": cold["last_output"]["metadata"],
    }


def _implementation_modules(
    modules: dict[str, object],
    implementation: str,
) -> tuple[object, object]:
    if implementation == "torch_rs":
        return modules["torch_rs"], modules["torch_rs_functional"]
    if implementation == "pytorch":
        return modules["pytorch"], modules["pytorch_functional"]
    raise AssertionError(f"unknown implementation {implementation!r}")


def _run_workload(
    *,
    modules: dict[str, object],
    workload: Workload,
    args: argparse.Namespace,
) -> dict[str, object]:
    np = modules["np"]
    arrays = _workload_arrays(np, workload)
    expected_inputs = _make_inputs(modules["pytorch"], arrays, workload)
    expected_output = _mse_sum(modules["pytorch_functional"], expected_inputs)
    expected_payload = _tensor_payload(np, expected_output)

    pass_results: dict[str, list[dict[str, object]]] = {
        "torch_rs": [],
        "pytorch": [],
    }
    for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
        for implementation in order:
            module, functional_module = _implementation_modules(modules, implementation)
            measured = _measure_implementation(
                np=np,
                module=module,
                functional_module=functional_module,
                implementation=implementation,
                arrays=arrays,
                workload=workload,
                args=args,
            )
            _assert_payloads_match(
                np,
                measured["cold"]["last_output"],
                expected_payload,
                workload=workload,
                implementation=implementation,
            )
            measured["order_index"] = order_index
            measured["order"] = order
            pass_results[implementation].append(measured)

    summaries = {}
    for implementation, measurements in pass_results.items():
        steady_medians = [
            measurement["steady"]["median_us"] for measurement in measurements
        ]
        steady_mads = [
            measurement["steady"]["mad_us"] for measurement in measurements
        ]
        steady_variances = [
            measurement["steady"]["variance_us2"] for measurement in measurements
        ]
        cold_values = [
            measurement["cold"]["per_call_us"] for measurement in measurements
        ]
        output_checksums = sorted(
            {
                str(measurement["cold"]["last_output_checksum"])
                for measurement in measurements
            }
        )
        block_checksums = sorted(
            {
                checksum
                for measurement in measurements
                for checksum in measurement["steady"]["block_checksums"]
            }
        )
        if len(output_checksums) != 1:
            raise AssertionError(
                f"{workload.name}/{implementation} unstable output checksum "
                f"across implementation-order passes: {output_checksums!r}"
            )
        if len(block_checksums) != 1:
            raise AssertionError(
                f"{workload.name}/{implementation} unstable block checksum "
                f"across implementation-order passes: {block_checksums!r}"
            )
        summaries[implementation] = {
            "cold_median_us": statistics.median(cold_values),
            "cold_values_us": cold_values,
            "steady_median_us": statistics.median(steady_medians),
            "steady_median_values_us": steady_medians,
            "steady_mad_us": statistics.median(steady_mads),
            "steady_variance_us2": statistics.median(steady_variances),
            "steady_sample_count": sum(
                measurement["steady"]["sample_count"] for measurement in measurements
            ),
            "output_checksums": output_checksums,
            "block_checksums": block_checksums,
        }

    torch_rs_steady = summaries["torch_rs"]["steady_median_us"]
    pytorch_steady = summaries["pytorch"]["steady_median_us"]
    torch_rs_cold = summaries["torch_rs"]["cold_median_us"]
    pytorch_cold = summaries["pytorch"]["cold_median_us"]
    return {
        "name": workload.name,
        "category": workload.category,
        "description": workload.description,
        "shape": list(workload.shape),
        "seed": workload.seed,
        "repeats": workload.repeats,
        "rtol": workload.rtol,
        "atol": workload.atol,
        "input_metadata": pass_results["torch_rs"][0]["input_metadata"],
        "input_checksums": pass_results["torch_rs"][0]["input_checksums"],
        "output_metadata": pass_results["torch_rs"][0]["output_metadata"],
        "expected_pytorch_output_checksum": _checksum_payload(expected_payload),
        "implementations": summaries,
        "ratios": {
            "steady_torch_rs_over_pytorch": torch_rs_steady / pytorch_steady,
            "cold_torch_rs_over_pytorch": torch_rs_cold / pytorch_cold,
        },
    }


def _geomean(values: list[float]) -> float | None:
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _select_workloads(selected_names: tuple[str, ...]) -> tuple[Workload, ...]:
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


def _output_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        raise SystemExit(f"output path must stay inside the worktree: {resolved}") from None
    return resolved


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    if args.warmups < 0 or args.samples <= 0 or args.threads <= 0:
        raise SystemExit("--warmups must be nonnegative; --samples and --threads must be positive")

    affinity_state = _pin_cpu(args.cpu)
    _configure_thread_environment(args.threads, args.cuda_visible_devices)
    modules = _load_modules(args)
    workloads = _select_workloads(tuple(args.workloads))

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        rows = [
            _run_workload(modules=modules, workload=workload, args=args)
            for workload in workloads
        ]
    finally:
        if gc_was_enabled:
            gc.enable()

    steady_ratios = [
        row["ratios"]["steady_torch_rs_over_pytorch"] for row in rows
    ]
    cold_ratios = [row["ratios"]["cold_torch_rs_over_pytorch"] for row in rows]
    return {
        "environment": _environment(
            args=args,
            modules=modules,
            affinity_state=affinity_state,
        ),
        "workloads": rows,
        "aggregates": {
            "workload_count": len(rows),
            "steady_geomean_torch_rs_over_pytorch": _geomean(steady_ratios),
            "steady_geomean_capped_0_10_10_0": _geomean(
                [min(10.0, max(0.10, ratio)) for ratio in steady_ratios]
            ),
            "cold_geomean_torch_rs_over_pytorch": _geomean(cold_ratios),
            "cold_geomean_capped_0_10_10_0": _geomean(
                [min(10.0, max(0.10, ratio)) for ratio in cold_ratios]
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--reference-version", default=REFERENCE_PYTORCH_VERSION)
    parser.add_argument("--workloads", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
