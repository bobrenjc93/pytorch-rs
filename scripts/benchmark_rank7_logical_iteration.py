#!/usr/bin/env python
"""Benchmark non-contiguous logical iteration for rank-5 through rank-7 tensors."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch as reference_torch
import torch_rs


@dataclass(frozen=True)
class Case:
    name: str
    source_shape: tuple[int, ...]
    permutation_before_select: tuple[int, ...]
    select_index: int
    operations: tuple[str, ...]
    repeats: int

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(
            self.source_shape[axis] for axis in self.permutation_before_select[1:]
        )


CASES = (
    Case(
        name="rank5-middle-select-permuted",
        source_shape=(3, 4, 2, 5, 6, 7),
        permutation_before_select=(2, 4, 1, 5, 0, 3),
        select_index=1,
        operations=("contiguous", "negative"),
        repeats=400,
    ),
    Case(
        name="rank6-middle-select-permuted",
        source_shape=(3, 4, 2, 5, 6, 7, 2),
        permutation_before_select=(2, 4, 1, 6, 0, 5, 3),
        select_index=1,
        operations=("contiguous", "negative"),
        repeats=240,
    ),
    Case(
        name="rank7-middle-select-permuted",
        source_shape=(3, 4, 2, 5, 6, 7, 2, 3),
        permutation_before_select=(2, 4, 1, 6, 0, 5, 3, 7),
        select_index=1,
        operations=("contiguous", "negative", "sqrt"),
        repeats=60,
    ),
)


def tensor_for(module, case: Case):
    elements = math.prod(case.source_shape)
    values = np.linspace(1.0, 2.0, num=elements, dtype=np.float32).reshape(
        case.source_shape
    )
    return module.tensor(values.tolist(), dtype=module.float32).permute(
        *case.permutation_before_select
    )[case.select_index]


def call_operation(tensor, operation: str):
    if operation == "contiguous":
        return tensor.contiguous()
    if operation == "negative":
        return tensor.negative()
    if operation == "sqrt":
        return tensor.sqrt()
    raise AssertionError(f"unknown operation: {operation}")


def consume(tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    flattened = tensor.reshape(-1)
    return float(flattened[0].item()) + float(flattened[-1].item())


def measure(function: Callable[[], object], *, warmups: int, samples: int, repeats: int):
    for _ in range(warmups):
        consume(function())

    timings = []
    checksum = 0.0
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(samples):
            started = time.perf_counter_ns()
            result = None
            for _ in range(repeats):
                result = function()
            elapsed = time.perf_counter_ns() - started
            timings.append(elapsed / repeats)
            checksum += consume(result)
    finally:
        if was_enabled:
            gc.enable()

    median = statistics.median(timings)
    deviations = [abs(sample - median) for sample in timings]
    mad = statistics.median(deviations)
    return {
        "median_us": median / 1_000.0,
        "mad_pct": 0.0 if median == 0.0 else 100.0 * mad / median,
        "min_us": min(timings) / 1_000.0,
        "max_us": max(timings) / 1_000.0,
        "checksum": checksum,
    }


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def cpu_affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--profile-label", default="release")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args()

    if reference_torch.__version__.split("+")[0] != "2.13.0":
        raise SystemExit(
            "rank-7 logical iteration benchmark requires pinned PyTorch 2.13.0"
        )
    if hasattr(reference_torch, "set_num_threads"):
        reference_torch.set_num_threads(1)

    metadata = {
        "profile": args.profile_label,
        "python": sys.version.split()[0],
        "torch_rs": getattr(torch_rs, "__version__", "unknown"),
        "pytorch": reference_torch.__version__,
        "rustc": command_output(["rustc", "--version"]),
        "cargo": command_output(["cargo", "--version"]),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": cpu_affinity(),
        "device": "cpu",
        "dtype": "float32",
        "env_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "pytorch_threads": reference_torch.get_num_threads(),
        "warmups": args.warmups,
        "samples": args.samples,
    }

    rows = []
    for case in CASES:
        tensors = {
            "torch_rs": tensor_for(torch_rs, case),
            "torch": tensor_for(reference_torch, case),
        }
        for operation in case.operations:
            measurements = {}
            for implementation, tensor in tensors.items():
                measurements[implementation] = measure(
                    lambda tensor=tensor, operation=operation: call_operation(
                        tensor, operation
                    ),
                    warmups=args.warmups,
                    samples=args.samples,
                    repeats=case.repeats,
                )
            rows.append(
                {
                    "workload": case.name,
                    "rank": len(case.output_shape),
                    "source_shape": case.source_shape,
                    "output_shape": case.output_shape,
                    "permutation_before_select": case.permutation_before_select,
                    "select_index": case.select_index,
                    "operation": operation,
                    "elements": math.prod(case.output_shape),
                    "repeats": case.repeats,
                    "torch_rs": measurements["torch_rs"],
                    "torch": measurements["torch"],
                    "torch_rs_over_torch": measurements["torch_rs"]["median_us"]
                    / measurements["torch"]["median_us"],
                }
            )

    report = {"metadata": metadata, "results": rows}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print("# Rank-7 Logical Iteration Benchmark")
    print()
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print()
    print(
        "| workload | op | elems | repeats | torch_rs median us | torch median us | rs/torch | rs MAD % | torch MAD % |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {workload} | {operation} | {elements} | {repeats} | "
            "{rs:.3f} | {torch:.3f} | {ratio:.3f} | {rs_mad:.2f} | {torch_mad:.2f} |".format(
                workload=row["workload"],
                operation=row["operation"],
                elements=row["elements"],
                repeats=row["repeats"],
                rs=row["torch_rs"]["median_us"],
                torch=row["torch"]["median_us"],
                ratio=row["torch_rs_over_torch"],
                rs_mad=row["torch_rs"]["mad_pct"],
                torch_mad=row["torch"]["mad_pct"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
