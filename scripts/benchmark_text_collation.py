#!/usr/bin/env python3
"""Paired public default_collate timings on precreated text metadata batches.

Run with the local built package (PYTHONPATH=python for a source-copy build).
No score, timing threshold, or evaluator definitions are changed by this tool.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

import torch
import torch_rs

from evaluate_cuda_math import source_provenance

ROOT = Path(__file__).resolve().parents[1]
Record = namedtuple("Record", "metadata")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(value):
    if isinstance(value, dict):
        children = [(key, snapshot(child)) for key, child in value.items()]
    elif isinstance(value, (list, tuple)):
        children = [snapshot(child) for child in value]
    else:
        children = None
    return type(value).__name__, id(value), children


def check(result, batch, values, kind):
    if kind.startswith("direct"):
        assert result is batch
        leaves = result
    elif kind == "dict":
        assert type(result) is dict and list(result) == ["metadata"]
        leaves = result["metadata"]
        assert type(leaves) is list
    else:
        assert type(result) is (Record if kind == "namedtuple" else list)
        leaves = result[0]
        assert type(leaves) is tuple
    assert len(leaves) == len(values)
    assert all(a is b for a, b in zip(leaves, values, strict=True))


def measure(function, batch, iterations):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        result = function(batch)
    elapsed = time.perf_counter_ns() - start
    return elapsed / iterations, result


def summary(samples):
    median = statistics.median(samples)
    return {"median_ns": median,
            "mad_ns": statistics.median(abs(x - median) for x in samples)}


def geomean(values):
    return math.exp(statistics.mean(math.log(x) for x in values))


def main():
    os.environ["GIT_OPTIONAL_LOCKS"] = "0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 8, 64, 1024, 8192])
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 2 or any(n < 1 for n in args.sizes):
        parser.error("use an even sample count >= 4 and positive sizes")
    assert torch.__version__.split("+")[0] == "2.13.0"
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    before = source_provenance()
    harness_hash = digest(__file__)
    started = datetime.now(timezone.utc).isoformat()
    rng = random.Random(args.seed)
    cases = [(size, text_type, container, kind) for size in args.sizes
             for text_type in (str, bytes) for container in (list, tuple)
             for kind in ("direct", "direct_mixed", "dict", "sequence", "namedtuple")]
    rng.shuffle(cases)
    functions = {"candidate": torch_rs.utils.data.default_collate,
                 "reference": torch.utils.data.default_collate}
    rows = []
    for size, text_type, container, kind in cases:
        values = [f"雪/image-{i % 17}.png" for i in range(size)]
        if text_type is bytes:
            values = [v.encode() for v in values]
        if kind == "direct_mixed":
            values[1:] = [None if i % 2 else i for i in range(1, size)]
        wrap = {"dict": lambda v: {"metadata": v}, "sequence": lambda v: [v],
                "namedtuple": Record}.get(kind, lambda v: v)
        batch = container(wrap(v) for v in values)
        original = snapshot(batch)
        iterations = 20000 if kind.startswith("direct") else max(8, 20000 // size)
        # Equal public calls, inputs, warmup counts, samples and iteration counts.
        # Exactly half the sample pairs execute each implementation first.
        firsts = ["candidate", "reference"] * (args.samples // 2)
        rng.shuffle(firsts)
        warmup_order = list(functions)
        rng.shuffle(warmup_order)
        for name in warmup_order:
            _, result = measure(functions[name], batch, iterations)
            check(result, batch, values, kind)
        raw = {name: [] for name in functions}
        for first in firsts:
            for name in (first, "reference" if first == "candidate" else "candidate"):
                elapsed, result = measure(functions[name], batch, iterations)
                raw[name].append(elapsed)
                check(result, batch, values, kind)
        assert snapshot(batch) == original
        stats = {name: summary(samples) for name, samples in raw.items()}
        rows.append({"size": size, "text": text_type.__name__,
                     "container": container.__name__, "kind": kind,
                     "iterations_per_sample": iterations,
                     "warmup_iterations_each": iterations, "warmup_order": warmup_order,
                     "first_in_pair": firsts, "raw_ns_per_call": raw, "summary": stats,
                     "reference_over_candidate": stats["reference"]["median_ns"] / stats["candidate"]["median_ns"],
                     "identity_order_container_inputs_checked": True})
    assert before == source_provenance() and harness_hash == digest(__file__)
    report = {"schema_version": 1, "started_at": started,
              "ended_at": datetime.now(timezone.utc).isoformat(),
              "source": before,
              "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
              "worktree": str(ROOT), "command": [sys.executable, *sys.argv],
              "python": sys.version, "executable": str(Path(sys.executable).resolve()),
              "platform": platform.platform(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "cpu": subprocess.check_output(["lscpu"], text=True),
              "reference_version": torch.__version__, "reference_package": torch.__file__,
              "candidate_package": torch_rs.__file__,
              "extension": {"path": torch_rs.torch_rs.__file__, "sha256": digest(torch_rs.torch_rs.__file__)},
              "harness_sha256": harness_hash, "seed": args.seed, "samples": args.samples,
              "method": "Precreated inputs; paired public calls; equal warmup/iterations; balanced randomized first order; per-call nanoseconds; semantic checks outside timing.",
              "cases": rows,
              "geomean_reference_over_candidate": {
                  **{kind: geomean(r["reference_over_candidate"] for r in rows if r["kind"] == kind)
                     for kind in sorted({r["kind"] for r in rows})},
                  "all": geomean(r["reference_over_candidate"] for r in rows)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["geomean_reference_over_candidate"], indent=2))


if __name__ == "__main__":
    main()
