#!/usr/bin/env python3
"""Run generated-shape validation for the creation factory benchmark.

The fixed release-timing matrix is useful for repeatable local evidence, but
merge decisions need a path that can receive a held-out seed from the reviewer
or evaluation runner. This script generates CPU float32 factory workloads from
that seed, excludes the public fixed matrix shapes, and reuses the benchmark
driver's symmetric timing and validation logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import secrets
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_creation_factories.py"
VALIDATOR_VERSION = "creation_factories_generated_shape_validator_v1"
DIMENSION_CHOICES = (0, 1, 2, 3, 4, 5, 7, 9, 16, 31, 33, 63, 65, 127, 257)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "_torch_rs_creation_factory_benchmark",
        BENCHMARK_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load benchmark script from {BENCHMARK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_creation_factories = _load_benchmark_module()


def _product(shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _shape_label(shape):
    if not shape:
        return "scalar"
    return "x".join(str(dimension) for dimension in shape)


def _candidate_shape(rng):
    rank = rng.choice((1, 2, 3, 4))
    force_zero = rng.random() < 0.3
    zero_axis = rng.randrange(rank) if force_zero else None
    dimensions = []
    for axis in range(rank):
        if axis == zero_axis:
            dimensions.append(0)
        else:
            dimensions.append(rng.choice(DIMENSION_CHOICES[1:]))
    return tuple(dimensions)


def generate_shapes(seed, count, max_elements):
    fixed_public_shapes = {
        tuple(shape)
        for _, shape, _, _ in benchmark_creation_factories.SHAPE_CASES
    }
    rng = random.Random(seed)
    shapes = []
    seen = set(fixed_public_shapes)
    attempts = 0
    while len(shapes) < count and attempts < count * 200:
        attempts += 1
        shape = _candidate_shape(rng)
        if shape in seen:
            continue
        if _product(shape) > max_elements:
            continue
        seen.add(shape)
        shapes.append(shape)
    if len(shapes) < count:
        raise SystemExit(
            f"could only generate {len(shapes)} held-out shapes below "
            f"max-elements={max_elements}; requested {count}"
        )
    return tuple(shapes)


def _repeats_for_shape(shape):
    elements = _product(shape)
    if elements <= 1:
        return 1024
    if elements <= 1024:
        return 256
    if elements <= 65536:
        return 32
    return 4


def _workloads_for_shapes(shapes):
    workloads = []
    for shape_index, shape in enumerate(shapes):
        label = f"generated_{shape_index:02d}_{_shape_label(shape)}"
        for api in benchmark_creation_factories.FACTORY_APIS:
            workloads.append(
                benchmark_creation_factories.Workload(
                    name=f"{api}_{label}",
                    api=api,
                    shape_label=label,
                    shape=shape,
                    repeats=_repeats_for_shape(shape),
                    description=(
                        f"held-out generated {api} factory shape {shape!r}"
                    ),
                    generated=True,
                )
            )
    return tuple(workloads)


def _positive_int(value, name):
    if value <= 0:
        raise SystemExit(f"{name} must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shape-count", type=int, default=8)
    parser.add_argument("--max-elements", type=int, default=262_144)
    parser.add_argument(
        "--warmups",
        type=int,
        default=benchmark_creation_factories.DEFAULT_WARMUPS,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=benchmark_creation_factories.DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=benchmark_creation_factories.DEFAULT_THREADS,
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    _positive_int(args.samples, "--samples")
    _positive_int(args.threads, "--threads")
    _positive_int(args.shape_count, "--shape-count")
    _positive_int(args.max_elements, "--max-elements")

    seed = args.seed if args.seed is not None else secrets.randbits(64)
    shapes = generate_shapes(seed, args.shape_count, args.max_elements)
    workloads = _workloads_for_shapes(shapes)
    args.workloads = ()

    validator_context = {
        "validator_version": VALIDATOR_VERSION,
        "independent_validator_path": str(
            Path("scripts") / "validate_creation_factory_benchmark.py"
        ),
        "seed": seed,
        "seed_source": "cli" if args.seed is not None else "secrets.randbits(64)",
        "shape_count": args.shape_count,
        "max_elements": args.max_elements,
        "generated_shapes": [list(shape) for shape in shapes],
        "fixed_public_shapes_excluded": True,
        "merge_decision_use": (
            "run with an evaluator-supplied held-out seed; do not use the "
            "fixed public matrix alone for merge scoring"
        ),
    }
    report = benchmark_creation_factories.run_benchmark(
        args,
        workloads=workloads,
        workload_set="generated_heldout_validator",
        validator_context=validator_context,
    )
    report["validator"] = validator_context

    encoded = json.dumps(report, indent=2, sort_keys=True)
    output = (
        benchmark_creation_factories._output_path(args.output)
        if args.output is not None
        else None
    )
    if output is None:
        print(encoded)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
