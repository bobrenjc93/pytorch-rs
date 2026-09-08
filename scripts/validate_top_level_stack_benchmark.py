#!/usr/bin/env python3
"""Run generated-shape validation for the top-level ``torch.stack`` benchmark.

The fixed release-timing matrix is repeatable public evidence. Merge decisions
also need an evaluator-supplied held-out path that cannot be targeted by an
implementation branch. This script generates CPU float32 same-shape stack
workloads from a seed, excludes the fixed public stack benchmark input shapes,
and reuses the benchmark driver's symmetric timing, checksum, and boundary-row
validation logic.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gc
import importlib
import importlib.machinery
import importlib.util
import json
import math
import random
import secrets
import statistics
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = REPOSITORY_ROOT / "scripts" / "benchmark_top_level_stack.py"
VALIDATOR_VERSION = "top_level_stack_generated_shape_validator_v1"
WORKLOAD_SET = "generated_heldout_validator"
REQUIRED_CATEGORIES = (
    "contiguous",
    "empty",
    "offset",
    "noncontiguous",
    "autograd forward",
    "autograd forward+backward",
)
NONZERO_DIMENSION_CHOICES = (
    1,
    2,
    3,
    4,
    5,
    7,
    9,
    11,
    13,
    16,
    17,
    31,
    33,
    63,
    65,
    127,
    129,
    257,
)
NONUNIT_DIMENSION_CHOICES = tuple(
    dimension for dimension in NONZERO_DIMENSION_CHOICES if dimension != 1
)


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "_torch_rs_top_level_stack_benchmark",
        BENCHMARK_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load benchmark script from {BENCHMARK_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark_top_level_stack = _load_benchmark_module()


@dataclass(frozen=True)
class GeneratedStackCase:
    name: str
    category: str
    layout: str
    mode: str
    shape: tuple[int, ...]
    dim: int
    input_source_indices: tuple[int, ...]
    seeds: tuple[int, ...]
    repeats: int
    source_shape: tuple[int, ...] | None = None
    offset_index: int | None = None


PUBLIC_INPUT_SHAPES = frozenset(
    {
        (),
        (257,),
        (257, 263),
        (2, 0, 3),
        (127, 131),
        (512, 1024),
        (32, 33),
    }
)


def _product(shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _shape_label(shape):
    if not shape:
        return "scalar"
    return "x".join(str(dimension) for dimension in shape)


def _category_slug(category):
    return category.replace("+", "_").replace(" ", "_")


def _dim_label(dim):
    return f"neg{abs(dim)}" if dim < 0 else str(dim)


def _stack_dim(rng, rank):
    return rng.choice(tuple(range(-(rank + 1), rank + 1)))


def _seed_tuple(rng, count):
    return tuple(rng.randrange(1, 2**31 - 1) for _ in range(count))


def _candidate_nonempty_shape(rng, max_elements, *, max_rank=4):
    for _ in range(1000):
        rank = rng.choice(tuple(range(1, max_rank + 1)))
        shape = tuple(rng.choice(NONZERO_DIMENSION_CHOICES) for _ in range(rank))
        if _product(shape) <= max_elements:
            return shape
    raise SystemExit(
        "could not generate a non-empty stack shape below "
        f"max-elements={max_elements}"
    )


def _candidate_empty_shape(rng):
    rank = rng.choice((1, 2, 3, 4))
    zero_axis = rng.randrange(rank)
    dimensions = []
    for axis in range(rank):
        if axis == zero_axis:
            dimensions.append(0)
        else:
            dimensions.append(rng.choice(NONZERO_DIMENSION_CHOICES))
    return tuple(dimensions)


def _case_repeats(shape, input_count, mode):
    elements = _product(shape) * input_count
    if elements == 0:
        base_repeats = 2048
    elif elements <= 128:
        base_repeats = 1024
    elif elements <= 4096:
        base_repeats = 128
    elif elements <= 65536:
        base_repeats = 16
    else:
        base_repeats = 4

    if mode == benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD:
        return max(1, base_repeats // 16)
    if mode == benchmark_top_level_stack.MODE_AUTOGRAD_FORWARD:
        return max(1, base_repeats // 4)
    return base_repeats


def _case_signature(case):
    return (
        case.category,
        case.layout,
        case.mode,
        case.shape,
        case.dim,
        case.input_source_indices,
        case.source_shape,
        case.offset_index,
    )


def _new_case_name(category, ordinal, shape, input_count, dim):
    return (
        f"generated_{_category_slug(category)}_{ordinal:02d}_"
        f"{_shape_label(shape)}_inputs{input_count}_dim{_dim_label(dim)}"
    )


def _contiguous_candidate(rng, max_elements, ordinal):
    input_count = rng.choice((2, 3, 4))
    shape = _candidate_nonempty_shape(rng, max_elements)
    dim = _stack_dim(rng, len(shape))
    mode = benchmark_top_level_stack.MODE_EAGER
    return GeneratedStackCase(
        name=_new_case_name("contiguous", ordinal, shape, input_count, dim),
        category="contiguous",
        layout="contiguous",
        mode=mode,
        shape=shape,
        dim=dim,
        input_source_indices=tuple(range(input_count)),
        seeds=_seed_tuple(rng, input_count),
        repeats=_case_repeats(shape, input_count, mode),
    )


def _empty_candidate(rng, max_elements, ordinal):
    del max_elements
    input_count = rng.choice((2, 3, 4))
    shape = _candidate_empty_shape(rng)
    dim = _stack_dim(rng, len(shape))
    mode = benchmark_top_level_stack.MODE_EAGER
    return GeneratedStackCase(
        name=_new_case_name("empty", ordinal, shape, input_count, dim),
        category="empty",
        layout="empty",
        mode=mode,
        shape=shape,
        dim=dim,
        input_source_indices=tuple(range(input_count)),
        seeds=_seed_tuple(rng, input_count),
        repeats=_case_repeats(shape, input_count, mode),
    )


def _offset_candidate(rng, max_elements, ordinal):
    input_count = rng.choice((2, 3))
    prefix = rng.choice((2, 3, 4))
    logical_limit = max(1, max_elements // (input_count * prefix))
    shape = _candidate_nonempty_shape(rng, logical_limit, max_rank=3)
    dim = _stack_dim(rng, len(shape))
    mode = benchmark_top_level_stack.MODE_EAGER
    offset_index = rng.randrange(1, prefix)
    return GeneratedStackCase(
        name=_new_case_name("offset", ordinal, shape, input_count, dim),
        category="offset",
        layout="offset",
        mode=mode,
        shape=shape,
        dim=dim,
        input_source_indices=tuple(range(input_count)),
        seeds=_seed_tuple(rng, input_count),
        repeats=_case_repeats(shape, input_count, mode),
        source_shape=(prefix, *shape),
        offset_index=offset_index,
    )


def _noncontiguous_candidate(rng, max_elements, ordinal):
    input_count = rng.choice((2, 3))
    logical_limit = max(1, max_elements // input_count)
    rows = rng.choice(NONUNIT_DIMENSION_CHOICES)
    cols = rng.choice(NONUNIT_DIMENSION_CHOICES)
    attempts = 0
    while rows * cols > logical_limit and attempts < 1000:
        rows = rng.choice(NONUNIT_DIMENSION_CHOICES)
        cols = rng.choice(NONUNIT_DIMENSION_CHOICES)
        attempts += 1
    if rows * cols > logical_limit:
        raise SystemExit(
            "could not generate a noncontiguous stack shape below "
            f"max-elements={max_elements}"
        )
    shape = (cols, rows)
    dim = _stack_dim(rng, len(shape))
    mode = benchmark_top_level_stack.MODE_EAGER
    return GeneratedStackCase(
        name=_new_case_name("noncontiguous", ordinal, shape, input_count, dim),
        category="noncontiguous",
        layout="noncontiguous",
        mode=mode,
        shape=shape,
        dim=dim,
        input_source_indices=tuple(range(input_count)),
        seeds=_seed_tuple(rng, input_count),
        repeats=_case_repeats(shape, input_count, mode),
        source_shape=(rows, cols),
    )


def _autograd_forward_candidate(rng, max_elements, ordinal):
    input_count = rng.choice((2, 3))
    logical_limit = max(1, min(max_elements // input_count, 65536))
    shape = _candidate_nonempty_shape(rng, logical_limit, max_rank=3)
    dim = _stack_dim(rng, len(shape))
    mode = benchmark_top_level_stack.MODE_AUTOGRAD_FORWARD
    return GeneratedStackCase(
        name=_new_case_name("autograd forward", ordinal, shape, input_count, dim),
        category="autograd forward",
        layout="contiguous",
        mode=mode,
        shape=shape,
        dim=dim,
        input_source_indices=tuple(range(input_count)),
        seeds=_seed_tuple(rng, input_count),
        repeats=_case_repeats(shape, input_count, mode),
    )


def _autograd_backward_candidate(rng, max_elements, ordinal):
    logical_limit = max(1, min(max_elements // 3, 16384))
    shape = _candidate_nonempty_shape(rng, logical_limit, max_rank=3)
    dim = _stack_dim(rng, len(shape))
    mode = benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD
    input_count = 3
    return GeneratedStackCase(
        name=_new_case_name(
            "autograd forward+backward",
            ordinal,
            shape,
            input_count,
            dim,
        ),
        category="autograd forward+backward",
        layout="contiguous",
        mode=mode,
        shape=shape,
        dim=dim,
        input_source_indices=(0, 1, 0),
        seeds=_seed_tuple(rng, 2),
        repeats=_case_repeats(shape, input_count, mode),
    )


CATEGORY_GENERATORS = {
    "contiguous": _contiguous_candidate,
    "empty": _empty_candidate,
    "offset": _offset_candidate,
    "noncontiguous": _noncontiguous_candidate,
    "autograd forward": _autograd_forward_candidate,
    "autograd forward+backward": _autograd_backward_candidate,
}


def generate_cases(seed, cases_per_category, max_elements):
    rng = random.Random(seed)
    cases = []
    seen_signatures = set()
    seen_names = set()
    for category in REQUIRED_CATEGORIES:
        generator = CATEGORY_GENERATORS[category]
        for ordinal in range(cases_per_category):
            for _ in range(1000):
                case = generator(rng, max_elements, ordinal)
                if case.shape in PUBLIC_INPUT_SHAPES:
                    continue
                signature = _case_signature(case)
                if signature in seen_signatures or case.name in seen_names:
                    continue
                seen_signatures.add(signature)
                seen_names.add(case.name)
                cases.append(case)
                break
            else:
                raise SystemExit(
                    "could not generate a held-out "
                    f"{category!r} stack case below max-elements={max_elements}"
                )
    return tuple(cases)


def _make_sources(module, np, case):
    sources = []
    for index, seed in enumerate(case.seeds):
        bias = (index - 1) * 0.125
        requires_grad = case.mode in (
            benchmark_top_level_stack.MODE_AUTOGRAD_FORWARD,
            benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD,
        )
        if case.layout == "offset":
            base = benchmark_top_level_stack._dense_tensor(
                module,
                np,
                case.source_shape,
                seed,
                bias=bias,
            )
            sources.append(base[case.offset_index])
        elif case.layout == "noncontiguous":
            base = benchmark_top_level_stack._dense_tensor(
                module,
                np,
                case.source_shape,
                seed,
                bias=bias,
            )
            sources.append(base.transpose(0, 1))
        else:
            sources.append(
                benchmark_top_level_stack._dense_tensor(
                    module,
                    np,
                    case.shape,
                    seed,
                    requires_grad=requires_grad,
                    bias=bias,
                )
            )
    return tuple(sources)


def _make_operands_factory(case):
    def make_operands(module, np):
        sources = _make_sources(module, np, case)
        tensors = [sources[index] for index in case.input_source_indices]
        leaves = ()
        if case.mode == benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD:
            leaves = tuple(
                (f"source_{index}", source) for index, source in enumerate(sources)
            )
        return benchmark_top_level_stack.Operands(tensors, case.dim, leaves)

    return make_operands


def _workloads_for_cases(cases):
    workloads = []
    for case in cases:
        repeated = (
            "; source_0 is stacked twice"
            if len(set(case.input_source_indices)) != len(case.input_source_indices)
            else ""
        )
        source = (
            f" from source shape {case.source_shape!r}"
            if case.source_shape is not None
            else ""
        )
        workloads.append(
            benchmark_top_level_stack.Workload(
                name=case.name,
                category=case.category,
                input_description=(
                    f"held-out generated CPU float32 {case.layout} same-shape "
                    f"stack: {len(case.input_source_indices)} inputs of shape "
                    f"{case.shape!r}, dim={case.dim}{source}{repeated}"
                ),
                output_description=(
                    "stack output plus accumulated leaf gradients"
                    if case.mode == benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD
                    else "stack output"
                ),
                repeats=case.repeats,
                mode=case.mode,
                seeds=case.seeds,
                make_operands=_make_operands_factory(case),
            )
        )
    return tuple(workloads)


def _case_record(case):
    return {
        "name": case.name,
        "category": case.category,
        "layout": case.layout,
        "mode": case.mode,
        "shape": list(case.shape),
        "dim": case.dim,
        "input_count": len(case.input_source_indices),
        "input_source_indices": list(case.input_source_indices),
        "seeds": list(case.seeds),
        "repeats": case.repeats,
        "source_shape": (
            list(case.source_shape) if case.source_shape is not None else None
        ),
        "offset_index": case.offset_index,
    }


def _expected_case_records(seed, cases_per_category, max_elements):
    return [
        _case_record(case)
        for case in generate_cases(seed, cases_per_category, max_elements)
    ]


def _positive_int(value, name):
    if value <= 0:
        raise SystemExit(f"{name} must be positive")
    return value


def _validator_context(args, seed, cases):
    return {
        "validator_version": VALIDATOR_VERSION,
        "independent_validator_path": str(
            Path("scripts") / "validate_top_level_stack_benchmark.py"
        ),
        "independent_validator_sha256": benchmark_top_level_stack._file_sha256(
            Path(__file__).resolve()
        ),
        "benchmark_driver_path": str(
            Path("scripts") / "benchmark_top_level_stack.py"
        ),
        "benchmark_driver_version": benchmark_top_level_stack.BENCHMARK_VERSION,
        "seed": seed,
        "seed_source": "cli" if args.seed is not None else "secrets.randbits(64)",
        "cases_per_category": args.cases_per_category,
        "max_elements": args.max_elements,
        "required_categories": list(REQUIRED_CATEGORIES),
        "generated_cases": [_case_record(case) for case in cases],
        "fixed_public_input_shapes": [
            list(shape) for shape in sorted(PUBLIC_INPUT_SHAPES, key=repr)
        ],
        "fixed_public_matrix_excluded": True,
        "same_shape_cpu_float32_only": True,
        "merge_decision_use": (
            "run with an evaluator-supplied held-out seed; do not use the "
            "fixed public stack matrix alone for merge scoring"
        ),
    }


def _annotate_supported_rows(rows, cases):
    by_name = {case.name: case for case in cases}
    for row in rows:
        case = by_name[row["workload"]]
        row["generated"] = True
        row["shape"] = list(case.shape)
        row["dim"] = case.dim
        row["input_count"] = len(case.input_source_indices)
        row["layout"] = case.layout
        row["validator_case"] = _case_record(case)
        row["validation"]["held_out_generated_shape"] = True
        row["validation"]["fixed_public_matrix_excluded"] = True
        row["validation"]["same_shape_cpu_float32_inputs"] = True


def run_validator(args, cases, workloads, validator_context):
    affinity = benchmark_top_level_stack._pin_cpu(args.cpu)
    benchmark_top_level_stack._configure_thread_environment(
        args.threads,
        args.cuda_visible_devices,
    )
    np, torch_rs, reference_torch = benchmark_top_level_stack._import_backends()
    benchmark_top_level_stack._validate_reference_version(reference_torch)
    benchmark_top_level_stack._configure_reference_threads(reference_torch, args.threads)
    benchmark_top_level_stack._validate_thread_configuration(
        torch_rs,
        reference_torch,
        args.threads,
    )

    gc_was_enabled = gc.isenabled()
    gc.disable()
    started = time.time()
    try:
        supported = benchmark_top_level_stack._run_supported_cells(
            np,
            torch_rs,
            reference_torch,
            workloads,
            args,
        )
        unsupported = benchmark_top_level_stack._run_unsupported_cells(
            np,
            torch_rs,
            reference_torch,
        )
    finally:
        if gc_was_enabled:
            gc.enable()

    _annotate_supported_rows(supported, cases)
    aggregates = benchmark_top_level_stack._aggregate_rows(supported)
    zero_credit = [
        row
        for row in unsupported
        if row["credit"] == benchmark_top_level_stack.CREDIT_ZERO
    ]
    error_parity = [
        row
        for row in unsupported
        if row["credit"] == benchmark_top_level_stack.CREDIT_ERROR_PARITY
    ]
    capped_with_zero_credit = [
        min(10.0, max(0.10, row["ratios"]["steady_torch_rs_over_pytorch"]))
        for row in supported
    ] + [10.0] * len(zero_credit)
    aggregates["zero_credit_unsupported_cell_count"] = len(zero_credit)
    aggregates["boundary_error_parity_cell_count"] = len(error_parity)
    aggregates["combined_capped_with_zero_credit_unsupported"] = (
        benchmark_top_level_stack._geomean(capped_with_zero_credit)
    )
    aggregates["generated_category_counts"] = dict(
        sorted(Counter(row["category"] for row in supported).items())
    )

    ended = time.time()
    args.workloads = tuple(workload.name for workload in workloads)
    environment = benchmark_top_level_stack._environment(
        torch_rs,
        reference_torch,
        np,
        affinity,
        args,
    )
    environment["benchmark_integrity"].update(
        {
            "workload_set": WORKLOAD_SET,
            "required_validator_path": str(
                Path("scripts") / "validate_top_level_stack_benchmark.py"
            ),
            "merge_decision_usage": (
                "candidate-local stack timing evidence only until paired with "
                "this generated-shape validator using an evaluator-held seed"
            ),
            "same_shape_cpu_float32_only": True,
            "fixed_public_matrix_excluded": True,
        }
    )
    environment["validator"] = validator_context
    return {
        "environment": environment,
        "started_epoch_seconds": started,
        "ended_epoch_seconds": ended,
        "duration_seconds": ended - started,
        "cases": supported,
        "zero_credit_unsupported_cells": zero_credit,
        "boundary_error_parity_cells": error_parity,
        "aggregates": aggregates,
        "validator": validator_context,
    }


def _load_artifact(path):
    with benchmark_top_level_stack._input_path(path).open(
        encoding="utf-8"
    ) as artifact_file:
        return json.load(artifact_file)


def _canonical_stack_dim(dim, rank):
    return dim + rank + 1 if dim < 0 else dim


def _same_shape_input_metadata(row):
    inputs = row.get("input_metadata") or []
    shapes = [tuple(item.get("shape", ())) for item in inputs]
    return inputs, shapes


def _compare_jsonish(errors, path, actual, expected):
    if isinstance(expected, float):
        if not isinstance(actual, (float, int)) or not math.isclose(
            float(actual),
            expected,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            errors.append(f"{path} mismatch: {actual!r} != {expected!r}")
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path} is not an object")
            return
        if set(actual) != set(expected):
            errors.append(
                f"{path} keys mismatch: "
                f"missing={sorted(set(expected) - set(actual))!r} "
                f"extra={sorted(set(actual) - set(expected))!r}"
            )
        for key in sorted(set(actual) & set(expected)):
            _compare_jsonish(errors, f"{path}.{key}", actual[key], expected[key])
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"{path} is not a list")
            return
        if len(actual) != len(expected):
            errors.append(f"{path} length mismatch: {len(actual)} != {len(expected)}")
            return
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _compare_jsonish(errors, f"{path}[{index}]", actual_item, expected_item)
        return
    if actual != expected:
        errors.append(f"{path} mismatch: {actual!r} != {expected!r}")


def _expected_aggregates(cases, unsupported, error_parity):
    aggregates = benchmark_top_level_stack._aggregate_rows(cases)
    capped_with_zero_credit = [
        min(10.0, max(0.10, row["ratios"]["steady_torch_rs_over_pytorch"]))
        for row in cases
    ] + [10.0] * len(unsupported)
    aggregates["zero_credit_unsupported_cell_count"] = len(unsupported)
    aggregates["boundary_error_parity_cell_count"] = len(error_parity)
    aggregates["combined_capped_with_zero_credit_unsupported"] = (
        benchmark_top_level_stack._geomean(capped_with_zero_credit)
    )
    aggregates["generated_category_counts"] = dict(
        sorted(Counter(row["category"] for row in cases).items())
    )
    return aggregates


def _numeric_list(errors, path, value, *, expected_count=None, positive=False):
    if not isinstance(value, list):
        errors.append(f"{path} is not a list")
        return None
    if expected_count is not None and len(value) != expected_count:
        errors.append(f"{path} length mismatch: {len(value)} != {expected_count}")
    numbers = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (float, int)):
            errors.append(f"{path}[{index}] is not numeric: {item!r}")
            continue
        number = float(item)
        if not math.isfinite(number):
            errors.append(f"{path}[{index}] is not finite: {item!r}")
        if positive and number <= 0.0:
            errors.append(f"{path}[{index}] is not positive: {item!r}")
        numbers.append(number)
    return numbers


def _expected_steady_summary(samples_us):
    median_us = statistics.median(samples_us)
    deviations = [abs(sample - median_us) for sample in samples_us]
    variance_us2 = statistics.pvariance(samples_us) if len(samples_us) > 1 else 0.0
    return {
        "median_us": median_us,
        "mad_us": statistics.median(deviations),
        "variance_us2": variance_us2,
        "sample_count": len(samples_us),
        "samples_us": samples_us,
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _validate_pass_timing(errors, path, pass_result, samples, warmups):
    if not isinstance(pass_result, dict):
        errors.append(f"{path} is not an object")
        return None
    steady = pass_result.get("steady")
    if not isinstance(steady, dict):
        errors.append(f"{path}.steady is not an object")
        return None
    samples_us = _numeric_list(
        errors,
        f"{path}.steady.samples_us",
        steady.get("samples_us"),
        expected_count=samples,
        positive=True,
    )
    if not samples_us:
        return None
    expected_steady = _expected_steady_summary(samples_us)
    _compare_jsonish(errors, f"{path}.steady", steady, expected_steady)

    cold_first_call_us = pass_result.get("cold_first_call_us")
    if (
        isinstance(cold_first_call_us, bool)
        or not isinstance(cold_first_call_us, (float, int))
        or not math.isfinite(float(cold_first_call_us))
        or float(cold_first_call_us) <= 0.0
    ):
        errors.append(f"{path}.cold_first_call_us is not a positive finite number")
        return None

    if warmups == 0:
        if pass_result.get("warmup_checksums") != []:
            errors.append(f"{path} unexpected warmup checksums")
    elif pass_result.get("warmup_checksums") != pass_result.get("steady_checksums"):
        errors.append(f"{path} warmup/steady checksum mismatch")

    return {
        "steady": expected_steady,
        "cold_first_call_us": (
            float(cold_first_call_us) if cold_first_call_us is not None else None
        ),
    }


def _validate_implementation_timing(
    errors,
    row_name,
    implementation,
    implementation_result,
    samples,
    warmups,
):
    path = f"{row_name}.{implementation}"
    if not isinstance(implementation_result, dict):
        errors.append(f"{path} is not an object")
        return None
    passes = implementation_result.get("passes", [])
    if len(passes) != len(benchmark_top_level_stack.IMPLEMENTATION_ORDERS):
        errors.append(f"{path} pass count mismatch")
    if implementation_result.get("steady_sample_count") != (
        samples * len(benchmark_top_level_stack.IMPLEMENTATION_ORDERS)
    ):
        errors.append(f"{path} sample count mismatch")
    if len(implementation_result.get("checksums", [])) != 1:
        errors.append(f"{path} unstable checksums")

    pass_timings = []
    for pass_index, pass_result in enumerate(passes):
        pass_path = f"{path}.passes[{pass_index}]"
        if isinstance(pass_result, dict):
            expected_order = (
                list(benchmark_top_level_stack.IMPLEMENTATION_ORDERS[pass_index])
                if pass_index < len(benchmark_top_level_stack.IMPLEMENTATION_ORDERS)
                else None
            )
            if pass_result.get("order_index") != pass_index:
                errors.append(f"{pass_path}.order_index mismatch")
            if expected_order is not None and pass_result.get("order") != expected_order:
                errors.append(f"{pass_path}.order mismatch")
        timing = _validate_pass_timing(
            errors,
            pass_path,
            pass_result,
            samples,
            warmups,
        )
        if timing is not None:
            pass_timings.append(timing)

    if len(pass_timings) != len(benchmark_top_level_stack.IMPLEMENTATION_ORDERS):
        return None

    medians = [item["steady"]["median_us"] for item in pass_timings]
    mads = [item["steady"]["mad_us"] for item in pass_timings]
    variances = [item["steady"]["variance_us2"] for item in pass_timings]
    cold_values = [item["cold_first_call_us"] for item in pass_timings]
    expected_fields = {
        "cold_first_call_median_us": statistics.median(cold_values),
        "cold_first_call_values_us": cold_values,
        "steady_median_us": statistics.median(medians),
        "steady_mad_us": statistics.median(mads),
        "steady_variance_us2": statistics.median(variances),
        "steady_sample_count": samples
        * len(benchmark_top_level_stack.IMPLEMENTATION_ORDERS),
        "checksums": sorted(
            {
                checksum
                for pass_result in passes
                for checksum in (
                    (pass_result.get("steady_checksums") or [])
                    + (pass_result.get("warmup_checksums") or [])
                    + [pass_result.get("cold_checksum")]
                )
            }
        ),
    }
    for key, expected in expected_fields.items():
        _compare_jsonish(
            errors,
            f"{path}.{key}",
            implementation_result.get(key),
            expected,
        )
    return expected_fields["steady_median_us"]


def _absolute_non_resolved(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return REPOSITORY_ROOT / path


def _require_recorded_path(errors, label, value, *, root):
    if not isinstance(value, str) or not value:
        errors.append(f"missing provenance field {label}")
        return None
    path = Path(value)
    if not path.is_absolute():
        errors.append(f"{label} is not absolute: {value!r}")
        return None
    absolute = _absolute_non_resolved(path)
    try:
        absolute.relative_to(root)
    except ValueError:
        errors.append(f"{label} is outside {root}: {value}")
    if not absolute.exists():
        errors.append(f"{label} does not exist: {value}")
    return absolute


def _module_path(module):
    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if origin in (None, "built-in", "frozen"):
        return None
    return str(Path(origin).resolve(strict=True))


def _validate_recorded_path(errors, label, recorded, current, *, root):
    recorded_path = _require_recorded_path(errors, label, recorded, root=root)
    if current in (None, ""):
        errors.append(f"current {label} is unavailable")
        return
    current_path = _absolute_non_resolved(current)
    if recorded_path is not None and recorded_path != current_path:
        errors.append(f"{label} mismatch: {recorded!r} != {str(current_path)!r}")


def _validate_current_runtime_paths(errors, environment):
    try:
        import numpy as np
        import torch as reference_torch
        import torch_rs
    except Exception as error:
        errors.append(f"could not import current benchmark runtime: {error}")
        return

    try:
        native = importlib.import_module("torch_rs.torch_rs")
    except Exception as error:
        errors.append(f"could not import current torch_rs native extension: {error}")
        native = None

    expected_venv = REPOSITORY_ROOT / ".venv"
    _validate_recorded_path(
        errors,
        "python_executable",
        environment.get("python_executable"),
        sys.executable,
        root=expected_venv,
    )
    runtime_paths = (
        ("numpy.path", environment.get("numpy", {}).get("path"), _module_path(np)),
        (
            "pytorch.path",
            environment.get("pytorch", {}).get("path"),
            _module_path(reference_torch),
        ),
        (
            "torch_rs.path",
            environment.get("torch_rs", {}).get("path"),
            _module_path(torch_rs),
        ),
        (
            "torch_rs.extension_path",
            environment.get("torch_rs", {}).get("extension_path"),
            _module_path(native) if native is not None else None,
        ),
    )
    for label, recorded, current in runtime_paths:
        _validate_recorded_path(
            errors,
            label,
            recorded,
            current,
            root=REPOSITORY_ROOT,
        )

    if native is not None:
        native_spec = getattr(native, "__spec__", None)
        if native_spec is None or not isinstance(
            native_spec.loader,
            importlib.machinery.ExtensionFileLoader,
        ):
            errors.append("torch_rs.torch_rs is not a native extension module")
        native_path = _module_path(native)
        if native_path is not None and not Path(native_path).name.endswith(
            tuple(importlib.machinery.EXTENSION_SUFFIXES)
        ):
            errors.append(f"native module has an unrecognized ABI suffix: {native_path}")
        torch_rs_c_path = _module_path(getattr(torch_rs, "_C", None))
        if native_path != torch_rs_c_path:
            errors.append("torch_rs._C does not point to torch_rs.torch_rs")

    current_versions = {
        "numpy.version": np.__version__,
        "pytorch.version": reference_torch.__version__,
        "torch_rs.version": benchmark_top_level_stack._package_version(
            "torch-rs",
            torch_rs,
        ),
    }
    recorded_versions = {
        "numpy.version": environment.get("numpy", {}).get("version"),
        "pytorch.version": environment.get("pytorch", {}).get("version"),
        "torch_rs.version": environment.get("torch_rs", {}).get("version"),
    }
    for label, current in current_versions.items():
        if recorded_versions[label] != current:
            errors.append(
                f"{label} mismatch: {recorded_versions[label]!r} != {current!r}"
            )


def _validate_environment_provenance(errors, environment, validator):
    if environment.get("benchmark_version") != (
        benchmark_top_level_stack.BENCHMARK_VERSION
    ):
        errors.append("benchmark version mismatch")
    if environment.get("validator") != validator:
        errors.append("environment validator context does not match top-level context")
    if environment.get("api") != f"torch.{benchmark_top_level_stack.API}":
        errors.append(f"API metadata mismatch: {environment.get('api')!r}")
    if environment.get("cwd") != str(REPOSITORY_ROOT):
        errors.append(f"cwd mismatch: {environment.get('cwd')!r}")

    driver = environment.get("driver", {})
    expected_driver_path = BENCHMARK_SCRIPT.relative_to(REPOSITORY_ROOT).as_posix()
    if driver.get("path") != expected_driver_path:
        errors.append(f"driver path mismatch: {driver.get('path')!r}")
    if driver.get("sha256") != benchmark_top_level_stack._file_sha256(
        BENCHMARK_SCRIPT
    ):
        errors.append("driver SHA-256 does not match the checked-in script")

    if environment.get("git") != benchmark_top_level_stack._git_provenance():
        errors.append("git provenance does not match the current worktree")

    build_profile = environment.get("build_profile", {})
    if build_profile.get("rust_profile") != "release":
        errors.append(f"build profile mismatch: {build_profile!r}")
    if build_profile.get("native_extension_required") is not True:
        errors.append("native extension requirement is not recorded")

    affinity = environment.get("cpu_affinity", {})
    pinned_affinity = affinity.get("pinned_affinity")
    if pinned_affinity is not None and len(pinned_affinity) != 1:
        errors.append(f"benchmark was not pinned to one CPU: {affinity!r}")

    threads = environment.get("threads")
    if not isinstance(threads, int) or threads <= 0:
        errors.append(f"invalid thread count: {threads!r}")
    else:
        env_threads = environment.get("env_threads", {})
        for name in benchmark_top_level_stack.THREAD_ENVIRONMENT_VARIABLES:
            if env_threads.get(name) != str(threads):
                errors.append(f"{name} mismatch: {env_threads.get(name)!r}")
        for section, key in (
            ("pytorch", "threads"),
            ("pytorch", "interop_threads"),
            ("torch_rs", "threads"),
            ("torch_rs", "interop_threads"),
        ):
            if environment.get(section, {}).get(key) != threads:
                errors.append(f"{section}.{key} mismatch")

    pytorch = environment.get("pytorch", {})
    if benchmark_top_level_stack._version_without_local(
        pytorch.get("version", "")
    ) != benchmark_top_level_stack.REFERENCE_PYTORCH_VERSION:
        errors.append(f"PyTorch version mismatch: {pytorch.get('version')!r}")

    for section, key in (
        ("rust", "rustc"),
        ("rust", "cargo"),
    ):
        if environment.get(section, {}).get(key) in (None, ""):
            errors.append(f"missing provenance field {section}.{key}")
    _validate_current_runtime_paths(errors, environment)


def _validate_unsupported_rows(errors, unsupported, error_parity):
    expected_by_name = {
        f"top_level_torch_stack_{unsupported_cell.name}": unsupported_cell
        for unsupported_cell in benchmark_top_level_stack.UNSUPPORTED_CELLS
    }
    for row in unsupported + error_parity:
        name = row.get("name")
        expected = expected_by_name.get(name)
        if expected is None:
            errors.append(f"{name} unknown boundary row")
            continue
        if row.get("api") != f"torch.{benchmark_top_level_stack.API}":
            errors.append(f"{name} API mismatch")
        if row.get("input_description") != expected.input_description:
            errors.append(f"{name} input description mismatch")
        if row.get("credit") != expected.credit:
            errors.append(
                f"{name} credit mismatch: {row.get('credit')!r} != {expected.credit!r}"
            )
        if row.get("reason") != expected.reason:
            errors.append(f"{name} reason mismatch")

        torch_rs_status = row.get("torch_rs", {})
        if torch_rs_status.get("kind") != "error":
            errors.append(f"{name} torch_rs status is not an error")
        if torch_rs_status.get("error_type") != expected.torch_rs_error_type:
            errors.append(f"{name} torch_rs error type mismatch")
        if (
            expected.torch_rs_message is not None
            and torch_rs_status.get("message") != expected.torch_rs_message
        ):
            errors.append(f"{name} torch_rs error message mismatch")

        pytorch_status = row.get("pytorch", {})
        if pytorch_status.get("kind") != expected.pytorch_expected_kind:
            errors.append(f"{name} PyTorch status mismatch")
        if expected.credit == benchmark_top_level_stack.CREDIT_ZERO:
            if pytorch_status.get("kind") != "supported":
                errors.append(f"{name} zero-credit row is not PyTorch-supported")
        elif expected.credit == benchmark_top_level_stack.CREDIT_ERROR_PARITY:
            if pytorch_status.get("kind") != "error":
                errors.append(f"{name} error-parity row is not a PyTorch error")
            if (
                pytorch_status.get("error_type") != torch_rs_status.get("error_type")
                or pytorch_status.get("message") != torch_rs_status.get("message")
            ):
                errors.append(f"{name} error-parity status mismatch")
        else:
            errors.append(f"{name} unknown credit policy {expected.credit!r}")

        validation = row.get("validation", {})
        if (
            validation.get("torch_rs_error_checked") is not True
            or validation.get("pytorch_status_checked") is not True
        ):
            errors.append(f"{name} missing unsupported-cell validation flags")


def _validate_case_row(errors, context_case, row, samples, warmups):
    row_name = f"{row.get('api')}/{row.get('workload')}"
    shape = tuple(context_case.get("shape") or ())
    dim = context_case.get("dim")
    input_count = context_case.get("input_count")
    mode = context_case.get("mode")
    category = context_case.get("category")
    layout = context_case.get("layout")
    if shape in PUBLIC_INPUT_SHAPES:
        errors.append(f"{row_name} uses fixed public input shape {shape!r}")
    for key in ("workload", "category", "mode", "repeats", "dim", "input_count"):
        expected_key = "name" if key == "workload" else key
        if row.get(key) != context_case.get(expected_key):
            errors.append(
                f"{row_name} {key} mismatch: "
                f"{row.get(key)!r} != {context_case.get(expected_key)!r}"
            )
    if row.get("seed_values") != context_case.get("seeds"):
        errors.append(f"{row_name} seed metadata mismatch")
    if row.get("api") != f"torch.{benchmark_top_level_stack.API}":
        errors.append(f"{row_name} API metadata mismatch")
    if row.get("generated") is not True:
        errors.append(f"{row_name} is not marked as generated")
    if row.get("validator_case") != context_case:
        errors.append(f"{row_name} validator case metadata mismatch")

    validation = row.get("validation", {})
    for required_key in (
        "metadata_checked",
        "value_bits_checked",
        "warmup_checksums_checked",
        "steady_checksums_checked",
        "held_out_generated_shape",
        "fixed_public_matrix_excluded",
        "same_shape_cpu_float32_inputs",
    ):
        if validation.get(required_key) is not True:
            errors.append(f"{row_name} missing validation flag {required_key}")

    inputs, input_shapes = _same_shape_input_metadata(row)
    if len(inputs) != input_count:
        errors.append(f"{row_name} input metadata count mismatch")
    if len(set(input_shapes)) != 1 or (input_shapes and input_shapes[0] != shape):
        errors.append(f"{row_name} inputs are not the generated same shape")
    for item in inputs:
        if item.get("dtype") != "torch.float32":
            errors.append(f"{row_name} input dtype is not torch.float32")
        if item.get("device") != "cpu":
            errors.append(f"{row_name} input device is not cpu")
        if item.get("layout") != "torch.strided":
            errors.append(f"{row_name} input layout is not torch.strided")
        if category == "offset" and item.get("storage_offset", 0) <= 0:
            errors.append(f"{row_name} offset input lacks nonzero storage offset")
        if layout == "noncontiguous" and item.get("is_contiguous") is not False:
            errors.append(f"{row_name} noncontiguous input is contiguous")
        if mode in (
            benchmark_top_level_stack.MODE_AUTOGRAD_FORWARD,
            benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD,
        ) and item.get("requires_grad") is not True:
            errors.append(f"{row_name} autograd input does not require grad")

    if category == "empty" and 0 not in shape:
        errors.append(f"{row_name} empty category has no zero dimension")
    if category == "contiguous":
        for item in inputs:
            if item.get("storage_offset") != 0 or item.get("is_contiguous") is not True:
                errors.append(f"{row_name} contiguous input metadata mismatch")

    output_metadata = row.get("output_metadata") or []
    if not output_metadata:
        errors.append(f"{row_name} missing output metadata")
    else:
        output_shape = list(shape)
        output_shape.insert(_canonical_stack_dim(dim, len(shape)), input_count)
        output = output_metadata[0]
        if output.get("shape") != output_shape:
            errors.append(
                f"{row_name} output shape mismatch: "
                f"{output.get('shape')!r} != {output_shape!r}"
            )
        if output.get("dtype") != "torch.float32" or output.get("device") != "cpu":
            errors.append(f"{row_name} output is not CPU float32")
        if mode in (
            benchmark_top_level_stack.MODE_AUTOGRAD_FORWARD,
            benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD,
        ) and output.get("requires_grad") is not True:
            errors.append(f"{row_name} autograd output does not require grad")

    if mode == benchmark_top_level_stack.MODE_AUTOGRAD_BACKWARD:
        labels = [entry.get("label") for entry in output_metadata]
        expected_labels = [
            "output",
            *[
                f"source_{index}_grad"
                for index in range(len(context_case.get("seeds") or ()))
            ],
        ]
        if labels != expected_labels:
            errors.append(f"{row_name} missing backward gradient artifacts")

    implementation_medians = {}
    for implementation in ("torch_rs", "pytorch"):
        implementation_result = row.get("implementations", {}).get(
            implementation,
            {},
        )
        median = _validate_implementation_timing(
            errors,
            row_name,
            implementation,
            implementation_result,
            samples,
            warmups,
        )
        if median is not None:
            implementation_medians[implementation] = median
    try:
        benchmark_top_level_stack._single_checksum_pair(row)
    except AssertionError as error:
        errors.append(str(error))
    if {"torch_rs", "pytorch"} - set(implementation_medians):
        return None
    pytorch_median = implementation_medians["pytorch"]
    if pytorch_median <= 0.0:
        errors.append(f"{row_name} PyTorch median is not positive")
        return None
    expected_ratios = {
        "steady_torch_rs_over_pytorch": (
            implementation_medians["torch_rs"] / pytorch_median
        ),
    }
    _compare_jsonish(errors, f"{row_name}.ratios", row.get("ratios"), expected_ratios)
    derived = dict(row)
    derived["ratios"] = expected_ratios
    return derived


def validate_artifact_dict(report):
    errors = []
    validator = report.get("validator", {})
    environment = report.get("environment", {})
    if validator.get("validator_version") != VALIDATOR_VERSION:
        errors.append("validator version mismatch")
    if validator.get("independent_validator_path") != (
        str(Path("scripts") / "validate_top_level_stack_benchmark.py")
    ):
        errors.append("validator path mismatch")
    if validator.get("independent_validator_sha256") != (
        benchmark_top_level_stack._file_sha256(Path(__file__).resolve())
    ):
        errors.append("validator SHA-256 does not match the checked-in script")
    if validator.get("benchmark_driver_version") != (
        benchmark_top_level_stack.BENCHMARK_VERSION
    ):
        errors.append("benchmark driver version mismatch")
    if validator.get("benchmark_driver_path") != (
        str(Path("scripts") / "benchmark_top_level_stack.py")
    ):
        errors.append("benchmark driver path mismatch")
    if validator.get("fixed_public_matrix_excluded") is not True:
        errors.append("public matrix exclusion is not recorded")
    if validator.get("same_shape_cpu_float32_only") is not True:
        errors.append("CPU float32 same-shape contract is not recorded")
    if validator.get("required_categories") != list(REQUIRED_CATEGORIES):
        errors.append("required category metadata mismatch")
    if validator.get("fixed_public_input_shapes") != [
        list(shape) for shape in sorted(PUBLIC_INPUT_SHAPES, key=repr)
    ]:
        errors.append("fixed public input shape metadata mismatch")
    if validator.get("seed_source") not in ("cli", "secrets.randbits(64)"):
        errors.append(f"seed source mismatch: {validator.get('seed_source')!r}")

    benchmark_integrity = environment.get("benchmark_integrity", {})
    if benchmark_integrity.get("workload_set") != WORKLOAD_SET:
        errors.append("workload set metadata mismatch")
    if benchmark_integrity.get("required_validator_path") != (
        str(Path("scripts") / "validate_top_level_stack_benchmark.py")
    ):
        errors.append("required validator path metadata mismatch")
    if benchmark_integrity.get("fixed_public_matrix_excluded") is not True:
        errors.append("environment does not record public matrix exclusion")
    if benchmark_integrity.get("same_shape_cpu_float32_only") is not True:
        errors.append("environment does not record CPU float32 same-shape contract")
    if environment.get("implementation_orders") != [
        list(order) for order in benchmark_top_level_stack.IMPLEMENTATION_ORDERS
    ]:
        errors.append("implementation order metadata mismatch")
    _validate_environment_provenance(errors, environment, validator)

    seed = validator.get("seed")
    cases_per_category = validator.get("cases_per_category")
    max_elements = validator.get("max_elements")
    if not isinstance(seed, int):
        errors.append(f"invalid seed: {seed!r}")
    if not isinstance(cases_per_category, int) or cases_per_category <= 0:
        errors.append(f"invalid cases_per_category: {cases_per_category!r}")
        cases_per_category = 0
    if not isinstance(max_elements, int) or max_elements <= 0:
        errors.append(f"invalid max_elements: {max_elements!r}")
        max_elements = 0
    expected_count = cases_per_category * len(REQUIRED_CATEGORIES)
    cases = report.get("cases") or []
    context_cases = validator.get("generated_cases") or []
    if len(cases) != expected_count:
        errors.append(f"case count mismatch: {len(cases)} != {expected_count}")
    if len(context_cases) != expected_count:
        errors.append(
            f"validator case count mismatch: {len(context_cases)} != {expected_count}"
        )
    if isinstance(seed, int) and cases_per_category > 0 and max_elements > 0:
        expected_context_cases = _expected_case_records(
            seed,
            cases_per_category,
            max_elements,
        )
        if context_cases != expected_context_cases:
            errors.append("validator generated cases do not match seed/config")

    category_counts = Counter(row.get("category") for row in cases)
    expected_category_counts = Counter(
        {category: cases_per_category for category in REQUIRED_CATEGORIES}
    )
    if category_counts != expected_category_counts:
        errors.append(f"category coverage mismatch: {dict(category_counts)!r}")
    if report.get("aggregates", {}).get("generated_category_counts") != dict(
        sorted(expected_category_counts.items())
    ):
        errors.append("aggregate generated category counts mismatch")

    by_name = {row.get("workload"): row for row in cases}
    if len(by_name) != len(cases):
        errors.append("duplicate workload names in cases")
    context_names = [context_case.get("name") for context_case in context_cases]
    if set(by_name) != set(context_names):
        errors.append("case set does not match validator context")
    if environment.get("workloads") != context_names:
        errors.append("environment workload list does not match validator context")

    samples = environment.get("samples")
    warmups = environment.get("warmups")
    if not isinstance(samples, int) or samples <= 0:
        errors.append(f"invalid sample count: {samples!r}")
        samples = 0
    if not isinstance(warmups, int) or warmups < 0:
        errors.append(f"invalid warmup count: {warmups!r}")
        warmups = 0
    derived_cases = []
    for context_case in context_cases:
        row = by_name.get(context_case.get("name"))
        if row is not None:
            derived_row = _validate_case_row(
                errors,
                context_case,
                row,
                samples,
                warmups,
            )
            if derived_row is not None:
                derived_cases.append(derived_row)

    unsupported = report.get("zero_credit_unsupported_cells", [])
    error_parity = report.get("boundary_error_parity_cells", [])
    for credit, field in (
        (
            benchmark_top_level_stack.CREDIT_ZERO,
            "zero_credit_unsupported_cells",
        ),
        (
            benchmark_top_level_stack.CREDIT_ERROR_PARITY,
            "boundary_error_parity_cells",
        ),
    ):
        expected_names = benchmark_top_level_stack._expected_unsupported_names(credit)
        actual_names = {row.get("name") for row in report.get(field, [])}
        if actual_names != expected_names:
            errors.append(
                f"{field} mismatch: "
                f"missing={sorted(expected_names - actual_names)!r} "
                f"extra={sorted(actual_names - expected_names)!r}"
            )
    _validate_unsupported_rows(errors, unsupported, error_parity)

    aggregates = report.get("aggregates", {})
    if aggregates.get("timed_supported_cell_count") != len(cases):
        errors.append("aggregate timed cell count does not match cases")
    if aggregates.get("zero_credit_unsupported_cell_count") != len(unsupported):
        errors.append("aggregate unsupported cell count mismatch")
    if aggregates.get("boundary_error_parity_cell_count") != len(error_parity):
        errors.append("aggregate error-parity cell count mismatch")
    try:
        expected_aggregate_values = _expected_aggregates(
            derived_cases,
            unsupported,
            error_parity,
        )
    except (KeyError, TypeError, ValueError) as error:
        errors.append(f"could not recompute aggregates: {error}")
    else:
        _compare_jsonish(errors, "aggregates", aggregates, expected_aggregate_values)

    if errors:
        raise AssertionError("\n".join(errors))


def validate_artifact(artifact_path):
    validate_artifact_dict(_load_artifact(artifact_path))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cases-per-category", type=int, default=2)
    parser.add_argument("--max-elements", type=int, default=262_144)
    parser.add_argument(
        "--warmups",
        type=int,
        default=benchmark_top_level_stack.DEFAULT_WARMUPS,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=benchmark_top_level_stack.DEFAULT_SAMPLES,
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=benchmark_top_level_stack.DEFAULT_THREADS,
    )
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--validate-artifact",
        type=Path,
        metavar="RAW_JSON",
        help="validate a generated stack-validator JSON artifact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validate_artifact is not None:
        validate_artifact(args.validate_artifact)
        return
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    _positive_int(args.samples, "--samples")
    _positive_int(args.threads, "--threads")
    _positive_int(args.cases_per_category, "--cases-per-category")
    _positive_int(args.max_elements, "--max-elements")

    seed = args.seed if args.seed is not None else secrets.randbits(64)
    cases = generate_cases(seed, args.cases_per_category, args.max_elements)
    workloads = _workloads_for_cases(cases)
    validator_context = _validator_context(args, seed, cases)
    report = run_validator(args, cases, workloads, validator_context)
    validate_artifact_dict(report)

    encoded = json.dumps(report, indent=2, sort_keys=True)
    output = (
        benchmark_top_level_stack._output_path(args.output)
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
