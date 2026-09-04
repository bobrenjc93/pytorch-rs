#!/usr/bin/env python3
"""Benchmark the supported CPU ``torch.compile`` eager/fullgraph corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import gc
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = (
    REPOSITORY_ROOT / "docs" / "benchmark-data" / "torch-compile-cpu-v4.json"
)
DEFAULT_MARKDOWN_REPORT_PATH = (
    REPOSITORY_ROOT / "docs" / "torch-compile-cpu-release-timings.md"
)
PROTECTED_OUTPUT_PATHS = {
    REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
    REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
}
REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "torch_compile_cpu_eager_benchmark_v3"
DEFAULT_WARMUPS = 7
DEFAULT_SAMPLES = 31
IMPLEMENTATION_ORDERS = (
    ("torch_rs", "pytorch"),
    ("pytorch", "torch_rs"),
)
CATEGORY_LABELS = {
    "tensor_arithmetic": "tensor-arithmetic",
    "broadcasting": "broadcasting",
    "inference": "inference",
    "training_autograd": "training-autograd",
    "recompilation_guards": "recompilation-guard",
}
CATEGORY_PHRASES = {
    "tensor_arithmetic": "tensor arithmetic",
    "broadcasting": "broadcasting",
    "inference": "inference",
    "training_autograd": "training autograd",
    "recompilation_guards": "recompilation guards",
}
CATEGORY_PROGRAM_LABELS = {
    "tensor_arithmetic": "tensor-arithmetic",
    "broadcasting": "broadcasting",
    "inference": "inference",
    "training_autograd": "training-autograd",
    "recompilation_guards": "recompilation-guard",
}


@dataclass(frozen=True)
class InputVariant:
    name: str
    category: str
    description: str
    repeats: int
    make_inputs: object
    input_count: int | None = 1


def _product(shape):
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _value(index):
    return ((index * 37) % 257 - 128) / 17.0


def _nested_values(shape, start=0):
    if not shape:
        return _value(start)
    if shape[0] == 0:
        return []
    stride = _product(shape[1:])
    return [_nested_values(shape[1:], start + index * stride) for index in range(shape[0])]


def _dense_input(shape):
    def make_inputs(module):
        return (module.tensor(_nested_values(shape), dtype=module.float32),)

    return make_inputs


def _dense_inputs(*shapes):
    def make_inputs(module):
        return tuple(
            module.tensor(_nested_values(shape), dtype=module.float32)
            for shape in shapes
        )

    return make_inputs


def _transposed_input(base_shape):
    def make_inputs(module):
        base = module.tensor(_nested_values(base_shape), dtype=module.float32)
        return (base.transpose(0, 1),)

    return make_inputs


def _transposed_matrix_vector_input(base_shape):
    def make_inputs(module):
        base = module.tensor(_nested_values(base_shape), dtype=module.float32)
        return (
            base.transpose(0, 1),
            module.tensor(_nested_values((base_shape[0],)), dtype=module.float32),
        )

    return make_inputs


INPUT_VARIANTS = (
    InputVariant(
        "case_default",
        "corpus_default",
        "case's checked-in input factory",
        256,
        None,
        None,
    ),
    InputVariant("scalar", "scalar", "rank-0 scalar", 2048, _dense_input(())),
    InputVariant("vector_17", "vector", "rank-1 prime length 17", 1024, _dense_input((17,))),
    InputVariant(
        "matrix_31x37",
        "small_matrix",
        "row-major rank-2 prime shape (31, 37)",
        128,
        _dense_input((31, 37)),
    ),
    InputVariant(
        "matrix_127x131",
        "medium_matrix",
        "row-major rank-2 prime shape (127, 131)",
        16,
        _dense_input((127, 131)),
    ),
    InputVariant(
        "empty_2x0",
        "empty",
        "rank-2 empty shape (2, 0)",
        2048,
        _dense_input((2, 0)),
    ),
    InputVariant(
        "transpose_37x31",
        "noncontiguous",
        "transpose view from base shape (31, 37)",
        128,
        _transposed_input((31, 37)),
    ),
    InputVariant(
        "matrix_vector_31x37_by_37",
        "broadcast_matrix_vector",
        "row-major rank-2 by trailing rank-1 broadcast, shape (31, 37) by (37,)",
        128,
        _dense_inputs((31, 37), (37,)),
        2,
    ),
    InputVariant(
        "matrix_vector_127x131_by_131",
        "broadcast_matrix_vector",
        "row-major rank-2 by trailing rank-1 broadcast, shape (127, 131) by (131,)",
        16,
        _dense_inputs((127, 131), (131,)),
        2,
    ),
    InputVariant(
        "tensor_scalar_31x37",
        "broadcast_tensor_scalar",
        "row-major rank-2 by scalar broadcast, shape (31, 37) by ()",
        128,
        _dense_inputs((31, 37), ()),
        2,
    ),
    InputVariant(
        "scalar_tensor_31x37",
        "broadcast_scalar_tensor",
        "scalar by row-major rank-2 broadcast, shape () by (31, 37)",
        128,
        _dense_inputs((), (31, 37)),
        2,
    ),
    InputVariant(
        "empty_2x0_by_0",
        "broadcast_empty",
        "rank-2 empty by trailing empty rank-1 broadcast, shape (2, 0) by (0,)",
        2048,
        _dense_inputs((2, 0), (0,)),
        2,
    ),
    InputVariant(
        "transpose_31x37_by_37",
        "broadcast_noncontiguous",
        "transpose view shape (31, 37) by trailing rank-1 broadcast",
        128,
        _transposed_matrix_vector_input((37, 31)),
        2,
    ),
)


def _load_compile_corpus_module():
    corpus_path = REPOSITORY_ROOT / "tests" / "test_compile_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "_torch_rs_compile_corpus_for_benchmark",
        corpus_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load compile corpus from {corpus_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def _package_version(distribution_name, module):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


def _exception_name(error):
    error_type = type(error)
    module = error_type.__module__
    name = error_type.__qualname__
    if module == "builtins":
        return name
    return f"{module}.{name}"


def _environment(torch_rs, reference_torch, corpus_version, args):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "corpus_version": corpus_version,
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cpu": _cpu_model_name(),
        "process_affinity": _affinity(),
        "env_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
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
        "git": _git_provenance(),
        "warmups": args.warmups,
        "samples": args.samples,
        "required_single_cpu_affinity": args.require_single_cpu_affinity,
        "implementation_orders": IMPLEMENTATION_ORDERS,
    }


def _corpus_metadata(corpus_module):
    def case_summary(case):
        return {
            "name": case.name,
            "category": case.category,
            "input_count": _program_input_count(case),
            "fullgraph": case.fullgraph,
            "dynamic": case.dynamic,
            "mode": case.mode,
            "options": case.options,
            "recompile_limit": case.recompile_limit,
        }

    def step_summary(step):
        return {
            "name": step.name,
            "guard_change": step.guard_change,
            "expected_compile_count": step.expected_compile_count,
            "reset_before": step.reset_before,
            "expect_limit_error": step.expect_limit_error,
        }

    def scenario_summary(scenario):
        return {
            "name": scenario.name,
            "case": scenario.case.name,
            "steps": [step_summary(step) for step in scenario.steps],
        }

    return {
        "version": corpus_module.COMPILE_CORPUS_VERSION,
        "public_cases": [
            case_summary(case) for case in corpus_module.compile_corpus_cases()
        ],
        "held_out_cases": [
            case_summary(case)
            for case in corpus_module.compile_corpus_cases(include_held_out=True)
            if case not in corpus_module.compile_corpus_cases()
        ],
        "public_recompilation_guard_scenarios": [
            scenario_summary(scenario)
            for scenario in corpus_module.compile_recompilation_guard_scenarios()
        ],
        "held_out_recompilation_guard_scenarios": [
            scenario_summary(scenario)
            for scenario in corpus_module.compile_recompilation_guard_scenarios(
                include_held_out=True
            )
            if scenario not in corpus_module.compile_recompilation_guard_scenarios()
        ],
    }


def _configure_reference_threads(reference_torch, threads):
    reference_torch.set_num_threads(threads)
    reference_torch.set_num_interop_threads(threads)


def _reset_compile_state(module, implementation, corpus_module):
    if implementation == "pytorch":
        corpus_module.reset_reference_compile_state()
        return
    compiler = getattr(module, "compiler", None)
    if compiler is not None:
        reset = getattr(compiler, "reset", None)
        if reset is not None:
            reset()


def _program_input_count(case):
    code = getattr(case.program, "__code__", None)
    if code is None:
        raise AssertionError(f"{case.name} program is not an exact Python function")
    return code.co_argcount


def _variant_applies_to_case(variant, case):
    return (
        variant.input_count is None
        or variant.input_count == _program_input_count(case)
    )


def _cell_input_factory(case, variant):
    return case.make_inputs if variant.make_inputs is None else variant.make_inputs


def _case_execution_context(module, case):
    if getattr(case, "category", None) == "inference":
        return module.no_grad()
    return nullcontext()


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
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _materialized_payload(tensor):
    return {
        "metadata": _tensor_metadata(tensor),
        "values": tensor.tolist(),
    }


def _checksum_tensor(tensor):
    payload = json.dumps(
        _materialized_payload(tensor),
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _assert_outputs_match(actual, expected, *, cell_name):
    actual_payload = _materialized_payload(actual)
    expected_payload = _materialized_payload(expected)
    if actual_payload != expected_payload:
        raise AssertionError(
            f"{cell_name} output mismatch:\n"
            f"actual={actual_payload!r}\nexpected={expected_payload!r}"
        )


def _assert_timing_checksums_match(measured, expected, *, cell_name):
    expected_checksum = _checksum_tensor(expected)
    if measured["cold_checksum"] != expected_checksum:
        raise AssertionError(
            f"{cell_name} cold checksum mismatch: "
            f"actual={measured['cold_checksum']!r} "
            f"expected={expected_checksum!r}"
        )
    if measured["steady_checksums"] != [expected_checksum]:
        raise AssertionError(
            f"{cell_name} steady checksum mismatch or instability: "
            f"actual={measured['steady_checksums']!r} "
            f"expected={[expected_checksum]!r}"
        )


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


def _time_once(compiled, inputs, module, case):
    with _case_execution_context(module, case):
        started_ns = time.perf_counter_ns()
        output = compiled(*inputs)
    _synchronize(module)
    checksum = _checksum_tensor(output)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, checksum, output


def _time_repeated(compiled, inputs, module, repeats, case):
    with _case_execution_context(module, case):
        started_ns = time.perf_counter_ns()
        output = None
        for _ in range(repeats):
            output = compiled(*inputs)
    _synchronize(module)
    checksum = _checksum_tensor(output)
    return time.perf_counter_ns() - started_ns, checksum


def _run_cell(
    *,
    module,
    implementation,
    case,
    variant,
    corpus_module,
    warmups,
    samples,
):
    make_inputs = _cell_input_factory(case, variant)
    inputs = make_inputs(module)
    expected_input_count = _program_input_count(case)
    if len(inputs) != expected_input_count:
        raise AssertionError(
            f"{case.name}/{variant.name} returned {len(inputs)} inputs, "
            f"expected {expected_input_count}"
        )
    if expected_input_count not in (1, 2):
        raise AssertionError(
            f"{case.name} has unsupported benchmark arity {expected_input_count}"
        )

    _reset_compile_state(module, implementation, corpus_module)
    compile_started_ns = time.perf_counter_ns()
    compiled = module.compile(case.program, **case.compile_kwargs("eager"))
    factory_ns = time.perf_counter_ns() - compile_started_ns
    cold_ns, cold_checksum, cold_output = _time_once(compiled, inputs, module, case)

    for _ in range(warmups):
        _time_repeated(compiled, inputs, module, variant.repeats, case)

    sample_ns = []
    sample_checksums = []
    for _ in range(samples):
        elapsed_ns, checksum = _time_repeated(
            compiled,
            inputs,
            module,
            variant.repeats,
            case,
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)

    return {
        "factory_us": factory_ns / 1000.0,
        "cold_first_call_us": cold_ns / 1000.0,
        "cold_checksum": cold_checksum,
        "steady": _summarize_samples(sample_ns, variant.repeats),
        "steady_checksums": sorted(set(sample_checksums)),
        "cold_output": cold_output,
        "input_metadata": [_tensor_metadata(input) for input in inputs],
        "output_metadata": _tensor_metadata(cold_output),
    }


def _run_guard_sequence_pass(
    *,
    module,
    implementation,
    scenario,
    corpus_module,
    reference_torch,
):
    case = scenario.case
    _reset_compile_state(module, implementation, corpus_module)
    compiled = module.compile(case.program, **case.compile_kwargs("eager"))
    steps = []

    for step in scenario.steps:
        if step.reset_before:
            _reset_compile_state(module, implementation, corpus_module)

        inputs = step.make_inputs(module)
        started_ns = time.perf_counter_ns()
        try:
            with _case_execution_context(module, case):
                output = compiled(*inputs)
                _synchronize(module)
                elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0
        except Exception as error:
            elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0
            if not step.expect_limit_error:
                raise
            steps.append(
                {
                    "step": step.name,
                    "guard_change": step.guard_change,
                    "status": "expected_error",
                    "elapsed_us": elapsed_us,
                    "input_metadata": [_tensor_metadata(input) for input in inputs],
                    "error_type": _exception_name(error),
                    "error_message": str(error).splitlines()[0],
                }
            )
            continue

        if step.expect_limit_error:
            raise AssertionError(
                f"{scenario.name}/{implementation}/{step.name} expected "
                "a recompile-limit error"
            )

        expected_inputs = step.make_inputs(module)
        with _case_execution_context(module, case):
            expected = case.program(*expected_inputs)
        cell_name = f"{scenario.name}/{step.name}/{implementation}"
        _assert_outputs_match(output, expected, cell_name=cell_name)
        checksum = _checksum_tensor(output)
        _assert_timing_checksums_match(
            {
                "cold_checksum": checksum,
                "steady_checksums": [checksum],
            },
            expected,
            cell_name=cell_name,
        )

        if implementation == "torch_rs":
            reference_inputs = step.make_inputs(reference_torch)
            with _case_execution_context(reference_torch, case):
                reference_expected = case.program(*reference_inputs)
            _assert_outputs_match(
                output,
                reference_expected,
                cell_name=f"{scenario.name}/{step.name}",
            )
            _assert_timing_checksums_match(
                {
                    "cold_checksum": checksum,
                    "steady_checksums": [checksum],
                },
                reference_expected,
                cell_name=f"{scenario.name}/{step.name}",
            )

        steps.append(
            {
                "step": step.name,
                "guard_change": step.guard_change,
                "status": "ok",
                "elapsed_us": elapsed_us,
                "input_metadata": [_tensor_metadata(input) for input in inputs],
                "output_metadata": _tensor_metadata(output),
                "checksum": checksum,
            }
        )

    return {
        "scenario": scenario.name,
        "case": case.name,
        "implementation": implementation,
        "recompile_limit": case.recompile_limit,
        "steps": steps,
    }


def _run_guard_sequences(corpus_module, torch_rs, reference_torch):
    rows = []
    scenarios = tuple(
        getattr(corpus_module, "COMPILE_RECOMPILATION_GUARD_SCENARIOS", ())
    )
    for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
        for implementation in order:
            module = torch_rs if implementation == "torch_rs" else reference_torch
            for scenario in scenarios:
                row = _run_guard_sequence_pass(
                    module=module,
                    implementation=implementation,
                    scenario=scenario,
                    corpus_module=corpus_module,
                    reference_torch=reference_torch,
                )
                row["order_index"] = order_index
                row["order"] = list(order)
                rows.append(row)
    return rows


def _geomean(values):
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _coverage_denominator(corpus_module, selected_cases):
    category_weights = dict(corpus_module.CATEGORY_WEIGHTS)
    selected_names = {case.name for case in selected_cases}
    public_cases = tuple(corpus_module.COMPILE_CORPUS)
    held_out_cases = tuple(getattr(corpus_module, "COMPILE_HELD_OUT_CORPUS", ()))

    supported_categories = []
    zero_credit_categories = []
    supported_weight = 0
    for category, weight in category_weights.items():
        category_cases = [
            case.name for case in public_cases if case.category == category
        ]
        if category_cases:
            supported_weight += weight
            supported_categories.append(
                {
                    "category": category,
                    "weight": weight,
                    "public_cases": category_cases,
                    "timed_public_cases": [
                        name for name in category_cases if name in selected_names
                    ],
                }
            )
        else:
            zero_credit_categories.append(
                {
                    "category": category,
                    "weight": weight,
                    "reason": (
                        "no native torch_rs eager/fullgraph compile cases are "
                        "implemented for this category in the checked-in corpus"
                    ),
                }
            )

    total_weight = sum(category_weights.values())
    zero_credit_weight = total_weight - supported_weight
    return {
        "category_weights": category_weights,
        "total_weight": total_weight,
        "supported_weight": supported_weight,
        "zero_credit_weight": zero_credit_weight,
        "weighted_supported_percent": (supported_weight / total_weight * 100.0)
        if total_weight
        else None,
        "public_supported_case_count": len(public_cases),
        "held_out_case_count": len(held_out_cases),
        "selected_public_case_count": len(selected_cases),
        "supported_categories": supported_categories,
        "zero_credit_categories": zero_credit_categories,
    }


def _select_named(items, selected_names, *, item_kind):
    if not selected_names:
        return tuple(items)
    by_name = {item.name: item for item in items}
    missing = [name for name in selected_names if name not in by_name]
    if missing:
        available = ", ".join(sorted(by_name))
        raise SystemExit(
            f"unknown {item_kind}: {', '.join(missing)}. Available: {available}"
        )
    return tuple(by_name[name] for name in selected_names)


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


def _input_path(path):
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        raise SystemExit(
            f"input path must stay inside the worktree: {resolved}"
        ) from None
    return resolved


def run_benchmark(args):
    corpus_module = _load_compile_corpus_module()
    torch_rs = corpus_module.torch
    reference_torch = corpus_module.reference_torch
    if reference_torch is None:
        raise SystemExit("install the reference dependency group to provide PyTorch")
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise SystemExit(
            "CPU compile benchmark requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )
    affinity = _affinity()
    if args.require_single_cpu_affinity and (affinity is None or len(affinity) != 1):
        raise SystemExit(
            "--require-single-cpu-affinity requires the process to be pinned "
            f"to exactly one CPU, got {affinity!r}"
        )

    _configure_reference_threads(reference_torch, args.threads)
    cases = _select_named(corpus_module.COMPILE_CORPUS, args.cases, item_kind="case")
    variants = _select_named(INPUT_VARIANTS, args.variants, item_kind="variant")

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        rows = _run_benchmark_flat(
            corpus_module,
            torch_rs,
            reference_torch,
            cases,
            variants,
            args,
        )
        guard_sequences = _run_guard_sequences(corpus_module, torch_rs, reference_torch)
    finally:
        if gc_was_enabled:
            gc.enable()

    steady_ratios = []
    cold_ratios = []
    for row in rows:
        steady_ratios.append(row["ratios"]["steady_torch_rs_over_pytorch"])
        cold_ratios.append(row["ratios"]["cold_torch_rs_over_pytorch"])

    return {
        "environment": _environment(
            torch_rs,
            reference_torch,
            corpus_module.COMPILE_CORPUS_VERSION,
            args,
        ),
        "corpus": _corpus_metadata(corpus_module),
        "cases": rows,
        "recompilation_guard_sequences": guard_sequences,
        "aggregates": {
            "timed_supported_cell_count": len(rows),
            "recompilation_guard_sequence_count": len(guard_sequences),
            "recompilation_guard_step_count": sum(
                len(row["steps"]) for row in guard_sequences
            ),
            "steady_geomean_torch_rs_over_pytorch": _geomean(steady_ratios),
            "steady_geomean_capped_0_10_10_0": _geomean(
                [min(10.0, max(0.10, ratio)) for ratio in steady_ratios]
            ),
            "cold_geomean_torch_rs_over_pytorch": _geomean(cold_ratios),
            "cold_geomean_capped_0_10_10_0": _geomean(
                [min(10.0, max(0.10, ratio)) for ratio in cold_ratios]
            ),
        },
        "coverage_denominator": _coverage_denominator(corpus_module, cases),
    }


def _measure_one_pass(
    *,
    module,
    implementation,
    case,
    variant,
    corpus_module,
    warmups,
    samples,
):
    result = _run_cell(
        module=module,
        implementation=implementation,
        case=case,
        variant=variant,
        corpus_module=corpus_module,
        warmups=warmups,
        samples=samples,
    )
    return {
        "factory_us": result["factory_us"],
        "cold_first_call_us": result["cold_first_call_us"],
        "steady": result["steady"],
        "steady_checksums": result["steady_checksums"],
        "cold_checksum": result["cold_checksum"],
        "input_metadata": result["input_metadata"],
        "output_metadata": result["output_metadata"],
        "cold_output": result["cold_output"],
    }


def _run_benchmark_flat(corpus_module, torch_rs, reference_torch, cases, variants, args):
    pass_results = {}
    for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
        for implementation in order:
            module = torch_rs if implementation == "torch_rs" else reference_torch
            for case in cases:
                for variant in variants:
                    if not _variant_applies_to_case(variant, case):
                        continue
                    cell_key = f"{case.name}/{variant.name}"
                    measured = _measure_one_pass(
                        module=module,
                        implementation=implementation,
                        case=case,
                        variant=variant,
                        corpus_module=corpus_module,
                        warmups=args.warmups,
                        samples=args.samples,
                    )
                    pass_results.setdefault(cell_key, {}).setdefault(
                        implementation,
                        [],
                    ).append(measured)
                    expected_inputs = _cell_input_factory(case, variant)(module)
                    with _case_execution_context(module, case):
                        expected = case.program(*expected_inputs)
                    _assert_outputs_match(
                        measured["cold_output"],
                        expected,
                        cell_name=f"{cell_key}/{implementation}",
                    )
                    _assert_timing_checksums_match(
                        measured,
                        expected,
                        cell_name=f"{cell_key}/{implementation}",
                    )
                    if implementation == "torch_rs":
                        reference_inputs = _cell_input_factory(
                            case,
                            variant,
                        )(reference_torch)
                        with _case_execution_context(reference_torch, case):
                            reference_expected = case.program(*reference_inputs)
                        _assert_outputs_match(
                            measured["cold_output"],
                            reference_expected,
                            cell_name=cell_key,
                        )
                        _assert_timing_checksums_match(
                            measured,
                            reference_expected,
                            cell_name=cell_key,
                        )

    rows = []
    for case in cases:
        for variant in variants:
            if not _variant_applies_to_case(variant, case):
                continue
            cell_key = f"{case.name}/{variant.name}"
            cell = pass_results[cell_key]
            implementations = {}
            for implementation in ("torch_rs", "pytorch"):
                passes = cell[implementation]
                checksums = set()
                factory_us = []
                cold_us = []
                for measured in passes:
                    factory_us.append(measured["factory_us"])
                    cold_us.append(measured["cold_first_call_us"])
                    checksums.update(measured["steady_checksums"])
                    checksums.add(measured["cold_checksum"])
                steady_summaries = [measured["steady"] for measured in passes]
                median_values = [
                    steady_summary["median_us"] for steady_summary in steady_summaries
                ]
                mad_values = [
                    steady_summary["mad_us"] for steady_summary in steady_summaries
                ]
                variance_values = [
                    steady_summary["variance_us2"]
                    for steady_summary in steady_summaries
                ]
                implementations[implementation] = {
                    "factory_median_us": statistics.median(factory_us),
                    "cold_first_call_median_us": statistics.median(cold_us),
                    "cold_first_call_values_us": cold_us,
                    "steady_median_us": statistics.median(median_values),
                    "steady_mad_us": statistics.median(mad_values),
                    "steady_variance_us2": statistics.median(variance_values),
                    "steady_sample_count": sum(
                        steady_summary["sample_count"]
                        for steady_summary in steady_summaries
                    ),
                    "checksums": sorted(checksums),
                }
            torch_rs_steady = implementations["torch_rs"]["steady_median_us"]
            pytorch_steady = implementations["pytorch"]["steady_median_us"]
            torch_rs_cold = implementations["torch_rs"][
                "cold_first_call_median_us"
            ]
            pytorch_cold = implementations["pytorch"]["cold_first_call_median_us"]
            rows.append(
                {
                    "case": case.name,
                    "category": case.category,
                    "variant": variant.name,
                    "variant_category": variant.category,
                    "input_count": _program_input_count(case),
                    "description": variant.description,
                    "repeats": variant.repeats,
                    "input_metadata": cell["torch_rs"][0]["input_metadata"],
                    "output_metadata": cell["torch_rs"][0]["output_metadata"],
                    "implementations": implementations,
                    "ratios": {
                        "steady_torch_rs_over_pytorch": (
                            torch_rs_steady / pytorch_steady
                        ),
                        "cold_torch_rs_over_pytorch": torch_rs_cold / pytorch_cold,
                    },
                }
            )
    if not rows:
        raise SystemExit("no benchmark cells selected")
    return rows


def _format_tuple(values):
    return str(tuple(values))


def _format_metadata(metadata):
    return (
        f"shape {_format_tuple(metadata['shape'])}, "
        f"stride {_format_tuple(metadata['stride'])}, "
        f"offset {metadata['storage_offset']}, "
        f"{metadata['dtype']}, {metadata['device']}, "
        f"requires_grad={metadata['requires_grad']}"
    )


def _single_checksum(row):
    torch_rs_checksums = row["implementations"]["torch_rs"]["checksums"]
    pytorch_checksums = row["implementations"]["pytorch"]["checksums"]
    if torch_rs_checksums != pytorch_checksums:
        raise AssertionError(
            f"{row['case']}/{row['variant']} checksum mismatch: "
            f"torch_rs={torch_rs_checksums!r} pytorch={pytorch_checksums!r}"
        )
    if len(torch_rs_checksums) != 1:
        raise AssertionError(
            f"{row['case']}/{row['variant']} has unstable checksums: "
            f"{torch_rs_checksums!r}"
        )
    return torch_rs_checksums[0]


def _format_timed_cell(row):
    torch_rs = row["implementations"]["torch_rs"]
    pytorch = row["implementations"]["pytorch"]
    return (
        f"| `{row['case']}` | `{row['variant']}` | {row['input_count']} | "
        f"{row['repeats']} | {_format_metadata(row['output_metadata'])} | "
        f"{torch_rs['cold_first_call_median_us']:.3f} | "
        f"{pytorch['cold_first_call_median_us']:.3f} | "
        f"{row['ratios']['cold_torch_rs_over_pytorch']:.3f}x | "
        f"{torch_rs['steady_median_us']:.3f} +/- "
        f"{torch_rs['steady_mad_us']:.3f} | "
        f"{pytorch['steady_median_us']:.3f} +/- "
        f"{pytorch['steady_mad_us']:.3f} | "
        f"{row['ratios']['steady_torch_rs_over_pytorch']:.3f}x | "
        f"`{_single_checksum(row)}` |"
    )


def _guard_error_name(step):
    return step.get("error_type", "").rsplit(".", 1)[-1]


def _format_guard_steps(steps):
    rendered = []
    for step in steps:
        if step["status"] == "expected_error":
            rendered.append(
                f"{step['step']} expected_error("
                f"{step['guard_change']}: {_guard_error_name(step)})"
            )
        else:
            rendered.append(f"{step['step']} ok({step['guard_change']})")
    return "; ".join(rendered)


def _format_guard_sequence(row):
    total_us = sum(step["elapsed_us"] for step in row["steps"])
    return (
        f"| `{row['scenario']}` | `{','.join(row['order'])}` | "
        f"`{row['implementation']}` | {row['recompile_limit']} | "
        f"{_format_guard_steps(row['steps'])} | {total_us:.3f} |"
    )


def _timed_category_counts(cases):
    counts = Counter(row["category"] for row in cases)
    ordered_categories = [
        category
        for category in CATEGORY_LABELS
        if category in counts
    ]
    ordered_categories.extend(
        sorted(category for category in counts if category not in CATEGORY_LABELS)
    )
    return ", ".join(
        f"{counts[category]} {CATEGORY_LABELS.get(category, category)}"
        for category in ordered_categories
    )


def _human_join(items):
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _supported_category_summary(coverage_denominator):
    categories = [
        CATEGORY_PHRASES.get(category["category"], category["category"])
        for category in coverage_denominator["supported_categories"]
    ]
    return _human_join(categories)


def _held_out_summary(report):
    corpus = report.get("corpus", {})
    category_weights = report.get("coverage_denominator", {}).get(
        "category_weights",
        {},
    )
    held_out_cases = corpus.get("held_out_cases", ())
    held_out_counts = Counter(case.get("category") for case in held_out_cases)
    ordered_categories = [
        category
        for category in category_weights
        if category in held_out_counts
    ]
    ordered_categories.extend(
        sorted(
            category
            for category in held_out_counts
            if category not in category_weights
        )
    )

    parts = []
    for category in ordered_categories:
        count = held_out_counts[category]
        program_word = "program" if count == 1 else "programs"
        phrase = CATEGORY_PROGRAM_LABELS.get(category, category)
        parts.append(f"{count} held-out {phrase} {program_word}")

    guard_count = len(corpus.get("held_out_recompilation_guard_scenarios", ()))
    if guard_count:
        scenario_word = "scenario" if guard_count == 1 else "scenarios"
        parts.append(f"{guard_count} held-out recompilation-guard {scenario_word}")

    if not parts:
        return "No held-out compile corpus cases are recorded in this artifact."
    return (
        f"The {corpus.get('version', 'compile')} corpus also keeps "
        f"{_human_join(parts)} in tests to guard "
        "against case-specific specialization; they are not included in the "
        "public timing table."
    )


def _supported_denominator_line(coverage_denominator):
    return (
        f"Supported category weight: {coverage_denominator['supported_weight']} / "
        f"{coverage_denominator['total_weight']}. Zero-credit unsupported category "
        f"weight: {coverage_denominator['zero_credit_weight']} / "
        f"{coverage_denominator['total_weight']}."
    )


def render_markdown_summary(report):
    cases = report["cases"]
    guard_sequences = report["recompilation_guard_sequences"]
    aggregates = report["aggregates"]
    coverage_denominator = report["coverage_denominator"]
    environment = report["environment"]
    statuses = sorted(
        {
            step["status"]
            for sequence in guard_sequences
            for step in sequence["steps"]
        }
    )

    lines = [
        "## Aggregate",
        "",
        f"- Raw JSON artifact: `{DEFAULT_ARTIFACT_PATH.relative_to(REPOSITORY_ROOT)}`",
        (
            f"- Benchmark/corpus: `{environment['benchmark_version']}` / "
            f"`{environment['corpus_version']}`"
        ),
        (
            "- Cold first compiled call: "
            f"{aggregates['cold_geomean_torch_rs_over_pytorch']:.3f}x uncapped, "
            f"{aggregates['cold_geomean_capped_0_10_10_0']:.3f}x capped"
        ),
        (
            "- Steady-state materialized compiled call: "
            f"{aggregates['steady_geomean_torch_rs_over_pytorch']:.3f}x uncapped, "
            f"{aggregates['steady_geomean_capped_0_10_10_0']:.3f}x capped"
        ),
        (
            f"- Timed supported cells: {aggregates['timed_supported_cell_count']} "
            f"({_timed_category_counts(cases)})"
        ),
        (
            "- Recompilation guard sequences: "
            f"{aggregates['recompilation_guard_sequence_count']} rows, "
            f"{aggregates['recompilation_guard_step_count']} checked steps, "
            f"statuses {', '.join(statuses)}"
        ),
        (
            "- Versioned denominator coverage: "
            f"{coverage_denominator['weighted_supported_percent']:.1f}% supported "
            "by native compile cases, "
            f"{coverage_denominator['zero_credit_weight']}% zero-credit unsupported "
            "category weight"
        ),
        "",
        "## Supported Timed Cells",
        "",
        (
            "| Program | Input variant | Inputs | Repeats | Output metadata | "
            "`torch_rs` cold us | PyTorch cold us | Cold ratio | "
            "`torch_rs` steady us +/- MAD | PyTorch steady us +/- MAD | "
            "Steady ratio | Checksum |"
        ),
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_format_timed_cell(row) for row in cases)
    lines.extend(
        [
            "",
            "## Recompilation Guard Sequences",
            "",
            (
                "These rows are behavioral evidence, not throughput cells. Each "
                "scenario runs once per implementation and once per implementation "
                "order. Steps marked `expected_error` are required fullgraph "
                "`recompile_limit` failures; the following cached call and reset "
                "call verify bounded-cache and reset semantics."
            ),
            "",
            "| Scenario | Order | Implementation | Limit | Steps | Total us |",
            "| --- | --- | --- | ---: | --- | ---: |",
        ]
    )
    lines.extend(_format_guard_sequence(row) for row in guard_sequences)
    lines.extend(
        [
            "",
            "## Zero-Credit Unsupported Denominator",
            "",
            (
                "The compile corpus keeps the full 100-point category denominator. "
                "The native `torch_rs` path currently has executable public cases "
                f"for {_supported_category_summary(coverage_denominator)}. "
                "Every remaining category below stays in the denominator as zero "
                "credit instead of being dropped from the report."
            ),
            "",
            "| Category | Weight | Accounting |",
            "| --- | ---: | --- |",
        ]
    )
    for category in coverage_denominator["supported_categories"]:
        timed_public_cases = ", ".join(
            f"`{name}`" for name in category["timed_public_cases"]
        )
        lines.append(
            f"| `{category['category']}` | {category['weight']} | "
            f"Supported and timed public cases: {timed_public_cases} |"
        )
    for category in coverage_denominator["zero_credit_categories"]:
        lines.append(
            f"| `{category['category']}` | {category['weight']} | "
            f"Zero credit: {category['reason']} |"
        )
    lines.extend(
        [
            "",
            _supported_denominator_line(coverage_denominator),
            _held_out_summary(report),
            "",
        ]
    )
    return "\n".join(lines)


def _load_artifact(path):
    with _input_path(path).open(encoding="utf-8") as artifact_file:
        return json.load(artifact_file)


def _markdown_summary(markdown_path):
    markdown = _input_path(markdown_path).read_text(encoding="utf-8")
    marker = "## Aggregate"
    try:
        return markdown[markdown.index(marker) :]
    except ValueError:
        raise AssertionError(f"{markdown_path} is missing {marker!r}") from None


def _validate_expected_artifact_shape(report):
    errors = []
    corpus_module = _load_compile_corpus_module()
    expected_corpus = _corpus_metadata(corpus_module)
    expected_public_cases = tuple(corpus_module.COMPILE_CORPUS)
    expected_coverage = _coverage_denominator(corpus_module, expected_public_cases)
    expected_category_counts = Counter()
    for case in expected_public_cases:
        for variant in INPUT_VARIANTS:
            if _variant_applies_to_case(variant, case):
                expected_category_counts[case.category] += 1
    expected_timed_cell_count = sum(expected_category_counts.values())

    environment = report.get("environment", {})
    if environment.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(
            "benchmark version mismatch: "
            f"{environment.get('benchmark_version')!r} != {BENCHMARK_VERSION!r}"
        )
    expected_corpus_version = getattr(corpus_module, "COMPILE_CORPUS_VERSION", None)
    if environment.get("corpus_version") != expected_corpus_version:
        errors.append(
            "corpus version mismatch: "
            f"{environment.get('corpus_version')!r} != {expected_corpus_version!r}"
        )

    cases = report.get("cases", [])
    category_counts = Counter(row.get("category") for row in cases)
    if (
        len(cases) != expected_timed_cell_count
        or category_counts != expected_category_counts
    ):
        errors.append(
            "timed cell count mismatch: "
            f"count={len(cases)} categories={dict(category_counts)!r}, "
            f"expected count={expected_timed_cell_count} "
            f"categories={dict(expected_category_counts)!r}"
        )
    if report.get("aggregates", {}).get("timed_supported_cell_count") != len(cases):
        errors.append("aggregate timed cell count does not match cases")

    guard_sequences = report.get("recompilation_guard_sequences", [])
    guard_step_count = sum(len(row.get("steps", ())) for row in guard_sequences)
    if len(guard_sequences) != 12 or guard_step_count != 60:
        errors.append(
            "guard sequence coverage mismatch: "
            f"rows={len(guard_sequences)} steps={guard_step_count}"
        )
    aggregates = report.get("aggregates", {})
    if aggregates.get("recompilation_guard_sequence_count") != len(guard_sequences):
        errors.append("aggregate guard sequence count does not match rows")
    if aggregates.get("recompilation_guard_step_count") != guard_step_count:
        errors.append("aggregate guard step count does not match rows")

    coverage = report.get("coverage_denominator", {})
    if coverage != expected_coverage:
        errors.append(
            "coverage denominator mismatch: "
            f"actual={coverage!r} expected={expected_coverage!r}"
        )

    corpus = report.get("corpus", {})
    if corpus != expected_corpus:
        errors.append(
            "corpus metadata mismatch: "
            f"actual={corpus!r} expected={expected_corpus!r}"
        )
    elif corpus.get("version") != environment.get("corpus_version"):
        errors.append("corpus metadata version does not match environment")

    for row in cases:
        try:
            _single_checksum(row)
        except AssertionError as error:
            errors.append(str(error))
    for sequence in guard_sequences:
        for step in sequence.get("steps", ()):
            if step.get("status") == "ok" and not step.get("checksum"):
                errors.append(
                    f"{sequence.get('scenario')}/{step.get('step')} missing checksum"
                )
            if step.get("status") == "expected_error" and not step.get("error_type"):
                errors.append(
                    f"{sequence.get('scenario')}/{step.get('step')} missing error type"
                )

    if errors:
        raise AssertionError("\n".join(errors))


def validate_artifact(artifact_path, markdown_path):
    report = _load_artifact(artifact_path)
    _validate_expected_artifact_shape(report)
    expected_summary = render_markdown_summary(report)
    actual_summary = _markdown_summary(markdown_path)
    if actual_summary != expected_summary:
        raise AssertionError(
            "markdown summary does not match raw benchmark artifact; "
            "regenerate the report summary from the checked-in JSON"
        )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--require-single-cpu-affinity", action="store_true")
    parser.add_argument("--cases", nargs="*", default=())
    parser.add_argument("--variants", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--render-markdown-summary",
        type=Path,
        metavar="RAW_JSON",
        help="render the markdown summary section for a benchmark JSON artifact",
    )
    parser.add_argument(
        "--validate-artifact",
        nargs="?",
        const=DEFAULT_ARTIFACT_PATH,
        type=Path,
        metavar="RAW_JSON",
        help="validate benchmark JSON and its rendered markdown summary",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT_PATH,
        help="markdown report to validate with --validate-artifact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if (
        args.render_markdown_summary is not None
        and args.validate_artifact is not None
    ):
        raise SystemExit(
            "--render-markdown-summary cannot be combined with --validate-artifact"
        )
    if args.render_markdown_summary is not None:
        print(
            render_markdown_summary(_load_artifact(args.render_markdown_summary)),
            end="",
        )
        return
    if args.validate_artifact is not None:
        validate_artifact(args.validate_artifact, args.markdown_report)
        return

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
