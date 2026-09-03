#!/usr/bin/env python3
"""Benchmark the supported CPU ``torch.compile`` eager/fullgraph corpus."""

from __future__ import annotations

import argparse
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
REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "torch_compile_cpu_eager_benchmark_v1"
DEFAULT_WARMUPS = 7
DEFAULT_SAMPLES = 31
IMPLEMENTATION_ORDERS = (
    ("torch_rs", "pytorch"),
    ("pytorch", "torch_rs"),
)


@dataclass(frozen=True)
class InputVariant:
    name: str
    category: str
    description: str
    repeats: int
    make_inputs: object


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


def _transposed_input(base_shape):
    def make_inputs(module):
        base = module.tensor(_nested_values(base_shape), dtype=module.float32)
        return (base.transpose(0, 1),)

    return make_inputs


INPUT_VARIANTS = (
    InputVariant(
        "case_default",
        "corpus_default",
        "case's checked-in input factory",
        256,
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
        "implementation_orders": IMPLEMENTATION_ORDERS,
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


def _time_once(compiled, inputs, module):
    started_ns = time.perf_counter_ns()
    output = compiled(*inputs)
    _synchronize(module)
    checksum = _checksum_tensor(output)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, checksum, output


def _time_repeated(compiled, inputs, module, repeats):
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
    make_inputs = case.make_inputs if variant.make_inputs is None else variant.make_inputs
    inputs = make_inputs(module)
    if len(inputs) != 1:
        raise AssertionError(f"{case.name} returned {len(inputs)} inputs, expected 1")

    _reset_compile_state(module, implementation, corpus_module)
    compile_started_ns = time.perf_counter_ns()
    compiled = module.compile(case.program, backend="eager", fullgraph=True)
    factory_ns = time.perf_counter_ns() - compile_started_ns
    cold_ns, cold_checksum, cold_output = _time_once(compiled, inputs, module)

    for _ in range(warmups):
        _time_repeated(compiled, inputs, module, variant.repeats)

    sample_ns = []
    sample_checksums = []
    for _ in range(samples):
        elapsed_ns, checksum = _time_repeated(
            compiled,
            inputs,
            module,
            variant.repeats,
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


def _geomean(values):
    if not values:
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


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
        raise SystemExit(f"output path must stay inside the worktree: {resolved}") from None
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
        "cases": rows,
        "aggregates": {
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
                    expected_inputs = (
                        case.make_inputs(module)
                        if variant.make_inputs is None
                        else variant.make_inputs(module)
                    )
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
                        reference_inputs = (
                            case.make_inputs(reference_torch)
                            if variant.make_inputs is None
                            else variant.make_inputs(reference_torch)
                        )
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
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cases", nargs="*", default=())
    parser.add_argument("--variants", nargs="*", default=())
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
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
