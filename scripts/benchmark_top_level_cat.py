#!/usr/bin/env python3
"""Benchmark supported eager CPU 1-D ``torch.cat`` cells against PyTorch."""

from __future__ import annotations

import argparse
from collections import Counter
import contextlib
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
DEFAULT_ARTIFACT_PATH = (
    REPOSITORY_ROOT / "docs" / "benchmark-data" / "top-level-cat-release-timings.json"
)
DEFAULT_MARKDOWN_REPORT_PATH = (
    REPOSITORY_ROOT / "docs" / "top-level-cat-release-timings.md"
)
PROTECTED_OUTPUT_PATHS = {
    REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
    REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
}
REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "top_level_cat_cpu_1d_benchmark_v1"
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
APIS = ("cat", "concat", "concatenate")

MODE_EAGER = "eager"
MODE_NO_GRAD = "no_grad"


@dataclass(frozen=True)
class Operands:
    tensors: object
    kwargs: dict[str, object]


@dataclass(frozen=True)
class Workload:
    name: str
    category: str
    call_form: str
    input_description: str
    output_description: str
    repeats: int
    mode: str
    input_seeds: tuple[int, ...]
    make_operands: object


@dataclass(frozen=True)
class UnsupportedCell:
    name: str
    category: str
    make_call: object
    error_type: str
    message: str


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


def _file_sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
        if requested_cpu is not None:
            raise SystemExit("CPU affinity pinning requires os.sched_setaffinity")
        return {
            "requested_cpu": requested_cpu,
            "selected_cpu": None,
            "initial_affinity": None,
            "pinned_affinity": None,
        }
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


def _validate_reference_version(reference_torch):
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise SystemExit(
            "top-level cat benchmark requires pinned PyTorch "
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


def _shape_product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _values(np, shape, seed, *, scale=0.03125, bias=0.0):
    rng = np.random.default_rng(seed)
    if _shape_product(shape) == 0:
        return np.zeros(shape, dtype=np.float32)
    raw = rng.integers(-4096, 4097, size=shape, dtype=np.int32)
    values = raw.astype(np.float32) * np.float32(scale)
    if bias:
        values = values + np.float32(bias)
    return values


def _tensor_from_array(module, array, *, requires_grad=False):
    kwargs = {"dtype": module.float32, "requires_grad": requires_grad}
    if array.shape == ():
        return module.tensor(float(array.reshape(()).item()), **kwargs)
    return module.tensor(array.tolist(), **kwargs)


def _dense_tensor(module, np, shape, seed, *, requires_grad=False, bias=0.0):
    return _tensor_from_array(
        module,
        _values(np, shape, seed, bias=bias),
        requires_grad=requires_grad,
    )


def _make_singleton_contiguous(module, np):
    return Operands(
        [_dense_tensor(module, np, (8192,), 2026090701)],
        {"dim": 0},
    )


def _make_multi_input_contiguous(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (257,), 2026090702),
            _dense_tensor(module, np, (263,), 2026090703, bias=0.25),
            _dense_tensor(module, np, (269,), 2026090704, bias=-0.5),
        ],
        {"dim": 0},
    )


def _make_empty_operand_middle(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (1024,), 2026090705),
            _dense_tensor(module, np, (0,), 2026090706),
            _dense_tensor(module, np, (511,), 2026090707, bias=0.125),
        ],
        {"dim": 0},
    )


def _make_all_empty_tuple_dim_negative_one(module, np):
    return Operands(
        (
            _dense_tensor(module, np, (0,), 2026090708),
            _dense_tensor(module, np, (0,), 2026090709),
        ),
        {"dim": -1},
    )


def _make_offset_contiguous_views(module, np):
    left_base = _dense_tensor(module, np, (3, 4096), 2026090710)
    right_base = _dense_tensor(module, np, (3, 4096), 2026090711, bias=0.5)
    return Operands([left_base[1], right_base[2]], {"dim": 0})


def _make_noncontiguous_stride2_views(module, np):
    left_base = _dense_tensor(module, np, (4096, 2), 2026090712)
    right_base = _dense_tensor(module, np, (4096, 2), 2026090713, bias=-0.25)
    return Operands(
        [left_base.transpose(0, 1)[1], right_base.transpose(0, 1)[0]],
        {"dim": 0},
    )


def _make_tuple_dim_negative_one(module, np):
    return Operands(
        (
            _dense_tensor(module, np, (513,), 2026090714),
            _dense_tensor(module, np, (509,), 2026090715, bias=0.75),
        ),
        {"dim": -1},
    )


def _make_axis_keyword(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (17,), 2026090716),
            _dense_tensor(module, np, (19,), 2026090717, bias=1.0),
        ],
        {"axis": 0},
    )


def _make_no_grad_grad_inputs(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (257,), 2026090718, requires_grad=True),
            _dense_tensor(
                module,
                np,
                (263,),
                2026090719,
                requires_grad=True,
                bias=-0.375,
            ),
        ],
        {"dim": 0},
    )


def _make_active_autograd_1d(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (257,), 2026090720, requires_grad=True),
            _dense_tensor(
                module,
                np,
                (263,),
                2026090721,
                requires_grad=True,
                bias=0.625,
            ),
        ],
        {"dim": 0},
    )


WORKLOADS = (
    Workload(
        "singleton_contiguous_8192",
        "singleton",
        "list dim=0",
        "one contiguous input (8192,), stride (1,)",
        "concatenation output",
        256,
        MODE_EAGER,
        (2026090701,),
        _make_singleton_contiguous,
    ),
    Workload(
        "multi_input_contiguous_257_263_269",
        "multi-input",
        "list dim=0",
        "three contiguous inputs with lengths 257, 263, and 269",
        "concatenation output",
        1024,
        MODE_EAGER,
        (2026090702, 2026090703, 2026090704),
        _make_multi_input_contiguous,
    ),
    Workload(
        "empty_operand_middle_1024_0_511",
        "empty operand",
        "list dim=0",
        "contiguous inputs with lengths 1024, 0, and 511",
        "concatenation output",
        512,
        MODE_EAGER,
        (2026090705, 2026090706, 2026090707),
        _make_empty_operand_middle,
    ),
    Workload(
        "all_empty_tuple_dim_negative_one",
        "empty operand",
        "tuple dim=-1",
        "two empty 1-D inputs with length 0",
        "empty concatenation output",
        5000,
        MODE_EAGER,
        (2026090708, 2026090709),
        _make_all_empty_tuple_dim_negative_one,
    ),
    Workload(
        "offset_contiguous_views_4096",
        "offset",
        "list dim=0",
        "left/right tensor((3, 4096))[1/2] -> (4096,), stride (1,), nonzero offsets",
        "concatenation output",
        128,
        MODE_EAGER,
        (2026090710, 2026090711),
        _make_offset_contiguous_views,
    ),
    Workload(
        "noncontiguous_stride2_views_4096",
        "noncontiguous",
        "list dim=0",
        "left/right tensor((4096, 2)).transpose(0, 1)[1/0] -> (4096,), stride (2,)",
        "concatenation output",
        128,
        MODE_EAGER,
        (2026090712, 2026090713),
        _make_noncontiguous_stride2_views,
    ),
    Workload(
        "tuple_dim_negative_one_513_509",
        "multi-input",
        "tuple dim=-1",
        "two contiguous tuple inputs with lengths 513 and 509",
        "concatenation output",
        1024,
        MODE_EAGER,
        (2026090714, 2026090715),
        _make_tuple_dim_negative_one,
    ),
    Workload(
        "axis_keyword_17_19",
        "axis keyword",
        "list axis=0",
        "two contiguous inputs with lengths 17 and 19",
        "concatenation output",
        5000,
        MODE_EAGER,
        (2026090716, 2026090717),
        _make_axis_keyword,
    ),
    Workload(
        "no_grad_grad_inputs_257_263",
        "no_grad",
        "list dim=0",
        "two grad-requiring contiguous inputs with lengths 257 and 263 inside no_grad",
        "concatenation output",
        1024,
        MODE_NO_GRAD,
        (2026090718, 2026090719),
        _make_no_grad_grad_inputs,
    ),
    Workload(
        "active_autograd_1d_257_263",
        "active autograd",
        "list dim=0",
        "two grad-requiring contiguous inputs with lengths 257 and 263; forward construction only",
        "concatenation output",
        512,
        MODE_EAGER,
        (2026090720, 2026090721),
        _make_active_autograd_1d,
    ),
)


def _unsupported_out_1d(module, api):
    output = module.zeros((3,), dtype=module.float32)
    return getattr(module, api)(
        [
            module.tensor([1.0], dtype=module.float32),
            module.tensor([2.0, 3.0], dtype=module.float32),
        ],
        dim=0,
        out=output,
    )


def _unsupported_rank3_dim0(module, api):
    return getattr(module, api)(
        [
            module.ones((2, 2, 3), dtype=module.float32),
            module.zeros((1, 2, 3), dtype=module.float32),
        ],
        dim=0,
    )


def _unsupported_rank3_dim2(module, api):
    return getattr(module, api)(
        [
            module.ones((2, 3, 2), dtype=module.float32),
            module.zeros((2, 3, 1), dtype=module.float32),
        ],
        dim=2,
    )


UNSUPPORTED_CELLS = (
    UnsupportedCell(
        "out_1d",
        "concrete out",
        _unsupported_out_1d,
        "RuntimeError",
        "cat(): the 'out' argument is not supported",
    ),
    UnsupportedCell(
        "rank3_dim0",
        "rank >=3 cat",
        _unsupported_rank3_dim0,
        "NotImplementedError",
        "cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported",
    ),
    UnsupportedCell(
        "rank3_dim2",
        "rank >=3 cat",
        _unsupported_rank3_dim2,
        "NotImplementedError",
        "cat(): only exact native CPU float32 rank-1 or rank-2 Tensor inputs are supported",
    ),
)


def _is_tensor(value):
    return all(hasattr(value, attr) for attr in ("shape", "stride", "storage_offset"))


def _operand_tensors(operands):
    tensors = []
    if isinstance(operands.tensors, (list, tuple)):
        for index, value in enumerate(operands.tensors):
            if _is_tensor(value):
                tensors.append((f"tensors[{index}]", value))
    elif _is_tensor(operands.tensors):
        tensors.append(("tensors", operands.tensors))
    return tuple(tensors)


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
    materialized = tensor
    detach = getattr(materialized, "detach", None)
    if callable(detach):
        materialized = detach()
    cpu = getattr(materialized, "cpu", None)
    if callable(cpu):
        materialized = cpu()
    return np.ascontiguousarray(np.asarray(materialized, dtype=np.float32))


def _tensor_value_bits(np, tensor):
    return _tensor_array(np, tensor).reshape(-1).view(np.uint32).tolist()


def _checksum_payload(payload):
    encoded = json.dumps(
        payload,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return str(int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big"))


def _checksum_tensor(np, tensor):
    array = _tensor_array(np, tensor)
    metadata = _tensor_metadata(tensor)
    hasher = hashlib.blake2b(digest_size=8)
    hasher.update(
        json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    hasher.update(b"\0")
    hasher.update(array.reshape(-1).view(np.uint8).tobytes())
    return str(int.from_bytes(hasher.digest(), "big"))


def _checksum_bundle(np, bundle):
    return _checksum_payload(
        [
            {
                "label": label,
                "metadata": _tensor_metadata(tensor),
                "checksum": _checksum_tensor(np, tensor),
            }
            for label, tensor in bundle
        ]
    )


def _roll_checksum(previous, checksum):
    return _checksum_payload({"previous": previous, "checksum": checksum})


def _bundle_metadata(bundle):
    return [
        {
            "label": label,
            **_tensor_metadata(tensor),
        }
        for label, tensor in bundle
    ]


def _assert_tensors_match(np, actual, expected, *, cell_name, label):
    actual_metadata = _tensor_metadata(actual)
    expected_metadata = _tensor_metadata(expected)
    if actual_metadata != expected_metadata:
        raise AssertionError(
            f"{cell_name}/{label} metadata mismatch:\n"
            f"actual={actual_metadata!r}\nexpected={expected_metadata!r}"
        )

    actual_bits = _tensor_value_bits(np, actual)
    expected_bits = _tensor_value_bits(np, expected)
    if actual_bits != expected_bits:
        raise AssertionError(
            f"{cell_name}/{label} value bits mismatch:\n"
            f"actual={actual_bits!r}\nexpected={expected_bits!r}"
        )


def _assert_bundles_match(np, actual, expected, *, cell_name):
    if [label for label, _ in actual] != [label for label, _ in expected]:
        raise AssertionError(
            f"{cell_name} materialized labels mismatch: "
            f"actual={[label for label, _ in actual]!r} "
            f"expected={[label for label, _ in expected]!r}"
        )
    for (actual_label, actual_tensor), (expected_label, expected_tensor) in zip(
        actual, expected
    ):
        _assert_tensors_match(
            np,
            actual_tensor,
            expected_tensor,
            cell_name=cell_name,
            label=actual_label or expected_label,
        )


def _execute_operation(module, api, workload, operands):
    if workload.mode == MODE_NO_GRAD:
        context = module.no_grad()
    else:
        context = contextlib.nullcontext()
    with context:
        output = getattr(module, api)(operands.tensors, **operands.kwargs)
    return (("output", output),)


def _time_block(np, module, api, workload, static_operands, repeats):
    started_ns = time.perf_counter_ns()
    last_bundle = None
    for _ in range(repeats):
        last_bundle = _execute_operation(module, api, workload, static_operands)
    _synchronize(module)
    elapsed_ns = time.perf_counter_ns() - started_ns
    checksum = _checksum_bundle(np, last_bundle)
    return elapsed_ns, checksum, last_bundle


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
        "samples_us": samples_us,
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _measure_one_pass(np, module, implementation, api, workload, args):
    static_operands = workload.make_operands(module, np)
    input_metadata = [
        {"label": label, **_tensor_metadata(tensor)}
        for label, tensor in _operand_tensors(static_operands)
    ]
    input_checksums_before = [
        {"label": label, "checksum": _checksum_tensor(np, tensor)}
        for label, tensor in _operand_tensors(static_operands)
    ]

    cold_ns, cold_checksum, cold_bundle = _time_block(
        np,
        module,
        api,
        workload,
        static_operands,
        1,
    )
    warmup_checksums = []
    warmup_sink = "0"
    for _ in range(args.warmups):
        _, checksum, _ = _time_block(
            np,
            module,
            api,
            workload,
            static_operands,
            workload.repeats,
        )
        warmup_checksums.append(checksum)
        warmup_sink = _roll_checksum(warmup_sink, checksum)

    sample_ns = []
    sample_checksums = []
    sample_sink = "0"
    last_bundle = cold_bundle
    for _ in range(args.samples):
        elapsed_ns, checksum, last_bundle = _time_block(
            np,
            module,
            api,
            workload,
            static_operands,
            workload.repeats,
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)
        sample_sink = _roll_checksum(sample_sink, checksum)

    input_checksums_after = [
        {"label": label, "checksum": _checksum_tensor(np, tensor)}
        for label, tensor in _operand_tensors(static_operands)
    ]
    if input_checksums_after != input_checksums_before:
        raise AssertionError(
            f"{workload.name}/{api}/{implementation} mutated benchmark operands: "
            f"before={input_checksums_before!r} after={input_checksums_after!r}"
        )

    return {
        "cold_first_call_us": cold_ns / 1000.0,
        "cold_checksum": cold_checksum,
        "warmup_checksums": sorted(set(warmup_checksums)),
        "warmup_checksum_sink": warmup_sink,
        "steady": _summarize_samples(sample_ns, workload.repeats),
        "steady_checksums": sorted(set(sample_checksums)),
        "steady_checksum_sink": sample_sink,
        "input_metadata": input_metadata,
        "input_checksums": input_checksums_before,
        "output_metadata": _bundle_metadata(last_bundle),
        "cold_bundle": cold_bundle,
        "operand_nonmutation_checked": True,
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


def _select_apis(selected_names):
    if not selected_names:
        return APIS
    missing = [name for name in selected_names if name not in APIS]
    if missing:
        raise SystemExit(
            f"unknown API: {', '.join(missing)}. Available: {', '.join(APIS)}"
        )
    return tuple(selected_names)


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


def _input_path(path):
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError:
        raise SystemExit(f"input path must stay inside the worktree: {resolved}") from None
    return resolved


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
        "input_generation": {
            "policy": "deterministic numpy default_rng seeds embedded per workload",
            "seed_values": sorted(
                {seed for workload in WORKLOADS for seed in workload.input_seeds}
            ),
        },
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
        "driver": {
            "path": str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT)),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "warmups": args.warmups,
        "samples": args.samples,
        "threads": args.threads,
        "implementation_orders": [list(order) for order in IMPLEMENTATION_ORDERS],
        "apis": list(_select_apis(args.apis)),
        "workloads": list(args.workloads) if args.workloads else "all",
    }


def _format_tuple(values):
    return str(tuple(values))


def _format_metadata(metadata):
    return (
        f"{_format_tuple(metadata['shape'])}, "
        f"stride {_format_tuple(metadata['stride'])}, "
        f"offset {metadata['storage_offset']}, "
        f"{metadata['dtype']}, {metadata['device']}, "
        f"requires_grad={metadata['requires_grad']}, "
        f"leaf={metadata['is_leaf']}"
    )


def _run_status(np, module, api, unsupported_cell):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            output = unsupported_cell.make_call(module, api)
    except Exception as error:
        return {
            "kind": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "status": f"{type(error).__name__}: {error}",
        }
    if _is_tensor(output):
        metadata = _tensor_metadata(output)
        checksum = _checksum_bundle(np, (("output", output),))
        return {
            "kind": "supported",
            "metadata": metadata,
            "checksum": checksum,
            "status": f"supported {_format_metadata(metadata)}",
        }
    return {
        "kind": "supported",
        "value": repr(output),
        "status": f"supported non-tensor {output!r}",
    }


def _run_unsupported_cells(np, torch_rs, reference_torch, apis):
    rows = []
    for api in apis:
        for unsupported_cell in UNSUPPORTED_CELLS:
            cell_name = f"top_level_torch_{api}_{unsupported_cell.name}"
            torch_rs_status = _run_status(np, torch_rs, api, unsupported_cell)
            if (
                torch_rs_status["kind"] != "error"
                or torch_rs_status["error_type"] != unsupported_cell.error_type
                or torch_rs_status["message"] != unsupported_cell.message
            ):
                raise AssertionError(
                    f"{cell_name} torch_rs status mismatch:\n"
                    f"actual={torch_rs_status!r}\n"
                    f"expected type={unsupported_cell.error_type!r} "
                    f"message={unsupported_cell.message!r}"
                )

            pytorch_status = _run_status(np, reference_torch, api, unsupported_cell)
            if pytorch_status["kind"] != "supported":
                raise AssertionError(
                    f"{cell_name} PyTorch status mismatch: {pytorch_status!r}"
                )

            rows.append(
                {
                    "name": cell_name,
                    "api": f"torch.{api}",
                    "category": unsupported_cell.category,
                    "torch_rs": torch_rs_status,
                    "pytorch": pytorch_status,
                    "credit": "zero",
                    "reason": "torch_rs does not support the equivalent PyTorch cell",
                    "validation": {
                        "torch_rs_error_checked": True,
                        "pytorch_supported_checked": True,
                    },
                }
            )
    return rows


def _expected_bundle(np, reference_torch, api, workload):
    operands = workload.make_operands(reference_torch, np)
    return _execute_operation(reference_torch, api, workload, operands)


def _run_supported_cells(np, torch_rs, reference_torch, workloads, apis, args):
    rows = []
    for api in apis:
        for workload in workloads:
            cell_name = f"torch.{api}/{workload.name}"
            expected = _expected_bundle(np, reference_torch, api, workload)
            expected_checksum = _checksum_bundle(np, expected)
            pass_results = {"torch_rs": [], "pytorch": []}

            for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
                for implementation in order:
                    module = torch_rs if implementation == "torch_rs" else reference_torch
                    measured = _measure_one_pass(
                        np,
                        module,
                        implementation,
                        api,
                        workload,
                        args,
                    )
                    _assert_bundles_match(
                        np,
                        measured["cold_bundle"],
                        expected,
                        cell_name=f"{cell_name}/{implementation}/cold",
                    )
                    expected_by_key = {
                        "warmup_checksums": []
                        if args.warmups == 0
                        else [expected_checksum],
                        "steady_checksums": [expected_checksum],
                    }
                    for key, expected_checksums in expected_by_key.items():
                        if measured[key] != expected_checksums:
                            raise AssertionError(
                                f"{cell_name}/{implementation} {key} mismatch "
                                f"or instability: actual={measured[key]!r} "
                                f"expected={expected_checksums!r}"
                            )

                    pass_record = {
                        "order_index": order_index,
                        "order": list(order),
                        "cold_first_call_us": measured["cold_first_call_us"],
                        "cold_checksum": measured["cold_checksum"],
                        "warmup_checksums": measured["warmup_checksums"],
                        "warmup_checksum_sink": measured["warmup_checksum_sink"],
                        "steady": measured["steady"],
                        "steady_checksums": measured["steady_checksums"],
                        "steady_checksum_sink": measured["steady_checksum_sink"],
                        "input_metadata": measured["input_metadata"],
                        "input_checksums": measured["input_checksums"],
                        "output_metadata": measured["output_metadata"],
                        "operand_nonmutation_checked": measured[
                            "operand_nonmutation_checked"
                        ],
                    }
                    pass_results[implementation].append(pass_record)

            implementations = {}
            for implementation, passes in pass_results.items():
                medians = [item["steady"]["median_us"] for item in passes]
                mads = [item["steady"]["mad_us"] for item in passes]
                variances = [item["steady"]["variance_us2"] for item in passes]
                cold_values = [item["cold_first_call_us"] for item in passes]
                checksums = sorted(
                    {
                        checksum
                        for item in passes
                        for checksum in (
                            item["steady_checksums"]
                            + item["warmup_checksums"]
                            + [item["cold_checksum"]]
                        )
                    }
                )
                implementations[implementation] = {
                    "cold_first_call_median_us": statistics.median(cold_values),
                    "cold_first_call_values_us": cold_values,
                    "steady_median_us": statistics.median(medians),
                    "steady_mad_us": statistics.median(mads),
                    "steady_variance_us2": statistics.median(variances),
                    "steady_sample_count": sum(
                        item["steady"]["sample_count"] for item in passes
                    ),
                    "checksums": checksums,
                    "passes": passes,
                }

            torch_rs_median = implementations["torch_rs"]["steady_median_us"]
            pytorch_median = implementations["pytorch"]["steady_median_us"]
            rows.append(
                {
                    "api": f"torch.{api}",
                    "workload": workload.name,
                    "category": workload.category,
                    "call_form": workload.call_form,
                    "input_description": workload.input_description,
                    "output_description": workload.output_description,
                    "mode": workload.mode,
                    "repeats": workload.repeats,
                    "input_seeds": list(workload.input_seeds),
                    "input_metadata": pass_results["torch_rs"][0]["input_metadata"],
                    "output_metadata": pass_results["torch_rs"][0]["output_metadata"],
                    "implementations": implementations,
                    "ratios": {
                        "steady_torch_rs_over_pytorch": torch_rs_median
                        / pytorch_median,
                    },
                    "validation": {
                        "reference_checksum": expected_checksum,
                        "metadata_checked": True,
                        "value_bits_checked": True,
                        "warmup_checksums_checked": True,
                        "steady_checksums_checked": True,
                        "operand_nonmutation_checked": all(
                            item["operand_nonmutation_checked"]
                            for implementation in pass_results.values()
                            for item in implementation
                        ),
                    },
                }
            )
    return rows


def _aggregate_rows(rows):
    ratios = [row["ratios"]["steady_torch_rs_over_pytorch"] for row in rows]
    by_api = {
        api: [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["api"] == api
        ]
        for api in (f"torch.{name}" for name in APIS)
    }
    by_category = {
        "singleton": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "singleton"
        ],
        "multi-input": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "multi-input"
        ],
        "empty operand": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "empty operand"
        ],
        "offset": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "offset"
        ],
        "noncontiguous": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "noncontiguous"
        ],
        "axis keyword": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "axis keyword"
        ],
        "active autograd": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "active autograd"
        ],
        "no_grad": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "no_grad"
        ],
    }
    named_groups = {"all supported cells": ratios}
    named_groups.update({f"{api} cells": values for api, values in by_api.items()})
    named_groups.update({f"{name} cells": values for name, values in by_category.items()})
    return {
        "timed_supported_cell_count": len(rows),
        "steady_geomean_torch_rs_over_pytorch": _geomean(ratios),
        "steady_geomean_capped_0_10_10_0": _geomean(
            [min(10.0, max(0.10, ratio)) for ratio in ratios]
        ),
        "groups": {
            name: {
                "cell_count": len(values),
                "geomean": _geomean(values),
                "geomean_capped_0_10_10_0": _geomean(
                    [min(10.0, max(0.10, value)) for value in values]
                ),
            }
            for name, values in named_groups.items()
            if values
        },
    }


def run_benchmark(args):
    affinity = _pin_cpu(args.cpu)
    _configure_thread_environment(args.threads, args.cuda_visible_devices)
    np, torch_rs, reference_torch = _import_backends()
    _validate_reference_version(reference_torch)
    _configure_reference_threads(reference_torch, args.threads)
    _validate_thread_configuration(torch_rs, reference_torch, args.threads)

    workloads = _select_workloads(args.workloads)
    apis = _select_apis(args.apis)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    started = time.time()
    try:
        supported = _run_supported_cells(
            np,
            torch_rs,
            reference_torch,
            workloads,
            apis,
            args,
        )
        unsupported = _run_unsupported_cells(np, torch_rs, reference_torch, apis)
    finally:
        if gc_was_enabled:
            gc.enable()

    aggregates = _aggregate_rows(supported)
    unsupported_penalty = [10.0] * len(unsupported)
    capped_with_zero_credit = [
        min(10.0, max(0.10, row["ratios"]["steady_torch_rs_over_pytorch"]))
        for row in supported
    ] + unsupported_penalty
    aggregates["zero_credit_unsupported_cell_count"] = len(unsupported)
    aggregates["combined_capped_with_zero_credit_unsupported"] = _geomean(
        capped_with_zero_credit
    )

    ended = time.time()
    return {
        "environment": _environment(torch_rs, reference_torch, np, affinity, args),
        "started_epoch_seconds": started,
        "ended_epoch_seconds": ended,
        "duration_seconds": ended - started,
        "cases": supported,
        "zero_credit_unsupported_cells": unsupported,
        "aggregates": aggregates,
    }


def _single_checksum_pair(row):
    torch_rs_checksums = row["implementations"]["torch_rs"]["checksums"]
    pytorch_checksums = row["implementations"]["pytorch"]["checksums"]
    if torch_rs_checksums != pytorch_checksums:
        raise AssertionError(
            f"{row['api']}/{row['workload']} checksum mismatch: "
            f"torch_rs={torch_rs_checksums!r} pytorch={pytorch_checksums!r}"
        )
    if len(torch_rs_checksums) != 1:
        raise AssertionError(
            f"{row['api']}/{row['workload']} has unstable checksums: "
            f"{torch_rs_checksums!r}"
        )
    return torch_rs_checksums[0], pytorch_checksums[0]


def _format_timed_cell(row):
    torch_rs = row["implementations"]["torch_rs"]
    pytorch = row["implementations"]["pytorch"]
    torch_rs_checksum, pytorch_checksum = _single_checksum_pair(row)
    output = row["output_metadata"][0]
    return (
        f"| `{row['workload']}` | {row['category']} | `{row['api']}` | "
        f"{row['call_form']} | {row['input_description']} | "
        f"{row['output_description']}; {_format_metadata(output)} | "
        f"{row['repeats']} | "
        f"{torch_rs['steady_median_us']:.3f} us +/- "
        f"{torch_rs['steady_mad_us']:.3f} us, var "
        f"{torch_rs['steady_variance_us2']:.3f} | "
        f"{pytorch['steady_median_us']:.3f} us +/- "
        f"{pytorch['steady_mad_us']:.3f} us, var "
        f"{pytorch['steady_variance_us2']:.3f} | "
        f"{row['ratios']['steady_torch_rs_over_pytorch']:.2f}x | "
        f"`{torch_rs_checksum}`/`{pytorch_checksum}` |"
    )


def _format_unsupported_cell(row):
    return (
        f"| `{row['name']}` | {row['category']} | `{row['torch_rs']['status']}` | "
        f"`{row['pytorch']['status']}` | {row['credit']} |"
    )


def _group_line(label, group):
    return (
        f"- {label}: {group['geomean']:.2f}x uncapped, "
        f"{group['geomean_capped_0_10_10_0']:.2f}x capped"
    )


def render_markdown_summary(report):
    cases = report["cases"]
    unsupported = report["zero_credit_unsupported_cells"]
    aggregates = report["aggregates"]
    groups = aggregates["groups"]
    environment = report["environment"]
    implementation_orders = ", ".join(
        " then ".join(order) for order in environment["implementation_orders"]
    )
    affinity = environment["cpu_affinity"]
    selected_cpu = affinity["selected_cpu"]
    pinned_affinity = affinity["pinned_affinity"]
    lines = [
        "## Aggregate",
        "",
        f"- Raw JSON artifact: `{DEFAULT_ARTIFACT_PATH.relative_to(REPOSITORY_ROOT)}`",
        f"- Benchmark: `{environment['benchmark_version']}`",
        (
            f"- Timed supported cells: {len(cases)} "
            f"({len(APIS)} APIs x {len(WORKLOADS)} workload shapes and modes)"
        ),
        f"- Zero-credit unsupported cells: {len(unsupported)}",
        (
            "- Implementation orders: "
            f"{implementation_orders}; each implementation appears once before "
            "and once after the other implementation"
        ),
        (
            "- Warmup and sampling: "
            f"{environment['warmups']} untimed warmup blocks and "
            f"{environment['samples']} measured blocks per implementation pass"
        ),
        (
            f"- CPU affinity: selected CPU {selected_cpu}, pinned affinity "
            f"{pinned_affinity}; threads={environment['threads']}"
        ),
        _group_line("All supported cells", groups["all supported cells"]),
        _group_line("`torch.cat` cells", groups["torch.cat cells"]),
        _group_line("`torch.concat` cells", groups["torch.concat cells"]),
        _group_line(
            "`torch.concatenate` cells",
            groups["torch.concatenate cells"],
        ),
        _group_line("Singleton cells", groups["singleton cells"]),
        _group_line("Multi-input cells", groups["multi-input cells"]),
        _group_line("Empty-operand cells", groups["empty operand cells"]),
        _group_line("Offset cells", groups["offset cells"]),
        _group_line("Noncontiguous cells", groups["noncontiguous cells"]),
        _group_line("Axis-keyword cells", groups["axis keyword cells"]),
        _group_line("Active-autograd cells", groups["active autograd cells"]),
        _group_line("`no_grad` cells", groups["no_grad cells"]),
        "",
        (
            "Including the unsupported cells below as zero-credit denominator "
            "entries with a 10.00x capped penalty gives a combined capped "
            "aggregate of "
            f"{aggregates['combined_capped_with_zero_credit_unsupported']:.2f}x."
        ),
        "",
        "## Supported Timed Cells",
        "",
        (
            "| Workload | Category | API | Call form | Input / mode | Output | "
            "Repeats | `torch_rs` median +/- MAD, variance | PyTorch median +/- "
            "MAD, variance | `torch_rs` / PyTorch | Materialized checksums |"
        ),
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_format_timed_cell(row) for row in cases)
    lines.extend(
        [
            "",
            "## Zero-Credit Unsupported Cells",
            "",
            (
                "These cells are not timed because `torch_rs` cannot execute the "
                "equivalent PyTorch operation. They are preserved as zero-credit "
                "cells instead of being removed from the evidence set."
            ),
            "",
            "| Workload | Category | `torch_rs` status | PyTorch status | Credit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(_format_unsupported_cell(row) for row in unsupported)
    lines.append("")
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


def _expected_case_names():
    return {f"torch.{api}/{workload.name}" for api in APIS for workload in WORKLOADS}


def _expected_unsupported_names():
    return {
        f"top_level_torch_{api}_{unsupported.name}"
        for api in APIS
        for unsupported in UNSUPPORTED_CELLS
    }


def _validate_expected_artifact_shape(report):
    errors = []
    environment = report.get("environment", {})
    if environment.get("benchmark_version") != BENCHMARK_VERSION:
        errors.append(
            "benchmark version mismatch: "
            f"{environment.get('benchmark_version')!r} != {BENCHMARK_VERSION!r}"
        )
    if environment.get("warmups") != DEFAULT_WARMUPS:
        errors.append(f"warmup count mismatch: {environment.get('warmups')!r}")
    if environment.get("samples") != DEFAULT_SAMPLES:
        errors.append(f"sample count mismatch: {environment.get('samples')!r}")
    if environment.get("threads") != DEFAULT_THREADS:
        errors.append(f"thread count mismatch: {environment.get('threads')!r}")
    if environment.get("cuda_visible_devices") != "":
        errors.append(
            f"CUDA visibility mismatch: {environment.get('cuda_visible_devices')!r}"
        )
    if environment.get("implementation_orders") != [
        list(order) for order in IMPLEMENTATION_ORDERS
    ]:
        errors.append("implementation order metadata mismatch")
    driver = environment.get("driver", {})
    expected_driver_path = (
        Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    )
    if driver.get("path") != expected_driver_path:
        errors.append(f"driver path mismatch: {driver.get('path')!r}")
    if driver.get("sha256") != _file_sha256(Path(__file__).resolve()):
        errors.append("driver SHA-256 does not match the checked-in script")

    affinity = environment.get("cpu_affinity", {})
    if len(affinity.get("pinned_affinity") or []) != 1:
        errors.append(f"benchmark was not pinned to one CPU: {affinity!r}")
    env_threads = environment.get("env_threads", {})
    for name in THREAD_ENVIRONMENT_VARIABLES:
        if env_threads.get(name) != str(DEFAULT_THREADS):
            errors.append(f"{name} mismatch: {env_threads.get(name)!r}")

    pytorch = environment.get("pytorch", {})
    if _version_without_local(pytorch.get("version", "")) != REFERENCE_PYTORCH_VERSION:
        errors.append(f"PyTorch version mismatch: {pytorch.get('version')!r}")
    if pytorch.get("cuda_available") is not False:
        errors.append("benchmark did not record CPU-only PyTorch execution")

    input_generation = environment.get("input_generation", {})
    expected_seeds = sorted(
        {seed for workload in WORKLOADS for seed in workload.input_seeds}
    )
    if input_generation.get("seed_values") != expected_seeds:
        errors.append("deterministic input seed metadata mismatch")

    cases = report.get("cases", [])
    actual_case_names = {f"{row.get('api')}/{row.get('workload')}" for row in cases}
    expected_case_names = _expected_case_names()
    if actual_case_names != expected_case_names:
        errors.append(
            "timed cell set mismatch: "
            f"missing={sorted(expected_case_names - actual_case_names)!r} "
            f"extra={sorted(actual_case_names - expected_case_names)!r}"
        )
    if len(cases) != len(APIS) * len(WORKLOADS):
        errors.append(f"timed cell count mismatch: {len(cases)}")
    api_counts = Counter(row.get("api") for row in cases)
    expected_api_counts = {f"torch.{api}": len(WORKLOADS) for api in APIS}
    if dict(api_counts) != expected_api_counts:
        errors.append(f"API coverage mismatch: {dict(api_counts)!r}")

    categories = {row.get("category") for row in cases}
    for required_category in (
        "singleton",
        "multi-input",
        "empty operand",
        "offset",
        "noncontiguous",
        "axis keyword",
        "active autograd",
        "no_grad",
    ):
        if required_category not in categories:
            errors.append(f"missing supported category {required_category!r}")

    unsupported = report.get("zero_credit_unsupported_cells", [])
    actual_unsupported_names = {row.get("name") for row in unsupported}
    expected_unsupported_names = _expected_unsupported_names()
    if actual_unsupported_names != expected_unsupported_names:
        errors.append(
            "unsupported cell set mismatch: "
            f"missing={sorted(expected_unsupported_names - actual_unsupported_names)!r} "
            f"extra={sorted(actual_unsupported_names - expected_unsupported_names)!r}"
        )

    aggregates = report.get("aggregates", {})
    if aggregates.get("timed_supported_cell_count") != len(cases):
        errors.append("aggregate timed cell count does not match cases")
    if aggregates.get("zero_credit_unsupported_cell_count") != len(unsupported):
        errors.append("aggregate unsupported cell count does not match rows")

    workloads_by_name = {workload.name: workload for workload in WORKLOADS}
    for row in cases:
        cell_name = f"{row.get('api')}/{row.get('workload')}"
        workload = workloads_by_name.get(row.get("workload"))
        if workload is None:
            errors.append(f"{cell_name} has unknown workload")
        elif row.get("input_seeds") != list(workload.input_seeds):
            errors.append(f"{cell_name} input seed metadata mismatch")
        for implementation in ("torch_rs", "pytorch"):
            implementation_result = row.get("implementations", {}).get(
                implementation,
                {},
            )
            passes = implementation_result.get("passes", [])
            if len(passes) != len(IMPLEMENTATION_ORDERS):
                errors.append(
                    f"{cell_name}/{implementation} pass count mismatch: {len(passes)}"
                )
            if implementation_result.get("steady_sample_count") != (
                DEFAULT_SAMPLES * len(IMPLEMENTATION_ORDERS)
            ):
                errors.append(
                    f"{cell_name}/{implementation} sample count mismatch: "
                    f"{implementation_result.get('steady_sample_count')!r}"
                )
            if len(implementation_result.get("checksums", [])) != 1:
                errors.append(
                    f"{cell_name}/{implementation} unstable checksums: "
                    f"{implementation_result.get('checksums')!r}"
                )
            for pass_result in passes:
                if pass_result.get("steady", {}).get("sample_count") != DEFAULT_SAMPLES:
                    errors.append(
                        f"{cell_name}/{implementation} pass sample count mismatch"
                    )
                if pass_result.get("warmup_checksums") != pass_result.get(
                    "steady_checksums"
                ):
                    errors.append(
                        f"{cell_name}/{implementation} warmup/steady checksum mismatch"
                    )
                if pass_result.get("order") not in [
                    list(order) for order in IMPLEMENTATION_ORDERS
                ]:
                    errors.append(f"{cell_name}/{implementation} order mismatch")
                if pass_result.get("operand_nonmutation_checked") is not True:
                    errors.append(
                        f"{cell_name}/{implementation} missing nonmutation check"
                    )
        try:
            _single_checksum_pair(row)
        except AssertionError as error:
            errors.append(str(error))
        validation = row.get("validation", {})
        for required_key in (
            "metadata_checked",
            "value_bits_checked",
            "warmup_checksums_checked",
            "steady_checksums_checked",
            "operand_nonmutation_checked",
        ):
            if validation.get(required_key) is not True:
                errors.append(f"{cell_name} missing validation flag {required_key}")

    for row in unsupported:
        name = row.get("name")
        if row.get("credit") != "zero":
            errors.append(f"{name} unsupported row is not zero credit")
        torch_rs_status = row.get("torch_rs", {})
        pytorch_status = row.get("pytorch", {})
        if torch_rs_status.get("kind") != "error":
            errors.append(f"{name} torch_rs status is not an error")
        if pytorch_status.get("kind") != "supported":
            errors.append(f"{name} PyTorch status is not supported")
        validation = row.get("validation", {})
        if (
            validation.get("torch_rs_error_checked") is not True
            or validation.get("pytorch_supported_checked") is not True
        ):
            errors.append(f"{name} missing unsupported-cell validation flags")

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
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--workloads", nargs="*", default=())
    parser.add_argument("--apis", nargs="*", default=())
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
