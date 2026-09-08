#!/usr/bin/env python3
"""Benchmark supported CPU ``torch.stack`` cells against PyTorch."""

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
    REPOSITORY_ROOT
    / "docs"
    / "benchmark-data"
    / "top-level-stack-release-timings.json"
)
DEFAULT_MARKDOWN_REPORT_PATH = (
    REPOSITORY_ROOT / "docs" / "top-level-stack-release-timings.md"
)
PROTECTED_OUTPUT_PATHS = {
    REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
    REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
}
REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "top_level_stack_cpu_benchmark_v2"
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
API = "stack"

CREDIT_ZERO = "zero"
CREDIT_ERROR_PARITY = "error_parity"

MODE_EAGER = "eager"
MODE_AUTOGRAD_FORWARD = "autograd_forward"
MODE_AUTOGRAD_BACKWARD = "autograd_backward"
MODE_NO_GRAD = "no_grad"


@dataclass(frozen=True)
class Operands:
    tensors: object
    dim: int
    leaves: tuple[tuple[str, object], ...] = ()


@dataclass(frozen=True)
class Workload:
    name: str
    category: str
    input_description: str
    output_description: str
    repeats: int
    mode: str
    seeds: tuple[int, ...]
    make_operands: object


@dataclass(frozen=True)
class UnsupportedCell:
    name: str
    input_description: str
    make_call: object
    torch_rs_error_type: str
    torch_rs_message: str | None
    pytorch_expected_kind: str
    credit: str
    reason: str


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
            "top-level stack benchmark requires pinned PyTorch "
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
    if any(dimension == 0 for dimension in array.shape):
        return module.zeros(tuple(array.shape), **kwargs)
    if array.shape == ():
        return module.tensor(float(array.reshape(()).item()), **kwargs)
    return module.tensor(array.tolist(), **kwargs)


def _dense_tensor(module, np, shape, seed, *, requires_grad=False, bias=0.0):
    return _tensor_from_array(
        module,
        _values(np, shape, seed, bias=bias),
        requires_grad=requires_grad,
    )


def _scalar_tensor(module, value, *, requires_grad=False):
    return module.tensor(float(value), dtype=module.float32, requires_grad=requires_grad)


def _make_scalar_stack(module, np):
    return Operands(
        [
            _scalar_tensor(module, 1.75),
            _scalar_tensor(module, -0.0),
            _scalar_tensor(module, -2.5),
        ],
        0,
    )


def _make_vector_stack(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (257,), 20260931),
            _dense_tensor(module, np, (257,), 20260932, bias=0.25),
            _dense_tensor(module, np, (257,), 20260933, bias=-0.5),
        ],
        -1,
    )


def _make_matrix_stack(module, np):
    return Operands(
        [
            _dense_tensor(module, np, (257, 263), 20260934),
            _dense_tensor(module, np, (257, 263), 20260935, bias=0.5),
            _dense_tensor(module, np, (257, 263), 20260936, bias=-0.25),
        ],
        1,
    )


def _make_empty_stack(module, np):
    return Operands(
        [
            module.zeros((2, 0, 3), dtype=module.float32),
            module.zeros((2, 0, 3), dtype=module.float32),
        ],
        2,
    )


def _make_offset_stack(module, np):
    first_base = _dense_tensor(module, np, (3, 127, 131), 20260937)
    second_base = _dense_tensor(module, np, (3, 127, 131), 20260938, bias=0.125)
    return Operands([first_base[1], second_base[1]], 0)


def _make_noncontiguous_stack(module, np):
    first_base = _dense_tensor(module, np, (1024, 512), 20260939)
    second_base = _dense_tensor(module, np, (1024, 512), 20260940, bias=-0.125)
    return Operands([first_base.transpose(0, 1), second_base.transpose(0, 1)], 0)


def _make_autograd_forward_stack(module, np):
    first = _dense_tensor(module, np, (127, 131), 20260941, requires_grad=True)
    second = _dense_tensor(
        module,
        np,
        (127, 131),
        20260942,
        requires_grad=True,
        bias=0.5,
    )
    return Operands([first, second], 1, (("first", first), ("second", second)))


def _make_autograd_backward_stack(module, np):
    left = _dense_tensor(module, np, (32, 33), 20260943, requires_grad=True)
    right = _dense_tensor(
        module,
        np,
        (32, 33),
        20260944,
        requires_grad=True,
        bias=-0.5,
    )
    return Operands([left, right, left], 1, (("left", left), ("right", right)))


WORKLOADS = (
    Workload(
        "scalar_three_inputs_dim0",
        "scalar",
        "three scalar tensors, dim=0",
        "stack output",
        10000,
        MODE_EAGER,
        (),
        _make_scalar_stack,
    ),
    Workload(
        "vector_three_inputs_dim_neg1_257",
        "vector",
        "three vectors of shape (257,), dim=-1",
        "stack output",
        512,
        MODE_EAGER,
        (20260931, 20260932, 20260933),
        _make_vector_stack,
    ),
    Workload(
        "matrix_three_inputs_dim1_257x263",
        "matrix",
        "three matrices of shape (257, 263), dim=1",
        "stack output",
        8,
        MODE_EAGER,
        (20260934, 20260935, 20260936),
        _make_matrix_stack,
    ),
    Workload(
        "empty_two_inputs_dim2_2x0x3",
        "empty",
        "two empty tensors of shape (2, 0, 3), dim=2",
        "stack output",
        5000,
        MODE_EAGER,
        (),
        _make_empty_stack,
    ),
    Workload(
        "offset_two_inputs_dim0_127x131",
        "offset",
        "two nonzero-storage-offset views from tensor((3, 127, 131))[1], dim=0",
        "stack output",
        20,
        MODE_EAGER,
        (20260937, 20260938),
        _make_offset_stack,
    ),
    Workload(
        "noncontig_two_inputs_dim0_512x1024",
        "noncontiguous",
        "two transposed views from tensor((1024, 512)).transpose(0, 1), dim=0",
        "stack output",
        3,
        MODE_EAGER,
        (20260939, 20260940),
        _make_noncontiguous_stack,
    ),
    Workload(
        "autograd_forward_two_inputs_dim1_127x131",
        "autograd forward",
        "two requires_grad=True leaves of shape (127, 131), dim=1; forward construction only",
        "stack output",
        20,
        MODE_AUTOGRAD_FORWARD,
        (20260941, 20260942),
        _make_autograd_forward_stack,
    ),
    Workload(
        "autograd_forward_backward_repeated_dim1_32x33",
        "autograd forward+backward",
        "left/right/left requires_grad=True leaves of shape (32, 33), dim=1; timed stack(...).sum().backward()",
        "stack output plus accumulated leaf gradients",
        5,
        MODE_AUTOGRAD_BACKWARD,
        (20260943, 20260944),
        _make_autograd_backward_stack,
    ),
)


def _unsupported_empty_input_sequence(module):
    return module.stack([])


def _unsupported_mixed_shapes(module):
    return module.stack(
        [
            module.tensor([1.0], dtype=module.float32),
            module.tensor([1.0, 2.0], dtype=module.float32),
        ]
    )


def _unsupported_mixed_metadata(module):
    float64 = getattr(module, "float64")
    return module.stack(
        [
            module.tensor([1.0], dtype=module.float32),
            module.tensor([2.0], dtype=float64),
        ]
    )


def _unsupported_concrete_out(module):
    output = module.zeros((2, 1), dtype=module.float32)
    return module.stack(
        [
            module.tensor([1.0], dtype=module.float32),
            module.tensor([2.0], dtype=module.float32),
        ],
        out=output,
    )


UNSUPPORTED_CELLS = (
    UnsupportedCell(
        "empty_input_sequence",
        "empty TensorList",
        _unsupported_empty_input_sequence,
        "RuntimeError",
        "stack expects a non-empty TensorList",
        "error",
        CREDIT_ERROR_PARITY,
        "torch_rs and PyTorch both reject an empty input sequence",
    ),
    UnsupportedCell(
        "mixed_shapes",
        "same-rank tensors with shapes (1,) and (2,)",
        _unsupported_mixed_shapes,
        "RuntimeError",
        "stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1",
        "error",
        CREDIT_ERROR_PARITY,
        "torch_rs and PyTorch both reject unequal input tensor shapes",
    ),
    UnsupportedCell(
        "mixed_metadata",
        "float32 tensor and float64 tensor requiring dtype promotion",
        _unsupported_mixed_metadata,
        "AttributeError",
        None,
        "supported",
        CREDIT_ZERO,
        "PyTorch supports dtype promotion for this call and torch_rs does not",
    ),
    UnsupportedCell(
        "concrete_out",
        "same-shape tensor sequence with a concrete out tensor",
        _unsupported_concrete_out,
        "RuntimeError",
        "stack(): the 'out' argument is not supported",
        "supported",
        CREDIT_ZERO,
        "PyTorch supports a concrete out tensor and torch_rs rejects it",
    ),
)


def _is_tensor(value):
    return all(hasattr(value, attr) for attr in ("shape", "stride", "storage_offset"))


def _operand_tensors(operands):
    return tuple((f"input_{index}", tensor) for index, tensor in enumerate(operands.tensors))


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


def _execute_operation(module, workload, operands):
    context = module.no_grad() if workload.mode == MODE_NO_GRAD else contextlib.nullcontext()
    with context:
        output = module.stack(operands.tensors, dim=operands.dim)

    if workload.mode == MODE_AUTOGRAD_BACKWARD:
        output.sum().backward()
        bundle = [("output", output)]
        for label, leaf in operands.leaves:
            grad = leaf.grad
            if grad is None:
                raise AssertionError(f"{workload.name} missing {label} gradient")
            bundle.append((f"{label}_grad", grad))
        return tuple(bundle)

    return (("output", output),)


def _make_block_operands(module, np, workload, static_operands, repeats):
    if workload.mode == MODE_AUTOGRAD_BACKWARD:
        return [workload.make_operands(module, np) for _ in range(repeats)]
    return [static_operands for _ in range(repeats)]


def _time_block(np, module, workload, static_operands, repeats):
    block_operands = _make_block_operands(
        module,
        np,
        workload,
        static_operands,
        repeats,
    )
    started_ns = time.perf_counter_ns()
    last_bundle = None
    for operands in block_operands:
        last_bundle = _execute_operation(module, workload, operands)
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


def _measure_one_pass(np, module, implementation, workload, args):
    static_operands = (
        None
        if workload.mode == MODE_AUTOGRAD_BACKWARD
        else workload.make_operands(module, np)
    )
    metadata_operands = (
        workload.make_operands(module, np)
        if workload.mode == MODE_AUTOGRAD_BACKWARD
        else static_operands
    )
    input_metadata = [
        {"label": label, **_tensor_metadata(tensor)}
        for label, tensor in _operand_tensors(metadata_operands)
    ]
    input_checksums_before = [
        {"label": label, "checksum": _checksum_tensor(np, tensor)}
        for label, tensor in _operand_tensors(static_operands or metadata_operands)
    ]

    cold_ns, cold_checksum, cold_bundle = _time_block(
        np,
        module,
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
            workload,
            static_operands,
            workload.repeats,
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)
        sample_sink = _roll_checksum(sample_sink, checksum)

    input_checksums_after = [
        {"label": label, "checksum": _checksum_tensor(np, tensor)}
        for label, tensor in _operand_tensors(static_operands or metadata_operands)
    ]
    operand_nonmutation_checked = workload.mode != MODE_AUTOGRAD_BACKWARD
    if operand_nonmutation_checked and input_checksums_after != input_checksums_before:
        raise AssertionError(
            f"{workload.name}/{implementation} mutated benchmark operands: "
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
        "operand_nonmutation_checked": operand_nonmutation_checked,
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
            "extension_path": getattr(getattr(torch_rs, "_C", None), "__file__", None),
            "threads": torch_rs.get_num_threads(),
            "interop_threads": torch_rs.get_num_interop_threads(),
        },
        "rust": {
            "rustc": _run_text(["rustc", "--version"]),
            "cargo": _run_text(["cargo", "--version"]),
        },
        "build_profile": {
            "rust_profile": "release",
            "cargo_profile_release_lto": "thin",
            "cargo_profile_release_codegen_units": 1,
            "native_extension_required": True,
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
        "api": f"torch.{API}",
        "workloads": list(args.workloads) if args.workloads else "all",
        "benchmark_integrity": {
            "timing_scope": (
                "torch.stack call with inputs constructed outside the timed region; "
                "autograd backward cells time stack(...).sum().backward()"
            ),
            "materialization": (
                "each cold, warmup, and measured block records a checksum over "
                "final output metadata and logical bytes"
            ),
            "unsupported_policy": (
                "PyTorch-supported boundary rows that torch_rs rejects are retained "
                "as zero-credit denominator entries; matching invalid-input error "
                "parity rows are checked but excluded from the unsupported denominator"
            ),
        },
    }


def _run_status(np, module, unsupported_cell):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            output = unsupported_cell.make_call(module)
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


def _run_unsupported_cells(np, torch_rs, reference_torch):
    rows = []
    for unsupported_cell in UNSUPPORTED_CELLS:
        cell_name = f"top_level_torch_stack_{unsupported_cell.name}"
        torch_rs_status = _run_status(np, torch_rs, unsupported_cell)
        message_matches = (
            unsupported_cell.torch_rs_message is None
            or torch_rs_status.get("message") == unsupported_cell.torch_rs_message
        )
        if (
            torch_rs_status["kind"] != "error"
            or torch_rs_status["error_type"] != unsupported_cell.torch_rs_error_type
            or not message_matches
        ):
            raise AssertionError(
                f"{cell_name} torch_rs status mismatch:\n"
                f"actual={torch_rs_status!r}\n"
                f"expected type={unsupported_cell.torch_rs_error_type!r} "
                f"message={unsupported_cell.torch_rs_message!r}"
            )

        pytorch_status = _run_status(np, reference_torch, unsupported_cell)
        if pytorch_status["kind"] != unsupported_cell.pytorch_expected_kind:
            raise AssertionError(
                f"{cell_name} PyTorch status mismatch: {pytorch_status!r}"
            )
        if unsupported_cell.credit == CREDIT_ZERO:
            if pytorch_status["kind"] != "supported":
                raise AssertionError(
                    f"{cell_name} zero-credit row is not PyTorch-supported: "
                    f"{pytorch_status!r}"
                )
        elif unsupported_cell.credit == CREDIT_ERROR_PARITY:
            if pytorch_status["kind"] != "error":
                raise AssertionError(
                    f"{cell_name} error-parity row is not a PyTorch error: "
                    f"{pytorch_status!r}"
                )
            if (
                pytorch_status.get("error_type") != torch_rs_status.get("error_type")
                or pytorch_status.get("message") != torch_rs_status.get("message")
            ):
                raise AssertionError(
                    f"{cell_name} error-parity mismatch:\n"
                    f"torch_rs={torch_rs_status!r}\npytorch={pytorch_status!r}"
                )
        else:
            raise AssertionError(
                f"{cell_name} unknown credit policy {unsupported_cell.credit!r}"
            )

        rows.append(
            {
                "name": cell_name,
                "api": f"torch.{API}",
                "input_description": unsupported_cell.input_description,
                "torch_rs": torch_rs_status,
                "pytorch": pytorch_status,
                "credit": unsupported_cell.credit,
                "reason": unsupported_cell.reason,
                "validation": {
                    "torch_rs_error_checked": True,
                    "pytorch_status_checked": True,
                },
            }
        )
    return rows


def _expected_bundle(np, reference_torch, workload):
    operands = workload.make_operands(reference_torch, np)
    return _execute_operation(reference_torch, workload, operands)


def _run_supported_cells(np, torch_rs, reference_torch, workloads, args):
    rows = []
    for workload in workloads:
        cell_name = f"torch.{API}/{workload.name}"
        expected = _expected_bundle(np, reference_torch, workload)
        expected_checksum = _checksum_bundle(np, expected)
        pass_results = {"torch_rs": [], "pytorch": []}

        for order_index, order in enumerate(IMPLEMENTATION_ORDERS):
            for implementation in order:
                module = torch_rs if implementation == "torch_rs" else reference_torch
                measured = _measure_one_pass(
                    np,
                    module,
                    implementation,
                    workload,
                    args,
                )
                _assert_bundles_match(
                    np,
                    measured["cold_bundle"],
                    expected,
                    cell_name=f"{cell_name}/{implementation}/cold",
                )
                for key in ("warmup_checksums", "steady_checksums"):
                    if key == "warmup_checksums" and args.warmups == 0:
                        expected_checksums = []
                    else:
                        expected_checksums = [expected_checksum]
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
                "api": f"torch.{API}",
                "workload": workload.name,
                "category": workload.category,
                "input_description": workload.input_description,
                "output_description": workload.output_description,
                "mode": workload.mode,
                "seed_values": list(workload.seeds),
                "repeats": workload.repeats,
                "input_metadata": pass_results["torch_rs"][0]["input_metadata"],
                "output_metadata": pass_results["torch_rs"][0]["output_metadata"],
                "implementations": implementations,
                "ratios": {
                    "steady_torch_rs_over_pytorch": torch_rs_median / pytorch_median,
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
    by_category = {
        "scalar": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "scalar"
        ],
        "vector": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "vector"
        ],
        "matrix": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "matrix"
        ],
        "empty": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "empty"
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
        "autograd forward": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "autograd forward"
        ],
        "autograd forward+backward": [
            row["ratios"]["steady_torch_rs_over_pytorch"]
            for row in rows
            if row["category"] == "autograd forward+backward"
        ],
    }
    named_groups = {"all supported cells": ratios}
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
    gc_was_enabled = gc.isenabled()
    gc.disable()
    started = time.time()
    try:
        supported = _run_supported_cells(
            np,
            torch_rs,
            reference_torch,
            workloads,
            args,
        )
        unsupported = _run_unsupported_cells(np, torch_rs, reference_torch)
    finally:
        if gc_was_enabled:
            gc.enable()

    aggregates = _aggregate_rows(supported)
    zero_credit = [
        row for row in unsupported if row["credit"] == CREDIT_ZERO
    ]
    error_parity = [
        row for row in unsupported if row["credit"] == CREDIT_ERROR_PARITY
    ]
    unsupported_penalty = [10.0] * len(zero_credit)
    capped_with_zero_credit = [
        min(10.0, max(0.10, row["ratios"]["steady_torch_rs_over_pytorch"]))
        for row in supported
    ] + unsupported_penalty
    aggregates["zero_credit_unsupported_cell_count"] = len(zero_credit)
    aggregates["boundary_error_parity_cell_count"] = len(error_parity)
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
        "zero_credit_unsupported_cells": zero_credit,
        "boundary_error_parity_cells": error_parity,
        "aggregates": aggregates,
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
        f"| `{row['workload']}` | {row['category']} | "
        f"{row['input_description']} | "
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
        f"| `{row['name']}` | {row['input_description']} | "
        f"`{row['torch_rs']['status']}` | `{row['pytorch']['status']}` | "
        f"{row['credit']} |"
    )


def _group_line(label, group):
    return (
        f"- {label}: {group['geomean']:.2f}x uncapped, "
        f"{group['geomean_capped_0_10_10_0']:.2f}x capped"
    )


def render_markdown_summary(report):
    cases = report["cases"]
    unsupported = report["zero_credit_unsupported_cells"]
    error_parity = report["boundary_error_parity_cells"]
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
        f"- Timed supported cells: {len(cases)} (1 API x {len(WORKLOADS)} workload shapes and modes)",
        f"- Zero-credit unsupported cells: {len(unsupported)}",
        f"- Error-parity boundary cells: {len(error_parity)}",
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
        _group_line("Scalar cells", groups["scalar cells"]),
        _group_line("Vector cells", groups["vector cells"]),
        _group_line("Matrix cells", groups["matrix cells"]),
        _group_line("Empty cells", groups["empty cells"]),
        _group_line("Offset cells", groups["offset cells"]),
        _group_line("Noncontiguous cells", groups["noncontiguous cells"]),
        _group_line("Autograd forward cells", groups["autograd forward cells"]),
        _group_line(
            "Autograd forward+backward cells",
            groups["autograd forward+backward cells"],
        ),
        "",
        (
            "Including the PyTorch-supported unsupported cells below as "
            "zero-credit denominator entries with a 10.00x capped penalty gives "
            "a combined capped aggregate of "
            f"{aggregates['combined_capped_with_zero_credit_unsupported']:.2f}x."
        ),
        "",
        "## Supported Timed Cells",
        "",
        (
            "| Workload | Category | Input / mode | Output | Repeats | "
            "`torch_rs` median +/- MAD, variance | PyTorch median +/- MAD, "
            "variance | `torch_rs` / PyTorch | Materialized checksums |"
        ),
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    lines.extend(_format_timed_cell(row) for row in cases)
    lines.extend(
        [
            "",
            "## Zero-Credit Unsupported Cells",
            "",
            (
                "These cells are not timed. PyTorch supports them and torch_rs "
                "rejects them, so they are preserved as zero-credit denominator "
                "entries instead of being removed from the evidence set."
            ),
            "",
            "| Workload | Input | `torch_rs` status | PyTorch status | Credit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(_format_unsupported_cell(row) for row in unsupported)
    lines.extend(
        [
            "",
            "## Error-Parity Boundary Cells",
            "",
            (
                "These cells are checked but excluded from the zero-credit "
                "denominator because PyTorch rejects the same invalid inputs."
            ),
            "",
            "| Workload | Input | `torch_rs` status | PyTorch status | Credit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    lines.extend(_format_unsupported_cell(row) for row in error_parity)
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
    return {f"torch.{API}/{workload.name}" for workload in WORKLOADS}


def _expected_unsupported_names(credit):
    return {
        f"top_level_torch_stack_{unsupported.name}"
        for unsupported in UNSUPPORTED_CELLS
        if unsupported.credit == credit
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
    if environment.get("implementation_orders") != [
        list(order) for order in IMPLEMENTATION_ORDERS
    ]:
        errors.append("implementation order metadata mismatch")
    if environment.get("api") != f"torch.{API}":
        errors.append(f"API metadata mismatch: {environment.get('api')!r}")
    driver = environment.get("driver", {})
    expected_driver_path = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    if driver.get("path") != expected_driver_path:
        errors.append(f"driver path mismatch: {driver.get('path')!r}")
    if driver.get("sha256") != _file_sha256(Path(__file__).resolve()):
        errors.append("driver SHA-256 does not match the checked-in script")

    git = environment.get("git", {})
    head = git.get("head")
    if not isinstance(head, str) or len(head) != 40:
        errors.append(f"git head is not a full commit hash: {head!r}")

    build_profile = environment.get("build_profile", {})
    if build_profile.get("rust_profile") != "release":
        errors.append(f"build profile mismatch: {build_profile!r}")
    if build_profile.get("native_extension_required") is not True:
        errors.append("native extension requirement is not recorded")

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
    for section, key in (
        ("rust", "rustc"),
        ("rust", "cargo"),
        ("pytorch", "threads"),
        ("pytorch", "interop_threads"),
        ("torch_rs", "threads"),
        ("torch_rs", "interop_threads"),
    ):
        value = environment.get(section, {}).get(key)
        if value in (None, ""):
            errors.append(f"missing provenance field {section}.{key}")

    cases = report.get("cases", [])
    actual_case_names = {f"{row.get('api')}/{row.get('workload')}" for row in cases}
    expected_case_names = _expected_case_names()
    expected_workloads_by_name = {workload.name: workload for workload in WORKLOADS}
    if actual_case_names != expected_case_names:
        errors.append(
            "timed cell set mismatch: "
            f"missing={sorted(expected_case_names - actual_case_names)!r} "
            f"extra={sorted(actual_case_names - expected_case_names)!r}"
        )
    if len(cases) != len(WORKLOADS):
        errors.append(f"timed cell count mismatch: {len(cases)}")
    category_counts = Counter(row.get("category") for row in cases)
    expected_category_counts = Counter(workload.category for workload in WORKLOADS)
    if category_counts != expected_category_counts:
        errors.append(f"category coverage mismatch: {dict(category_counts)!r}")

    unsupported = report.get("zero_credit_unsupported_cells", [])
    error_parity = report.get("boundary_error_parity_cells", [])
    actual_unsupported_names = {row.get("name") for row in unsupported}
    expected_unsupported_names = _expected_unsupported_names(CREDIT_ZERO)
    if actual_unsupported_names != expected_unsupported_names:
        errors.append(
            "unsupported cell set mismatch: "
            f"missing={sorted(expected_unsupported_names - actual_unsupported_names)!r} "
            f"extra={sorted(actual_unsupported_names - expected_unsupported_names)!r}"
        )
    actual_error_parity_names = {row.get("name") for row in error_parity}
    expected_error_parity_names = _expected_unsupported_names(CREDIT_ERROR_PARITY)
    if actual_error_parity_names != expected_error_parity_names:
        errors.append(
            "error-parity boundary cell set mismatch: "
            f"missing={sorted(expected_error_parity_names - actual_error_parity_names)!r} "
            f"extra={sorted(actual_error_parity_names - expected_error_parity_names)!r}"
        )

    aggregates = report.get("aggregates", {})
    if aggregates.get("timed_supported_cell_count") != len(cases):
        errors.append("aggregate timed cell count does not match cases")
    if aggregates.get("zero_credit_unsupported_cell_count") != len(unsupported):
        errors.append("aggregate unsupported cell count does not match rows")
    if aggregates.get("boundary_error_parity_cell_count") != len(error_parity):
        errors.append("aggregate error-parity cell count does not match rows")
    expected_combined = _geomean(
        [
            min(10.0, max(0.10, row["ratios"]["steady_torch_rs_over_pytorch"]))
            for row in cases
        ]
        + [10.0] * len(unsupported)
    )
    if not math.isclose(
        aggregates.get("combined_capped_with_zero_credit_unsupported", math.nan),
        expected_combined,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        errors.append("combined zero-credit aggregate does not match row set")

    for row in cases:
        cell_name = f"{row.get('api')}/{row.get('workload')}"
        expected_workload = expected_workloads_by_name.get(row.get("workload"))
        if expected_workload is not None and row.get("seed_values") != list(
            expected_workload.seeds
        ):
            errors.append(f"{cell_name} seed metadata mismatch")
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
        ):
            if validation.get(required_key) is not True:
                errors.append(f"{cell_name} missing validation flag {required_key}")
        if row.get("mode") == MODE_AUTOGRAD_BACKWARD:
            labels = [entry.get("label") for entry in row.get("output_metadata", [])]
            if labels != ["output", "left_grad", "right_grad"]:
                errors.append(f"{cell_name} missing backward gradient artifacts")

    expected_by_name = {
        f"top_level_torch_stack_{unsupported.name}": unsupported
        for unsupported in UNSUPPORTED_CELLS
    }
    for row in unsupported + error_parity:
        name = row.get("name")
        expected_unsupported = expected_by_name.get(name)
        if expected_unsupported is None:
            errors.append(f"{name} unknown boundary row")
            continue
        if row.get("credit") != expected_unsupported.credit:
            errors.append(
                f"{name} credit mismatch: "
                f"{row.get('credit')!r} != {expected_unsupported.credit!r}"
            )
        torch_rs_status = row.get("torch_rs", {})
        if torch_rs_status.get("kind") != "error":
            errors.append(f"{name} torch_rs status is not an error")
        if torch_rs_status.get("error_type") != expected_unsupported.torch_rs_error_type:
            errors.append(f"{name} torch_rs error type mismatch")
        expected_message = expected_unsupported.torch_rs_message
        if (
            expected_message is not None
            and torch_rs_status.get("message") != expected_message
        ):
            errors.append(f"{name} torch_rs error message mismatch")
        pytorch_status = row.get("pytorch", {})
        if pytorch_status.get("kind") != expected_unsupported.pytorch_expected_kind:
            errors.append(f"{name} PyTorch status mismatch")
        if row.get("credit") == CREDIT_ZERO and pytorch_status.get("kind") != "supported":
            errors.append(f"{name} zero-credit row is not PyTorch-supported")
        if row.get("credit") == CREDIT_ERROR_PARITY:
            if pytorch_status.get("kind") != "error":
                errors.append(f"{name} error-parity row is not a PyTorch error")
            if (
                pytorch_status.get("error_type") != torch_rs_status.get("error_type")
                or pytorch_status.get("message") != torch_rs_status.get("message")
            ):
                errors.append(f"{name} error-parity status mismatch")
        validation = row.get("validation", {})
        if (
            validation.get("torch_rs_error_checked") is not True
            or validation.get("pytorch_status_checked") is not True
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

    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")

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
