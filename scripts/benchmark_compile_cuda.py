#!/usr/bin/env python3
"""Benchmark one CUDA ``torch.compile`` reference workload.

The benchmark is intentionally narrow: it measures a single PyTorch 2.13
CUDA/H100 reference workload and one matching private ``torch_rs`` CUDA compile
path. CPU execution, eager fallback, and forwarding to installed PyTorch are
fail-closed and never count as eligible CUDA compile evidence.
"""

from __future__ import annotations

import argparse
import array
import gc
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_OUTPUT_PATHS = {
    REPOSITORY_ROOT / "docs" / "burner-evaluation-history.json",
    REPOSITORY_ROOT / "docs" / "burner-evaluation-progress.svg",
}

REFERENCE_PYTORCH_VERSION = "2.13.0"
BENCHMARK_VERSION = "torch_compile_cuda_h100_reference_benchmark_v8"
WORKLOAD_VERSION = "h100_cuda_pointwise_reduce_float32_v1"
PREPARED_EXECUTOR_SCHEMA_VERSION = (
    "torch_rs_private_cuda_pointwise_reduce_compile_executor_v1"
)
WORKLOAD_SHAPE = (1024, 1024)
WORKLOAD_SEED = 20260904

REFERENCE_COMPILE_CONFIG = {
    "backend": "inductor",
    "fullgraph": True,
    "dynamic": False,
    "mode": None,
    "options": None,
}

DEFAULT_WARMUPS = 5
DEFAULT_SAMPLES = 17
DEFAULT_REPEATS = 3
DEFAULT_REQUIRED_CUDA_VISIBLE_DEVICES = "0"


def h100_cuda_pointwise_reduce_float32(x, bias):
    """One versioned CUDA reference workload for PyTorch 2.13."""
    mixed = (x + bias).sin() * (x - bias).cos()
    return (mixed + x.relu()).sum(dim=1)


h100_cuda_pointwise_reduce_float32._torch_rs_cuda_compile_workload_version = (
    WORKLOAD_VERSION
)
h100_cuda_pointwise_reduce_float32._torch_rs_cuda_compile_workload_shape = (
    WORKLOAD_SHAPE
)
h100_cuda_pointwise_reduce_float32._torch_rs_cuda_compile_output_shape = (
    WORKLOAD_SHAPE[0],
)
h100_cuda_pointwise_reduce_float32._torch_rs_cuda_compile_dtype = "torch.float32"


def _version_without_local(version):
    return version.split("+", 1)[0]


def _package_version(distribution_name, module):
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return getattr(module, "__version__", None)


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


def _nvidia_smi_query():
    output = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
            "-i",
            "0",
        ]
    )
    if output is None:
        return None
    rows = []
    for line in output.splitlines():
        columns = [column.strip() for column in line.split(",")]
        if len(columns) == 5:
            rows.append(
                {
                    "index": columns[0],
                    "name": columns[1],
                    "memory_total_mib": columns[2],
                    "driver_version": columns[3],
                    "compute_capability": columns[4],
                }
            )
    return rows


def _nvcc_version():
    return _run_text(["nvcc", "--version"])


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


def _require_reference_environment(reference_torch, args):
    if _version_without_local(reference_torch.__version__) != REFERENCE_PYTORCH_VERSION:
        raise SystemExit(
            "CUDA compile benchmark requires pinned PyTorch "
            f"{REFERENCE_PYTORCH_VERSION}, got {reference_torch.__version__}"
        )

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        args.required_cuda_visible_devices is not None
        and cuda_visible_devices != args.required_cuda_visible_devices
    ):
        raise SystemExit(
            "CUDA compile benchmark must run with "
            f"CUDA_VISIBLE_DEVICES={args.required_cuda_visible_devices!r}; "
            f"got {cuda_visible_devices!r}"
        )

    if not reference_torch.cuda.is_available():
        raise SystemExit(
            "CUDA compile benchmark requires a CUDA-visible PyTorch runtime"
        )
    if reference_torch.cuda.device_count() < 1:
        raise SystemExit("CUDA compile benchmark requires at least one visible GPU")

    reference_torch.cuda.set_device(0)
    device_name = reference_torch.cuda.get_device_name(0)
    if not args.allow_non_h100 and "H100" not in device_name:
        raise SystemExit(
            "CUDA compile benchmark is calibrated for NVIDIA H100; "
            f"visible device 0 is {device_name!r}"
        )


def _validate_counts(args):
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")


def _make_reference_inputs(reference_torch):
    reference_torch.manual_seed(WORKLOAD_SEED)
    x = reference_torch.randn(
        WORKLOAD_SHAPE,
        device="cuda",
        dtype=reference_torch.float32,
    )
    bias = reference_torch.randn(
        (WORKLOAD_SHAPE[1],),
        device="cuda",
        dtype=reference_torch.float32,
    )
    return x, bias


def _synchronize(reference_torch):
    reference_torch.cuda.synchronize(0)


def _tensor_metadata(tensor):
    device = tensor.device
    return {
        "shape": list(tuple(tensor.shape)),
        "stride": list(tuple(tensor.stride())),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "requires_grad": bool(tensor.requires_grad),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def _checksum_tensor(tensor):
    payload = {
        "metadata": _tensor_metadata(tensor),
        "values": tensor.detach().cpu().contiguous().tolist(),
    }
    encoded = json.dumps(
        payload,
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _tensor_float32_bytes(tensor):
    if str(tensor.dtype) != "torch.float32":
        raise AssertionError(f"expected torch.float32 tensor, got {tensor.dtype}")
    contiguous = tensor.detach().cpu().contiguous().view(-1)
    values = array.array("f", contiguous.tolist())
    if values.itemsize != 4:
        raise AssertionError("array('f') is not a 32-bit float on this platform")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


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


def _time_once(reference_torch, compiled, inputs):
    _synchronize(reference_torch)
    started_ns = time.perf_counter_ns()
    output = compiled(*inputs)
    _synchronize(reference_torch)
    checksum = _checksum_tensor(output)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, checksum, output


def _time_repeated(reference_torch, compiled, inputs, repeats):
    _synchronize(reference_torch)
    started_ns = time.perf_counter_ns()
    output = None
    for _ in range(repeats):
        output = compiled(*inputs)
    _synchronize(reference_torch)
    checksum = _checksum_tensor(output)
    elapsed_ns = time.perf_counter_ns() - started_ns
    return elapsed_ns, checksum


def _compile_reference(reference_torch):
    kwargs = {
        "backend": REFERENCE_COMPILE_CONFIG["backend"],
        "fullgraph": REFERENCE_COMPILE_CONFIG["fullgraph"],
        "dynamic": REFERENCE_COMPILE_CONFIG["dynamic"],
    }
    if REFERENCE_COMPILE_CONFIG["mode"] is not None:
        kwargs["mode"] = REFERENCE_COMPILE_CONFIG["mode"]
    if REFERENCE_COMPILE_CONFIG["options"] is not None:
        kwargs["options"] = REFERENCE_COMPILE_CONFIG["options"]
    return reference_torch.compile(h100_cuda_pointwise_reduce_float32, **kwargs)


def _run_pytorch_reference(reference_torch, args):
    inputs = _make_reference_inputs(reference_torch)
    expected = h100_cuda_pointwise_reduce_float32(*inputs)
    _synchronize(reference_torch)

    factory_started_ns = time.perf_counter_ns()
    compiled = _compile_reference(reference_torch)
    factory_ns = time.perf_counter_ns() - factory_started_ns

    cold_ns, cold_checksum, cold_output = _time_once(reference_torch, compiled, inputs)
    reference_torch.testing.assert_close(cold_output, expected)

    for _ in range(args.warmups):
        _time_repeated(reference_torch, compiled, inputs, args.repeats)

    sample_ns = []
    sample_checksums = []
    for _ in range(args.samples):
        elapsed_ns, checksum = _time_repeated(
            reference_torch,
            compiled,
            inputs,
            args.repeats,
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)

    expected_checksum = _checksum_tensor(expected)
    checksums = sorted(set([cold_checksum, *sample_checksums]))
    if len(checksums) != 1:
        raise AssertionError(
            "compiled CUDA workload produced unstable checksums: "
            f"{checksums!r}"
        )

    report = {
        "implementation": "pytorch",
        "status": "ok",
        "workload_version": WORKLOAD_VERSION,
        "compile_config": dict(REFERENCE_COMPILE_CONFIG),
        "factory_us": factory_ns / 1000.0,
        "cold_first_call_us": cold_ns / 1000.0,
        "cold_checksum": cold_checksum,
        "steady": _summarize_samples(sample_ns, args.repeats),
        "steady_checksums": sorted(set(sample_checksums)),
        "input_metadata": [_tensor_metadata(input) for input in inputs],
        "output_metadata": _tensor_metadata(cold_output),
        "correctness": {
            "eager_reference_checksum": expected_checksum,
            "assert_close": True,
        },
    }
    private_workload_buffers = {
        "x_host_bytes": _tensor_float32_bytes(inputs[0]),
        "bias_host_bytes": _tensor_float32_bytes(inputs[1]),
        "expected_output_bytes": _tensor_float32_bytes(cold_output),
        "expected_output_checksum": cold_checksum,
        "expected_output_metadata": _tensor_metadata(cold_output),
    }
    return report, private_workload_buffers


def classify_torch_rs_cuda_compile_evidence(evidence):
    """Return fail-closed CUDA compile eligibility for a torch_rs evidence row."""
    reasons = []
    compile_config = evidence.get("compile_config")
    if compile_config is None:
        compile_config = {}
    elif type(compile_config) is not dict:
        reasons.append("compile_config is not a dict")
        compile_config = {}

    if evidence.get("implementation") != "torch_rs":
        reasons.append("implementation is not torch_rs")
    if evidence.get("status") != "ok":
        reasons.append("execution status is not ok")
    if evidence.get("workload_version") != WORKLOAD_VERSION:
        reasons.append("workload version does not match the CUDA benchmark")
    if evidence.get("compile_backend") != REFERENCE_COMPILE_CONFIG["backend"]:
        reasons.append(
            "compile backend is not the declared CUDA reference backend "
            f"{REFERENCE_COMPILE_CONFIG['backend']!r}"
        )
    if (
        "backend" in compile_config
        and compile_config.get("backend") != REFERENCE_COMPILE_CONFIG["backend"]
    ):
        reasons.append(
            "compile_config backend is not the declared CUDA reference backend "
            f"{REFERENCE_COMPILE_CONFIG['backend']!r}"
        )

    fullgraph_values = []
    if "compile_fullgraph" in evidence:
        fullgraph_values.append(evidence.get("compile_fullgraph"))
    if "fullgraph" in compile_config:
        fullgraph_values.append(compile_config.get("fullgraph"))
    if not fullgraph_values or any(value is not True for value in fullgraph_values):
        reasons.append("compile fullgraph setting is not True")

    dynamic_values = []
    if "compile_dynamic" in evidence:
        dynamic_values.append(evidence.get("compile_dynamic"))
    if "dynamic" in compile_config:
        dynamic_values.append(compile_config.get("dynamic"))
    if not dynamic_values or any(value is not False for value in dynamic_values):
        reasons.append("compile dynamic setting is not False")
    if evidence.get("input_device_type") != "cuda":
        reasons.append("inputs did not execute on CUDA")
    if evidence.get("output_device_type") != "cuda":
        reasons.append("outputs did not materialize on CUDA")
    if evidence.get("native_cuda_compile") is not True:
        reasons.append("native CUDA compile execution was not demonstrated")
    if evidence.get("eager_fallback") is not False:
        reasons.append("eager fallback is not eligible CUDA compile evidence")
    if evidence.get("forwarded_to_pytorch") is not False:
        reasons.append("forwarding to installed PyTorch is not eligible evidence")

    return {
        "eligible_cuda_compile_evidence": not reasons,
        "score_credit": 1.0 if not reasons else 0.0,
        "rejection_reasons": reasons,
    }


def _torch_rs_public_cuda_probes(torch_rs):
    return {
        "cuda_is_available": bool(torch_rs.cuda.is_available()),
        "cuda_device_count": int(torch_rs.cuda.device_count()),
        "cuda_is_initialized": bool(torch_rs.cuda.is_initialized()),
        "accelerator_is_available": bool(torch_rs.accelerator.is_available()),
        "accelerator_device_count": int(torch_rs.accelerator.device_count()),
    }


def _torch_rs_private_cuda_driver_probe():
    from torch_rs import _cuda_driver_probe

    return _cuda_driver_probe.probe_cuda_driver_device0()


def _torch_rs_private_cuda_runtime_roundtrip(required_cuda_visible_devices):
    from torch_rs import _cuda_runtime_roundtrip

    return _cuda_runtime_roundtrip.roundtrip_float32_device0(
        required_cuda_visible_devices=required_cuda_visible_devices,
    )


def _torch_rs_private_cuda_pointwise_kernel(required_cuda_visible_devices):
    from torch_rs import _cuda_pointwise_kernel

    return _cuda_pointwise_kernel.launch_float32_pointwise_device0(
        required_cuda_visible_devices=required_cuda_visible_devices,
    )


def _torch_rs_private_cuda_pointwise_reduce_workload(
    required_cuda_visible_devices,
    workload_buffers,
):
    from torch_rs import _cuda_pointwise_reduce_workload

    return _cuda_pointwise_reduce_workload.launch_h100_float32_pointwise_reduce_device0(
        workload_buffers["x_host_bytes"],
        workload_buffers["bias_host_bytes"],
        rows=WORKLOAD_SHAPE[0],
        columns=WORKLOAD_SHAPE[1],
        expected_output_bytes=workload_buffers["expected_output_bytes"],
        expected_output_checksum=workload_buffers["expected_output_checksum"],
        expected_output_metadata=workload_buffers["expected_output_metadata"],
        required_cuda_visible_devices=required_cuda_visible_devices,
    )


def _torch_rs_private_cuda_pointwise_reduce_inputs(
    required_cuda_visible_devices,
    workload_buffers,
):
    from torch_rs import _cuda_pointwise_reduce_workload

    return _cuda_pointwise_reduce_workload.make_h100_float32_pointwise_reduce_inputs_device0(
        workload_buffers["x_host_bytes"],
        workload_buffers["bias_host_bytes"],
        rows=WORKLOAD_SHAPE[0],
        columns=WORKLOAD_SHAPE[1],
        required_cuda_visible_devices=required_cuda_visible_devices,
    )


def _require_private_cuda_runtime_roundtrip(roundtrip):
    if roundtrip.get("status") != "ok":
        raise AssertionError(
            "private torch_rs CUDA runtime roundtrip failed: "
            f"{roundtrip.get('reason')}"
        )
    if roundtrip.get("cpu_fallback") is not False:
        raise AssertionError(
            "private torch_rs CUDA runtime roundtrip used CPU fallback"
        )
    if roundtrip.get("device_type") != "cuda" or roundtrip.get("device_index") != 0:
        raise AssertionError(
            "private torch_rs CUDA runtime roundtrip did not run on CUDA device 0"
        )
    if roundtrip.get("checksum_match") is not True:
        raise AssertionError("private torch_rs CUDA runtime checksum did not match")
    if (
        roundtrip.get("device_roundtrip_checksum")
        != roundtrip.get("expected_checksum")
    ):
        raise AssertionError("private torch_rs CUDA runtime checksum is unexpected")


def _require_private_cuda_pointwise_kernel(pointwise):
    if pointwise.get("status") != "ok":
        raise AssertionError(
            "private torch_rs CUDA pointwise kernel failed: "
            f"{pointwise.get('reason')}"
        )
    if pointwise.get("cpu_fallback") is not False:
        raise AssertionError("private torch_rs CUDA pointwise kernel used CPU fallback")
    if pointwise.get("device_type") != "cuda" or pointwise.get("device_index") != 0:
        raise AssertionError(
            "private torch_rs CUDA pointwise kernel did not run on CUDA device 0"
        )
    if pointwise.get("checksum_match") is not True:
        raise AssertionError("private torch_rs CUDA pointwise checksum did not match")
    if pointwise.get("device_output_checksum") != pointwise.get("expected_checksum"):
        raise AssertionError("private torch_rs CUDA pointwise checksum is unexpected")
    if (pointwise.get("launch") or {}).get("sync_error", {}).get("result") != 0:
        raise AssertionError("private torch_rs CUDA pointwise kernel did not sync")


def _require_private_cuda_pointwise_reduce_workload(pointwise_reduce):
    if pointwise_reduce.get("status") != "ok":
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce workload failed: "
            f"{pointwise_reduce.get('reason')}"
        )
    if pointwise_reduce.get("cpu_fallback") is not False:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce workload used CPU fallback"
        )
    if (
        pointwise_reduce.get("device_type") != "cuda"
        or pointwise_reduce.get("device_index") != 0
    ):
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce workload did not run on "
            "CUDA device 0"
        )
    if pointwise_reduce.get("workload_shape") != list(WORKLOAD_SHAPE):
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce workload shape changed"
        )
    if pointwise_reduce.get("output_shape") != [WORKLOAD_SHAPE[0]]:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce output shape changed"
        )
    if pointwise_reduce.get("output_metadata_match") is not True:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce output metadata did not "
            "match PyTorch"
        )
    if pointwise_reduce.get("checksum_match") is not True:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce checksum did not match "
            "PyTorch"
        )
    if (
        pointwise_reduce.get("device_output_checksum")
        != pointwise_reduce.get("pytorch_reference_output_checksum")
    ):
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce checksum is unexpected"
        )
    if (pointwise_reduce.get("launch") or {}).get("sync_error", {}).get("result") != 0:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce kernel did not sync"
        )


def _require_private_cuda_pointwise_reduce_inputs(inputs):
    if inputs.get("status") != "ok":
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce inputs failed: "
            f"{inputs.get('reason')}"
        )
    if inputs.get("cpu_fallback") is not False:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce inputs used CPU fallback"
        )
    if inputs.get("device_type") != "cuda" or inputs.get("device_index") != 0:
        raise AssertionError(
            "private torch_rs CUDA pointwise-reduce inputs did not allocate on "
            "CUDA device 0"
        )
    if inputs.get("workload_shape") != list(WORKLOAD_SHAPE):
        raise AssertionError("private torch_rs CUDA input workload shape changed")
    expected_metadata = [
        {
            "shape": [WORKLOAD_SHAPE[0], WORKLOAD_SHAPE[1]],
            "stride": [WORKLOAD_SHAPE[1], 1],
            "storage_offset": 0,
            "dtype": "torch.float32",
            "device": "cuda:0",
            "device_type": "cuda",
            "device_index": 0,
            "requires_grad": False,
            "is_contiguous": True,
        },
        {
            "shape": [WORKLOAD_SHAPE[1]],
            "stride": [1],
            "storage_offset": 0,
            "dtype": "torch.float32",
            "device": "cuda:0",
            "device_type": "cuda",
            "device_index": 0,
            "requires_grad": False,
            "is_contiguous": True,
        },
    ]
    if inputs.get("input_metadata") != expected_metadata:
        raise AssertionError("private torch_rs CUDA input metadata changed")
    wrappers = inputs.get("public_cuda_tensor_inputs")
    if not isinstance(wrappers, list) or len(wrappers) != 2:
        raise AssertionError("public torch_rs CUDA input wrappers are missing")
    for wrapper, expected in zip(wrappers, expected_metadata):
        if wrapper.get("schema_version") != (
            "torch_rs_public_cuda_benchmark_tensor_v1"
        ):
            raise AssertionError("public torch_rs CUDA input wrapper schema changed")
        if wrapper.get("status") != "ok":
            raise AssertionError("public torch_rs CUDA input wrapper status is not ok")
        if wrapper.get("cpu_fallback") is not False:
            raise AssertionError("public torch_rs CUDA input wrapper used CPU fallback")
        if wrapper.get("device_type") != "cuda" or wrapper.get("device_index") != 0:
            raise AssertionError(
                "public torch_rs CUDA input wrapper did not observe CUDA device 0"
            )
        if wrapper.get("dtype") != "torch.float32":
            raise AssertionError(
                "public torch_rs CUDA input wrapper did not expose torch.float32"
            )
        if wrapper.get("is_cuda") is not True:
            raise AssertionError(
                "public torch_rs CUDA input wrapper did not report CUDA residency"
            )
        if wrapper.get("shape") != expected["shape"]:
            raise AssertionError("public torch_rs CUDA input wrapper shape changed")
        if wrapper.get("stride") != expected["stride"]:
            raise AssertionError("public torch_rs CUDA input wrapper stride changed")
        if (wrapper.get("readback") or {}).get("synchronized") is not True:
            raise AssertionError(
                "public torch_rs CUDA input wrapper readback was not synchronized"
            )
        if not wrapper.get("checksum"):
            raise AssertionError("public torch_rs CUDA input wrapper checksum missing")
    if inputs.get("checksum_match") is not True:
        raise AssertionError("private torch_rs CUDA input checksums did not match")


def _require_public_cuda_tensor_wrapper_evidence(wrapper):
    if wrapper is None:
        raise AssertionError("public torch_rs CUDA tensor wrapper evidence is missing")
    if wrapper.get("schema_version") != (
        "torch_rs_public_cuda_benchmark_tensor_v1"
    ):
        raise AssertionError("public torch_rs CUDA tensor wrapper schema changed")
    if wrapper.get("status") != "ok":
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper status is not ok"
        )
    if wrapper.get("cpu_fallback") is not False:
        raise AssertionError("public torch_rs CUDA tensor wrapper used CPU fallback")
    if wrapper.get("device_type") != "cuda" or wrapper.get("device_index") != 0:
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper did not observe CUDA device 0"
        )
    if wrapper.get("device") != "cuda:0":
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper device is unexpected"
        )
    if wrapper.get("dtype") != "torch.float32":
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper did not expose torch.float32"
        )
    if wrapper.get("is_cuda") is not True:
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper did not report CUDA residency"
        )
    if wrapper.get("shape") != [WORKLOAD_SHAPE[0]]:
        raise AssertionError("public torch_rs CUDA tensor wrapper shape changed")
    if wrapper.get("stride") != [1]:
        raise AssertionError("public torch_rs CUDA tensor wrapper stride changed")
    if (wrapper.get("readback") or {}).get("synchronized") is not True:
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper readback was not synchronized"
        )
    if not wrapper.get("checksum"):
        raise AssertionError(
            "public torch_rs CUDA tensor wrapper checksum is missing"
        )


def _compile_execution_from_output(output):
    metadata = output.metadata()
    compile_execution = metadata.get("compile_execution")
    if type(compile_execution) is not dict:
        raise AssertionError("compiled CUDA output is missing execution metadata")
    return metadata, compile_execution


def _compiled_executor_metadata(compiled):
    executor = getattr(compiled, "_torch_rs_cuda_compile_executor", None)
    if executor is None or not hasattr(executor, "metadata"):
        return None
    metadata = executor.metadata()
    if type(metadata) is not dict:
        return None
    return metadata


def _require_torch_rs_cuda_compile_prepared_executor(
    metadata,
    *,
    expected_invocation_count=None,
):
    if type(metadata) is not dict:
        raise AssertionError("compiled CUDA wrapper is missing prepared executor")
    if metadata.get("schema_version") != PREPARED_EXECUTOR_SCHEMA_VERSION:
        raise AssertionError("compiled CUDA prepared executor schema changed")
    if metadata.get("status") != "ok":
        raise AssertionError(
            "compiled CUDA prepared executor failed: "
            f"{metadata.get('reason')}"
        )
    if metadata.get("prepared") is not True:
        raise AssertionError("compiled CUDA executor was not prepared")
    if metadata.get("setup_hoisted_to_compile_wrapper") is not True:
        raise AssertionError("compiled CUDA setup was not hoisted to the wrapper")
    if not metadata.get("preparation_id"):
        raise AssertionError("compiled CUDA prepared executor id is missing")
    if metadata.get("native_cuda_compile") is not True:
        raise AssertionError("prepared executor did not mark native CUDA compile")
    if metadata.get("eager_fallback") is not False:
        raise AssertionError("prepared executor used eager fallback")
    if metadata.get("forwarded_to_pytorch") is not False:
        raise AssertionError("prepared executor forwarded to PyTorch")
    if metadata.get("compile_backend") != REFERENCE_COMPILE_CONFIG["backend"]:
        raise AssertionError("prepared executor used the wrong backend")
    if metadata.get("compile_fullgraph") is not True:
        raise AssertionError("prepared executor did not use fullgraph=True")
    if metadata.get("compile_dynamic") is not False:
        raise AssertionError("prepared executor did not use dynamic=False")
    if metadata.get("device_type") != "cuda" or metadata.get("device_index") != 0:
        raise AssertionError("prepared executor did not select CUDA device 0")
    if metadata.get("workload_shape") != list(WORKLOAD_SHAPE):
        raise AssertionError("prepared executor workload shape changed")
    if metadata.get("output_shape") != [WORKLOAD_SHAPE[0]]:
        raise AssertionError("prepared executor output shape changed")
    if metadata.get("cpu_fallback") is not False:
        raise AssertionError("prepared executor used CPU fallback")
    if metadata.get("nvcc", {}).get("available") is not True:
        raise AssertionError("prepared executor did not record nvcc")
    if metadata.get("build") is None:
        raise AssertionError("prepared executor did not record kernel build")
    if metadata.get("kernel_library", {}).get("loaded") is not True:
        raise AssertionError("prepared executor did not load the kernel library")
    if (
        expected_invocation_count is not None
        and metadata.get("invocation_count") != expected_invocation_count
    ):
        raise AssertionError(
            "prepared executor invocation count changed: "
            f"expected {expected_invocation_count}, "
            f"got {metadata.get('invocation_count')}"
        )


def _require_torch_rs_cuda_compile_output(
    output,
    *,
    expected_checksum,
    expected_output_bytes,
    expected_output_metadata,
):
    metadata, compile_execution = _compile_execution_from_output(output)
    _require_public_cuda_tensor_wrapper_evidence(metadata)
    if metadata.get("native_cuda_compile") is not True:
        raise AssertionError("compiled CUDA output did not mark native execution")
    if metadata.get("eager_fallback") is not False:
        raise AssertionError("compiled CUDA output used eager fallback")
    if metadata.get("forwarded_to_pytorch") is not False:
        raise AssertionError("compiled CUDA output forwarded to PyTorch")
    if metadata.get("compile_backend") != REFERENCE_COMPILE_CONFIG["backend"]:
        raise AssertionError("compiled CUDA output used the wrong backend")
    if metadata.get("workload_version") != WORKLOAD_VERSION:
        raise AssertionError("compiled CUDA output workload version changed")
    if metadata.get("compile_fullgraph") is not True:
        raise AssertionError("compiled CUDA output did not use fullgraph=True")
    if metadata.get("compile_dynamic") is not False:
        raise AssertionError("compiled CUDA output did not use dynamic=False")

    if compile_execution.get("schema_version") != (
        "torch_rs_private_cuda_pointwise_reduce_compile_execution_v1"
    ):
        raise AssertionError("compiled CUDA execution schema changed")
    if compile_execution.get("status") != "ok":
        raise AssertionError(
            "compiled CUDA execution failed: "
            f"{compile_execution.get('reason')}"
        )
    if compile_execution.get("native_cuda_compile") is not True:
        raise AssertionError("native CUDA compile execution was not demonstrated")
    if compile_execution.get("eager_fallback") is not False:
        raise AssertionError("compiled CUDA execution used eager fallback")
    if compile_execution.get("forwarded_to_pytorch") is not False:
        raise AssertionError("compiled CUDA execution forwarded to PyTorch")
    if compile_execution.get("compile_backend") != REFERENCE_COMPILE_CONFIG["backend"]:
        raise AssertionError("compiled CUDA execution used the wrong backend")
    if compile_execution.get("workload_version") != WORKLOAD_VERSION:
        raise AssertionError("compiled CUDA execution workload version changed")
    if compile_execution.get("compile_fullgraph") is not True:
        raise AssertionError("compiled CUDA execution did not use fullgraph=True")
    if compile_execution.get("compile_dynamic") is not False:
        raise AssertionError("compiled CUDA execution did not use dynamic=False")
    if compile_execution.get("input_device_type") != "cuda":
        raise AssertionError("compiled CUDA execution inputs were not CUDA")
    if compile_execution.get("output_device_type") != "cuda":
        raise AssertionError("compiled CUDA execution output was not CUDA")
    if compile_execution.get("output_metadata") != expected_output_metadata:
        raise AssertionError("compiled CUDA output metadata changed")
    if compile_execution.get("device_output_checksum") != expected_checksum:
        raise AssertionError("compiled CUDA output checksum did not match PyTorch")
    if compile_execution.get("readback_synchronized") is not True:
        raise AssertionError("compiled CUDA output readback was not synchronized")
    if compile_execution.get("launch", {}).get("sync_error", {}).get("result") != 0:
        raise AssertionError("compiled CUDA kernel did not synchronize")
    executor = compile_execution.get("executor")
    if type(executor) is not dict:
        raise AssertionError("compiled CUDA execution is missing executor evidence")
    if executor.get("schema_version") != PREPARED_EXECUTOR_SCHEMA_VERSION:
        raise AssertionError("compiled CUDA execution executor schema changed")
    if executor.get("prepared") is not True:
        raise AssertionError("compiled CUDA execution did not use a prepared executor")
    if executor.get("setup_hoisted_to_compile_wrapper") is not True:
        raise AssertionError("compiled CUDA execution repeated wrapper setup")
    if not executor.get("preparation_id"):
        raise AssertionError("compiled CUDA execution executor id is missing")
    if type(executor.get("invocation_index")) is not int:
        raise AssertionError("compiled CUDA execution invocation index is missing")
    if type(executor.get("reused_prepared_executor")) is not bool:
        raise AssertionError("compiled CUDA execution reuse marker is missing")

    comparison = _compare_compiled_output_bytes(
        output,
        compile_execution,
        expected_output_bytes,
    )
    compile_execution["expected_output_checksum"] = expected_checksum
    compile_execution["output_comparison"] = comparison
    if comparison["exact_bytes_match"] is not True:
        raise AssertionError("compiled CUDA output bytes did not match PyTorch")
    return metadata, compile_execution


def _compare_compiled_output_bytes(output, compile_execution, expected_output_bytes):
    from torch_rs import _cuda_pointwise_reduce_workload

    output_buffer = output._torch_rs_private_cuda_buffer()
    readback = output_buffer.checksum_readback(
        lambda payload: _cuda_pointwise_reduce_workload._checksum_output_bytes(
            payload,
            WORKLOAD_SHAPE[0],
            WORKLOAD_SHAPE[1],
        )
    )
    if readback.copy_call.get("result") != 0:
        return {
            "exact_bytes_match": False,
            "reason": "compiled output verification readback failed",
            "copy_call": readback.copy_call,
            "sync_call": readback.sync_call,
        }
    if readback.sync_call is None or readback.sync_call.get("result") != 0:
        return {
            "exact_bytes_match": False,
            "reason": "compiled output verification sync failed",
            "copy_call": readback.copy_call,
            "sync_call": readback.sync_call,
        }
    comparison = _cuda_pointwise_reduce_workload._float32_comparison(
        readback.payload or b"",
        expected_output_bytes,
    )
    comparison.update(
        {
            "copy_call": readback.copy_call,
            "sync_call": readback.sync_call,
            "synchronized": True,
            "private_output_bytes_checksum": readback.checksum,
            "expected_byte_count": len(expected_output_bytes),
        }
    )
    return {
        **comparison,
        "recorded_private_output_bytes_checksum": compile_execution.get(
            "device_output_bytes_checksum"
        ),
    }


def _time_torch_rs_cuda_compile_once(compiled, inputs):
    started_ns = time.perf_counter_ns()
    output = compiled(*inputs)
    elapsed_ns = time.perf_counter_ns() - started_ns
    metadata, compile_execution = _compile_execution_from_output(output)
    return (
        elapsed_ns,
        compile_execution["device_output_checksum"],
        output,
        metadata,
        compile_execution,
    )


def _time_torch_rs_cuda_compile_repeated(compiled, inputs, repeats):
    started_ns = time.perf_counter_ns()
    output = None
    for _ in range(repeats):
        output = compiled(*inputs)
    elapsed_ns = time.perf_counter_ns() - started_ns
    metadata, compile_execution = _compile_execution_from_output(output)
    return (
        elapsed_ns,
        compile_execution["device_output_checksum"],
        metadata,
        compile_execution,
    )


def _time_torch_rs_cuda_unprepared_compatibility_repeated(
    inputs,
    repeats,
    required_cuda_visible_devices,
):
    from torch_rs import _cuda_pointwise_reduce_workload

    started_ns = time.perf_counter_ns()
    output = None
    for _ in range(repeats):
        if output is not None:
            output._torch_rs_close_private_cuda_buffer()
        output = (
            _cuda_pointwise_reduce_workload
            .execute_h100_float32_pointwise_reduce_compiled_device0(
                *inputs,
                required_cuda_visible_devices=required_cuda_visible_devices,
            )
        )
    elapsed_ns = time.perf_counter_ns() - started_ns
    try:
        _metadata, compile_execution = _compile_execution_from_output(output)
        return elapsed_ns, compile_execution["device_output_checksum"], compile_execution
    finally:
        if output is not None:
            output._torch_rs_close_private_cuda_buffer()


def _run_torch_rs_cuda_unprepared_compatibility_comparison(
    args,
    input_bundle,
    expected_checksum,
):
    sample_ns = []
    sample_checksums = []
    last_execution = None

    for _ in range(args.warmups):
        _elapsed_ns, _checksum, last_execution = (
            _time_torch_rs_cuda_unprepared_compatibility_repeated(
                input_bundle.inputs,
                args.repeats,
                args.required_cuda_visible_devices,
            )
        )

    for _ in range(args.samples):
        elapsed_ns, checksum, last_execution = (
            _time_torch_rs_cuda_unprepared_compatibility_repeated(
                input_bundle.inputs,
                args.repeats,
                args.required_cuda_visible_devices,
            )
        )
        sample_ns.append(elapsed_ns)
        sample_checksums.append(checksum)

    checksums = sorted(set(sample_checksums))
    if checksums != [expected_checksum]:
        raise AssertionError(
            "unprepared torch_rs CUDA workload produced unstable checksums: "
            f"{checksums!r}"
        )

    return {
        "implementation": "torch_rs",
        "status": "ok",
        "measurement": "compatibility_execute_prepares_each_call",
        "workload_version": WORKLOAD_VERSION,
        "compile_backend": REFERENCE_COMPILE_CONFIG["backend"],
        "compile_fullgraph": True,
        "compile_dynamic": False,
        "native_cuda_compile": True,
        "eager_fallback": False,
        "forwarded_to_pytorch": False,
        "steady": _summarize_samples(sample_ns, args.repeats),
        "steady_checksums": checksums,
        "last_compile_execution": last_execution,
        "setup_repeated_per_invocation": True,
    }


def _run_torch_rs_cuda_compile(torch_rs, args, workload_buffers):
    input_bundle, input_evidence = _torch_rs_private_cuda_pointwise_reduce_inputs(
        args.required_cuda_visible_devices,
        workload_buffers,
    )
    _require_private_cuda_pointwise_reduce_inputs(input_evidence)
    assert input_bundle is not None

    mask_attribute = "_torch_rs_cuda_compile_required_cuda_visible_devices"
    missing_attribute = object()
    previous_mask = getattr(
        h100_cuda_pointwise_reduce_float32,
        mask_attribute,
        missing_attribute,
    )
    setattr(
        h100_cuda_pointwise_reduce_float32,
        mask_attribute,
        args.required_cuda_visible_devices,
    )
    try:
        factory_started_ns = time.perf_counter_ns()
        compiled = torch_rs.compile(
            h100_cuda_pointwise_reduce_float32,
            backend=REFERENCE_COMPILE_CONFIG["backend"],
            fullgraph=REFERENCE_COMPILE_CONFIG["fullgraph"],
            dynamic=REFERENCE_COMPILE_CONFIG["dynamic"],
        )
        factory_ns = time.perf_counter_ns() - factory_started_ns
        prepared_executor_before_first_call = _compiled_executor_metadata(compiled)
        _require_torch_rs_cuda_compile_prepared_executor(
            prepared_executor_before_first_call,
            expected_invocation_count=0,
        )

        (
            cold_ns,
            cold_checksum,
            cold_output,
            cold_metadata,
            cold_execution,
        ) = _time_torch_rs_cuda_compile_once(compiled, input_bundle.inputs)
        cold_metadata, cold_execution = _require_torch_rs_cuda_compile_output(
            cold_output,
            expected_checksum=workload_buffers["expected_output_checksum"],
            expected_output_bytes=workload_buffers["expected_output_bytes"],
            expected_output_metadata=workload_buffers["expected_output_metadata"],
        )
        last_execution = cold_execution

        for _ in range(args.warmups):
            (
                _warmup_ns,
                _warmup_checksum,
                _warmup_metadata,
                warmup_execution,
            ) = _time_torch_rs_cuda_compile_repeated(
                compiled,
                input_bundle.inputs,
                args.repeats,
            )
            last_execution = warmup_execution

        sample_ns = []
        sample_checksums = []
        for _ in range(args.samples):
            elapsed_ns, checksum, _sample_metadata, sample_execution = (
                _time_torch_rs_cuda_compile_repeated(
                    compiled,
                    input_bundle.inputs,
                    args.repeats,
                )
            )
            last_execution = sample_execution
            sample_ns.append(elapsed_ns)
            sample_checksums.append(checksum)

        checksums = sorted(set([cold_checksum, *sample_checksums]))
        if checksums != [workload_buffers["expected_output_checksum"]]:
            raise AssertionError(
                "compiled torch_rs CUDA workload produced unstable checksums: "
                f"{checksums!r}"
            )

        expected_invocations = (
            1 + args.warmups * args.repeats + args.samples * args.repeats
        )
        prepared_executor_after_timing = _compiled_executor_metadata(compiled)
        _require_torch_rs_cuda_compile_prepared_executor(
            prepared_executor_after_timing,
            expected_invocation_count=expected_invocations,
        )
        cold_executor = cold_execution.get("executor") or {}
        last_executor = last_execution.get("executor") or {}
        prepared_executor_reuse = {
            "preparation_id": prepared_executor_after_timing["preparation_id"],
            "before_first_call_invocation_count": (
                prepared_executor_before_first_call["invocation_count"]
            ),
            "expected_invocation_count": expected_invocations,
            "observed_invocation_count": (
                prepared_executor_after_timing["invocation_count"]
            ),
            "cold_invocation_index": cold_executor.get("invocation_index"),
            "last_invocation_index": last_executor.get("invocation_index"),
            "cold_reused_prepared_executor": cold_executor.get(
                "reused_prepared_executor"
            ),
            "steady_state_reused_prepared_executor": last_executor.get(
                "reused_prepared_executor"
            )
            is True,
            "setup_hoisted_to_compile_wrapper": (
                prepared_executor_after_timing.get(
                    "setup_hoisted_to_compile_wrapper"
                )
                is True
            ),
        }
        if (
            prepared_executor_reuse["observed_invocation_count"]
            != expected_invocations
            or prepared_executor_reuse["last_invocation_index"]
            != expected_invocations
            or not prepared_executor_reuse["steady_state_reused_prepared_executor"]
        ):
            raise AssertionError("compiled CUDA prepared executor was not reused")

        unprepared_comparison = None
        if args.include_unprepared_comparison:
            unprepared_comparison = (
                _run_torch_rs_cuda_unprepared_compatibility_comparison(
                    args,
                    input_bundle,
                    workload_buffers["expected_output_checksum"],
                )
            )
            unprepared_median = unprepared_comparison["steady"]["median_us"]
            prepared_median = _summarize_samples(
                sample_ns,
                args.repeats,
            )["median_us"]
            prepared_executor_reuse[
                "unprepared_compatibility_steady_median_us"
            ] = unprepared_median
            prepared_executor_reuse["prepared_steady_median_us"] = prepared_median
            prepared_executor_reuse[
                "prepared_vs_unprepared_steady_speedup"
            ] = (unprepared_median / prepared_median if prepared_median else None)

        evidence = {
            "implementation": "torch_rs",
            "status": "ok",
            "workload_version": WORKLOAD_VERSION,
            "compile_backend": REFERENCE_COMPILE_CONFIG["backend"],
            "compile_config": dict(REFERENCE_COMPILE_CONFIG),
            "compile_fullgraph": True,
            "compile_dynamic": False,
            "input_device_type": "cuda",
            "output_device_type": "cuda",
            "native_cuda_compile": True,
            "eager_fallback": False,
            "forwarded_to_pytorch": False,
            "factory_us": factory_ns / 1000.0,
            "cold_first_call_us": cold_ns / 1000.0,
            "cold_checksum": cold_checksum,
            "steady": _summarize_samples(sample_ns, args.repeats),
            "steady_checksums": sorted(set(sample_checksums)),
            "input_metadata": cold_execution["input_metadata"],
            "input_tensor_evidence": input_evidence,
            "output_metadata": cold_execution["output_metadata"],
            "output_tensor_wrapper": cold_metadata,
            "prepared_executor": prepared_executor_before_first_call,
            "prepared_executor_after_timing": prepared_executor_after_timing,
            "prepared_executor_reuse": prepared_executor_reuse,
            "compile_execution": cold_execution,
            "last_compile_execution": last_execution,
            "unprepared_compatibility_comparison": unprepared_comparison,
            "correctness": {
                "pytorch_reference_checksum": workload_buffers[
                    "expected_output_checksum"
                ],
                "exact_checksum_match": True,
            },
        }
        classification = classify_torch_rs_cuda_compile_evidence(evidence)
        evidence["eligibility"] = classification
        evidence["score_credit"] = classification["score_credit"]
        if not classification["eligible_cuda_compile_evidence"]:
            evidence["status"] = "zero_credit_rejected"
        return evidence
    finally:
        if previous_mask is missing_attribute:
            try:
                delattr(h100_cuda_pointwise_reduce_float32, mask_attribute)
            except AttributeError:
                pass
        else:
            setattr(h100_cuda_pointwise_reduce_float32, mask_attribute, previous_mask)
        input_bundle.close()


def torch_rs_zero_credit_unsupported_row(torch_rs, cuda_tensor_evidence=None):
    evidence = {
        "implementation": "torch_rs",
        "status": "unsupported",
        "workload_version": WORKLOAD_VERSION,
        "compile_backend": None,
        "compile_fullgraph": None,
        "compile_dynamic": None,
        "input_device_type": None,
        "output_device_type": None,
        "native_cuda_compile": False,
        "eager_fallback": False,
        "forwarded_to_pytorch": False,
    }
    classification = classify_torch_rs_cuda_compile_evidence(evidence)
    return {
        "implementation": "torch_rs",
        "status": "zero_credit_unsupported",
        "workload_version": WORKLOAD_VERSION,
        "score_credit": 0.0,
        "reason": (
            "this unsupported row helper records missing CUDA compile evidence "
            "explicitly instead of substituting CPU execution, backend='eager', "
            "eager fallback, or installed-PyTorch forwarding. Private "
            "benchmark-only CUDA driver/runtime/kernel/workload evidence and "
            "the public benchmark tensor wrapper alone are prerequisites, not "
            "compiled torch_rs CUDA output; prerequisite-only evidence does "
            "not receive compile credit."
        ),
        "cuda_probes": _torch_rs_public_cuda_probes(torch_rs),
        "prerequisite_cuda_tensor_evidence": cuda_tensor_evidence,
        "rejected_fallbacks": [
            "CPU tensor execution",
            "backend='eager' compile execution",
            "eager fallback",
            "forwarding to installed PyTorch",
            "skipped or missing execution",
        ],
        "eligibility": classification,
    }


def _device_provenance(reference_torch):
    properties = reference_torch.cuda.get_device_properties(0)
    memory_free, memory_total = reference_torch.cuda.mem_get_info(0)
    return {
        "torch_cuda_device_index": 0,
        "torch_cuda_device_name": reference_torch.cuda.get_device_name(0),
        "torch_cuda_device_capability": list(
            reference_torch.cuda.get_device_capability(0)
        ),
        "torch_cuda_total_memory_bytes": int(properties.total_memory),
        "torch_cuda_multiprocessor_count": int(properties.multi_processor_count),
        "torch_cuda_memory_free_bytes": int(memory_free),
        "torch_cuda_memory_total_bytes": int(memory_total),
        "nvidia_smi": _nvidia_smi_query(),
        "nvcc_version": _nvcc_version(),
    }


def _environment(reference_torch, torch_rs, args):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "workload_version": WORKLOAD_VERSION,
        "python": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "warmups": args.warmups,
        "samples": args.samples,
        "repeats": args.repeats,
        "pytorch": {
            "version": reference_torch.__version__,
            "path": getattr(reference_torch, "__file__", None),
            "cuda_runtime": getattr(
                getattr(reference_torch, "version", None),
                "cuda",
                None,
            ),
            "git_version": getattr(
                getattr(reference_torch, "version", None),
                "git_version",
                None,
            ),
            "cuda_available": bool(reference_torch.cuda.is_available()),
            "cuda_device_count": int(reference_torch.cuda.device_count()),
            "cudnn_version": reference_torch.backends.cudnn.version(),
        },
        "torch_rs": {
            "version": _package_version("torch-rs", torch_rs),
            "path": getattr(torch_rs, "__file__", None),
        },
        "reference_compile_config": dict(REFERENCE_COMPILE_CONFIG),
        "workload": {
            "name": h100_cuda_pointwise_reduce_float32.__name__,
            "shape": list(WORKLOAD_SHAPE),
            "dtype": "torch.float32",
            "seed": WORKLOAD_SEED,
        },
        "gpu": _device_provenance(reference_torch),
        "git": _git_provenance(),
    }


def run_benchmark(args):
    _validate_counts(args)
    import torch as reference_torch
    import torch_rs

    torch_rs_cuda_driver_probe = _torch_rs_private_cuda_driver_probe()
    _require_reference_environment(reference_torch, args)
    torch_rs_cuda_runtime_roundtrip = _torch_rs_private_cuda_runtime_roundtrip(
        args.required_cuda_visible_devices,
    )
    _require_private_cuda_runtime_roundtrip(torch_rs_cuda_runtime_roundtrip)
    torch_rs_cuda_pointwise_kernel = _torch_rs_private_cuda_pointwise_kernel(
        args.required_cuda_visible_devices,
    )
    _require_private_cuda_pointwise_kernel(torch_rs_cuda_pointwise_kernel)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        pytorch_reference, private_workload_buffers = _run_pytorch_reference(
            reference_torch,
            args,
        )
        torch_rs_cuda_pointwise_reduce_workload = (
            _torch_rs_private_cuda_pointwise_reduce_workload(
                args.required_cuda_visible_devices,
                private_workload_buffers,
            )
        )
        _require_private_cuda_pointwise_reduce_workload(
            torch_rs_cuda_pointwise_reduce_workload,
        )
        prerequisite_cuda_tensor_wrapper = torch_rs_cuda_pointwise_reduce_workload[
            "public_cuda_tensor_wrapper"
        ]
        _require_public_cuda_tensor_wrapper_evidence(
            prerequisite_cuda_tensor_wrapper
        )
        torch_rs_row = _run_torch_rs_cuda_compile(
            torch_rs,
            args,
            private_workload_buffers,
        )
    finally:
        if gc_was_enabled:
            gc.enable()

    eligible = torch_rs_row["eligibility"]["eligible_cuda_compile_evidence"]
    if eligible:
        torch_rs_median = torch_rs_row["steady"]["median_us"]
        reference_median = pytorch_reference["steady"]["median_us"]
        speed_ratio = reference_median / torch_rs_median if torch_rs_median else 0.0
        score_percent = min(1.0, speed_ratio) * 100.0
    else:
        speed_ratio = None
        score_percent = 0.0

    return {
        "environment": _environment(reference_torch, torch_rs, args),
        "reference_workload": pytorch_reference,
        "torch_rs_cuda_driver_probe": torch_rs_cuda_driver_probe,
        "torch_rs_cuda_runtime_roundtrip": torch_rs_cuda_runtime_roundtrip,
        "torch_rs_cuda_pointwise_kernel": torch_rs_cuda_pointwise_kernel,
        "torch_rs_cuda_pointwise_reduce_workload": (
            torch_rs_cuda_pointwise_reduce_workload
        ),
        "torch_rs_cuda_compile_inputs": torch_rs_row["input_tensor_evidence"],
        "torch_rs_cuda_tensor_wrapper": torch_rs_row["output_tensor_wrapper"],
        "torch_rs_cuda_prerequisite_tensor_wrapper": prerequisite_cuda_tensor_wrapper,
        "candidate": torch_rs_row,
        "aggregates": {
            "common_success_geomean_speed_ratio": speed_ratio,
            "coverage_adjusted_overall_percent": score_percent,
            "torch_rs_cuda_compile_score_percent": score_percent,
            "zero_credit_unsupported_cell_count": 0 if eligible else 1,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--required-cuda-visible-devices",
        default=DEFAULT_REQUIRED_CUDA_VISIBLE_DEVICES,
        help=(
            "required CUDA_VISIBLE_DEVICES value; use an empty string to skip "
            "this check for local experimentation"
        ),
    )
    parser.add_argument(
        "--allow-non-h100",
        action="store_true",
        help="allow running the reference benchmark on a non-H100 CUDA device",
    )
    parser.add_argument(
        "--include-unprepared-comparison",
        action="store_true",
        help=(
            "also time the compatibility path that prepares CUDA compile setup "
            "on each invocation; this is recorded as non-scoring evidence"
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.required_cuda_visible_devices == "":
        args.required_cuda_visible_devices = None
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
