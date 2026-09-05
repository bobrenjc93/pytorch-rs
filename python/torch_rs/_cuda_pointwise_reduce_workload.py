"""Private benchmark-only CUDA pointwise-plus-row-reduction workload.

This module is intentionally separate from ``torch_rs.cuda``. It verifies that
benchmark code can allocate torch_rs-owned CUDA buffers and run the H100 CUDA
``torch.compile`` reference workload shape without claiming public CUDA tensor
or compile support.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

from . import _cuda_driver_probe
from . import _cuda_pointwise_kernel as _cuda_kernel_support


POINTWISE_REDUCE_SCHEMA_VERSION = (
    "torch_rs_private_cuda_pointwise_reduce_workload_v1"
)
POINTWISE_REDUCE_KERNEL_VERSION = (
    "h100_cuda_pointwise_plus_row_reduce_float32_v1"
)
WORKLOAD_SHAPE = (1024, 1024)
OUTPUT_SHAPE = (1024,)

_FLOAT32_SIZE = 4
_THREADS_PER_BLOCK = 32
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_UNAVAILABLE_ERRORS = {
    "cudaErrorInsufficientDriver",
    "cudaErrorNoDevice",
}
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void torch_rs_private_pointwise_reduce_kernel_v1(
    const float* x,
    const float* bias,
    float* output,
    int rows,
    int columns
) {
    int row = blockIdx.x;
    int lane = threadIdx.x & 31;
    if (row >= rows) {
        return;
    }

    float partial = 0.0f;
    int row_offset = row * columns;
    for (int base_column = lane * 4; base_column < columns; base_column += 128) {
#pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
            int column = base_column + offset;
            if (column < columns) {
                float x_value = x[row_offset + column];
                float bias_value = bias[column];
                float mixed = sinf(x_value + bias_value)
                    * cosf(x_value - bias_value);
                float relu = x_value > 0.0f ? x_value : 0.0f;
                partial += mixed + relu;
            }
        }
    }

    unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(mask, partial, offset, 32);
    }

    if (lane == 0) {
        output[row] = partial;
    }
}

extern "C" int torch_rs_private_h100_pointwise_reduce_float32_v1(
    const float* x,
    const float* bias,
    float* output,
    int rows,
    int columns,
    int* blocks_out,
    int* threads_out,
    int* launch_error_out,
    int* sync_error_out
) {
    if (x == nullptr || bias == nullptr || output == nullptr) {
        return -1;
    }
    if (rows <= 0 || columns <= 0) {
        return -2;
    }

    int threads = 32;
    int blocks = rows;
    if (blocks_out != nullptr) {
        *blocks_out = blocks;
    }
    if (threads_out != nullptr) {
        *threads_out = threads;
    }

    torch_rs_private_pointwise_reduce_kernel_v1<<<blocks, threads>>>(
        x,
        bias,
        output,
        rows,
        columns
    );
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error_out != nullptr) {
        *launch_error_out = static_cast<int>(launch_error);
    }
    if (launch_error != cudaSuccess) {
        if (sync_error_out != nullptr) {
            *sync_error_out = 0;
        }
        return static_cast<int>(launch_error);
    }

    cudaError_t sync_error = cudaDeviceSynchronize();
    if (sync_error_out != nullptr) {
        *sync_error_out = static_cast<int>(sync_error);
    }
    if (sync_error != cudaSuccess) {
        return static_cast<int>(sync_error);
    }

    return 0;
}
"""


def _checksum_payload(domain: bytes, payload: bytes, rows: int, columns: int) -> str:
    checksum = hashlib.blake2b(digest_size=16)
    checksum.update(domain)
    checksum.update(rows.to_bytes(8, "little"))
    checksum.update(columns.to_bytes(8, "little"))
    checksum.update(payload)
    return checksum.hexdigest()


def _checksum_x_bytes(payload: bytes, rows: int, columns: int) -> str:
    return _checksum_payload(
        b"torch_rs_private_cuda_pointwise_reduce_x_v1",
        payload,
        rows,
        columns,
    )


def _checksum_bias_bytes(payload: bytes, rows: int, columns: int) -> str:
    return _checksum_payload(
        b"torch_rs_private_cuda_pointwise_reduce_bias_v1",
        payload,
        rows,
        columns,
    )


def _checksum_output_bytes(payload: bytes, rows: int, columns: int) -> str:
    return _checksum_payload(
        b"torch_rs_private_cuda_pointwise_reduce_output_v1",
        payload,
        rows,
        columns,
    )


def _output_metadata(rows: int, device_index: int | None) -> dict[str, Any]:
    return {
        "shape": [rows],
        "stride": [1],
        "storage_offset": 0,
        "dtype": "torch.float32",
        "device": f"cuda:{device_index}" if device_index is not None else None,
        "device_type": "cuda" if device_index is not None else None,
        "device_index": device_index,
        "requires_grad": False,
        "is_contiguous": True,
    }


def _input_metadata(
    rows: int,
    columns: int,
    device_index: int | None,
) -> list[dict[str, Any]]:
    device = f"cuda:{device_index}" if device_index is not None else None
    device_type = "cuda" if device_index is not None else None
    return [
        {
            "shape": [rows, columns],
            "stride": [columns, 1],
            "storage_offset": 0,
            "dtype": "torch.float32",
            "device": device,
            "device_type": device_type,
            "device_index": device_index,
            "requires_grad": False,
            "is_contiguous": True,
        },
        {
            "shape": [columns],
            "stride": [1],
            "storage_offset": 0,
            "dtype": "torch.float32",
            "device": device,
            "device_type": device_type,
            "device_index": device_index,
            "requires_grad": False,
            "is_contiguous": True,
        },
    ]


def _checksum_tensor_metadata_values(payload: bytes, metadata: dict[str, Any]) -> str:
    element_count = len(payload) // _FLOAT32_SIZE
    values = list(struct.unpack(f"<{element_count}f", payload))
    encoded = json.dumps(
        {
            "metadata": metadata,
            "values": values,
        },
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=8).hexdigest()


def _float32_comparison(
    actual: bytes,
    expected: bytes | None,
) -> dict[str, Any]:
    if expected is None:
        return {
            "expected_provided": False,
            "exact_bytes_match": None,
            "mismatched_element_count": None,
            "max_abs_diff": None,
            "max_rel_diff": None,
        }
    if len(actual) != len(expected):
        return {
            "expected_provided": True,
            "exact_bytes_match": False,
            "mismatched_element_count": None,
            "max_abs_diff": None,
            "max_rel_diff": None,
            "reason": "expected byte length does not match output byte length",
        }

    element_count = len(actual) // _FLOAT32_SIZE
    actual_values = struct.unpack(f"<{element_count}f", actual)
    expected_values = struct.unpack(f"<{element_count}f", expected)
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatched = 0
    for actual_value, expected_value in zip(actual_values, expected_values):
        absolute = abs(actual_value - expected_value)
        if absolute:
            mismatched += 1
            max_abs_diff = max(max_abs_diff, absolute)
            denominator = max(abs(expected_value), 1.0e-30)
            max_rel_diff = max(max_rel_diff, absolute / denominator)

    return {
        "expected_provided": True,
        "exact_bytes_match": actual == expected,
        "mismatched_element_count": mismatched,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
    }


def _runtime_call(
    runtime: ctypes.CDLL,
    function_name: str,
    *arguments: Any,
) -> dict[str, Any]:
    return _cuda_kernel_support._runtime_call(runtime, function_name, *arguments)


def _configure_runtime_symbols(runtime: ctypes.CDLL) -> list[str]:
    return _cuda_kernel_support._configure_runtime_symbols(runtime)


def _target_build_directory(root: Path, key: str) -> Path:
    target = root / "target"
    if target.is_symlink():
        raise RuntimeError(f"refusing symlinked target directory: {target}")
    target.mkdir(exist_ok=True)
    pointwise_reduce_target = target / "torch_rs_private_cuda_pointwise_reduce"
    if pointwise_reduce_target.is_symlink():
        raise RuntimeError(
            "refusing symlinked CUDA pointwise-reduce build directory: "
            f"{pointwise_reduce_target}"
        )
    pointwise_reduce_target.mkdir(exist_ok=True)
    build_directory = pointwise_reduce_target / key
    if build_directory.is_symlink():
        raise RuntimeError(
            "refusing symlinked CUDA pointwise-reduce build directory: "
            f"{build_directory}"
        )
    build_directory.mkdir(parents=True, exist_ok=True)
    resolved = build_directory.resolve()
    if root not in (resolved, *resolved.parents):
        raise RuntimeError(
            "CUDA pointwise-reduce build directory resolved outside worktree: "
            f"{resolved}"
        )
    return resolved


def _build_kernel(
    *,
    repository_root: Path,
    driver_probe: dict[str, Any],
    nvcc: dict[str, Any],
) -> tuple[Path | None, dict[str, Any]]:
    nvcc_path = nvcc.get("path")
    source_checksum = hashlib.blake2b(
        _CUDA_SOURCE.encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    nvcc_version_output = (nvcc.get("version") or {}).get("stdout") or ""
    build_key_payload = "\n".join(
        [
            source_checksum,
            str(nvcc_path),
            nvcc_version_output,
            _cuda_kernel_support._compute_capability_arch(driver_probe),
        ]
    ).encode("utf-8")
    build_key = hashlib.blake2b(build_key_payload, digest_size=8).hexdigest()
    build: dict[str, Any] = {
        "source_checksum": source_checksum,
        "architecture": _cuda_kernel_support._compute_capability_arch(driver_probe),
        "build_key": build_key,
        "source_path": None,
        "library_path": None,
        "reused": False,
        "compile": None,
    }
    if not nvcc_path:
        return None, build

    build_directory = _target_build_directory(repository_root, build_key)
    source_path = build_directory / "torch_rs_private_pointwise_reduce.cu"
    library_path = build_directory / "libtorch_rs_private_pointwise_reduce.so"
    build["source_path"] = str(source_path)
    build["library_path"] = str(library_path)

    for artifact_path in (source_path, library_path):
        if artifact_path.is_symlink():
            raise RuntimeError(
                "refusing symlinked CUDA pointwise-reduce artifact path: "
                f"{artifact_path}"
            )

    if (
        not source_path.exists()
        or source_path.read_text(encoding="utf-8") != _CUDA_SOURCE
    ):
        source_path.write_text(_CUDA_SOURCE, encoding="utf-8")

    if library_path.exists():
        build["reused"] = True
        return library_path, build

    command = [
        str(nvcc_path),
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-O2",
        "-std=c++17",
        f"-arch={build['architecture']}",
        str(source_path),
        "-o",
        str(library_path),
    ]
    compile_result = _cuda_kernel_support._run_text(command, cwd=repository_root)
    build["compile"] = compile_result
    if compile_result["returncode"] != 0:
        return None, build
    return library_path, build


def _load_kernel_library(
    library_path: Path,
    runtime_library: str | None,
) -> tuple[ctypes.CDLL | None, dict[str, Any]]:
    if runtime_library is not None:
        try:
            ctypes.CDLL(runtime_library, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    try:
        library = ctypes.CDLL(str(library_path))
    except OSError as error:
        return None, {"loaded": False, "error": str(error)}

    function_name = "torch_rs_private_h100_pointwise_reduce_float32_v1"
    try:
        function = getattr(library, function_name)
    except AttributeError as error:
        return None, {
            "loaded": False,
            "error": str(error),
            "function": function_name,
        }

    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    function.restype = ctypes.c_int
    return library, {
        "loaded": True,
        "error": None,
        "function": function_name,
    }


def _base_result(
    *,
    rows: int,
    columns: int,
    x_bytes: bytes,
    bias_bytes: bytes,
    expected_output_bytes: bytes | None,
    expected_output_checksum: str | None,
    expected_output_metadata: dict[str, Any] | None,
    required_cuda_visible_devices: str | None,
    runtime: ctypes.CDLL | None,
    runtime_library: str | None,
    runtime_load_error: str | None,
    driver_probe: dict[str, Any],
    nvcc: dict[str, Any],
) -> dict[str, Any]:
    output_byte_count = rows * _FLOAT32_SIZE
    return {
        "schema_version": POINTWISE_REDUCE_SCHEMA_VERSION,
        "primitive": "torch_rs_private_cuda_h100_pointwise_reduce_float32_device0",
        "public_torch_cuda_api": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_cuda_visible_devices": required_cuda_visible_devices,
        "cuda_visible_devices_match": (
            required_cuda_visible_devices is None
            or os.environ.get("CUDA_VISIBLE_DEVICES") == required_cuda_visible_devices
        ),
        "single_visible_cuda_device": None,
        "status": "unavailable",
        "reason": None,
        "cpu_fallback": False,
        "device_type": None,
        "device_index": None,
        "dtype": "float32",
        "workload_shape": [rows, columns],
        "output_shape": [rows],
        "x_byte_count": len(x_bytes),
        "bias_byte_count": len(bias_bytes),
        "output_byte_count": output_byte_count,
        "pointwise_reduce_kernel_version": POINTWISE_REDUCE_KERNEL_VERSION,
        "host_x_checksum": _checksum_x_bytes(x_bytes, rows, columns),
        "host_bias_checksum": _checksum_bias_bytes(bias_bytes, rows, columns),
        "expected_output_bytes_checksum": (
            _checksum_output_bytes(expected_output_bytes, rows, columns)
            if expected_output_bytes is not None
            else None
        ),
        "device_output_bytes_checksum": None,
        "pytorch_reference_output_checksum": expected_output_checksum,
        "device_output_checksum": None,
        "checksum_match": False,
        "expected_output_metadata": expected_output_metadata,
        "output_metadata": None,
        "output_metadata_match": False,
        "output_comparison": _float32_comparison(b"", None),
        "device_x_pointer_nonzero": False,
        "device_bias_pointer_nonzero": False,
        "device_output_pointer_nonzero": False,
        "driver": driver_probe["driver"],
        "runtime": _cuda_kernel_support._runtime_versions(
            runtime,
            runtime_library,
            runtime_load_error,
        ),
        "device_0": driver_probe["device_0"],
        "gpu": _cuda_kernel_support._gpu_provenance(driver_probe),
        "nvcc": nvcc,
        "build": None,
        "kernel_library": None,
        "launch": None,
        "calls": {},
    }


def _validate_inputs(
    x_host_bytes: bytes,
    bias_host_bytes: bytes,
    rows: int,
    columns: int,
    expected_output_bytes: bytes | None,
    expected_output_checksum: str | None,
    expected_output_metadata: dict[str, Any] | None,
    required_cuda_visible_devices: str | None,
) -> None:
    if type(x_host_bytes) is not bytes:
        raise TypeError("x_host_bytes must be bytes")
    if type(bias_host_bytes) is not bytes:
        raise TypeError("bias_host_bytes must be bytes")
    if type(rows) is not int:
        raise TypeError("rows must be int")
    if type(columns) is not int:
        raise TypeError("columns must be int")
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")
    if len(x_host_bytes) != rows * columns * _FLOAT32_SIZE:
        raise ValueError("x_host_bytes length does not match rows * columns")
    if len(bias_host_bytes) != columns * _FLOAT32_SIZE:
        raise ValueError("bias_host_bytes length does not match columns")
    if (
        expected_output_bytes is not None
        and type(expected_output_bytes) is not bytes
    ):
        raise TypeError("expected_output_bytes must be bytes or None")
    if (
        expected_output_bytes is not None
        and len(expected_output_bytes) != rows * _FLOAT32_SIZE
    ):
        raise ValueError("expected_output_bytes length does not match rows")
    if (
        expected_output_checksum is not None
        and type(expected_output_checksum) is not str
    ):
        raise TypeError("expected_output_checksum must be str or None")
    if (
        expected_output_metadata is not None
        and type(expected_output_metadata) is not dict
    ):
        raise TypeError("expected_output_metadata must be dict or None")
    has_any_expected = (
        expected_output_bytes is not None
        or expected_output_checksum is not None
        or expected_output_metadata is not None
    )
    has_all_expected = (
        expected_output_bytes is not None
        and expected_output_checksum is not None
        and expected_output_metadata is not None
    )
    if has_any_expected and not has_all_expected:
        raise ValueError(
            "expected_output_bytes, expected_output_checksum, and "
            "expected_output_metadata must be provided together"
        )
    if (
        required_cuda_visible_devices is not None
        and type(required_cuda_visible_devices) is not str
    ):
        raise TypeError("required_cuda_visible_devices must be str or None")


def launch_h100_float32_pointwise_reduce_device0(
    x_host_bytes: bytes,
    bias_host_bytes: bytes,
    *,
    rows: int = WORKLOAD_SHAPE[0],
    columns: int = WORKLOAD_SHAPE[1],
    expected_output_bytes: bytes | None = None,
    expected_output_checksum: str | None = None,
    expected_output_metadata: dict[str, Any] | None = None,
    required_cuda_visible_devices: str | None = "0",
) -> dict[str, Any]:
    """Run the H100 reference pointwise-plus-row-reduction CUDA kernel.

    The helper owns all CUDA allocations it uses. It never substitutes CPU
    execution; CUDA failures are returned in ``status``/``reason`` fields so
    tests can skip hardware-only checks cleanly.
    """
    _validate_inputs(
        x_host_bytes,
        bias_host_bytes,
        rows,
        columns,
        expected_output_bytes,
        expected_output_checksum,
        expected_output_metadata,
        required_cuda_visible_devices,
    )

    driver_probe = _cuda_driver_probe.probe_cuda_driver_device0()
    (
        runtime,
        runtime_library,
        runtime_load_error,
    ) = _cuda_driver_probe._load_shared_library(
        "cudart",
        _cuda_driver_probe._CUDA_RUNTIME_NAMES,
    )
    nvcc = _cuda_kernel_support._nvcc_provenance()
    result = _base_result(
        rows=rows,
        columns=columns,
        x_bytes=x_host_bytes,
        bias_bytes=bias_host_bytes,
        expected_output_bytes=expected_output_bytes,
        expected_output_checksum=expected_output_checksum,
        expected_output_metadata=expected_output_metadata,
        required_cuda_visible_devices=required_cuda_visible_devices,
        runtime=runtime,
        runtime_library=runtime_library,
        runtime_load_error=runtime_load_error,
        driver_probe=driver_probe,
        nvcc=nvcc,
    )

    if (
        required_cuda_visible_devices is not None
        and os.environ.get("CUDA_VISIBLE_DEVICES") != required_cuda_visible_devices
    ):
        result["reason"] = (
            "CUDA_VISIBLE_DEVICES="
            f"{required_cuda_visible_devices} is required"
        )
        return result
    if runtime is None:
        result["reason"] = "CUDA runtime shared library was not loaded"
        return result

    missing_symbols = _configure_runtime_symbols(runtime)
    if missing_symbols:
        result["status"] = "error"
        result["reason"] = "CUDA runtime is missing required symbols"
        result["missing_symbols"] = missing_symbols
        return result

    device_count = ctypes.c_int()
    device_count_call = _runtime_call(
        runtime,
        "cudaGetDeviceCount",
        ctypes.byref(device_count),
    )
    device_count_call["value"] = (
        int(device_count.value) if device_count_call["result"] == 0 else None
    )
    result["calls"]["cudaGetDeviceCount"] = device_count_call
    if device_count_call["result"] != 0:
        if device_count_call["error_name"] in _CUDA_UNAVAILABLE_ERRORS:
            result["status"] = "unavailable"
        else:
            result["status"] = "error"
        result["reason"] = "cudaGetDeviceCount failed"
        return result
    if device_count.value < 1:
        result["reason"] = "no CUDA runtime devices are visible"
        return result
    result["single_visible_cuda_device"] = int(device_count.value) == 1
    if not nvcc["available"]:
        result["status"] = "error"
        result["reason"] = "nvcc was not found"
        return result

    repository_root = _cuda_kernel_support._repository_root()
    if repository_root is None:
        result["status"] = "error"
        result["reason"] = "worktree root was not found"
        return result

    try:
        library_path, build = _build_kernel(
            repository_root=repository_root,
            driver_probe=driver_probe,
            nvcc=nvcc,
        )
    except RuntimeError as error:
        result["status"] = "error"
        result["reason"] = str(error)
        return result
    result["build"] = build
    if library_path is None:
        result["status"] = "error"
        result["reason"] = (
            "nvcc failed to compile the private CUDA pointwise-reduce kernel"
        )
        return result

    kernel_library, load = _load_kernel_library(library_path, runtime_library)
    result["kernel_library"] = load
    if kernel_library is None:
        result["status"] = "error"
        result["reason"] = "compiled CUDA pointwise-reduce library was not loaded"
        return result

    set_device_call = _runtime_call(runtime, "cudaSetDevice", 0)
    result["calls"]["cudaSetDevice"] = set_device_call
    if set_device_call["result"] != 0:
        result["status"] = "error"
        result["reason"] = "cudaSetDevice(0) failed"
        return result

    current_device = ctypes.c_int(-1)
    get_device_call = _runtime_call(
        runtime,
        "cudaGetDevice",
        ctypes.byref(current_device),
    )
    get_device_call["value"] = (
        int(current_device.value) if get_device_call["result"] == 0 else None
    )
    result["calls"]["cudaGetDevice"] = get_device_call
    if get_device_call["result"] != 0:
        result["status"] = "error"
        result["reason"] = "cudaGetDevice failed after cudaSetDevice(0)"
        return result

    result["device_type"] = "cuda"
    result["device_index"] = int(current_device.value)
    result["input_metadata"] = _input_metadata(
        rows,
        columns,
        result["device_index"],
    )
    result["output_metadata"] = _output_metadata(rows, result["device_index"])
    result["output_metadata_match"] = (
        expected_output_metadata is not None
        and result["output_metadata"] == expected_output_metadata
    )

    x_byte_count = len(x_host_bytes)
    bias_byte_count = len(bias_host_bytes)
    output_byte_count = rows * _FLOAT32_SIZE
    device_x = ctypes.c_void_p()
    device_bias = ctypes.c_void_p()
    device_output = ctypes.c_void_p()

    try:
        x_malloc = _runtime_call(
            runtime,
            "cudaMalloc",
            ctypes.byref(device_x),
            x_byte_count,
        )
        x_malloc["byte_count"] = x_byte_count
        result["calls"]["cudaMalloc_x"] = x_malloc
        result["device_x_pointer_nonzero"] = bool(device_x.value)
        if x_malloc["result"] != 0 or not device_x.value:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for x"
            return result

        bias_malloc = _runtime_call(
            runtime,
            "cudaMalloc",
            ctypes.byref(device_bias),
            bias_byte_count,
        )
        bias_malloc["byte_count"] = bias_byte_count
        result["calls"]["cudaMalloc_bias"] = bias_malloc
        result["device_bias_pointer_nonzero"] = bool(device_bias.value)
        if bias_malloc["result"] != 0 or not device_bias.value:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for bias"
            return result

        output_malloc = _runtime_call(
            runtime,
            "cudaMalloc",
            ctypes.byref(device_output),
            output_byte_count,
        )
        output_malloc["byte_count"] = output_byte_count
        result["calls"]["cudaMalloc_output"] = output_malloc
        result["device_output_pointer_nonzero"] = bool(device_output.value)
        if output_malloc["result"] != 0 or not device_output.value:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for output"
            return result

        host_x = ctypes.create_string_buffer(x_host_bytes, x_byte_count)
        host_bias = ctypes.create_string_buffer(bias_host_bytes, bias_byte_count)
        host_output = ctypes.create_string_buffer(output_byte_count)

        x_h2d_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            device_x,
            ctypes.c_void_p(ctypes.addressof(host_x)),
            x_byte_count,
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        x_h2d_call["kind"] = "cudaMemcpyHostToDevice"
        x_h2d_call["byte_count"] = x_byte_count
        result["calls"]["cudaMemcpyHostToDevice_x"] = x_h2d_call
        if x_h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed for x"
            return result

        bias_h2d_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            device_bias,
            ctypes.c_void_p(ctypes.addressof(host_bias)),
            bias_byte_count,
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        bias_h2d_call["kind"] = "cudaMemcpyHostToDevice"
        bias_h2d_call["byte_count"] = bias_byte_count
        result["calls"]["cudaMemcpyHostToDevice_bias"] = bias_h2d_call
        if bias_h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed for bias"
            return result

        sync_after_h2d = _runtime_call(runtime, "cudaDeviceSynchronize")
        result["calls"]["cudaDeviceSynchronize_after_host_to_device"] = (
            sync_after_h2d
        )
        if sync_after_h2d["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaDeviceSynchronize failed after host-to-device"
            return result

        blocks = ctypes.c_int(0)
        threads = ctypes.c_int(0)
        launch_error = ctypes.c_int(0)
        sync_error = ctypes.c_int(0)
        kernel_function = getattr(
            kernel_library,
            "torch_rs_private_h100_pointwise_reduce_float32_v1",
        )
        kernel_result = int(
            kernel_function(
                device_x,
                device_bias,
                device_output,
                rows,
                columns,
                ctypes.byref(blocks),
                ctypes.byref(threads),
                ctypes.byref(launch_error),
                ctypes.byref(sync_error),
            )
        )
        launch = {
            "result": kernel_result,
            "blocks": int(blocks.value),
            "threads_per_block": int(threads.value),
            "launch_error": {
                "result": int(launch_error.value),
                "error_name": _cuda_driver_probe._runtime_error_name(
                    runtime,
                    int(launch_error.value),
                )
                if launch_error.value
                else None,
            },
            "sync_error": {
                "result": int(sync_error.value),
                "error_name": _cuda_driver_probe._runtime_error_name(
                    runtime,
                    int(sync_error.value),
                )
                if sync_error.value
                else None,
            },
        }
        result["launch"] = launch
        result["calls"]["torchRsPrivateH100PointwiseReduceFloat32"] = launch
        if kernel_result != 0:
            result["status"] = "error"
            result["reason"] = "private CUDA pointwise-reduce kernel launch failed"
            return result

        d2h_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            ctypes.c_void_p(ctypes.addressof(host_output)),
            device_output,
            output_byte_count,
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        d2h_call["kind"] = "cudaMemcpyDeviceToHost"
        d2h_call["byte_count"] = output_byte_count
        result["calls"]["cudaMemcpyDeviceToHost_output"] = d2h_call
        if d2h_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy device-to-host failed for output"
            return result

        sync_after_d2h = _runtime_call(runtime, "cudaDeviceSynchronize")
        result["calls"]["cudaDeviceSynchronize_after_device_to_host"] = (
            sync_after_d2h
        )
        if sync_after_d2h["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaDeviceSynchronize failed after device-to-host"
            return result

        host_output_bytes = ctypes.string_at(
            ctypes.addressof(host_output),
            output_byte_count,
        )
        result["device_output_bytes_checksum"] = _checksum_output_bytes(
            host_output_bytes,
            rows,
            columns,
        )
        result["device_output_checksum"] = _checksum_tensor_metadata_values(
            host_output_bytes,
            result["output_metadata"],
        )
        result["output_comparison"] = _float32_comparison(
            host_output_bytes,
            expected_output_bytes,
        )
        if expected_output_checksum is None:
            result["checksum_match"] = None
            result["output_metadata_match"] = None
            result["status"] = "ok"
            result["reason"] = (
                "pointwise-reduce workload executed without reference comparison"
            )
            return result

        result["checksum_match"] = (
            result["device_output_checksum"] == expected_output_checksum
            and result["output_comparison"]["exact_bytes_match"] is True
            and result["output_metadata_match"] is True
        )
        if not result["checksum_match"]:
            result["status"] = "error"
            result["reason"] = "pointwise-reduce workload checksum mismatch"
            return result

        result["status"] = "ok"
        result["reason"] = "pointwise-reduce workload checksum verified"
        return result
    finally:
        if device_output.value:
            free_output = _runtime_call(runtime, "cudaFree", device_output)
            result["calls"]["cudaFree_output"] = free_output
            if result["status"] == "ok" and free_output["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for output"
        if device_bias.value:
            free_bias = _runtime_call(runtime, "cudaFree", device_bias)
            result["calls"]["cudaFree_bias"] = free_bias
            if result["status"] == "ok" and free_bias["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for bias"
        if device_x.value:
            free_x = _runtime_call(runtime, "cudaFree", device_x)
            result["calls"]["cudaFree_x"] = free_x
            if result["status"] == "ok" and free_x["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for x"


__all__ = [
    "OUTPUT_SHAPE",
    "POINTWISE_REDUCE_KERNEL_VERSION",
    "POINTWISE_REDUCE_SCHEMA_VERSION",
    "WORKLOAD_SHAPE",
    "launch_h100_float32_pointwise_reduce_device0",
]
