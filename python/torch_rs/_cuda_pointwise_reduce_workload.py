"""Private benchmark-only CUDA pointwise-plus-row-reduction workload.

This module is intentionally separate from ``torch_rs.cuda``. It verifies that
benchmark code can allocate torch_rs-owned CUDA buffers and run the H100 CUDA
``torch.compile`` reference workload shapes without claiming public CUDA tensor
or compile support.
"""

from __future__ import annotations

import ctypes
import copy
import hashlib
import json
import os
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _cuda_buffer
from . import _cuda_driver_probe
from . import _cuda_pointwise_kernel as _cuda_kernel_support
from . import _cuda_runtime_ownership
from ._cuda_benchmark_tensor import CudaBenchmarkTensor


POINTWISE_REDUCE_SCHEMA_VERSION = (
    "torch_rs_private_cuda_pointwise_reduce_workload_v1"
)
POINTWISE_REDUCE_INPUTS_SCHEMA_VERSION = (
    "torch_rs_private_cuda_pointwise_reduce_inputs_v1"
)
POINTWISE_REDUCE_COMPILE_EXECUTION_SCHEMA_VERSION = (
    "torch_rs_private_cuda_pointwise_reduce_compile_execution_v1"
)
POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION = (
    "torch_rs_private_cuda_pointwise_reduce_compile_executor_v1"
)
POINTWISE_REDUCE_OUTPUT_POOL_SCHEMA_VERSION = (
    _cuda_runtime_ownership.CUDA_BUFFER_POOL_SCHEMA_VERSION
)
POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION = "h100_cuda_pointwise_reduce_float32_v1"
POINTWISE_REDUCE_KERNEL_VERSION = (
    "h100_cuda_pointwise_plus_row_reduce_float32_v3"
)
WORKLOAD_SHAPE = (1024, 1024)
OUTPUT_SHAPE = (1024,)
SUPPORTED_WORKLOAD_SHAPES = (
    (256, 256),
    (1024, 1024),
    (4096, 256),
    (256, 4096),
)
_WORKLOAD_X_METADATA = _cuda_buffer.float32_metadata(WORKLOAD_SHAPE, device_index=0)
_WORKLOAD_BIAS_METADATA = _cuda_buffer.float32_metadata(
    (WORKLOAD_SHAPE[1],),
    device_index=0,
)
_WORKLOAD_OUTPUT_METADATA = _cuda_buffer.float32_metadata(
    OUTPUT_SHAPE,
    device_index=0,
)

_FLOAT32_SIZE = 4
_COMPILE_DEFAULT_THREADS_PER_BLOCK = 32
_OUTPUT_POOL_INITIAL_CAPACITY = 4
_CUDA_UNAVAILABLE_ERRORS = _cuda_buffer.CUDA_UNAVAILABLE_ERRORS
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float torch_rs_private_pointwise_reduce_value_v1(
    float x_value,
    float bias_value
) {
    float mixed = sinf(x_value + bias_value)
        * cosf(x_value - bias_value);
    float relu = x_value > 0.0f ? x_value : 0.0f;
    return mixed + relu;
}

__device__ __forceinline__ float torch_rs_private_warp_reduce_sum_v1(
    float partial
) {
    unsigned int mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_xor_sync(mask, partial, offset, 32);
    }
    return partial;
}

extern "C" __global__ void torch_rs_private_pointwise_reduce_kernel_v1(
    const float* x,
    const float* bias,
    float* output,
    int rows,
    int columns
) {
    int row = blockIdx.x;
    int thread = threadIdx.x;
    int lane = thread & 31;
    if (row >= rows) {
        return;
    }

    if ((columns == 256 && rows == 256) || columns == 4096) {
        __shared__ float warp_partials[16];
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        int row_offset = row * columns;
        int columns_per_iteration = blockDim.x * 4;
        int first_column = thread * 4;
        for (
            int tile_column = first_column;
            tile_column < columns;
            tile_column += columns_per_iteration
        ) {
            int column = tile_column;
            if (column < columns) {
                acc0 += torch_rs_private_pointwise_reduce_value_v1(
                    x[row_offset + column],
                    bias[column]
                );
            }
            column = tile_column + 1;
            if (column < columns) {
                acc1 += torch_rs_private_pointwise_reduce_value_v1(
                    x[row_offset + column],
                    bias[column]
                );
            }
            column = tile_column + 2;
            if (column < columns) {
                acc2 += torch_rs_private_pointwise_reduce_value_v1(
                    x[row_offset + column],
                    bias[column]
                );
            }
            column = tile_column + 3;
            if (column < columns) {
                acc3 += torch_rs_private_pointwise_reduce_value_v1(
                    x[row_offset + column],
                    bias[column]
                );
            }
        }

        float partial = acc0 + acc1;
        partial += acc2;
        partial += acc3;
        partial = torch_rs_private_warp_reduce_sum_v1(partial);

        int warp = thread >> 5;
        if (lane == 0) {
            warp_partials[warp] = partial;
        }
        __syncthreads();

        int warp_count = blockDim.x >> 5;
        float block_partial = thread < warp_count ? warp_partials[thread] : 0.0f;
        unsigned int mask = 0xffffffffu;
        for (int offset = warp_count >> 1; offset > 0; offset >>= 1) {
            block_partial += __shfl_xor_sync(mask, block_partial, offset, 32);
        }
        if (thread == 0) {
            output[row] = block_partial;
        }
        return;
    }

    float partial = 0.0f;
    int row_offset = row * columns;
    for (int base_column = lane * 4; base_column < columns; base_column += 128) {
#pragma unroll
        for (int offset = 0; offset < 4; ++offset) {
            int column = base_column + offset;
            if (column < columns) {
                partial += torch_rs_private_pointwise_reduce_value_v1(
                    x[row_offset + column],
                    bias[column]
                );
            }
        }
    }

    partial = torch_rs_private_warp_reduce_sum_v1(partial);

    if (lane == 0) {
        output[row] = partial;
    }
}

extern "C" int torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
    const float* x,
    const float* bias,
    float* output,
    int rows,
    int columns,
    int* blocks_out,
    int* threads_out,
    int* launch_error_out
) {
    if (x == nullptr || bias == nullptr || output == nullptr) {
        return -1;
    }
    if (rows <= 0 || columns <= 0) {
        return -2;
    }

    int threads = 32;
    if (columns == 4096) {
        threads = 512;
    } else if (columns == 256 && rows == 256) {
        threads = 64;
    }
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
    return static_cast<int>(launch_error);
}

struct torch_rs_private_h100_pointwise_reduce_launch_report_v1 {
    int set_device_error;
    int get_device_error;
    int observed_device;
    int launch_error;
    int blocks;
    int threads_per_block;
};

extern "C" int torch_rs_private_h100_pointwise_reduce_float32_device0_guarded_launch_async_v1(
    const float* x,
    const float* bias,
    float* output,
    int rows,
    int columns,
    torch_rs_private_h100_pointwise_reduce_launch_report_v1* report
) {
    if (report != nullptr) {
        report->set_device_error = 0;
        report->get_device_error = 0;
        report->observed_device = -1;
        report->launch_error = 0;
        report->blocks = 0;
        report->threads_per_block = 0;
    }

    cudaError_t set_device_error = cudaSetDevice(0);
    if (report != nullptr) {
        report->set_device_error = static_cast<int>(set_device_error);
    }
    if (set_device_error != cudaSuccess) {
        return static_cast<int>(set_device_error);
    }

    int observed_device = -1;
    cudaError_t get_device_error = cudaGetDevice(&observed_device);
    if (report != nullptr) {
        report->get_device_error = static_cast<int>(get_device_error);
        report->observed_device = observed_device;
    }
    if (get_device_error != cudaSuccess) {
        return static_cast<int>(get_device_error);
    }
    if (observed_device != 0) {
        return -3;
    }

    return torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
        x,
        bias,
        output,
        rows,
        columns,
        report == nullptr ? nullptr : &report->blocks,
        report == nullptr ? nullptr : &report->threads_per_block,
        report == nullptr ? nullptr : &report->launch_error
    );
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
    int launch_result = torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1(
        x,
        bias,
        output,
        rows,
        columns,
        blocks_out,
        threads_out,
        launch_error_out
    );
    if (launch_result != 0) {
        if (sync_error_out != nullptr) {
            *sync_error_out = 0;
        }
        return launch_result;
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


def _validate_compile_workload_shape(
    workload_shape: tuple[int, int],
) -> tuple[int, int]:
    if type(workload_shape) is not tuple:
        raise TypeError("workload_shape must be tuple[int, int]")
    if len(workload_shape) != 2:
        raise ValueError("workload_shape must have exactly two dimensions")
    rows, columns = workload_shape
    if type(rows) is not int or type(columns) is not int:
        raise TypeError("workload_shape dimensions must be int")
    if workload_shape not in SUPPORTED_WORKLOAD_SHAPES:
        raise _cuda_compile_unsupported(
            f"unsupported workload shape {list(workload_shape)}"
        )
    return rows, columns


def _compile_bias_shape(workload_shape: tuple[int, int]) -> tuple[int]:
    _rows, columns = _validate_compile_workload_shape(workload_shape)
    return (columns,)


def _compile_output_shape(workload_shape: tuple[int, int]) -> tuple[int]:
    rows, _columns = _validate_compile_workload_shape(workload_shape)
    return (rows,)


def _compile_input_metadata(
    workload_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    rows, columns = _validate_compile_workload_shape(workload_shape)
    return [
        _cuda_buffer.float32_metadata((rows, columns), device_index=0),
        _cuda_buffer.float32_metadata((columns,), device_index=0),
    ]


def _compile_output_metadata(workload_shape: tuple[int, int]) -> dict[str, Any]:
    return _cuda_buffer.float32_metadata(
        _compile_output_shape(workload_shape),
        device_index=0,
    )


def _compile_launch_dimensions(workload_shape: tuple[int, int]) -> tuple[int, int]:
    rows, columns = _validate_compile_workload_shape(workload_shape)
    threads = _COMPILE_DEFAULT_THREADS_PER_BLOCK
    if columns == 4096:
        threads = 512
    elif columns == 256 and rows == 256:
        threads = 64
    return rows, threads


class _GuardedLaunchReport(ctypes.Structure):
    _fields_ = [
        ("set_device_error", ctypes.c_int),
        ("get_device_error", ctypes.c_int),
        ("observed_device", ctypes.c_int),
        ("launch_error", ctypes.c_int),
        ("blocks", ctypes.c_int),
        ("threads_per_block", ctypes.c_int),
    ]


@dataclass(frozen=True, slots=True)
class _CompiledExecutionToken:
    invocation_index: int
    reused_prepared_executor: bool
    output_pool_acquisition: (
        _cuda_runtime_ownership.PrivateCudaBufferPoolAcquisition
    )
    device_output_pointer_nonzero: bool
    result: int
    launch_error: int
    blocks: int
    threads_per_block: int

    def __deepcopy__(self, memo: dict[int, Any]) -> "_CompiledExecutionToken":
        del memo
        return self


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
    async_function_name = (
        "torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1"
    )
    guarded_async_function_name = (
        "torch_rs_private_h100_pointwise_reduce_float32_device0_guarded_launch_async_v1"
    )
    try:
        function = getattr(library, function_name)
        async_function = getattr(library, async_function_name)
        guarded_async_function = getattr(library, guarded_async_function_name)
    except AttributeError as error:
        return None, {
            "loaded": False,
            "error": str(error),
            "function": function_name,
            "async_function": async_function_name,
            "guarded_async_function": guarded_async_function_name,
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
    async_function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    async_function.restype = ctypes.c_int
    guarded_async_function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_GuardedLaunchReport),
    ]
    guarded_async_function.restype = ctypes.c_int
    return library, {
        "loaded": True,
        "error": None,
        "function": function_name,
        "async_function": async_function_name,
        "guarded_async_function": guarded_async_function_name,
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
        "buffer_schema_version": _cuda_buffer.BUFFER_SCHEMA_VERSION,
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
        "input_metadata": None,
        "expected_output_metadata": expected_output_metadata,
        "output_metadata": None,
        "public_cuda_tensor_wrapper": None,
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


class H100Float32PointwiseReduceInputBundle:
    """Lifetime owner for private CUDA inputs wrapped as benchmark tensors."""

    __slots__ = ("x", "bias", "_device_x", "_device_bias", "_evidence", "_closed")

    def __init__(
        self,
        *,
        x: CudaBenchmarkTensor,
        bias: CudaBenchmarkTensor,
        device_x: _cuda_buffer.PrivateCudaFloat32Buffer,
        device_bias: _cuda_buffer.PrivateCudaFloat32Buffer,
        evidence: dict[str, Any],
    ) -> None:
        self.x = x
        self.bias = bias
        self._device_x = device_x
        self._device_bias = device_bias
        self._evidence = evidence
        self._closed = False

    @property
    def inputs(self) -> tuple[CudaBenchmarkTensor, CudaBenchmarkTensor]:
        return (self.x, self.bias)

    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._evidence)

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {}
        self._closed = True
        calls: dict[str, Any] = {}
        free_bias = self._device_bias.close()
        if free_bias is not None:
            calls["cudaFree_bias"] = free_bias
            self._evidence["calls"]["cudaFree_bias"] = free_bias
        free_x = self._device_x.close()
        if free_x is not None:
            calls["cudaFree_x"] = free_x
            self._evidence["calls"]["cudaFree_x"] = free_x
        return copy.deepcopy(calls)

    def __enter__(self) -> "H100Float32PointwiseReduceInputBundle":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _input_bundle_base_result(
    *,
    rows: int,
    columns: int,
    x_bytes: bytes,
    bias_bytes: bytes,
    required_cuda_visible_devices: str | None,
    runtime: ctypes.CDLL | None,
    runtime_library: str | None,
    runtime_load_error: str | None,
    driver_probe: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": POINTWISE_REDUCE_INPUTS_SCHEMA_VERSION,
        "primitive": (
            "torch_rs_private_cuda_h100_pointwise_reduce_float32_inputs_device0"
        ),
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
        "buffer_schema_version": _cuda_buffer.BUFFER_SCHEMA_VERSION,
        "workload_shape": [rows, columns],
        "input_metadata": None,
        "public_cuda_tensor_inputs": None,
        "x_byte_count": len(x_bytes),
        "bias_byte_count": len(bias_bytes),
        "host_x_checksum": _checksum_x_bytes(x_bytes, rows, columns),
        "host_bias_checksum": _checksum_bias_bytes(bias_bytes, rows, columns),
        "device_x_checksum": None,
        "device_bias_checksum": None,
        "checksum_match": False,
        "device_x_pointer_nonzero": False,
        "device_bias_pointer_nonzero": False,
        "driver": driver_probe["driver"],
        "runtime": _cuda_kernel_support._runtime_versions(
            runtime,
            runtime_library,
            runtime_load_error,
        ),
        "device_0": driver_probe["device_0"],
        "gpu": _cuda_kernel_support._gpu_provenance(driver_probe),
        "calls": {},
    }


def _close_input_buffers_on_failure(
    result: dict[str, Any],
    device_x: _cuda_buffer.PrivateCudaFloat32Buffer | None,
    device_bias: _cuda_buffer.PrivateCudaFloat32Buffer | None,
) -> None:
    if device_bias is not None:
        free_bias = device_bias.close()
        if free_bias is not None:
            result["calls"]["cudaFree_bias"] = free_bias
    if device_x is not None:
        free_x = device_x.close()
        if free_x is not None:
            result["calls"]["cudaFree_x"] = free_x


def make_h100_float32_pointwise_reduce_inputs_device0(
    x_host_bytes: bytes,
    bias_host_bytes: bytes,
    *,
    rows: int = WORKLOAD_SHAPE[0],
    columns: int = WORKLOAD_SHAPE[1],
    required_cuda_visible_devices: str | None = "0",
) -> tuple[H100Float32PointwiseReduceInputBundle | None, dict[str, Any]]:
    """Allocate private CUDA inputs and expose them as benchmark tensor views."""
    _validate_inputs(
        x_host_bytes,
        bias_host_bytes,
        rows,
        columns,
        None,
        None,
        None,
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
    result = _input_bundle_base_result(
        rows=rows,
        columns=columns,
        x_bytes=x_host_bytes,
        bias_bytes=bias_host_bytes,
        required_cuda_visible_devices=required_cuda_visible_devices,
        runtime=runtime,
        runtime_library=runtime_library,
        runtime_load_error=runtime_load_error,
        driver_probe=driver_probe,
    )

    if (
        required_cuda_visible_devices is not None
        and os.environ.get("CUDA_VISIBLE_DEVICES") != required_cuda_visible_devices
    ):
        result["reason"] = (
            "CUDA_VISIBLE_DEVICES="
            f"{required_cuda_visible_devices} is required"
        )
        return None, result
    if runtime is None:
        result["reason"] = "CUDA runtime shared library was not loaded"
        return None, result

    missing_symbols = _configure_runtime_symbols(runtime)
    if missing_symbols:
        result["status"] = "error"
        result["reason"] = "CUDA runtime is missing required symbols"
        result["missing_symbols"] = missing_symbols
        return None, result

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
        return None, result
    if device_count.value < 1:
        result["reason"] = "no CUDA runtime devices are visible"
        return None, result
    result["single_visible_cuda_device"] = int(device_count.value) == 1

    set_device_call = _runtime_call(runtime, "cudaSetDevice", 0)
    result["calls"]["cudaSetDevice"] = set_device_call
    if set_device_call["result"] != 0:
        result["status"] = "error"
        result["reason"] = "cudaSetDevice(0) failed"
        return None, result

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
        return None, result

    result["device_type"] = "cuda"
    result["device_index"] = int(current_device.value)
    device_x = _cuda_buffer.PrivateCudaFloat32Buffer(
        runtime,
        (rows, columns),
        name="x",
        device_index=result["device_index"],
    )
    device_bias = None
    try:
        result["calls"]["cudaMalloc_x"] = device_x.malloc_call
        result["device_x_pointer_nonzero"] = device_x.pointer_nonzero
        if not device_x.allocation_ok:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for x"
            return None, result

        device_bias = _cuda_buffer.PrivateCudaFloat32Buffer(
            runtime,
            (columns,),
            name="bias",
            device_index=result["device_index"],
        )
        result["calls"]["cudaMalloc_bias"] = device_bias.malloc_call
        result["device_bias_pointer_nonzero"] = device_bias.pointer_nonzero
        if not device_bias.allocation_ok:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for bias"
            return None, result

        x_h2d_call = device_x.copy_from_host(x_host_bytes)
        result["calls"]["cudaMemcpyHostToDevice_x"] = x_h2d_call
        if x_h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed for x"
            return None, result

        bias_h2d_call = device_bias.copy_from_host(bias_host_bytes)
        result["calls"]["cudaMemcpyHostToDevice_bias"] = bias_h2d_call
        if bias_h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed for bias"
            return None, result

        sync_after_h2d = device_x.synchronize()
        result["calls"]["cudaDeviceSynchronize_after_host_to_device"] = (
            sync_after_h2d
        )
        if sync_after_h2d["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaDeviceSynchronize failed after host-to-device"
            return None, result

        x_readback = device_x.checksum_readback(
            lambda payload: _checksum_x_bytes(payload, rows, columns),
        )
        result["calls"]["cudaMemcpyDeviceToHost_x_input"] = x_readback.copy_call
        result["calls"]["cudaDeviceSynchronize_after_x_input_readback"] = (
            x_readback.sync_call
        )
        if x_readback.copy_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy device-to-host failed for x input"
            return None, result
        if x_readback.sync_call is None or x_readback.sync_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaDeviceSynchronize failed after x input readback"
            return None, result

        bias_readback = device_bias.checksum_readback(
            lambda payload: _checksum_bias_bytes(payload, rows, columns),
        )
        result["calls"]["cudaMemcpyDeviceToHost_bias_input"] = (
            bias_readback.copy_call
        )
        result["calls"]["cudaDeviceSynchronize_after_bias_input_readback"] = (
            bias_readback.sync_call
        )
        if bias_readback.copy_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy device-to-host failed for bias input"
            return None, result
        if (
            bias_readback.sync_call is None
            or bias_readback.sync_call["result"] != 0
        ):
            result["status"] = "error"
            result["reason"] = (
                "cudaDeviceSynchronize failed after bias input readback"
            )
            return None, result

        x_tensor = CudaBenchmarkTensor(
            device_x,
            readback=x_readback,
            checksum_name="torch_rs_private_cuda_pointwise_reduce_x_v1",
        )
        bias_tensor = CudaBenchmarkTensor(
            device_bias,
            readback=bias_readback,
            checksum_name="torch_rs_private_cuda_pointwise_reduce_bias_v1",
        )
        result["input_metadata"] = [device_x.metadata(), device_bias.metadata()]
        result["public_cuda_tensor_inputs"] = [
            x_tensor.metadata(),
            bias_tensor.metadata(),
        ]
        result["device_x_checksum"] = x_readback.checksum
        result["device_bias_checksum"] = bias_readback.checksum
        result["checksum_match"] = (
            result["device_x_checksum"] == result["host_x_checksum"]
            and result["device_bias_checksum"] == result["host_bias_checksum"]
        )
        if not result["checksum_match"]:
            result["status"] = "error"
            result["reason"] = "input CUDA tensor checksum mismatch"
            return None, result

        result["status"] = "ok"
        result["reason"] = "pointwise-reduce CUDA inputs allocated and verified"
        return (
            H100Float32PointwiseReduceInputBundle(
                x=x_tensor,
                bias=bias_tensor,
                device_x=device_x,
                device_bias=device_bias,
                evidence=result,
            ),
            result,
        )
    finally:
        if result["status"] != "ok":
            _close_input_buffers_on_failure(result, device_x, device_bias)


def _cuda_compile_unsupported(reason: str) -> NotImplementedError:
    shape_family = ", ".join(
        f"[{rows}, {columns}]" for rows, columns in SUPPORTED_WORKLOAD_SHAPES
    )
    return NotImplementedError(
        "torch.compile(): native CUDA inductor execution is supported only "
        "for the versioned H100 float32 pointwise-reduce benchmark with "
        "two live CUDA benchmark tensor inputs matching one explicit "
        f"workload marker shape in {{{shape_family}}}; {reason}"
    )


def _require_compiled_cuda_benchmark_input(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> tuple[_cuda_buffer.PrivateCudaFloat32Buffer, dict[str, Any]]:
    buffer = _require_compiled_cuda_benchmark_input_buffer(
        value,
        name=name,
        expected_shape=expected_shape,
    )
    expected_metadata = _cuda_buffer.float32_metadata(expected_shape, device_index=0)
    return buffer, copy.deepcopy(expected_metadata)


def _require_compiled_cuda_benchmark_input_buffer(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> _cuda_buffer.PrivateCudaFloat32Buffer:
    if type(value) is not CudaBenchmarkTensor:
        raise _cuda_compile_unsupported(
            f"{name} is not a torch_rs CudaBenchmarkTensor"
        )

    expected_metadata = _cuda_buffer.float32_metadata(expected_shape, device_index=0)
    mismatched = []
    if value.shape != expected_shape:
        mismatched.append("shape")
    if value.stride != tuple(expected_metadata["stride"]):
        mismatched.append("stride")
    if str(value.dtype) != _cuda_buffer.FLOAT32_DTYPE:
        mismatched.append("dtype")
    if value.device != "cuda:0":
        mismatched.append("device")
    if value.device_type != "cuda":
        mismatched.append("device_type")
    if value.device_index != 0:
        mismatched.append("device_index")
    if value.requires_grad is not False:
        mismatched.append("requires_grad")
    if value.is_contiguous is not True:
        mismatched.append("is_contiguous")
    if value.is_cuda is not True:
        mismatched.append("is_cuda")
    if mismatched:
        joined = ", ".join(sorted(set(mismatched)))
        raise _cuda_compile_unsupported(f"{name} metadata mismatch: {joined}")

    try:
        buffer = value._torch_rs_private_cuda_buffer()
    except ValueError as error:
        raise _cuda_compile_unsupported(f"{name} private buffer is not live") from error
    if buffer.shape != expected_shape:
        raise _cuda_compile_unsupported(f"{name} private buffer shape mismatch")
    if buffer.stride != tuple(expected_metadata["stride"]):
        raise _cuda_compile_unsupported(f"{name} private buffer stride mismatch")
    if buffer.device_index != 0:
        raise _cuda_compile_unsupported(f"{name} private buffer device mismatch")
    if buffer.element_count != _cuda_buffer.element_count(expected_shape):
        raise _cuda_compile_unsupported(
            f"{name} private buffer element count mismatch"
        )
    if buffer.byte_count != buffer.element_count * _FLOAT32_SIZE:
        raise _cuda_compile_unsupported(f"{name} private buffer byte count mismatch")
    if not buffer.allocation_ok:
        raise _cuda_compile_unsupported(f"{name} private buffer is not live")
    return buffer


class H100Float32PointwiseReduceCompiledExecutor:
    """Prepared executor for the narrow H100 CUDA compile benchmark."""

    __slots__ = (
        "_build",
        "_closed",
        "_driver_probe",
        "_execution_static_evidence",
        "_executor_key",
        "_executor_static_evidence",
        "_invocation_count",
        "_kernel_function",
        "_kernel_library",
        "_kernel_library_evidence",
        "_launch_report_tls",
        "_lifecycle_lock",
        "_nvcc",
        "_output_metadata",
        "_output_pool",
        "_rows",
        "_columns",
        "_workload_shape",
        "_bias_shape",
        "_output_shape",
        "_input_metadata",
        "_launch_blocks",
        "_threads_per_block",
        "_preparation",
        "_required_cuda_visible_devices",
        "_runtime",
        "_runtime_library",
        "_runtime_load_error",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        runtime: ctypes.CDLL,
        runtime_library: str | None,
        runtime_load_error: str | None,
        driver_probe: dict[str, Any],
        nvcc: dict[str, Any],
        build: dict[str, Any],
        kernel_library: ctypes.CDLL,
        kernel_library_evidence: dict[str, Any],
        required_cuda_visible_devices: str | None,
        preparation: dict[str, Any],
        workload_shape: tuple[int, int] = WORKLOAD_SHAPE,
    ) -> None:
        rows, columns = _validate_compile_workload_shape(workload_shape)
        self._runtime = runtime
        self._runtime_library = runtime_library
        self._runtime_load_error = runtime_load_error
        self._driver_probe = copy.deepcopy(driver_probe)
        self._nvcc = copy.deepcopy(nvcc)
        self._build = copy.deepcopy(build)
        self._kernel_library = kernel_library
        self._kernel_library_evidence = copy.deepcopy(kernel_library_evidence)
        self._required_cuda_visible_devices = required_cuda_visible_devices
        self._invocation_count = 0
        self._closed = False
        self._lifecycle_lock = threading.RLock()
        self._rows = rows
        self._columns = columns
        self._workload_shape = (rows, columns)
        self._bias_shape = _compile_bias_shape(self._workload_shape)
        self._output_shape = _compile_output_shape(self._workload_shape)
        self._input_metadata = _compile_input_metadata(self._workload_shape)
        self._output_metadata = _compile_output_metadata(self._workload_shape)
        self._launch_blocks, self._threads_per_block = _compile_launch_dimensions(
            self._workload_shape
        )
        self._kernel_function = getattr(
            self._kernel_library,
            (
                "torch_rs_private_h100_pointwise_reduce_float32_"
                "launch_async_v1"
            ),
        )
        self._launch_report_tls = threading.local()
        key_payload = json.dumps(
            {
                "workload_version": POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION,
                "workload_shape": list(self._workload_shape),
                "kernel_version": POINTWISE_REDUCE_KERNEL_VERSION,
                "build_key": self._build.get("build_key"),
                "library_path": self._build.get("library_path"),
                "runtime_library": self._runtime_library,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "required_cuda_visible_devices": required_cuda_visible_devices,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._executor_key = hashlib.blake2b(
            key_payload,
            digest_size=8,
        ).hexdigest()
        self._preparation = copy.deepcopy(preparation)
        self._output_pool = _cuda_runtime_ownership.PrivateCudaBufferPool(
            self._runtime,
            self._output_shape,
            name_prefix="compiled_output_pool",
            device_index=0,
            owner_id=self._executor_key,
        )
        output_pool_allocations = self._output_pool.preallocate(
            _OUTPUT_POOL_INITIAL_CAPACITY,
        )
        for index, allocation in enumerate(output_pool_allocations):
            self._preparation["calls"][f"cudaMalloc_output_pool_{index}"] = (
                allocation["malloc_call"]
            )
        self._preparation["preparation_id"] = self._executor_key
        self._preparation["prepared"] = True
        self._preparation["invocation_count"] = 0
        self._preparation["closed"] = False
        self._preparation["output_pool"] = self._output_pool.metadata()
        self._preparation["launch_dimensions"] = {
            "blocks": self._launch_blocks,
            "threads_per_block": self._threads_per_block,
        }
        self._preparation["device_selection_hoisted_to_compile_wrapper"] = True
        self._preparation["device_selection"] = {
            "cudaSetDevice": "preparation",
            "cudaGetDevice": "preparation",
            "per_launch_cudaSetDevice": False,
            "per_launch_cudaGetDevice": False,
        }
        self._executor_static_evidence = {
            "schema_version": POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION,
            "preparation_id": self._executor_key,
            "prepared": True,
            "setup_hoisted_to_compile_wrapper": True,
        }
        self._execution_static_evidence = {
            "schema_version": POINTWISE_REDUCE_COMPILE_EXECUTION_SCHEMA_VERSION,
            "implementation": "torch_rs",
            "status": "ok",
            "workload_version": POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION,
            "compile_backend": "inductor",
            "compile_fullgraph": True,
            "compile_dynamic": False,
            "native_cuda_compile": True,
            "eager_fallback": False,
            "forwarded_to_pytorch": False,
            "public_torch_cuda_api": False,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "required_cuda_visible_devices": required_cuda_visible_devices,
            "cuda_visible_devices_match": (
                required_cuda_visible_devices is None
                or os.environ.get("CUDA_VISIBLE_DEVICES")
                == required_cuda_visible_devices
            ),
            "single_visible_cuda_device": self._preparation.get(
                "single_visible_cuda_device"
            ),
            "cpu_fallback": False,
            "input_device_type": "cuda",
            "output_device_type": "cuda",
            "device_type": "cuda",
            "device_index": 0,
            "dtype": "float32",
            "workload_shape": list(self._workload_shape),
            "output_shape": list(self._output_shape),
            "input_metadata": copy.deepcopy(self._input_metadata),
            "output_metadata": copy.deepcopy(self._output_metadata),
            "launch_dimensions": {
                "blocks": self._launch_blocks,
                "threads_per_block": self._threads_per_block,
            },
            "device_output_bytes_checksum": None,
            "device_output_checksum": None,
            "readback_synchronized": False,
            "driver": copy.deepcopy(self._driver_probe["driver"]),
            "runtime": copy.deepcopy(self._preparation["runtime"]),
            "device_0": copy.deepcopy(self._driver_probe["device_0"]),
            "gpu": copy.deepcopy(self._preparation["gpu"]),
            "nvcc": copy.deepcopy(self._nvcc),
            "build": copy.deepcopy(self._build),
            "kernel_library": copy.deepcopy(self._kernel_library_evidence),
            "kernel_synchronized_in_call": False,
            "output_materialized": False,
            "readback_deferred": True,
            "device_selection_hoisted_to_preparation": True,
            "device_selection": {
                "cudaSetDevice": "preparation",
                "cudaGetDevice": "preparation",
                "per_launch_cudaSetDevice": False,
                "per_launch_cudaGetDevice": False,
            },
        }

    def metadata(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            metadata = copy.deepcopy(self._preparation)
            metadata["invocation_count"] = self._invocation_count
            metadata["closed"] = self._closed
            metadata["output_pool"] = self._output_pool.metadata()
            return metadata

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("compiled CUDA executor is closed")

    def _require_visible_mask(self) -> None:
        required_cuda_visible_devices = self._required_cuda_visible_devices
        if (
            required_cuda_visible_devices is not None
            and os.environ.get("CUDA_VISIBLE_DEVICES")
            != required_cuda_visible_devices
        ):
            raise _cuda_compile_unsupported(
                "CUDA_VISIBLE_DEVICES="
                f"{required_cuda_visible_devices} is required"
            )

    def _require_visible_device0(self) -> None:
        self._require_visible_mask()
        _cuda_runtime_ownership.verify_current_device(
            self._runtime,
            0,
            context="compiled workload",
        )

    def _require_materializable_output(self) -> None:
        with self._lifecycle_lock:
            if self._closed and self._output_pool.metadata()["live_buffers"] == 0:
                raise RuntimeError("compiled CUDA executor is closed")
            self._require_visible_device0()

    def _release_output_lease(
        self,
        lease: _cuda_runtime_ownership.PrivateCudaBufferLease,
        *,
        close_executor: bool = False,
    ) -> dict[str, Any] | None:
        with self._lifecycle_lock:
            release = lease.release(compact=not close_executor)
            if release is None:
                return None
            freed_after_executor_close = release.get(
                "freed_after_pool_close",
                False,
            )
            if close_executor:
                release = dict(release)
                release["executor_close"] = self.close()
                release["freed_after_executor_close"] = (
                    freed_after_executor_close
                    or release["buffer_name"] in {
                        key.removeprefix("cudaFree_")
                        for key in release["executor_close"].get("calls", {})
                    }
                )
                return release
            if hasattr(release, "with_executor_close"):
                return release.with_executor_close(
                    freed_after_executor_close=freed_after_executor_close,
                )
            release["freed_after_executor_close"] = freed_after_executor_close
            return release

    def _thread_launch_parameters(
        self,
    ) -> tuple[ctypes.c_int, ctypes.c_int, ctypes.c_int, Any, Any, Any]:
        parameters = getattr(self._launch_report_tls, "parameters", None)
        if parameters is None:
            blocks = ctypes.c_int()
            threads_per_block = ctypes.c_int()
            launch_error = ctypes.c_int()
            parameters = (
                blocks,
                threads_per_block,
                launch_error,
                ctypes.byref(blocks),
                ctypes.byref(threads_per_block),
                ctypes.byref(launch_error),
            )
            self._launch_report_tls.parameters = parameters
        blocks, threads_per_block, launch_error, *_refs = parameters
        blocks.value = 0
        threads_per_block.value = 0
        launch_error.value = 0
        return parameters

    def _runtime_error_evidence(self, result: int) -> dict[str, Any]:
        return {
            "result": result,
            "error_name": (
                _cuda_driver_probe._runtime_error_name(self._runtime, result)
                if result > 0
                else None
            ),
        }

    def _launch_evidence(self, token: _CompiledExecutionToken) -> dict[str, Any]:
        return {
            "result": token.result,
            "blocks": token.blocks,
            "threads_per_block": token.threads_per_block,
            "launch_error": self._runtime_error_evidence(token.launch_error),
            "sync_error": {
                "result": None,
                "error_name": None,
                "deferred_to_explicit_timing_boundary": True,
            },
        }

    def _output_pool_evidence(
        self,
        token: _CompiledExecutionToken,
    ) -> dict[str, Any]:
        output_pool_evidence = token.output_pool_acquisition.to_dict()
        allocated_in_execute = output_pool_evidence["allocated_in_acquire"]
        output_pool_evidence["allocated_in_execute"] = allocated_in_execute
        if allocated_in_execute:
            output_pool_evidence["source"] = "allocated_during_execute"
        return output_pool_evidence

    def _materialized_compile_execution(
        self,
        token: _CompiledExecutionToken,
        readback: _cuda_buffer.PrivateCudaHostReadback,
        device_output: _cuda_buffer.PrivateCudaFloat32Buffer,
    ) -> dict[str, Any]:
        if readback.payload is None:
            raise RuntimeError("compiled output readback payload is missing")

        launch = self._launch_evidence(token)
        output_pool_evidence = self._output_pool_evidence(token)
        calls = {
            "torchRsPrivateH100PointwiseReduceFloat32": launch,
            "cudaMemcpyDeviceToHost_output": readback.copy_call,
            "cudaDeviceSynchronize_after_device_to_host": readback.sync_call,
        }
        if token.output_pool_acquisition.allocated_in_acquire:
            calls["cudaMalloc_output"] = device_output.malloc_call

        materialized_execution = copy.deepcopy(self._execution_static_evidence)
        materialized_execution.update(
            {
                "reason": "compiled pointwise-reduce workload checksum read back",
                "device_output_bytes_checksum": readback.checksum,
                "device_output_checksum": _checksum_tensor_metadata_values(
                    readback.payload,
                    self._output_metadata,
                ),
                "readback_synchronized": True,
                "output_materialized": True,
                "executor": {
                    **self._executor_static_evidence,
                    "invocation_index": token.invocation_index,
                    "reused_prepared_executor": token.reused_prepared_executor,
                },
                "launch": launch,
                "calls": calls,
                "device_output_pointer_nonzero": (
                    token.device_output_pointer_nonzero
                ),
                "output_buffer_pool": output_pool_evidence,
            }
        )
        materialized_execution["output_buffer_pool"][
            "pool_snapshot_after_materialization"
        ] = self._output_pool.metadata()
        return materialized_execution

    def close(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            self._closed = True
            self._preparation["closed"] = True
            return self._output_pool.close()

    def synchronize(self, *, label: str) -> dict[str, Any]:
        with self._lifecycle_lock:
            self._require_open()
            self._require_visible_mask()
        return _cuda_runtime_ownership.synchronize_current_device(
            self._runtime,
            0,
            label=label,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def execute(
        self,
        x: CudaBenchmarkTensor,
        bias: CudaBenchmarkTensor,
        *,
        close_executor_on_output_release: bool = False,
    ) -> CudaBenchmarkTensor:
        device_x = _require_compiled_cuda_benchmark_input_buffer(
            x,
            name="x",
            expected_shape=self._workload_shape,
        )
        device_bias = _require_compiled_cuda_benchmark_input_buffer(
            bias,
            name="bias",
            expected_shape=self._bias_shape,
        )
        if device_x.runtime is not device_bias.runtime:
            raise _cuda_compile_unsupported(
                "input buffers do not share a CUDA runtime"
            )

        with self._lifecycle_lock:
            self._require_open()
            self._require_visible_mask()
            self._invocation_count += 1
            invocation_index = self._invocation_count
            output_lease = self._output_pool.acquire()
        device_output = output_lease.buffer
        if not device_output.allocation_ok:
            output_lease.release()
            raise RuntimeError("cudaMalloc failed for compiled output")

        try:
            (
                blocks,
                threads_per_block,
                launch_error,
                blocks_ref,
                threads_per_block_ref,
                launch_error_ref,
            ) = self._thread_launch_parameters()
            kernel_result = int(
                self._kernel_function(
                    device_x.pointer,
                    device_bias.pointer,
                    device_output.pointer,
                    self._rows,
                    self._columns,
                    blocks_ref,
                    threads_per_block_ref,
                    launch_error_ref,
                )
            )
            token = _CompiledExecutionToken(
                invocation_index=invocation_index,
                reused_prepared_executor=invocation_index > 1,
                output_pool_acquisition=output_lease.acquisition_token,
                device_output_pointer_nonzero=device_output.pointer_nonzero,
                result=kernel_result,
                launch_error=int(launch_error.value),
                blocks=int(blocks.value),
                threads_per_block=int(threads_per_block.value),
            )
            if kernel_result != 0:
                raise RuntimeError(
                    "private CUDA pointwise-reduce kernel launch failed"
                )
            if (
                token.blocks != self._launch_blocks
                or token.threads_per_block != self._threads_per_block
            ):
                raise RuntimeError(
                    "private CUDA pointwise-reduce launch dimensions changed"
                )

            def materialized_metadata_updates(
                readback: _cuda_buffer.PrivateCudaHostReadback,
                wrapper_metadata: dict[str, Any],
            ) -> dict[str, Any]:
                del wrapper_metadata
                return {
                    "compile_execution": self._materialized_compile_execution(
                        token,
                        readback,
                        device_output,
                    )
                }

            pending_execution = token
            return CudaBenchmarkTensor(
                device_output,
                checksum=(
                    lambda payload: _checksum_output_bytes(
                        payload,
                        self._rows,
                        self._columns,
                    )
                ),
                checksum_name="torch_rs_private_cuda_pointwise_reduce_output_v1",
                metadata_updates={
                    "workload_version": POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION,
                    "pointwise_reduce_kernel_version": POINTWISE_REDUCE_KERNEL_VERSION,
                    "native_cuda_compile": True,
                    "compile_backend": "inductor",
                    "compile_fullgraph": True,
                    "compile_dynamic": False,
                    "eager_fallback": False,
                    "forwarded_to_pytorch": False,
                    "compile_execution": pending_execution,
                },
                materialized_metadata_updates=materialized_metadata_updates,
                materialization_guard=self._require_materializable_output,
                release_buffer=(
                    lambda lease=output_lease: self._release_output_lease(
                        lease,
                        close_executor=close_executor_on_output_release,
                    )
                ),
            )
        except Exception:
            output_lease.release()
            raise


def prepare_h100_float32_pointwise_reduce_compiled_executor_device0(
    *,
    required_cuda_visible_devices: str | None = "0",
    workload_shape: tuple[int, int] = WORKLOAD_SHAPE,
) -> H100Float32PointwiseReduceCompiledExecutor:
    """Prepare invariant CUDA compile executor state once per wrapper."""
    rows, columns = _validate_compile_workload_shape(workload_shape)
    workload_shape = (rows, columns)
    output_shape = _compile_output_shape(workload_shape)
    launch_blocks, launch_threads = _compile_launch_dimensions(workload_shape)
    if (
        required_cuda_visible_devices is not None
        and type(required_cuda_visible_devices) is not str
    ):
        raise TypeError("required_cuda_visible_devices must be str or None")
    if (
        required_cuda_visible_devices is not None
        and os.environ.get("CUDA_VISIBLE_DEVICES") != required_cuda_visible_devices
    ):
        raise _cuda_compile_unsupported(
            "CUDA_VISIBLE_DEVICES="
            f"{required_cuda_visible_devices} is required"
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
    preparation: dict[str, Any] = {
        "schema_version": POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION,
        "implementation": "torch_rs",
        "status": "error",
        "reason": None,
        "workload_version": POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION,
        "compile_backend": "inductor",
        "compile_fullgraph": True,
        "compile_dynamic": False,
        "native_cuda_compile": True,
        "eager_fallback": False,
        "forwarded_to_pytorch": False,
        "public_torch_cuda_api": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_cuda_visible_devices": required_cuda_visible_devices,
        "cuda_visible_devices_match": (
            required_cuda_visible_devices is None
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            == required_cuda_visible_devices
        ),
        "cpu_fallback": False,
        "device_type": None,
        "device_index": None,
        "dtype": "float32",
        "workload_shape": list(workload_shape),
        "output_shape": list(output_shape),
        "launch_dimensions": {
            "blocks": launch_blocks,
            "threads_per_block": launch_threads,
        },
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
        "prepared": False,
        "setup_hoisted_to_compile_wrapper": True,
        "device_selection_hoisted_to_compile_wrapper": True,
        "device_selection": {
            "cudaSetDevice": "preparation",
            "cudaGetDevice": "preparation",
            "per_launch_cudaSetDevice": False,
            "per_launch_cudaGetDevice": False,
        },
        "invocation_count": 0,
        "calls": {},
    }

    if runtime is None:
        raise RuntimeError("CUDA runtime shared library was not loaded")

    missing_symbols = _configure_runtime_symbols(runtime)
    if missing_symbols:
        raise RuntimeError(
            "CUDA runtime is missing required symbols: "
            + ", ".join(missing_symbols)
        )

    device_count = ctypes.c_int()
    device_count_call = _runtime_call(
        runtime,
        "cudaGetDeviceCount",
        ctypes.byref(device_count),
    )
    device_count_call["value"] = (
        int(device_count.value) if device_count_call["result"] == 0 else None
    )
    preparation["calls"]["cudaGetDeviceCount"] = device_count_call
    if device_count_call["result"] != 0 or device_count.value < 1:
        raise RuntimeError("CUDA runtime has no visible device for compiled workload")
    preparation["single_visible_cuda_device"] = int(device_count.value) == 1

    set_device_call = _runtime_call(runtime, "cudaSetDevice", 0)
    preparation["calls"]["cudaSetDevice"] = set_device_call
    if set_device_call["result"] != 0:
        raise RuntimeError("cudaSetDevice(0) failed for compiled workload")

    current_device = ctypes.c_int(-1)
    get_device_call = _runtime_call(
        runtime,
        "cudaGetDevice",
        ctypes.byref(current_device),
    )
    get_device_call["value"] = (
        int(current_device.value) if get_device_call["result"] == 0 else None
    )
    preparation["calls"]["cudaGetDevice"] = get_device_call
    if get_device_call["result"] != 0:
        raise RuntimeError("cudaGetDevice failed for compiled workload")
    if int(current_device.value) != 0:
        raise RuntimeError("compiled workload did not select CUDA device 0")
    preparation["device_type"] = "cuda"
    preparation["device_index"] = 0

    if not nvcc["available"]:
        raise RuntimeError("nvcc was not found for compiled CUDA workload")

    repository_root = _cuda_kernel_support._repository_root()
    if repository_root is None:
        raise RuntimeError("worktree root was not found")

    library_path, build = _build_kernel(
        repository_root=repository_root,
        driver_probe=driver_probe,
        nvcc=nvcc,
    )
    preparation["build"] = build
    if library_path is None:
        raise RuntimeError(
            "nvcc failed to compile the private CUDA pointwise-reduce kernel"
        )

    kernel_library, load = _load_kernel_library(library_path, runtime_library)
    preparation["kernel_library"] = load
    if kernel_library is None:
        raise RuntimeError("compiled CUDA pointwise-reduce library was not loaded")

    preparation["status"] = "ok"
    preparation["reason"] = "compiled pointwise-reduce executor prepared"
    return H100Float32PointwiseReduceCompiledExecutor(
        runtime=runtime,
        runtime_library=runtime_library,
        runtime_load_error=runtime_load_error,
        driver_probe=driver_probe,
        nvcc=nvcc,
        build=build,
        kernel_library=kernel_library,
        kernel_library_evidence=load,
        required_cuda_visible_devices=required_cuda_visible_devices,
        preparation=preparation,
        workload_shape=workload_shape,
    )


def execute_h100_float32_pointwise_reduce_compiled_device0(
    x: CudaBenchmarkTensor,
    bias: CudaBenchmarkTensor,
    *,
    required_cuda_visible_devices: str | None = "0",
    workload_shape: tuple[int, int] = WORKLOAD_SHAPE,
) -> CudaBenchmarkTensor:
    """Execute the benchmark pointwise-reduce kernel for the CUDA compile path."""
    executor = prepare_h100_float32_pointwise_reduce_compiled_executor_device0(
        required_cuda_visible_devices=required_cuda_visible_devices,
        workload_shape=workload_shape,
    )
    return executor.execute(x, bias, close_executor_on_output_release=True)


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
    device_x = _cuda_buffer.PrivateCudaFloat32Buffer(
        runtime,
        (rows, columns),
        name="x",
        device_index=result["device_index"],
    )
    result["calls"]["cudaMalloc_x"] = device_x.malloc_call
    result["device_x_pointer_nonzero"] = device_x.pointer_nonzero
    if not device_x.allocation_ok:
        result["status"] = "error"
        result["reason"] = "cudaMalloc failed for x"
        return result

    device_bias = None
    device_output = None
    try:
        device_bias = _cuda_buffer.PrivateCudaFloat32Buffer(
            runtime,
            (columns,),
            name="bias",
            device_index=result["device_index"],
        )
        result["calls"]["cudaMalloc_bias"] = device_bias.malloc_call
        result["device_bias_pointer_nonzero"] = device_bias.pointer_nonzero
        if not device_bias.allocation_ok:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for bias"
            return result

        device_output = _cuda_buffer.PrivateCudaFloat32Buffer(
            runtime,
            (rows,),
            name="output",
            device_index=result["device_index"],
        )
        result["calls"]["cudaMalloc_output"] = device_output.malloc_call
        result["device_output_pointer_nonzero"] = device_output.pointer_nonzero
        if not device_output.allocation_ok:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for output"
            return result

        result["input_metadata"] = [device_x.metadata(), device_bias.metadata()]
        result["output_metadata"] = device_output.metadata()
        result["output_metadata_match"] = (
            expected_output_metadata is not None
            and result["output_metadata"] == expected_output_metadata
        )

        x_h2d_call = device_x.copy_from_host(x_host_bytes)
        result["calls"]["cudaMemcpyHostToDevice_x"] = x_h2d_call
        if x_h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed for x"
            return result

        bias_h2d_call = device_bias.copy_from_host(bias_host_bytes)
        result["calls"]["cudaMemcpyHostToDevice_bias"] = bias_h2d_call
        if bias_h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed for bias"
            return result

        sync_after_h2d = device_x.synchronize()
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
                device_x.pointer,
                device_bias.pointer,
                device_output.pointer,
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

        readback = device_output.checksum_readback(
            lambda payload: _checksum_output_bytes(payload, rows, columns),
        )
        d2h_call = readback.copy_call
        result["calls"]["cudaMemcpyDeviceToHost_output"] = d2h_call
        if d2h_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy device-to-host failed for output"
            return result

        sync_after_d2h = readback.sync_call
        result["calls"]["cudaDeviceSynchronize_after_device_to_host"] = (
            sync_after_d2h
        )
        if sync_after_d2h is None or sync_after_d2h["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaDeviceSynchronize failed after device-to-host"
            return result

        output_wrapper = CudaBenchmarkTensor(
            device_output,
            readback=readback,
            checksum_name="torch_rs_private_cuda_pointwise_reduce_output_v1",
        )
        result["public_cuda_tensor_wrapper"] = output_wrapper.metadata()

        host_output_bytes = readback.payload
        result["device_output_bytes_checksum"] = readback.checksum
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
        if device_output is not None:
            free_output = device_output.close()
        else:
            free_output = None
        if free_output is not None:
            result["calls"]["cudaFree_output"] = free_output
            if result["status"] == "ok" and free_output["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for output"
        if device_bias is not None:
            free_bias = device_bias.close()
        else:
            free_bias = None
        if free_bias is not None:
            result["calls"]["cudaFree_bias"] = free_bias
            if result["status"] == "ok" and free_bias["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for bias"
        free_x = device_x.close()
        if free_x is not None:
            result["calls"]["cudaFree_x"] = free_x
            if result["status"] == "ok" and free_x["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for x"


__all__ = [
    "H100Float32PointwiseReduceInputBundle",
    "H100Float32PointwiseReduceCompiledExecutor",
    "OUTPUT_SHAPE",
    "POINTWISE_REDUCE_COMPILE_EXECUTION_SCHEMA_VERSION",
    "POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION",
    "POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION",
    "POINTWISE_REDUCE_INPUTS_SCHEMA_VERSION",
    "POINTWISE_REDUCE_KERNEL_VERSION",
    "POINTWISE_REDUCE_OUTPUT_POOL_SCHEMA_VERSION",
    "POINTWISE_REDUCE_SCHEMA_VERSION",
    "SUPPORTED_WORKLOAD_SHAPES",
    "WORKLOAD_SHAPE",
    "execute_h100_float32_pointwise_reduce_compiled_device0",
    "launch_h100_float32_pointwise_reduce_device0",
    "make_h100_float32_pointwise_reduce_inputs_device0",
    "prepare_h100_float32_pointwise_reduce_compiled_executor_device0",
]
