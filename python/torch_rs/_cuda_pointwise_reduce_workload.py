"""Private benchmark-only CUDA pointwise-plus-row-reduction workload.

This module is intentionally separate from ``torch_rs.cuda``. It verifies that
benchmark code can allocate torch_rs-owned CUDA buffers and run the H100 CUDA
``torch.compile`` reference workload shape without claiming public CUDA tensor
or compile support.
"""

from __future__ import annotations

import ctypes
import copy
import hashlib
import json
import os
import struct
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
    "h100_cuda_pointwise_plus_row_reduce_float32_v2"
)
WORKLOAD_SHAPE = (1024, 1024)
OUTPUT_SHAPE = (1024,)
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
_THREADS_PER_BLOCK = 32
_OUTPUT_POOL_INITIAL_CAPACITY = 2
_CUDA_UNAVAILABLE_ERRORS = _cuda_buffer.CUDA_UNAVAILABLE_ERRORS
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
    try:
        function = getattr(library, function_name)
        async_function = getattr(library, async_function_name)
    except AttributeError as error:
        return None, {
            "loaded": False,
            "error": str(error),
            "function": function_name,
            "async_function": async_function_name,
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
    return library, {
        "loaded": True,
        "error": None,
        "function": function_name,
        "async_function": async_function_name,
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
    return NotImplementedError(
        "torch.compile(): native CUDA inductor execution is supported only "
        "for the versioned H100 float32 pointwise-reduce benchmark with "
        "two live CUDA benchmark tensor inputs shaped [1024, 1024] and "
        f"[1024]; {reason}"
    )


def _require_compiled_cuda_benchmark_input(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> tuple[_cuda_buffer.PrivateCudaFloat32Buffer, dict[str, Any]]:
    if type(value) is not CudaBenchmarkTensor:
        raise _cuda_compile_unsupported(
            f"{name} is not a torch_rs CudaBenchmarkTensor"
        )

    expected_metadata = (
        _WORKLOAD_X_METADATA
        if expected_shape == WORKLOAD_SHAPE
        else _WORKLOAD_BIAS_METADATA
    )
    metadata = {
        "shape": list(value.shape),
        "stride": list(value.stride),
        "storage_offset": 0,
        "dtype": str(value.dtype),
        "device": value.device,
        "device_type": value.device_type,
        "device_index": value.device_index,
        "requires_grad": value.requires_grad,
        "is_contiguous": value.is_contiguous,
        "is_cuda": value.is_cuda,
        "cpu_fallback": False,
    }
    mismatched = [
        key
        for key, expected_value in expected_metadata.items()
        if metadata.get(key) != expected_value
    ]
    if metadata.get("is_cuda") is not True:
        mismatched.append("is_cuda")
    if metadata.get("cpu_fallback") is not False:
        mismatched.append("cpu_fallback")
    if mismatched:
        joined = ", ".join(sorted(set(mismatched)))
        raise _cuda_compile_unsupported(f"{name} metadata mismatch: {joined}")

    try:
        buffer = value._torch_rs_private_cuda_buffer()
    except ValueError as error:
        raise _cuda_compile_unsupported(f"{name} private buffer is not live") from error
    if buffer.metadata() != expected_metadata:
        raise _cuda_compile_unsupported(f"{name} private buffer metadata mismatch")
    return buffer, copy.deepcopy(expected_metadata)


class H100Float32PointwiseReduceCompiledExecutor:
    """Prepared executor for the narrow H100 CUDA compile benchmark."""

    __slots__ = (
        "_build",
        "_closed",
        "_driver_probe",
        "_executor_key",
        "_invocation_count",
        "_kernel_function",
        "_kernel_library",
        "_kernel_library_evidence",
        "_nvcc",
        "_output_pool",
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
    ) -> None:
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
        self._kernel_function = getattr(
            self._kernel_library,
            "torch_rs_private_h100_pointwise_reduce_float32_launch_async_v1",
        )
        key_payload = json.dumps(
            {
                "workload_version": POINTWISE_REDUCE_COMPILE_WORKLOAD_VERSION,
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
            OUTPUT_SHAPE,
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

    def metadata(self) -> dict[str, Any]:
        metadata = copy.deepcopy(self._preparation)
        metadata["invocation_count"] = self._invocation_count
        metadata["closed"] = self._closed
        metadata["output_pool"] = self._output_pool.metadata()
        return metadata

    @property
    def closed(self) -> bool:
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

    def _set_and_require_visible_device0(self) -> dict[str, Any]:
        self._require_visible_mask()
        calls = _cuda_runtime_ownership.set_and_verify_current_device(
            self._runtime,
            0,
            context="compiled workload",
        )
        return {
            "cudaSetDevice_before_launch": calls["cudaSetDevice"],
            "cudaGetDevice_before_launch": calls["cudaGetDevice"],
        }

    def _require_materializable_output(self) -> None:
        if self._closed and self._output_pool.metadata()["live_buffers"] == 0:
            raise RuntimeError("compiled CUDA executor is closed")
        self._require_visible_device0()

    def _release_output_lease(
        self,
        lease: _cuda_runtime_ownership.PrivateCudaBufferLease,
        *,
        close_executor: bool = False,
    ) -> dict[str, Any] | None:
        release = lease.release()
        if release is None:
            return None
        release["freed_after_executor_close"] = release.get(
            "freed_after_pool_close",
            False,
        )
        if close_executor:
            release["executor_close"] = self.close()
        return release

    def close(self) -> dict[str, Any]:
        self._closed = True
        self._preparation["closed"] = True
        return self._output_pool.close()

    def synchronize(self, *, label: str) -> dict[str, Any]:
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
        self._require_open()
        required_cuda_visible_devices = self._required_cuda_visible_devices
        self._require_visible_mask()

        device_x, x_metadata = _require_compiled_cuda_benchmark_input(
            x,
            name="x",
            expected_shape=WORKLOAD_SHAPE,
        )
        device_bias, bias_metadata = _require_compiled_cuda_benchmark_input(
            bias,
            name="bias",
            expected_shape=(WORKLOAD_SHAPE[1],),
        )
        if device_x.runtime is not device_bias.runtime:
            raise _cuda_compile_unsupported(
                "input buffers do not share a CUDA runtime"
            )

        self._invocation_count += 1
        invocation_index = self._invocation_count
        runtime = self._runtime
        evidence: dict[str, Any] = {
            "schema_version": POINTWISE_REDUCE_COMPILE_EXECUTION_SCHEMA_VERSION,
            "implementation": "torch_rs",
            "status": "error",
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
            "input_device_type": "cuda",
            "output_device_type": None,
            "device_type": None,
            "device_index": None,
            "dtype": "float32",
            "workload_shape": list(WORKLOAD_SHAPE),
            "output_shape": list(OUTPUT_SHAPE),
            "input_metadata": [x_metadata, bias_metadata],
            "output_metadata": None,
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
            "executor": {
                "schema_version": POINTWISE_REDUCE_COMPILE_EXECUTOR_SCHEMA_VERSION,
                "preparation_id": self._executor_key,
                "prepared": True,
                "setup_hoisted_to_compile_wrapper": True,
                "invocation_index": invocation_index,
                "reused_prepared_executor": invocation_index > 1,
            },
            "launch": None,
            "calls": {},
        }

        evidence["single_visible_cuda_device"] = self._preparation.get(
            "single_visible_cuda_device"
        )
        evidence["device_type"] = "cuda"
        evidence["device_index"] = 0
        evidence["calls"].update(self._set_and_require_visible_device0())

        output_lease = self._output_pool.acquire()
        device_output = output_lease.buffer
        output_pool_evidence = copy.deepcopy(output_lease.acquisition)
        output_pool_evidence["allocated_in_execute"] = output_pool_evidence[
            "allocated_in_acquire"
        ]
        if output_pool_evidence["allocated_in_execute"]:
            output_pool_evidence["source"] = "allocated_during_execute"
        evidence["output_buffer_pool"] = output_pool_evidence
        if output_pool_evidence["allocated_in_execute"]:
            evidence["calls"]["cudaMalloc_output"] = device_output.malloc_call
        evidence["device_output_pointer_nonzero"] = device_output.pointer_nonzero
        if not device_output.allocation_ok:
            output_lease.release()
            raise RuntimeError("cudaMalloc failed for compiled output")

        try:
            blocks = ctypes.c_int(0)
            threads = ctypes.c_int(0)
            launch_error = ctypes.c_int(0)
            kernel_result = int(
                self._kernel_function(
                    device_x.pointer,
                    device_bias.pointer,
                    device_output.pointer,
                    WORKLOAD_SHAPE[0],
                    WORKLOAD_SHAPE[1],
                    ctypes.byref(blocks),
                    ctypes.byref(threads),
                    ctypes.byref(launch_error),
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
                    "result": None,
                    "error_name": None,
                    "deferred_to_explicit_timing_boundary": True,
                },
            }
            evidence["launch"] = launch
            evidence["calls"]["torchRsPrivateH100PointwiseReduceFloat32"] = launch
            if kernel_result != 0:
                raise RuntimeError(
                    "private CUDA pointwise-reduce kernel launch failed"
                )

            output_metadata = copy.deepcopy(_WORKLOAD_OUTPUT_METADATA)
            evidence["status"] = "ok"
            evidence["reason"] = (
                "compiled pointwise-reduce workload launched with output "
                "checksum readback pending"
            )
            evidence["output_device_type"] = "cuda"
            evidence["output_metadata"] = output_metadata
            evidence["kernel_synchronized_in_call"] = False
            evidence["output_materialized"] = False
            evidence["readback_deferred"] = True
            evidence["readback_synchronized"] = False

            def materialized_metadata_updates(
                readback: _cuda_buffer.PrivateCudaHostReadback,
                wrapper_metadata: dict[str, Any],
            ) -> dict[str, Any]:
                del wrapper_metadata
                if readback.payload is None:
                    raise RuntimeError("compiled output readback payload is missing")
                materialized_execution = copy.deepcopy(evidence)
                materialized_execution["reason"] = (
                    "compiled pointwise-reduce workload checksum read back"
                )
                materialized_execution["device_output_bytes_checksum"] = (
                    readback.checksum
                )
                materialized_execution["device_output_checksum"] = (
                    _checksum_tensor_metadata_values(
                        readback.payload,
                        output_metadata,
                    )
                )
                materialized_execution["readback_synchronized"] = True
                materialized_execution["output_materialized"] = True
                materialized_execution["driver"] = copy.deepcopy(
                    self._driver_probe["driver"]
                )
                materialized_execution["runtime"] = copy.deepcopy(
                    self._preparation["runtime"]
                )
                materialized_execution["device_0"] = copy.deepcopy(
                    self._driver_probe["device_0"]
                )
                materialized_execution["gpu"] = copy.deepcopy(
                    self._preparation["gpu"]
                )
                materialized_execution["nvcc"] = copy.deepcopy(self._nvcc)
                materialized_execution["build"] = copy.deepcopy(self._build)
                materialized_execution["kernel_library"] = copy.deepcopy(
                    self._kernel_library_evidence
                )
                materialized_execution["calls"][
                    "cudaMemcpyDeviceToHost_output"
                ] = readback.copy_call
                materialized_execution["calls"][
                    "cudaDeviceSynchronize_after_device_to_host"
                ] = readback.sync_call
                materialized_execution["output_buffer_pool"] = (
                    copy.deepcopy(output_pool_evidence)
                )
                materialized_execution["output_buffer_pool"][
                    "pool_snapshot_after_materialization"
                ] = self._output_pool.metadata()
                return {"compile_execution": materialized_execution}

            return CudaBenchmarkTensor(
                device_output,
                checksum=(
                    lambda payload: _checksum_output_bytes(
                        payload,
                        WORKLOAD_SHAPE[0],
                        WORKLOAD_SHAPE[1],
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
                    "compile_execution": evidence,
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
) -> H100Float32PointwiseReduceCompiledExecutor:
    """Prepare invariant CUDA compile executor state once per wrapper."""
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
        "workload_shape": list(WORKLOAD_SHAPE),
        "output_shape": list(OUTPUT_SHAPE),
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
    )


def execute_h100_float32_pointwise_reduce_compiled_device0(
    x: CudaBenchmarkTensor,
    bias: CudaBenchmarkTensor,
    *,
    required_cuda_visible_devices: str | None = "0",
) -> CudaBenchmarkTensor:
    """Execute the benchmark pointwise-reduce kernel for the CUDA compile path."""
    executor = prepare_h100_float32_pointwise_reduce_compiled_executor_device0(
        required_cuda_visible_devices=required_cuda_visible_devices,
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
    "WORKLOAD_SHAPE",
    "execute_h100_float32_pointwise_reduce_compiled_device0",
    "launch_h100_float32_pointwise_reduce_device0",
    "make_h100_float32_pointwise_reduce_inputs_device0",
    "prepare_h100_float32_pointwise_reduce_compiled_executor_device0",
]
