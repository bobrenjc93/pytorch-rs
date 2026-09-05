"""Private benchmark-only CUDA pointwise kernel launch.

This module is intentionally separate from ``torch_rs.cuda``. It proves that
benchmark-only torch_rs-owned CUDA code can compile, launch, synchronize, and
verify one deterministic float32 pointwise kernel without claiming public CUDA
tensor or compile support.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

from . import _cuda_driver_probe


POINTWISE_SCHEMA_VERSION = "torch_rs_private_cuda_pointwise_kernel_v1"
DETERMINISTIC_INPUT_VERSION = "float32_modulated_arange_v1"
POINTWISE_KERNEL_VERSION = "scale2_add_index_offset_v1"
DEFAULT_ELEMENT_COUNT = 4096
DEFAULT_INPUT_CHECKSUM = "8922573efa224da11cd33b7dd0401a60"
DEFAULT_POINTWISE_CHECKSUM = "859e8e6c64e796d56eee827f79e23386"

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_UNAVAILABLE_ERRORS = {
    "cudaErrorInsufficientDriver",
    "cudaErrorNoDevice",
}
_CUDA_RUNTIME_SYMBOLS = (
    "cudaGetDeviceCount",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaMalloc",
    "cudaMemcpy",
    "cudaDeviceSynchronize",
    "cudaFree",
)
_CUDA_SOURCE = r"""
#include <cuda_runtime.h>

extern "C" __global__ void torch_rs_private_pointwise_kernel_v1(
    const float* input,
    float* output,
    int element_count
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) {
        return;
    }

    float offset = static_cast<float>((index % 29) - 14) * 0.00390625f;
    output[index] = input[index] * 2.0f + offset;
}

extern "C" int torch_rs_private_pointwise_float32_v1(
    const float* input,
    float* output,
    int element_count,
    int* blocks_out,
    int* threads_out,
    int* launch_error_out,
    int* sync_error_out
) {
    if (input == nullptr || output == nullptr || element_count <= 0) {
        return -1;
    }

    int threads = 256;
    int blocks = (element_count + threads - 1) / threads;
    if (blocks_out != nullptr) {
        *blocks_out = blocks;
    }
    if (threads_out != nullptr) {
        *threads_out = threads;
    }

    torch_rs_private_pointwise_kernel_v1<<<blocks, threads>>>(
        input,
        output,
        element_count
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


def _deterministic_input_values(element_count: int) -> list[float]:
    return [
        ((index % 251) - 125) * 0.03125
        + ((index * 17) % 19) * 0.0009765625
        for index in range(element_count)
    ]


def _deterministic_input_bytes(element_count: int) -> bytes:
    return struct.pack(f"<{element_count}f", *_deterministic_input_values(element_count))


def _expected_output_bytes(element_count: int) -> bytes:
    values = [
        value * 2.0 + ((index % 29) - 14) * 0.00390625
        for index, value in enumerate(_deterministic_input_values(element_count))
    ]
    return struct.pack(f"<{element_count}f", *values)


def _checksum_payload(domain: bytes, payload: bytes, element_count: int) -> str:
    checksum = hashlib.blake2b(digest_size=16)
    checksum.update(domain)
    checksum.update(element_count.to_bytes(8, "little"))
    checksum.update(payload)
    return checksum.hexdigest()


def _checksum_input_bytes(payload: bytes, element_count: int) -> str:
    return _checksum_payload(
        b"torch_rs_private_cuda_pointwise_input_v1",
        payload,
        element_count,
    )


def _checksum_output_bytes(payload: bytes, element_count: int) -> str:
    return _checksum_payload(
        b"torch_rs_private_cuda_pointwise_kernel_v1",
        payload,
        element_count,
    )


def _runtime_call(
    runtime: ctypes.CDLL,
    function_name: str,
    *arguments: Any,
) -> dict[str, Any]:
    function = getattr(runtime, function_name)
    code = int(function(*arguments))
    return {
        "result": code,
        "error_name": (
            _cuda_driver_probe._runtime_error_name(runtime, code) if code else None
        ),
    }


def _runtime_versions(
    runtime: ctypes.CDLL | None,
    library: str | None,
    load_error: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "library": library,
        "loaded": runtime is not None,
        "load_error": load_error,
        "runtime_version": None,
        "runtime_version_text": None,
        "driver_version": None,
        "driver_version_text": None,
        "calls": {},
    }
    if runtime is None:
        return result

    try:
        runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
    except AttributeError:
        result["calls"]["cudaRuntimeGetVersion"] = {
            "result": None,
            "error_name": "missing symbol",
        }
    else:
        runtime_version = ctypes.c_int()
        runtime_call = _runtime_call(
            runtime,
            "cudaRuntimeGetVersion",
            ctypes.byref(runtime_version),
        )
        result["calls"]["cudaRuntimeGetVersion"] = runtime_call
        if runtime_call["result"] == 0:
            result["runtime_version"] = int(runtime_version.value)
            result["runtime_version_text"] = _cuda_driver_probe._cuda_version_text(
                int(runtime_version.value)
            )

    try:
        runtime.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaDriverGetVersion.restype = ctypes.c_int
    except AttributeError:
        result["calls"]["cudaDriverGetVersion"] = {
            "result": None,
            "error_name": "missing symbol",
        }
    else:
        driver_version = ctypes.c_int()
        driver_call = _runtime_call(
            runtime,
            "cudaDriverGetVersion",
            ctypes.byref(driver_version),
        )
        result["calls"]["cudaDriverGetVersion"] = driver_call
        if driver_call["result"] == 0:
            result["driver_version"] = int(driver_version.value)
            result["driver_version_text"] = _cuda_driver_probe._cuda_version_text(
                int(driver_version.value)
            )

    return result


def _configure_runtime_symbols(runtime: ctypes.CDLL) -> list[str]:
    missing: list[str] = []
    for name in _CUDA_RUNTIME_SYMBOLS:
        if not hasattr(runtime, name):
            missing.append(name)

    if missing:
        return missing

    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDeviceCount.restype = ctypes.c_int
    runtime.cudaSetDevice.argtypes = [ctypes.c_int]
    runtime.cudaSetDevice.restype = ctypes.c_int
    runtime.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaGetDevice.restype = ctypes.c_int
    runtime.cudaMalloc.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    runtime.cudaMalloc.restype = ctypes.c_int
    runtime.cudaMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    runtime.cudaMemcpy.restype = ctypes.c_int
    runtime.cudaDeviceSynchronize.argtypes = []
    runtime.cudaDeviceSynchronize.restype = ctypes.c_int
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaFree.restype = ctypes.c_int
    return []


def _run_text(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError as error:
        return {
            "command": command,
            "returncode": None,
            "stdout": "",
            "stderr": str(error),
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _nvcc_provenance() -> dict[str, Any]:
    path = shutil.which("nvcc")
    result: dict[str, Any] = {
        "path": path,
        "available": path is not None,
        "version": None,
    }
    if path is not None:
        result["version"] = _run_text([path, "--version"])
    return result


def _repository_root() -> Path | None:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "Cargo.toml").is_file() and (
            candidate / "pyproject.toml"
        ).is_file():
            return candidate.resolve()
    return None


def _target_build_directory(root: Path, key: str) -> Path:
    target = root / "target"
    if target.is_symlink():
        raise RuntimeError(f"refusing symlinked target directory: {target}")
    target.mkdir(exist_ok=True)
    build_directory = target / "torch_rs_private_cuda_pointwise" / key
    build_directory.mkdir(parents=True, exist_ok=True)
    resolved = build_directory.resolve()
    if root not in (resolved, *resolved.parents):
        raise RuntimeError(
            f"CUDA pointwise build directory resolved outside worktree: {resolved}"
        )
    return resolved


def _compute_capability_arch(driver_probe: dict[str, Any]) -> str:
    capability = (driver_probe.get("device_0") or {}).get("compute_capability")
    if (
        isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(value, int) for value in capability)
    ):
        return f"sm_{capability[0]}{capability[1]}"
    return "sm_90"


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
            _compute_capability_arch(driver_probe),
        ]
    ).encode("utf-8")
    build_key = hashlib.blake2b(build_key_payload, digest_size=8).hexdigest()
    build: dict[str, Any] = {
        "source_checksum": source_checksum,
        "architecture": _compute_capability_arch(driver_probe),
        "build_key": build_key,
        "source_path": None,
        "library_path": None,
        "reused": False,
        "compile": None,
    }
    if not nvcc_path:
        return None, build

    build_directory = _target_build_directory(repository_root, build_key)
    source_path = build_directory / "torch_rs_private_pointwise.cu"
    library_path = build_directory / "libtorch_rs_private_pointwise.so"
    build["source_path"] = str(source_path)
    build["library_path"] = str(library_path)

    if not source_path.exists() or source_path.read_text(encoding="utf-8") != _CUDA_SOURCE:
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
    compile_result = _run_text(command, cwd=repository_root)
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

    function_name = "torch_rs_private_pointwise_float32_v1"
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


def _gpu_provenance(driver_probe: dict[str, Any]) -> dict[str, Any]:
    device = driver_probe.get("device_0") or {}
    device_count = (driver_probe.get("driver") or {}).get("device_count") or {}
    return {
        "cuda_visible_device_index": 0,
        "driver_device_handle": device.get("driver_device_handle"),
        "name": device.get("name"),
        "compute_capability": device.get("compute_capability"),
        "total_memory_bytes": device.get("total_memory_bytes"),
        "driver_visible_device_count": device_count.get("value"),
    }


def _base_result(
    *,
    element_count: int,
    byte_count: int,
    input_checksum: str,
    expected_checksum: str,
    required_cuda_visible_devices: str | None,
    runtime: ctypes.CDLL | None,
    runtime_library: str | None,
    runtime_load_error: str | None,
    driver_probe: dict[str, Any],
    nvcc: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": POINTWISE_SCHEMA_VERSION,
        "primitive": "torch_rs_private_cuda_pointwise_float32_kernel_device0",
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
        "element_count": element_count,
        "byte_count": byte_count,
        "deterministic_input_version": DETERMINISTIC_INPUT_VERSION,
        "pointwise_kernel_version": POINTWISE_KERNEL_VERSION,
        "expected_checksum": expected_checksum,
        "host_input_checksum": input_checksum,
        "device_output_checksum": None,
        "checksum_match": False,
        "device_input_pointer_nonzero": False,
        "device_output_pointer_nonzero": False,
        "driver": driver_probe["driver"],
        "runtime": _runtime_versions(runtime, runtime_library, runtime_load_error),
        "device_0": driver_probe["device_0"],
        "gpu": _gpu_provenance(driver_probe),
        "nvcc": nvcc,
        "build": None,
        "kernel_library": None,
        "launch": None,
        "calls": {},
    }


def launch_float32_pointwise_device0(
    element_count: int = DEFAULT_ELEMENT_COUNT,
    *,
    required_cuda_visible_devices: str | None = "0",
) -> dict[str, Any]:
    """Compile, launch, synchronize, and verify one CUDA pointwise kernel.

    CUDA failures are reported in ``status``/``reason`` fields so tests can
    skip hardware-only checks cleanly. The helper never substitutes CPU
    execution for the CUDA kernel.
    """
    if type(element_count) is not int:
        raise TypeError("element_count must be int")
    if element_count <= 0:
        raise ValueError("element_count must be positive")
    if (
        required_cuda_visible_devices is not None
        and type(required_cuda_visible_devices) is not str
    ):
        raise TypeError("required_cuda_visible_devices must be str or None")

    host_input_bytes = _deterministic_input_bytes(element_count)
    expected_output = _expected_output_bytes(element_count)
    byte_count = len(host_input_bytes)
    input_checksum = _checksum_input_bytes(host_input_bytes, element_count)
    expected_checksum = (
        DEFAULT_POINTWISE_CHECKSUM
        if element_count == DEFAULT_ELEMENT_COUNT
        else _checksum_output_bytes(expected_output, element_count)
    )
    driver_probe = _cuda_driver_probe.probe_cuda_driver_device0()
    runtime, runtime_library, runtime_load_error = _cuda_driver_probe._load_shared_library(
        "cudart",
        _cuda_driver_probe._CUDA_RUNTIME_NAMES,
    )
    nvcc = _nvcc_provenance()
    result = _base_result(
        element_count=element_count,
        byte_count=byte_count,
        input_checksum=input_checksum,
        expected_checksum=expected_checksum,
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

    repository_root = _repository_root()
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
        result["reason"] = "nvcc failed to compile the private CUDA pointwise kernel"
        return result

    kernel_library, load = _load_kernel_library(library_path, runtime_library)
    result["kernel_library"] = load
    if kernel_library is None:
        result["status"] = "error"
        result["reason"] = "compiled CUDA pointwise library was not loaded"
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

    device_input = ctypes.c_void_p()
    device_output = ctypes.c_void_p()
    input_malloc = _runtime_call(
        runtime,
        "cudaMalloc",
        ctypes.byref(device_input),
        byte_count,
    )
    input_malloc["byte_count"] = byte_count
    result["calls"]["cudaMalloc_input"] = input_malloc
    result["device_input_pointer_nonzero"] = bool(device_input.value)
    if input_malloc["result"] != 0 or not device_input.value:
        result["status"] = "error"
        result["reason"] = "cudaMalloc failed for input"
        return result

    try:
        output_malloc = _runtime_call(
            runtime,
            "cudaMalloc",
            ctypes.byref(device_output),
            byte_count,
        )
        output_malloc["byte_count"] = byte_count
        result["calls"]["cudaMalloc_output"] = output_malloc
        result["device_output_pointer_nonzero"] = bool(device_output.value)
        if output_malloc["result"] != 0 or not device_output.value:
            result["status"] = "error"
            result["reason"] = "cudaMalloc failed for output"
            return result

        host_input = ctypes.create_string_buffer(host_input_bytes, byte_count)
        host_output = ctypes.create_string_buffer(byte_count)

        h2d_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            device_input,
            ctypes.c_void_p(ctypes.addressof(host_input)),
            byte_count,
            _CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        h2d_call["kind"] = "cudaMemcpyHostToDevice"
        h2d_call["byte_count"] = byte_count
        result["calls"]["cudaMemcpyHostToDevice"] = h2d_call
        if h2d_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy host-to-device failed"
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
            "torch_rs_private_pointwise_float32_v1",
        )
        kernel_result = int(
            kernel_function(
                device_input,
                device_output,
                element_count,
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
        result["calls"]["torchRsPrivatePointwiseFloat32"] = launch
        if kernel_result != 0:
            result["status"] = "error"
            result["reason"] = "private CUDA pointwise kernel launch failed"
            return result

        d2h_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            ctypes.c_void_p(ctypes.addressof(host_output)),
            device_output,
            byte_count,
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        d2h_call["kind"] = "cudaMemcpyDeviceToHost"
        d2h_call["byte_count"] = byte_count
        result["calls"]["cudaMemcpyDeviceToHost"] = d2h_call
        if d2h_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaMemcpy device-to-host failed"
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
            byte_count,
        )
        output_checksum = _checksum_output_bytes(host_output_bytes, element_count)
        result["device_output_checksum"] = output_checksum
        result["checksum_match"] = (
            host_output_bytes == expected_output
            and output_checksum == result["expected_checksum"]
        )
        if not result["checksum_match"]:
            result["status"] = "error"
            result["reason"] = "pointwise kernel checksum mismatch"
            return result

        result["status"] = "ok"
        result["reason"] = "pointwise kernel checksum verified"
        return result
    finally:
        if device_output.value:
            free_output = _runtime_call(runtime, "cudaFree", device_output)
            result["calls"]["cudaFree_output"] = free_output
            if result["status"] == "ok" and free_output["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for output"
        if device_input.value:
            free_input = _runtime_call(runtime, "cudaFree", device_input)
            result["calls"]["cudaFree_input"] = free_input
            if result["status"] == "ok" and free_input["result"] != 0:
                result["status"] = "error"
                result["reason"] = "cudaFree failed for input"


__all__ = [
    "DEFAULT_ELEMENT_COUNT",
    "DEFAULT_INPUT_CHECKSUM",
    "DEFAULT_POINTWISE_CHECKSUM",
    "DETERMINISTIC_INPUT_VERSION",
    "POINTWISE_KERNEL_VERSION",
    "POINTWISE_SCHEMA_VERSION",
    "launch_float32_pointwise_device0",
]
