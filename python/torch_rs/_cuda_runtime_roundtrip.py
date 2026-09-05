"""Private benchmark-only CUDA runtime allocation/copy roundtrip.

This module is intentionally separate from ``torch_rs.cuda``. It verifies that
benchmark code can move bytes through a CUDA runtime primitive without claiming
public CUDA tensor or compile support.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
from typing import Any

from . import _cuda_driver_probe


ROUNDTRIP_SCHEMA_VERSION = "torch_rs_private_cuda_runtime_roundtrip_v1"
DETERMINISTIC_VECTOR_VERSION = "float32_modulated_arange_v1"
DEFAULT_ELEMENT_COUNT = 1024
DEFAULT_ROUNDTRIP_CHECKSUM = "89c5ee9507c6f91487b4bad190da4a7f"

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


def _deterministic_float32_bytes(element_count: int) -> bytes:
    values = [
        ((index % 251) - 125) * 0.03125
        + ((index * 17) % 19) * 0.0009765625
        for index in range(element_count)
    ]
    return struct.pack(f"<{element_count}f", *values)


def _checksum_float32_bytes(payload: bytes, element_count: int) -> str:
    checksum = hashlib.blake2b(digest_size=16)
    checksum.update(b"torch_rs_private_cuda_float32_roundtrip_v1")
    checksum.update(element_count.to_bytes(8, "little"))
    checksum.update(payload)
    return checksum.hexdigest()


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
) -> dict[str, Any]:
    return {
        "schema_version": ROUNDTRIP_SCHEMA_VERSION,
        "primitive": "torch_rs_private_cuda_runtime_float32_roundtrip_device0",
        "public_torch_cuda_api": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_cuda_visible_devices": required_cuda_visible_devices,
        "cuda_visible_devices_match": (
            required_cuda_visible_devices is None
            or os.environ.get("CUDA_VISIBLE_DEVICES") == required_cuda_visible_devices
        ),
        "status": "unavailable",
        "reason": None,
        "cpu_fallback": False,
        "device_type": None,
        "device_index": None,
        "dtype": "float32",
        "element_count": element_count,
        "byte_count": byte_count,
        "deterministic_vector_version": DETERMINISTIC_VECTOR_VERSION,
        "expected_checksum": expected_checksum,
        "host_input_checksum": input_checksum,
        "device_roundtrip_checksum": None,
        "checksum_match": False,
        "device_pointer_nonzero": False,
        "driver": driver_probe["driver"],
        "runtime": _runtime_versions(runtime, runtime_library, runtime_load_error),
        "device_0": driver_probe["device_0"],
        "calls": {},
    }


def roundtrip_float32_device0(
    element_count: int = DEFAULT_ELEMENT_COUNT,
    *,
    required_cuda_visible_devices: str | None = "0",
) -> dict[str, Any]:
    """Allocate device-0 CUDA memory, copy deterministic float32 bytes, and sync.

    The returned dictionary is benchmark evidence. CUDA failures are returned as
    ``status``/``reason`` fields so tests can skip hardware-only checks cleanly
    on machines without a visible CUDA runtime.
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

    host_input_bytes = _deterministic_float32_bytes(element_count)
    byte_count = len(host_input_bytes)
    input_checksum = _checksum_float32_bytes(host_input_bytes, element_count)
    driver_probe = _cuda_driver_probe.probe_cuda_driver_device0()
    runtime, runtime_library, runtime_load_error = _cuda_driver_probe._load_shared_library(
        "cudart",
        _cuda_driver_probe._CUDA_RUNTIME_NAMES,
    )
    result = _base_result(
        element_count=element_count,
        byte_count=byte_count,
        input_checksum=input_checksum,
        expected_checksum=DEFAULT_ROUNDTRIP_CHECKSUM
        if element_count == DEFAULT_ELEMENT_COUNT
        else input_checksum,
        required_cuda_visible_devices=required_cuda_visible_devices,
        runtime=runtime,
        runtime_library=runtime_library,
        runtime_load_error=runtime_load_error,
        driver_probe=driver_probe,
    )

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

    device_pointer = ctypes.c_void_p()
    malloc_call = _runtime_call(
        runtime,
        "cudaMalloc",
        ctypes.byref(device_pointer),
        byte_count,
    )
    malloc_call["byte_count"] = byte_count
    result["calls"]["cudaMalloc"] = malloc_call
    result["device_pointer_nonzero"] = bool(device_pointer.value)
    if malloc_call["result"] != 0 or not device_pointer.value:
        result["status"] = "error"
        result["reason"] = "cudaMalloc failed"
        return result

    host_input = ctypes.create_string_buffer(host_input_bytes, byte_count)
    host_output = ctypes.create_string_buffer(byte_count)

    try:
        h2d_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            device_pointer,
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

        d2h_call = _runtime_call(
            runtime,
            "cudaMemcpy",
            ctypes.c_void_p(ctypes.addressof(host_output)),
            device_pointer,
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
        output_checksum = _checksum_float32_bytes(host_output_bytes, element_count)
        result["device_roundtrip_checksum"] = output_checksum
        result["checksum_match"] = (
            host_output_bytes == host_input_bytes
            and output_checksum == result["expected_checksum"]
        )
        if not result["checksum_match"]:
            result["status"] = "error"
            result["reason"] = "roundtrip checksum mismatch"
            return result

        result["status"] = "ok"
        result["reason"] = "roundtrip checksum verified"
        return result
    finally:
        free_call = _runtime_call(runtime, "cudaFree", device_pointer)
        result["calls"]["cudaFree"] = free_call
        if result["status"] == "ok" and free_call["result"] != 0:
            result["status"] = "error"
            result["reason"] = "cudaFree failed after roundtrip"


__all__ = [
    "DEFAULT_ELEMENT_COUNT",
    "DEFAULT_ROUNDTRIP_CHECKSUM",
    "DETERMINISTIC_VECTOR_VERSION",
    "ROUNDTRIP_SCHEMA_VERSION",
    "roundtrip_float32_device0",
]
