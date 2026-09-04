"""Private benchmark-only CUDA driver metadata probe.

This module is intentionally not used by ``torch_rs.cuda``. It exists only so
benchmark drivers can record whether this process can initialize CUDA driver
metadata without claiming public CUDA tensor or compile support.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import os
from pathlib import Path
from typing import Any


PROBE_SCHEMA_VERSION = "torch_rs_private_cuda_driver_probe_v1"

_CUDA_DRIVER_NAMES = ("libcuda.so.1", "libcuda.so")
_CUDA_RUNTIME_NAMES = ("libcudart.so.13", "libcudart.so.12", "libcudart.so")
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76


def _cuda_version_text(version: int | None) -> str | None:
    if version is None:
        return None
    major = version // 1000
    minor = (version % 1000) // 10
    patch = version % 10
    if patch:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}"


def _load_shared_library(
    kind: str,
    names: tuple[str, ...],
) -> tuple[ctypes.CDLL | None, str | None, str | None]:
    candidates: list[str] = []
    found = ctypes.util.find_library(kind)
    if found is not None:
        candidates.append(found)
    candidates.extend(names)

    if kind == "cudart":
        try:
            spec = importlib.util.find_spec("nvidia.cuda_runtime")
        except ModuleNotFoundError:
            spec = None
        if spec is not None and spec.submodule_search_locations is not None:
            for location in spec.submodule_search_locations:
                runtime_library_dir = Path(location) / "lib"
                candidates.extend(
                    str(candidate)
                    for candidate in sorted(
                        runtime_library_dir.glob("libcudart.so*"),
                        reverse=True,
                    )
                )

    seen: set[str] = set()
    errors: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return ctypes.CDLL(candidate), candidate, None
        except OSError as error:
            errors.append(f"{candidate}: {error}")

    if errors:
        return None, None, "; ".join(errors)
    return None, None, f"{kind} shared library was not found"


def _driver_error_name(driver: ctypes.CDLL, code: int) -> str | None:
    try:
        cu_get_error_name = driver.cuGetErrorName
    except AttributeError:
        return None
    name = ctypes.c_char_p()
    cu_get_error_name.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
    cu_get_error_name.restype = ctypes.c_int
    if cu_get_error_name(code, ctypes.byref(name)) != 0 or not name.value:
        return None
    return name.value.decode("utf-8", errors="replace")


def _runtime_error_name(runtime: ctypes.CDLL, code: int) -> str | None:
    try:
        cuda_get_error_name = runtime.cudaGetErrorName
    except AttributeError:
        return None
    cuda_get_error_name.argtypes = [ctypes.c_int]
    cuda_get_error_name.restype = ctypes.c_char_p
    name = cuda_get_error_name(code)
    if not name:
        return None
    return name.decode("utf-8", errors="replace")


def _driver_call(
    driver: ctypes.CDLL,
    function_name: str,
    *arguments: Any,
) -> tuple[int, str | None]:
    function = getattr(driver, function_name)
    code = int(function(*arguments))
    return code, _driver_error_name(driver, code) if code else None


def _runtime_call(
    runtime: ctypes.CDLL,
    function_name: str,
    *arguments: Any,
) -> tuple[int, str | None]:
    function = getattr(runtime, function_name)
    code = int(function(*arguments))
    return code, _runtime_error_name(runtime, code) if code else None


def _driver_version(driver: ctypes.CDLL) -> dict[str, Any]:
    version = ctypes.c_int()
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    code, error_name = _driver_call(
        driver,
        "cuDriverGetVersion",
        ctypes.byref(version),
    )
    return {
        "call": "cuDriverGetVersion",
        "result": code,
        "error_name": error_name,
        "version": int(version.value) if code == 0 else None,
        "version_text": _cuda_version_text(int(version.value)) if code == 0 else None,
    }


def _runtime_versions() -> dict[str, Any]:
    runtime, library, load_error = _load_shared_library("cudart", _CUDA_RUNTIME_NAMES)
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
        runtime_code, runtime_error = _runtime_call(
            runtime,
            "cudaRuntimeGetVersion",
            ctypes.byref(runtime_version),
        )
        result["calls"]["cudaRuntimeGetVersion"] = {
            "result": runtime_code,
            "error_name": runtime_error,
        }
        if runtime_code == 0:
            result["runtime_version"] = int(runtime_version.value)
            result["runtime_version_text"] = _cuda_version_text(
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
        driver_code, driver_error = _runtime_call(
            runtime,
            "cudaDriverGetVersion",
            ctypes.byref(driver_version),
        )
        result["calls"]["cudaDriverGetVersion"] = {
            "result": driver_code,
            "error_name": driver_error,
        }
        if driver_code == 0:
            result["driver_version"] = int(driver_version.value)
            result["driver_version_text"] = _cuda_version_text(
                int(driver_version.value)
            )

    return result


def _device_attribute(
    driver: ctypes.CDLL,
    device: ctypes.c_int,
    attribute: int,
) -> tuple[int | None, int, str | None]:
    value = ctypes.c_int()
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetAttribute.restype = ctypes.c_int
    code, error_name = _driver_call(
        driver,
        "cuDeviceGetAttribute",
        ctypes.byref(value),
        attribute,
        device,
    )
    return (int(value.value) if code == 0 else None, code, error_name)


def _device_zero_metadata(driver: ctypes.CDLL) -> dict[str, Any]:
    device = ctypes.c_int()
    driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    driver.cuDeviceGet.restype = ctypes.c_int
    device_code, device_error = _driver_call(
        driver,
        "cuDeviceGet",
        ctypes.byref(device),
        0,
    )
    if device_code != 0:
        return {
            "index": 0,
            "available": False,
            "cuDeviceGet": {
                "result": device_code,
                "error_name": device_error,
            },
        }

    name_buffer = ctypes.create_string_buffer(256)
    driver.cuDeviceGetName.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_int,
        ctypes.c_int,
    ]
    driver.cuDeviceGetName.restype = ctypes.c_int
    name_code, name_error = _driver_call(
        driver,
        "cuDeviceGetName",
        name_buffer,
        len(name_buffer),
        device,
    )

    major, major_code, major_error = _device_attribute(
        driver,
        device,
        _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )
    minor, minor_code, minor_error = _device_attribute(
        driver,
        device,
        _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )

    total_memory = ctypes.c_size_t()
    memory_function_name = None
    try:
        memory_function = driver.cuDeviceTotalMem_v2
        memory_function_name = "cuDeviceTotalMem_v2"
    except AttributeError:
        try:
            memory_function = driver.cuDeviceTotalMem
            memory_function_name = "cuDeviceTotalMem"
        except AttributeError:
            memory_function = None
    memory_call = {"result": None, "error_name": None}
    if memory_function is not None:
        memory_function.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        memory_function.restype = ctypes.c_int
        memory_code = int(memory_function(ctypes.byref(total_memory), device))
        memory_error = _driver_error_name(driver, memory_code) if memory_code else None
        memory_call = {
            "call": memory_function_name,
            "result": memory_code,
            "error_name": memory_error,
        }

    return {
        "index": 0,
        "available": name_code == 0 and major_code == 0 and minor_code == 0,
        "driver_device_handle": int(device.value),
        "name": (
            name_buffer.value.decode("utf-8", errors="replace")
            if name_code == 0
            else None
        ),
        "compute_capability": [major, minor]
        if major is not None and minor is not None
        else None,
        "total_memory_bytes": int(total_memory.value)
        if memory_call["result"] == 0
        else None,
        "calls": {
            "cuDeviceGet": {
                "result": device_code,
                "error_name": device_error,
            },
            "cuDeviceGetName": {
                "result": name_code,
                "error_name": name_error,
            },
            "cuDeviceGetAttribute_major": {
                "result": major_code,
                "error_name": major_error,
            },
            "cuDeviceGetAttribute_minor": {
                "result": minor_code,
                "error_name": minor_error,
            },
            "cuDeviceTotalMem": memory_call,
        },
    }


def probe_cuda_driver_device0() -> dict[str, Any]:
    """Probe CUDA driver metadata for visible device 0 for benchmark reports."""
    driver, library, load_error = _load_shared_library("cuda", _CUDA_DRIVER_NAMES)
    result: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe": "torch_rs_private_cuda_driver_device0",
        "public_torch_cuda_api": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "status": "unavailable",
        "driver_initialized": False,
        "driver": {
            "library": library,
            "loaded": driver is not None,
            "load_error": load_error,
            "cuInit": None,
            "version": None,
            "device_count": None,
        },
        "runtime": _runtime_versions(),
        "device_0": None,
    }
    if driver is None:
        return result

    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    init_code, init_error = _driver_call(driver, "cuInit", 0)
    result["driver"]["cuInit"] = {
        "result": init_code,
        "error_name": init_error,
    }
    if init_code != 0:
        result["status"] = "error"
        return result

    result["driver_initialized"] = True
    result["driver"]["version"] = _driver_version(driver)

    count = ctypes.c_int()
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    count_code, count_error = _driver_call(
        driver,
        "cuDeviceGetCount",
        ctypes.byref(count),
    )
    result["driver"]["device_count"] = {
        "result": count_code,
        "error_name": count_error,
        "value": int(count.value) if count_code == 0 else None,
    }
    if count_code != 0:
        result["status"] = "error"
        return result
    if count.value < 1:
        return result

    result["device_0"] = _device_zero_metadata(driver)
    if result["device_0"].get("available"):
        result["status"] = "ok"
    else:
        result["status"] = "error"
    return result


__all__ = ["PROBE_SCHEMA_VERSION", "probe_cuda_driver_device0"]
