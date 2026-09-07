"""Private benchmark-only CUDA float32 buffer ownership helpers.

This module deliberately stays outside ``torch_rs.cuda``. It gives private
benchmark code a small CUDA buffer boundary without advertising public CUDA
tensor support.
"""

from __future__ import annotations

import ctypes
import math
import struct
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from . import _cuda_driver_probe


BUFFER_SCHEMA_VERSION = "torch_rs_private_cuda_float32_buffer_v1"
DETERMINISTIC_FLOAT32_VECTOR_VERSION = "float32_modulated_arange_v1"
FLOAT32_DTYPE = "torch.float32"
FLOAT32_ITEMSIZE = 4

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_UNAVAILABLE_ERRORS = {
    "cudaErrorInsufficientDriver",
    "cudaErrorNoDevice",
}
CUDA_RUNTIME_SYMBOLS = (
    "cudaGetDeviceCount",
    "cudaSetDevice",
    "cudaGetDevice",
    "cudaMalloc",
    "cudaMemcpy",
    "cudaDeviceSynchronize",
    "cudaFree",
)


@dataclass(frozen=True)
class PrivateCudaHostReadback:
    payload: bytes | None
    copy_call: dict[str, Any]
    sync_call: dict[str, Any] | None
    checksum: str | None


def deterministic_float32_values(element_count: int) -> list[float]:
    if type(element_count) is not int:
        raise TypeError("element_count must be int")
    if element_count <= 0:
        raise ValueError("element_count must be positive")
    return [
        ((index % 251) - 125) * 0.03125
        + ((index * 17) % 19) * 0.0009765625
        for index in range(element_count)
    ]


def deterministic_float32_bytes(element_count: int) -> bytes:
    values = deterministic_float32_values(element_count)
    return struct.pack(f"<{element_count}f", *values)


def runtime_call(
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


def runtime_versions(
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
        runtime_call_result = runtime_call(
            runtime,
            "cudaRuntimeGetVersion",
            ctypes.byref(runtime_version),
        )
        result["calls"]["cudaRuntimeGetVersion"] = runtime_call_result
        if runtime_call_result["result"] == 0:
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
        driver_call = runtime_call(
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


def configure_runtime_symbols(runtime: ctypes.CDLL) -> list[str]:
    missing: list[str] = []
    for name in CUDA_RUNTIME_SYMBOLS:
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


def _shape_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    if type(shape) is int:
        dimensions = (shape,)
    elif type(shape) in {list, tuple}:
        dimensions = tuple(shape)
    else:
        raise TypeError("shape must be an int, list, or tuple")

    if not dimensions:
        raise ValueError("shape must have at least one dimension")
    for dimension in dimensions:
        if type(dimension) is not int:
            raise TypeError("shape dimensions must be int")
        if dimension <= 0:
            raise ValueError("shape dimensions must be positive")
    return dimensions


def contiguous_stride(shape: int | Sequence[int]) -> tuple[int, ...]:
    dimensions = _shape_tuple(shape)
    stride: list[int] = []
    running = 1
    for dimension in reversed(dimensions):
        stride.append(running)
        running *= dimension
    return tuple(reversed(stride))


def element_count(shape: int | Sequence[int]) -> int:
    return math.prod(_shape_tuple(shape))


def float32_metadata(
    shape: int | Sequence[int],
    *,
    device_index: int = 0,
) -> dict[str, Any]:
    if type(device_index) is not int:
        raise TypeError("device_index must be int")
    if device_index != 0:
        raise ValueError("only CUDA device 0 is supported")

    dimensions = _shape_tuple(shape)
    return {
        "shape": list(dimensions),
        "stride": list(contiguous_stride(dimensions)),
        "storage_offset": 0,
        "dtype": FLOAT32_DTYPE,
        "device": f"cuda:{device_index}",
        "device_type": "cuda",
        "device_index": device_index,
        "requires_grad": False,
        "is_contiguous": True,
    }


class PrivateCudaFloat32Buffer:
    """Owner for one private benchmark-only contiguous CUDA float32 buffer."""

    def __init__(
        self,
        runtime: ctypes.CDLL,
        shape: int | Sequence[int],
        *,
        name: str,
        device_index: int = 0,
    ) -> None:
        if type(name) is not str:
            raise TypeError("name must be str")
        if type(device_index) is not int:
            raise TypeError("device_index must be int")
        if device_index != 0:
            raise ValueError("only CUDA device 0 is supported")

        self.runtime = runtime
        self.name = name
        self.shape = _shape_tuple(shape)
        self.stride = contiguous_stride(self.shape)
        self.element_count = math.prod(self.shape)
        self.byte_count = self.element_count * FLOAT32_ITEMSIZE
        self.device_index = device_index
        self._pointer = ctypes.c_void_p()
        self._closed = False
        self.malloc_call = runtime_call(
            self.runtime,
            "cudaMalloc",
            ctypes.byref(self._pointer),
            self.byte_count,
        )
        self.malloc_call.update(
            {
                "byte_count": self.byte_count,
                "buffer_name": self.name,
                "device_index": self.device_index,
            }
        )

    @property
    def pointer(self) -> ctypes.c_void_p:
        return self._pointer

    @property
    def pointer_nonzero(self) -> bool:
        return bool(self._pointer.value)

    @property
    def allocation_ok(self) -> bool:
        return self.malloc_call["result"] == 0 and self.pointer_nonzero

    def metadata(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": 0,
            "dtype": FLOAT32_DTYPE,
            "device": f"cuda:{self.device_index}",
            "device_type": "cuda",
            "device_index": self.device_index,
            "requires_grad": False,
            "is_contiguous": True,
        }

    def copy_from_host(self, host_bytes: bytes) -> dict[str, Any]:
        if type(host_bytes) is not bytes:
            raise TypeError("host_bytes must be bytes")
        if len(host_bytes) != self.byte_count:
            raise ValueError("host_bytes length does not match buffer byte count")
        host_input = ctypes.create_string_buffer(host_bytes, self.byte_count)
        call = runtime_call(
            self.runtime,
            "cudaMemcpy",
            self._pointer,
            ctypes.c_void_p(ctypes.addressof(host_input)),
            self.byte_count,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        )
        call.update(
            {
                "kind": "cudaMemcpyHostToDevice",
                "byte_count": self.byte_count,
                "buffer_name": self.name,
                "device_index": self.device_index,
            }
        )
        return call

    def copy_to_host(self) -> tuple[bytes | None, dict[str, Any]]:
        host_output = ctypes.create_string_buffer(self.byte_count)
        call = runtime_call(
            self.runtime,
            "cudaMemcpy",
            ctypes.c_void_p(ctypes.addressof(host_output)),
            self._pointer,
            self.byte_count,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        call.update(
            {
                "kind": "cudaMemcpyDeviceToHost",
                "byte_count": self.byte_count,
                "buffer_name": self.name,
                "device_index": self.device_index,
            }
        )
        if call["result"] != 0:
            return None, call
        return ctypes.string_at(ctypes.addressof(host_output), self.byte_count), call

    def synchronize(self) -> dict[str, Any]:
        call = runtime_call(self.runtime, "cudaDeviceSynchronize")
        call.update({"buffer_name": self.name, "device_index": self.device_index})
        return call

    def checksum_readback(
        self,
        checksum: Callable[[bytes], str],
    ) -> PrivateCudaHostReadback:
        payload, copy_call = self.copy_to_host()
        if copy_call["result"] != 0 or payload is None:
            return PrivateCudaHostReadback(
                payload=None,
                copy_call=copy_call,
                sync_call=None,
                checksum=None,
            )

        sync_call = self.synchronize()
        if sync_call["result"] != 0:
            return PrivateCudaHostReadback(
                payload=payload,
                copy_call=copy_call,
                sync_call=sync_call,
                checksum=None,
            )

        return PrivateCudaHostReadback(
            payload=payload,
            copy_call=copy_call,
            sync_call=sync_call,
            checksum=checksum(payload),
        )

    def close(self) -> dict[str, Any] | None:
        if self._closed or not self._pointer.value:
            self._closed = True
            return None
        call = runtime_call(self.runtime, "cudaFree", self._pointer)
        call.update(
            {
                "byte_count": self.byte_count,
                "buffer_name": self.name,
                "device_index": self.device_index,
            }
        )
        self._pointer = ctypes.c_void_p()
        self._closed = True
        return call

    def __enter__(self) -> "PrivateCudaFloat32Buffer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BUFFER_SCHEMA_VERSION",
    "CUDA_UNAVAILABLE_ERRORS",
    "DETERMINISTIC_FLOAT32_VECTOR_VERSION",
    "FLOAT32_DTYPE",
    "FLOAT32_ITEMSIZE",
    "PrivateCudaFloat32Buffer",
    "PrivateCudaHostReadback",
    "configure_runtime_symbols",
    "contiguous_stride",
    "deterministic_float32_bytes",
    "deterministic_float32_values",
    "element_count",
    "float32_metadata",
    "runtime_call",
    "runtime_versions",
]
