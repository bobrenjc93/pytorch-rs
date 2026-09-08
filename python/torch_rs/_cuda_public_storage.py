"""CUDA runtime helpers for torch_rs' narrow public tensor storage path."""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import pathlib
import threading

_CUDA_SUCCESS = 0
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_FLOAT32_BYTES = 4

_RUNTIME_LOCK = threading.Lock()
_RUNTIME: ctypes.CDLL | None = None
_RUNTIME_LOAD_ERROR: OSError | None = None
_RUNTIME_CONFIGURED = False
_INITIALIZED = False


def _candidate_libraries() -> list[str]:
    candidates: list[str] = []
    found = ctypes.util.find_library("cudart")
    if found is not None:
        candidates.append(found)
    candidates.extend(("libcudart.so.13", "libcudart.so.12", "libcudart.so"))

    try:
        spec = importlib.util.find_spec("nvidia.cuda_runtime")
    except Exception:
        spec = None
    if spec is not None and spec.submodule_search_locations is not None:
        for location in spec.submodule_search_locations:
            library_dir = pathlib.Path(location) / "lib"
            candidates.extend(str(path) for path in library_dir.glob("libcudart.so*"))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def _load_runtime() -> ctypes.CDLL:
    global _RUNTIME, _RUNTIME_LOAD_ERROR
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        last_error: OSError | None = None
        for library in _candidate_libraries():
            try:
                _RUNTIME = ctypes.CDLL(library)
                return _RUNTIME
            except OSError as error:
                last_error = error
        _RUNTIME_LOAD_ERROR = last_error
    if last_error is None:
        raise OSError("CUDA runtime library could not be found")
    raise OSError(f"CUDA runtime library could not be loaded: {last_error}")


def _runtime_or_none() -> ctypes.CDLL | None:
    try:
        return _load_runtime()
    except OSError:
        return None


def _configure_runtime(runtime: ctypes.CDLL) -> None:
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return
    with _RUNTIME_LOCK:
        if _RUNTIME_CONFIGURED:
            return
        runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        runtime.cudaGetDeviceCount.restype = ctypes.c_int
        runtime.cudaSetDevice.argtypes = [ctypes.c_int]
        runtime.cudaSetDevice.restype = ctypes.c_int
        runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        runtime.cudaMalloc.restype = ctypes.c_int
        runtime.cudaFree.argtypes = [ctypes.c_void_p]
        runtime.cudaFree.restype = ctypes.c_int
        runtime.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        runtime.cudaMemset.restype = ctypes.c_int
        runtime.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        runtime.cudaMemcpy.restype = ctypes.c_int
        runtime.cudaDeviceSynchronize.argtypes = []
        runtime.cudaDeviceSynchronize.restype = ctypes.c_int
        runtime.cudaGetErrorName.argtypes = [ctypes.c_int]
        runtime.cudaGetErrorName.restype = ctypes.c_char_p
        _RUNTIME_CONFIGURED = True


def _configured_runtime() -> ctypes.CDLL:
    runtime = _load_runtime()
    try:
        _configure_runtime(runtime)
    except AttributeError as error:
        raise RuntimeError(f"CUDA runtime is missing a required symbol: {error}") from error
    return runtime


def _runtime_error_name(runtime: ctypes.CDLL, code: int) -> str:
    try:
        name = runtime.cudaGetErrorName(code)
    except Exception:
        return f"cudaError{code}"
    if name is None:
        return f"cudaError{code}"
    return name.decode("ascii", "replace")


def _check_call(runtime: ctypes.CDLL, function: str, *arguments: object) -> None:
    code = int(getattr(runtime, function)(*arguments))
    if code != _CUDA_SUCCESS:
        name = _runtime_error_name(runtime, code)
        raise RuntimeError(f"{function} failed with {name} ({code})")


def _device_count(runtime: ctypes.CDLL) -> int:
    count = ctypes.c_int()
    code = int(runtime.cudaGetDeviceCount(ctypes.byref(count)))
    if code != _CUDA_SUCCESS:
        return 0
    return max(0, int(count.value))


def is_available() -> bool:
    return device_count() > 0


def device_count() -> int:
    runtime = _runtime_or_none()
    if runtime is None:
        return 0
    try:
        _configure_runtime(runtime)
    except AttributeError:
        return 0
    return _device_count(runtime)


def is_initialized() -> bool:
    return _INITIALIZED


class CudaFloat32Storage:
    """Owning CUDA float32 buffer used by torch_rs.Tensor zeros only."""

    __slots__ = ("_runtime", "_pointer", "_elements", "_device_index", "_closed")

    @classmethod
    def zeros(cls, element_count: int, device_index: int) -> "CudaFloat32Storage":
        return cls(element_count, device_index)

    def __init__(self, element_count: int, device_index: int) -> None:
        global _INITIALIZED
        if type(element_count) is not int or element_count < 0:
            raise ValueError("CUDA storage element_count must be a non-negative int")
        if type(device_index) is not int or device_index < 0:
            raise ValueError("CUDA storage device_index must be a non-negative int")

        runtime = _configured_runtime()
        count = _device_count(runtime)
        if device_index >= count:
            raise RuntimeError(
                f"CUDA device {device_index} is unavailable; device_count() is {count}"
            )

        _check_call(runtime, "cudaSetDevice", device_index)
        self._runtime = runtime
        self._pointer = ctypes.c_void_p()
        self._elements = element_count
        self._device_index = device_index
        self._closed = False

        byte_count = element_count * _FLOAT32_BYTES
        if byte_count:
            _check_call(runtime, "cudaMalloc", ctypes.byref(self._pointer), byte_count)
            try:
                _check_call(runtime, "cudaMemset", self._pointer, 0, byte_count)
                _check_call(runtime, "cudaDeviceSynchronize")
            except Exception:
                self.close()
                raise
        else:
            _check_call(runtime, "cudaDeviceSynchronize")
        _INITIALIZED = True

    @property
    def data_ptr(self) -> int:
        return int(self._pointer.value or 0)

    @property
    def element_count(self) -> int:
        return self._elements

    @property
    def device_index(self) -> int:
        return self._device_index

    def copy_to_host(self) -> bytes:
        if self._closed:
            raise RuntimeError("CUDA storage has been freed")
        _check_call(self._runtime, "cudaSetDevice", self._device_index)
        byte_count = self._elements * _FLOAT32_BYTES
        if byte_count == 0:
            _check_call(self._runtime, "cudaDeviceSynchronize")
            return b""
        output = ctypes.create_string_buffer(byte_count)
        _check_call(
            self._runtime,
            "cudaMemcpy",
            output,
            self._pointer,
            byte_count,
            _CUDA_MEMCPY_DEVICE_TO_HOST,
        )
        _check_call(self._runtime, "cudaDeviceSynchronize")
        return output.raw

    def close(self) -> None:
        if self._closed:
            return
        pointer = self._pointer
        self._pointer = ctypes.c_void_p()
        self._closed = True
        if pointer.value:
            self._runtime.cudaFree(pointer)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
