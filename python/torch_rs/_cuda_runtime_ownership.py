"""Private CUDA runtime ownership primitives.

The helpers in this module deliberately do not encode benchmark names,
workload shapes, checksum domains, or public ``torch.cuda`` behavior.  They
only track process-local ownership for torch_rs-created CUDA runtime buffers.

Invariants:
* a buffer can be handed out by at most one live lease at a time;
* released leases are idempotent, but stale active leases fail closed;
* closing the pool frees only idle buffers and defers live buffers until their
  leases are released;
* every tracked buffer is passed to ``close()`` at most once by the pool;
* current-device helpers always verify the logical device observed by cudart.
"""

from __future__ import annotations

import ctypes
import hashlib
from typing import Any, Callable, Sequence

from . import _cuda_buffer


CUDA_BUFFER_POOL_SCHEMA_VERSION = "torch_rs_private_cuda_buffer_pool_v1"


def _validate_device_index(device_index: int) -> None:
    if type(device_index) is not int:
        raise TypeError("device_index must be int")
    if device_index < 0:
        raise ValueError("device_index must be non-negative")


def _current_device_call(runtime: ctypes.CDLL) -> tuple[int | None, dict[str, Any]]:
    current_device = ctypes.c_int(-1)
    call = _cuda_buffer.runtime_call(
        runtime,
        "cudaGetDevice",
        ctypes.byref(current_device),
    )
    call["value"] = int(current_device.value) if call["result"] == 0 else None
    return call["value"], call


def verify_current_device(
    runtime: ctypes.CDLL,
    device_index: int,
    *,
    context: str,
) -> dict[str, Any]:
    """Verify that cudart's current logical device matches ``device_index``."""
    if type(context) is not str:
        raise TypeError("context must be str")
    _validate_device_index(device_index)
    current_device, call = _current_device_call(runtime)
    if call["result"] != 0:
        raise RuntimeError(f"cudaGetDevice failed for {context}")
    if current_device != device_index:
        raise RuntimeError(
            f"{context} CUDA device changed: expected {device_index}, "
            f"got {current_device}"
        )
    return call


def set_and_verify_current_device(
    runtime: ctypes.CDLL,
    device_index: int,
    *,
    context: str,
) -> dict[str, Any]:
    """Set cudart's current logical device and verify the selected device."""
    if type(context) is not str:
        raise TypeError("context must be str")
    _validate_device_index(device_index)
    set_device_call = _cuda_buffer.runtime_call(
        runtime,
        "cudaSetDevice",
        device_index,
    )
    if set_device_call["result"] != 0:
        raise RuntimeError(f"cudaSetDevice({device_index}) failed for {context}")
    get_device_call = verify_current_device(
        runtime,
        device_index,
        context=context,
    )
    return {
        "cudaSetDevice": set_device_call,
        "cudaGetDevice": get_device_call,
    }


def synchronize_current_device(
    runtime: ctypes.CDLL,
    device_index: int,
    *,
    label: str,
) -> dict[str, Any]:
    """Synchronize and tag the call with the expected logical CUDA device."""
    if type(label) is not str:
        raise TypeError("label must be str")
    verify_current_device(runtime, device_index, context=label)
    call = _cuda_buffer.runtime_call(runtime, "cudaDeviceSynchronize")
    call.update({"buffer_name": label, "device_index": device_index})
    if call["result"] != 0:
        raise RuntimeError(f"cudaDeviceSynchronize failed for {label}")
    return call


class PrivateCudaBufferLease:
    """One live borrow from a ``PrivateCudaBufferPool``."""

    __slots__ = ("_pool", "_released", "acquisition", "buffer", "lease_id")

    def __init__(
        self,
        pool: "PrivateCudaBufferPool",
        buffer: _cuda_buffer.PrivateCudaFloat32Buffer,
        lease_id: str,
        acquisition: dict[str, Any],
    ) -> None:
        self._pool = pool
        self.buffer = buffer
        self.lease_id = lease_id
        self.acquisition = acquisition
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> dict[str, Any] | None:
        return self._pool.release(self)


class PrivateCudaBufferPool:
    """Reusable private CUDA buffer pool with deferred live-buffer close."""

    __slots__ = (
        "_allocation_count",
        "_available",
        "_buffer_factory",
        "_buffers",
        "_closed",
        "_device_index",
        "_free_count",
        "_initial_capacity",
        "_lease_counts",
        "_lease_sequence",
        "_live_leases",
        "_max_live",
        "_name_prefix",
        "_next_index",
        "_owner_id",
        "_release_count",
        "_reuse_count",
        "_runtime",
        "_shape",
    )

    def __init__(
        self,
        runtime: ctypes.CDLL,
        shape: int | Sequence[int],
        *,
        name_prefix: str,
        device_index: int = 0,
        owner_id: str | None = None,
        buffer_factory: Callable[..., _cuda_buffer.PrivateCudaFloat32Buffer]
        | None = None,
    ) -> None:
        if type(name_prefix) is not str:
            raise TypeError("name_prefix must be str")
        if not name_prefix:
            raise ValueError("name_prefix must not be empty")
        if owner_id is not None and type(owner_id) is not str:
            raise TypeError("owner_id must be str or None")
        _validate_device_index(device_index)

        self._runtime = runtime
        self._shape = tuple(_cuda_buffer.float32_metadata(shape)["shape"])
        self._name_prefix = name_prefix
        self._device_index = device_index
        key_payload = f"{id(runtime)}:{name_prefix}:{self._shape}:{device_index}"
        self._owner_id = owner_id or hashlib.blake2b(
            key_payload.encode("utf-8"),
            digest_size=8,
        ).hexdigest()
        self._buffer_factory = buffer_factory or _cuda_buffer.PrivateCudaFloat32Buffer
        self._buffers: list[_cuda_buffer.PrivateCudaFloat32Buffer] = []
        self._available: list[_cuda_buffer.PrivateCudaFloat32Buffer] = []
        self._live_leases: dict[int, PrivateCudaBufferLease] = {}
        self._lease_counts: dict[int, int] = {}
        self._initial_capacity = 0
        self._next_index = 0
        self._allocation_count = 0
        self._free_count = 0
        self._lease_sequence = 0
        self._max_live = 0
        self._release_count = 0
        self._reuse_count = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def preallocate(self, capacity: int) -> list[dict[str, Any]]:
        if type(capacity) is not int:
            raise TypeError("capacity must be int")
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if self._buffers:
            raise RuntimeError("CUDA buffer pool preallocation already started")
        self._require_open()

        allocations = []
        try:
            for _ in range(capacity):
                buffer, allocation = self._allocate_buffer()
                self._available.append(buffer)
                allocations.append(allocation)
        except Exception:
            self.close()
            raise
        self._initial_capacity = capacity
        return allocations

    def _allocate_buffer(
        self,
    ) -> tuple[_cuda_buffer.PrivateCudaFloat32Buffer, dict[str, Any]]:
        index = self._next_index
        self._next_index += 1
        buffer = self._buffer_factory(
            self._runtime,
            self._shape,
            name=f"{self._name_prefix}_{index}",
            device_index=self._device_index,
        )
        self._buffers.append(buffer)
        self._lease_counts[id(buffer)] = 0
        self._allocation_count += 1
        allocation = {
            "buffer_name": buffer.name,
            "byte_count": buffer.byte_count,
            "device_index": buffer.device_index,
            "malloc_call": dict(buffer.malloc_call),
        }
        if not buffer.allocation_ok:
            free_call = buffer.close()
            if free_call is not None:
                self._free_count += 1
                allocation["free_call_after_failed_allocation"] = free_call
            raise RuntimeError("cudaMalloc failed for CUDA buffer pool")
        return buffer, allocation

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("CUDA buffer pool is closed")

    def acquire(self) -> PrivateCudaBufferLease:
        self._require_open()
        allocated_in_acquire = False
        if self._available:
            buffer = self._available.pop(0)
        else:
            buffer, _allocation = self._allocate_buffer()
            allocated_in_acquire = True

        previous_lease_count = self._lease_counts.get(id(buffer), 0)
        self._lease_counts[id(buffer)] = previous_lease_count + 1
        self._lease_sequence += 1
        lease_id = f"{self._owner_id}:{self._lease_sequence}"
        reused_released_allocation = previous_lease_count > 0
        if reused_released_allocation:
            self._reuse_count += 1

        source = "preallocated"
        if allocated_in_acquire:
            source = "allocated_during_acquire"
        elif reused_released_allocation:
            source = "reused_released_allocation"

        acquisition = {
            "schema_version": CUDA_BUFFER_POOL_SCHEMA_VERSION,
            "enabled": True,
            "lease_id": lease_id,
            "buffer_name": buffer.name,
            "source": source,
            "allocated_in_acquire": allocated_in_acquire,
            "reused_released_allocation": reused_released_allocation,
            "pool_initial_capacity": self._initial_capacity,
            "pool_allocation_count": self._allocation_count,
            "pool_reuse_count": self._reuse_count,
            "pool_release_count": self._release_count,
            "pool_live_buffers_after_acquire": len(self._live_leases) + 1,
            "pool_available_buffers_after_acquire": len(self._available),
        }
        lease = PrivateCudaBufferLease(self, buffer, lease_id, acquisition)
        self._live_leases[id(buffer)] = lease
        self._max_live = max(self._max_live, len(self._live_leases))
        return lease

    def release(self, lease: PrivateCudaBufferLease) -> dict[str, Any] | None:
        if type(lease) is not PrivateCudaBufferLease:
            raise TypeError("lease must be PrivateCudaBufferLease")
        if lease._pool is not self:
            raise ValueError("CUDA buffer lease belongs to a different pool")
        if lease._released:
            return None

        buffer = lease.buffer
        active_lease = self._live_leases.get(id(buffer))
        if active_lease is not lease:
            raise ValueError("CUDA buffer lease is stale")
        if not buffer.allocation_ok:
            del self._live_leases[id(buffer)]
            lease._released = True
            if self._closed:
                return None
            raise ValueError("CUDA buffer lease target is no longer live")

        del self._live_leases[id(buffer)]
        lease._released = True
        self._release_count += 1
        release = {
            "schema_version": CUDA_BUFFER_POOL_SCHEMA_VERSION,
            "lease_id": lease.lease_id,
            "buffer_name": buffer.name,
            "pool_live_buffers_after_release": len(self._live_leases),
            "pool_release_count": self._release_count,
        }
        if self._closed:
            free_call = buffer.close()
            if free_call is not None:
                self._free_count += 1
            release.update(
                {
                    "released_to_pool": False,
                    "freed_after_pool_close": free_call is not None,
                    "pool_available_buffers_after_release": len(self._available),
                    "pool_free_count": self._free_count,
                    "calls": {}
                    if free_call is None
                    else {f"cudaFree_{buffer.name}": free_call},
                }
            )
            return release

        self._available.insert(0, buffer)
        release.update(
            {
                "released_to_pool": True,
                "freed_after_pool_close": False,
                "pool_available_buffers_after_release": len(self._available),
                "pool_free_count": self._free_count,
            }
        )
        return release

    def close(self) -> dict[str, Any]:
        if self._closed:
            return {
                "schema_version": CUDA_BUFFER_POOL_SCHEMA_VERSION,
                "closed": True,
                "already_closed": True,
                "free_count": self._free_count,
                "deferred_live_buffers": len(self._live_leases),
                "calls": {},
            }

        self._closed = True
        calls: dict[str, Any] = {}
        live_buffer_ids = set(self._live_leases)
        for buffer in self._buffers:
            if id(buffer) in live_buffer_ids:
                continue
            free_call = buffer.close()
            if free_call is not None:
                self._free_count += 1
                calls[f"cudaFree_{buffer.name}"] = free_call
        self._available.clear()
        return {
            "schema_version": CUDA_BUFFER_POOL_SCHEMA_VERSION,
            "closed": True,
            "already_closed": False,
            "free_count": self._free_count,
            "deferred_live_buffers": len(self._live_leases),
            "calls": calls,
        }

    def metadata(self) -> dict[str, Any]:
        live_ids = set(self._live_leases)
        available_ids = {id(buffer) for buffer in self._available}
        return {
            "schema_version": CUDA_BUFFER_POOL_SCHEMA_VERSION,
            "enabled": True,
            "closed": self._closed,
            "initial_capacity": self._initial_capacity,
            "total_buffers": len(self._buffers),
            "available_buffers": len(self._available),
            "live_buffers": len(self._live_leases),
            "allocation_count": self._allocation_count,
            "reuse_count": self._reuse_count,
            "release_count": self._release_count,
            "free_count": self._free_count,
            "max_live_buffers": self._max_live,
            "buffers": [
                {
                    "name": buffer.name,
                    "shape": list(buffer.shape),
                    "byte_count": buffer.byte_count,
                    "pointer_nonzero": buffer.pointer_nonzero,
                    "available": id(buffer) in available_ids,
                    "live": id(buffer) in live_ids,
                    "lease_count": self._lease_counts.get(id(buffer), 0),
                }
                for buffer in self._buffers
            ],
        }

    def __enter__(self) -> "PrivateCudaBufferPool":
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
    "CUDA_BUFFER_POOL_SCHEMA_VERSION",
    "PrivateCudaBufferLease",
    "PrivateCudaBufferPool",
    "set_and_verify_current_device",
    "synchronize_current_device",
    "verify_current_device",
]
