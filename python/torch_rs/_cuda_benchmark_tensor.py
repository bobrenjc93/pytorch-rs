"""Public metadata wrapper for torch_rs-owned CUDA benchmark buffers.

This is intentionally narrower than a CUDA tensor backend. It can only wrap
existing private benchmark buffers that already live on CUDA device 0, and it
exposes synchronized checksum/readback metadata without exposing host bytes,
CUDA pointers, factories, transfers, or arithmetic.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from . import _cuda_buffer
from .torch_rs import float32 as _torch_float32


CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION = "torch_rs_public_cuda_benchmark_tensor_v1"


class CudaBenchmarkTensor:
    """Tensor-shaped public view of one private CUDA benchmark float32 buffer."""

    __slots__ = (
        "_byte_count",
        "_checksum",
        "_checksum_name",
        "_device",
        "_device_index",
        "_device_type",
        "_dtype",
        "_element_count",
        "_metadata",
        "_name",
        "_readback_metadata",
        "_shape",
        "_stride",
    )

    def __init__(
        self,
        buffer: _cuda_buffer.PrivateCudaFloat32Buffer,
        *,
        checksum: Callable[[bytes], str] | None = None,
        readback: _cuda_buffer.PrivateCudaHostReadback | None = None,
        checksum_name: str | None = None,
    ) -> None:
        if type(buffer) is not _cuda_buffer.PrivateCudaFloat32Buffer:
            raise TypeError(
                "CudaBenchmarkTensor can only wrap torch_rs-owned private "
                "CUDA float32 benchmark buffers"
            )
        if readback is not None and checksum is not None:
            raise TypeError("pass either checksum or readback, not both")
        if readback is None and not callable(checksum):
            raise TypeError("checksum must be callable when readback is not provided")
        if checksum_name is not None and type(checksum_name) is not str:
            raise TypeError("checksum_name must be str or None")

        metadata = dict(buffer.metadata())
        self._validate_buffer(buffer, metadata)

        if readback is None:
            readback = buffer.checksum_readback(checksum)
        if type(readback) is not _cuda_buffer.PrivateCudaHostReadback:
            raise TypeError("readback must be PrivateCudaHostReadback")

        readback_metadata = self._validate_readback(buffer, readback)
        readback_metadata["checksum_name"] = checksum_name
        readback_metadata["payload_exposed"] = False

        self._name = buffer.name
        self._shape = tuple(metadata["shape"])
        self._stride = tuple(metadata["stride"])
        self._dtype = _torch_float32
        self._device = metadata["device"]
        self._device_type = metadata["device_type"]
        self._device_index = metadata["device_index"]
        self._element_count = int(buffer.element_count)
        self._byte_count = int(buffer.byte_count)
        self._checksum = readback_metadata["checksum"]
        self._checksum_name = checksum_name
        self._metadata = {
            "schema_version": CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION,
            "status": "ok",
            "reason": "synchronized CUDA benchmark buffer checksum readback",
            "public_torch_cuda_api": False,
            "public_cuda_tensor_wrapper": True,
            "source_buffer_schema_version": _cuda_buffer.BUFFER_SCHEMA_VERSION,
            "source_buffer_name": self._name,
            "shape": list(self._shape),
            "stride": list(self._stride),
            "storage_offset": 0,
            "dtype": str(self._dtype),
            "device": self._device,
            "device_type": self._device_type,
            "device_index": self._device_index,
            "is_cuda": True,
            "requires_grad": False,
            "is_contiguous": True,
            "element_count": self._element_count,
            "byte_count": self._byte_count,
            "checksum": self._checksum,
            "checksum_name": self._checksum_name,
            "readback": copy.deepcopy(readback_metadata),
            "cpu_fallback": False,
            "operations_supported": [],
        }
        self._readback_metadata = copy.deepcopy(readback_metadata)

    @staticmethod
    def _validate_buffer(
        buffer: _cuda_buffer.PrivateCudaFloat32Buffer,
        metadata: dict[str, Any],
    ) -> None:
        if not buffer.allocation_ok:
            raise ValueError("CudaBenchmarkTensor requires a live CUDA allocation")
        if getattr(buffer, "device_index", None) != 0:
            raise ValueError("CudaBenchmarkTensor supports only CUDA device 0")
        if metadata.get("device_type") != "cuda":
            raise ValueError("CudaBenchmarkTensor requires CUDA buffer metadata")
        if metadata.get("device_index") != 0 or metadata.get("device") != "cuda:0":
            raise ValueError("CudaBenchmarkTensor supports only CUDA device 0")
        if metadata.get("dtype") != _cuda_buffer.FLOAT32_DTYPE:
            raise TypeError("CudaBenchmarkTensor supports only torch.float32 buffers")
        if metadata.get("storage_offset") != 0:
            raise ValueError("CudaBenchmarkTensor supports only storage_offset=0")
        if metadata.get("is_contiguous") is not True:
            raise ValueError("CudaBenchmarkTensor supports only contiguous buffers")
        if metadata.get("requires_grad") is not False:
            raise ValueError("CudaBenchmarkTensor does not support gradients")
        if metadata.get("shape") != list(buffer.shape):
            raise ValueError("CudaBenchmarkTensor buffer shape metadata changed")
        if metadata.get("stride") != list(buffer.stride):
            raise ValueError("CudaBenchmarkTensor buffer stride metadata changed")

    @staticmethod
    def _validate_readback(
        buffer: _cuda_buffer.PrivateCudaFloat32Buffer,
        readback: _cuda_buffer.PrivateCudaHostReadback,
    ) -> dict[str, Any]:
        copy_call = dict(readback.copy_call)
        sync_call = None if readback.sync_call is None else dict(readback.sync_call)
        if copy_call.get("result") != 0:
            raise ValueError("CudaBenchmarkTensor readback copy failed")
        if copy_call.get("kind") != "cudaMemcpyDeviceToHost":
            raise ValueError("CudaBenchmarkTensor requires device-to-host readback")
        if copy_call.get("device_index") != 0:
            raise ValueError("CudaBenchmarkTensor readback must use CUDA device 0")
        if copy_call.get("byte_count") != buffer.byte_count:
            raise ValueError("CudaBenchmarkTensor readback byte count changed")
        if copy_call.get("buffer_name") != buffer.name:
            raise ValueError("CudaBenchmarkTensor readback buffer name changed")
        if sync_call is None or sync_call.get("result") != 0:
            raise ValueError("CudaBenchmarkTensor readback was not synchronized")
        if sync_call.get("device_index") != 0:
            raise ValueError("CudaBenchmarkTensor sync must use CUDA device 0")
        if sync_call.get("buffer_name") != buffer.name:
            raise ValueError("CudaBenchmarkTensor sync buffer name changed")
        if (
            type(readback.payload) is not bytes
            or len(readback.payload) != buffer.byte_count
        ):
            raise ValueError("CudaBenchmarkTensor readback payload size changed")
        if type(readback.checksum) is not str or not readback.checksum:
            raise ValueError("CudaBenchmarkTensor requires a readback checksum")

        return {
            "copy_call": copy_call,
            "sync_call": sync_call,
            "synchronized": True,
            "checksum": readback.checksum,
            "byte_count": buffer.byte_count,
        }

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def stride(self) -> tuple[int, ...]:
        return self._stride

    @property
    def dtype(self) -> object:
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    @property
    def device_type(self) -> str:
        return self._device_type

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def is_cuda(self) -> bool:
        return True

    @property
    def is_contiguous(self) -> bool:
        return True

    @property
    def requires_grad(self) -> bool:
        return False

    @property
    def checksum(self) -> str:
        return self._checksum

    @property
    def checksum_name(self) -> str | None:
        return self._checksum_name

    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    def readback_metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._readback_metadata)

    def __repr__(self) -> str:
        return (
            "CudaBenchmarkTensor("
            f"shape={self._shape!r}, "
            f"stride={self._stride!r}, "
            f"dtype={self._metadata['dtype']}, "
            f"device={self._device!r}, "
            f"checksum={self._checksum!r})"
        )


CudaBenchmarkTensor.__module__ = "torch_rs"


__all__ = [
    "CUDA_BENCHMARK_TENSOR_SCHEMA_VERSION",
    "CudaBenchmarkTensor",
]
