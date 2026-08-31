# mypy: allow-untyped-defs
from contextlib import contextmanager as _contextmanager
from functools import wraps as _wraps
from typing import Any as _Any
import warnings as _warnings

import torch_rs as torch


__all__ = [
    "is_built",
    "cuBLASModule",
    "is_ck_sdpa_available",
    "matmul",
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "enable_mem_efficient_sdp",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "enable_math_sdp",
    "allow_fp16_bf16_reduction_math_sdp",
    "fp16_bf16_reduction_math_sdp_allowed",
    "is_flash_attention_available",
    "sdp_kernel",
]


def is_built():
    r"""
    Return whether PyTorch is built with CUDA support.

    Note that this doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run on a machine with working CUDA drivers and devices, we would be able to use it.
    """
    return torch._C._has_cuda


class cuBLASModule:
    @staticmethod
    def _parse_reduction_setting(value: _Any, attr_name: str) -> tuple[bool, bool]:
        def _ensure_bool(obj: _Any, which: str) -> bool:
            if isinstance(obj, bool):
                return obj
            raise TypeError(
                f"{attr_name} expects a bool for {which}, but got {type(obj)!r}"
            )

        if isinstance(value, bool):
            return value, True
        if isinstance(value, (list, tuple)):
            if not value:
                raise TypeError(f"{attr_name} expects at least one boolean argument")
            if len(value) > 2:
                raise TypeError(f"{attr_name} expects at most two boolean arguments")
            allow_reduced_precision = _ensure_bool(value[0], "allow_reduced_precision")
            if len(value) == 1:
                return allow_reduced_precision, True
            allow_splitk = _ensure_bool(value[1], "allow_splitk")
            return allow_reduced_precision, allow_splitk
        raise TypeError(
            f"{attr_name} expects a bool or a tuple/list of bools, but got {type(value)!r}"
        )

    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch._C._get_cublas_allow_tf32()
        if name == "allow_fp16_reduced_precision_reduction":
            allow_reduced_precision, _ = (
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
            )
            return allow_reduced_precision
        if name == "allow_bf16_reduced_precision_reduction":
            allow_reduced_precision, _ = (
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
            )
            return allow_reduced_precision
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch._C._set_cublas_allow_tf32(value)
        if name == "allow_fp16_reduced_precision_reduction":
            allow_reduced_precision, allow_splitk = self._parse_reduction_setting(
                value, "allow_fp16_reduced_precision_reduction"
            )
            return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(
                allow_reduced_precision,
                allow_splitk,
            )
        if name == "allow_bf16_reduced_precision_reduction":
            allow_reduced_precision, allow_splitk = self._parse_reduction_setting(
                value, "allow_bf16_reduced_precision_reduction"
            )
            return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(
                allow_reduced_precision,
                allow_splitk,
            )
        raise AttributeError("Unknown attribute " + name)


matmul = cuBLASModule()


_SDP_KERNEL_DEPRECATION = (
    "`torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this "
    "context manager will be removed. Please see "
    "`torch.nn.attention.sdpa_kernel()` for the new context manager, with "
    "updated signature."
)


def _deprecated_sdp_kernel(arg):
    msg = _SDP_KERNEL_DEPRECATION
    category = FutureWarning
    stacklevel = 2

    @_wraps(arg)
    def wrapped(*args, **kwargs):
        _warnings.warn(msg, category=category, stacklevel=stacklevel)
        return arg(*args, **kwargs)

    wrapped.__deprecated__ = msg
    return wrapped


def is_ck_sdpa_available() -> bool:
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether composable_kernel may be used as the backend for
    scaled-dot-product-attention.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._is_ck_sdpa_available()


def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
    torch._C._set_sdp_use_flash(enabled)


def flash_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """
    return torch._C._get_flash_sdp_enabled()


def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    torch._C._set_sdp_use_mem_efficient(enabled)


def mem_efficient_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    return torch._C._get_mem_efficient_sdp_enabled()


def math_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
    return torch._C._get_math_sdp_enabled()


def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    torch._C._set_sdp_use_math(enabled)


def allow_fp16_bf16_reduction_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables fp16/bf16 reduction in math scaled dot product attention.
    """
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(enabled)


def fp16_bf16_reduction_math_sdp_allowed():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether fp16/bf16 reduction in math scaled dot product attention is enabled or not.
    """
    return torch._C._get_math_sdp_allow_fp16_bf16_reduction()


@_contextmanager
@_deprecated_sdp_kernel
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    requested_flash = bool(enable_flash)
    requested_mem_efficient = bool(enable_mem_efficient)
    requested_math = bool(enable_math)
    bool(enable_cudnn)

    previous_flash = flash_sdp_enabled()
    previous_math = math_sdp_enabled()
    previous_mem_efficient = mem_efficient_sdp_enabled()
    try:
        enable_flash_sdp(requested_flash)
        enable_mem_efficient_sdp(requested_mem_efficient)
        enable_math_sdp(requested_math)
        yield {}
    finally:
        enable_flash_sdp(previous_flash)
        enable_mem_efficient_sdp(previous_mem_efficient)
        enable_math_sdp(previous_math)


def is_flash_attention_available() -> bool:
    r"""Check if PyTorch was built with FlashAttention for scaled_dot_product_attention.

    Returns:
        True if FlashAttention is built and available; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._is_flash_attention_available()
