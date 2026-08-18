"""Internal formatting helpers for PyTorch-compatible diagnostics."""

import math
import sys as _sys

from .torch_rs import _nn_functional_dropout_tensor_autograd_suffix


def _format_single_element_tensor(tensor, value):
    finite = math.isfinite(value)
    if finite and value != 0.0:
        integer_mode = value == math.ceil(value)
        scientific = abs(value) > 1.0e8 or abs(value) < 1.0e-4
    else:
        integer_mode = True
        scientific = False

    if scientific:
        formatted = f"{value:.4e}"
    elif integer_mode:
        formatted = f"{value:.0f}"
        if finite:
            formatted += "."
    else:
        formatted = f"{value:.4f}"

    try:
        formatted = _format_recursive_singleton_contents(
            len(tensor.shape), formatted
        )
    except RecursionError as error:
        # Before CPython 3.12, the pure-Python recursion check reports the
        # boundary where it happened (for example, ``in comparison``), while
        # PyTorch's native tensor formatter reaches the callable boundary.
        # Preserve the original exception and traceback while matching that
        # public diagnostic.
        if _sys.version_info < (3, 12):
            error.args = (
                "maximum recursion depth exceeded while calling a Python object",
            )
        raise
    suffix = _nn_functional_dropout_tensor_autograd_suffix(tensor)
    return f"tensor({formatted}{suffix})"


def _format_recursive_singleton_contents(
    dimensions, formatted, formatter_frames=6
):
    # PyTorch reaches its recursive tensor printer through six additional
    # Tensor.__format__/repr helper frames. Preserve that recursion headroom so
    # deeply ranked diagnostics fail at the same resource limit.
    if formatter_frames:
        return _format_recursive_singleton_contents(
            dimensions, formatted, formatter_frames - 1
        )
    if dimensions == 1:
        return f"[{formatted}]"
    return (
        "["
        + _format_recursive_singleton_contents(
            dimensions - 1, formatted, 0
        )
        + "]"
    )
