"""Tracing.

This module contains functionality to support the JIT's tracing frontend, notably:
    * torch.jit.trace
    * torch.jit.trace_module

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""

import functools as _functools
from collections.abc import Callable as _Callable
from typing import TypeVar as _TypeVar

from typing_extensions import ParamSpec as _ParamSpec


_R = _TypeVar("R", covariant=True)
_P = _ParamSpec("P")


def is_tracing():
    """Return a boolean value.

    Returns ``True`` in tracing (if a function is called during the
    tracing of code with ``torch.jit.trace``) and ``False`` otherwise.
    """
    return False


def _script_if_tracing(fn: _Callable[_P, _R]) -> _Callable[_P, _R]:
    @_functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not is_tracing():
            return fn(*args, **kwargs)

        # ``torch_rs.jit.script`` is intentionally not part of the supported
        # surface yet. Keep the import on the tracing-only path so eager use
        # remains independent of PyTorch and becomes ready for a future local
        # scripting implementation without exposing a placeholder API today.
        from torch_rs.jit import script as _script

        compiled_fn: _Callable[_P, _R] = _script(wrapper.__original_fn)
        return compiled_fn(*args, **kwargs)

    wrapper.__original_fn = fn
    wrapper.__script_if_tracing_wrapper = True

    return wrapper
