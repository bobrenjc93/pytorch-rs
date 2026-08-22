import functools as _functools
import types as _types
from collections.abc import Callable
from typing import Any

from .. import _compiler_state as _state


__all__ = [
    "assume_constant_result",
    "reset",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
]


def assume_constant_result(fn):
    """
    This function is used to mark a function `fn` as having a constant result.
    This allows the compiler to optimize away your function.
    Returns The same function `fn`

    Args:
        fn: The function to be marked as having a constant result.

    .. warning::
        `assume_constant_result` can if invalid cause safety and soundness issues, :func:`torch.compile`
        will not attempt to validate whether the constant assumption is true or not

    """
    fn._dynamo_marked_constant = True
    return fn


def reset() -> None:
    """
    Reset the in-process compiler state.

    This function clears Dynamo's in-memory compilation caches and related
    process-local state used by :func:`torch.compile`. It does not delete
    filesystem caches, such as Inductor's disk cache.
    """


def disable(fn=None, recursive=True, *, reason=None):
    """
    This function provides a decorator to disable compilation on a function.
    It also provides the option of recursively disabling called functions.

    Args:
        fn (optional): The function to disable
        recursive (optional): A boolean value indicating whether the disabling should be recursive.
        reason (optional): A string value indicating the reason for disabling the function.
    """
    if fn is None:
        raise NotImplementedError(
            "torch_rs.compiler.disable only supports direct calls with a Python "
            "function"
        )
    if not callable(fn):
        raise AssertionError("fn must be callable")
    if not isinstance(fn, (_types.FunctionType, _types.MethodType)):
        raise NotImplementedError(
            "torch_rs.compiler.disable only supports direct calls with a Python "
            "function"
        )

    while hasattr(fn, "_torchdynamo_orig_callable") and getattr(
        fn, "_torchdynamo_wrapper_id", None
    ) == id(fn):
        fn = fn._torchdynamo_orig_callable
        if not callable(fn):
            raise AssertionError(
                f"A callable function is expected, but {type(fn)} is provided."
            )
        if not isinstance(fn, (_types.FunctionType, _types.MethodType)):
            raise NotImplementedError(
                "torch_rs.compiler.disable only supports direct calls with a Python "
                "function"
            )

    @_functools.wraps(fn)
    def disabled(*args, **kwargs):
        return fn(*args, **kwargs)

    disabled._torchdynamo_disable = True
    disabled._torchdynamo_disable_msg = reason
    disabled._torchdynamo_orig_callable = fn
    disabled._torchdynamo_wrapper_id = id(disabled)
    disabled._torchdynamo_disable_recursive = bool(recursive)
    return disabled


def set_default_backend(backend: str | Callable[..., Any] | None) -> None:
    """Set the default backend for ``torch.compile`` when no ``backend`` argument is specified.

    Passing ``None`` resets the default back to ``"inductor"``.

    Args:
        backend: A backend name (string), a callable backend, or ``None``.

    Example::

        >>> torch.compiler.set_default_backend("eager")
        >>> torch.compiler.get_default_backend()
        'eager'
        >>> torch.compiler.set_default_backend(None)  # reset
        >>> torch.compiler.get_default_backend()
        'inductor'
    """
    if backend is None:
        _state.default_backend = "inductor"
        return
    if not isinstance(backend, str) and not callable(backend):
        raise TypeError(f"backend must be a string or callable, got {type(backend)}")
    _state.default_backend = backend


def get_default_backend() -> str | Callable[..., Any]:
    """Return the current default backend for ``torch.compile``.

    Returns:
        The current default backend (string or callable). Initially ``"inductor"``.
    """
    return _state.default_backend


def is_compiling() -> bool:
    """
    Indicates whether a graph is executed/traced as part of torch.compile() or torch.export().

    Note that there are 2 other related flags that should deprecated eventually:
      * torch._dynamo.external_utils.is_compiling()
      * torch._utils.is_compiling()

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_compiling():
        >>>        pass # ...logic that is not needed in a compiled/traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    return False


def is_dynamo_compiling() -> bool:
    """
    Indicates whether a graph is traced via TorchDynamo.

    It's stricter than is_compiling() flag, as it would only be set to True when
    TorchDynamo is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_dynamo_compiling():
        >>>        pass # ...logic that is not needed in a TorchDynamo-traced graph...
        >>>
        >>>     # ...rest of the function...
    """
    return False


def is_exporting() -> bool:
    """
    Indicated whether we're under exporting.

    It's stricter than is_compiling() flag, as it would only be set to True when
    torch.export is used.

    Example::

        >>> def forward(self, x):
        >>>     if not torch.compiler.is_exporting():
        >>>        pass # ...logic that is not needed in export...
        >>>
        >>>     # ...rest of the function...
    """
    return False
