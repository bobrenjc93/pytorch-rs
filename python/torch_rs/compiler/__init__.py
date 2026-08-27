from collections.abc import Callable
import functools
import types
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
    "keep_portable_guards_unsafe",
    "skip_guard_on_globals_unsafe",
    "skip_all_guards_unsafe",
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


def _disable_function(fn, recursive, reason):
    supported_function_types = (types.FunctionType, types.MethodType)
    if not isinstance(fn, supported_function_types):
        raise NotImplementedError(
            "torch.compiler.disable() currently supports only Python functions"
        )

    unwrapped_fn = fn
    while (
        hasattr(unwrapped_fn, "_torchdynamo_orig_callable")
        and getattr(unwrapped_fn, "_torchdynamo_wrapper_id", None)
        == id(unwrapped_fn)
    ):
        unwrapped_fn = unwrapped_fn._torchdynamo_orig_callable
        if not isinstance(unwrapped_fn, supported_function_types):
            raise NotImplementedError(
                "torch.compiler.disable() currently supports only Python functions"
            )

    disable_recursive = bool(recursive)

    @functools.wraps(unwrapped_fn)
    def disabled_function(*args, **kwargs):
        return unwrapped_fn(*args, **kwargs)

    disabled_function._torchdynamo_disable = True
    disabled_function._torchdynamo_disable_msg = reason
    disabled_function._torchdynamo_orig_callable = unwrapped_fn
    disabled_function._torchdynamo_wrapper_id = id(disabled_function)
    disabled_function._torchdynamo_disable_recursive = disable_recursive
    return disabled_function


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
        disable_recursive = bool(recursive)

        def decorator(fn):
            return _disable_function(fn, disable_recursive, reason)

        return decorator

    return _disable_function(fn, recursive, reason)


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


def keep_portable_guards_unsafe(guard_entries):
    """
    A common function to only keep guards that can be used in both Python and non-Python environments.
    This includes:
    - Tensor metadata and dynamic shape information.
    - Global contexts state (e.g. autocast, no_grad, etc.)

    This is unsafe to use by default.
    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_global_context_and_tensor_guards_unsafe},
    >> )
    """
    return [
        (
            g.guard_type in ("GLOBAL_STATE", "SHAPE_ENV")
            or (g.guard_type == "TENSOR_MATCH" and not g.is_global)
        )
        for g in guard_entries
    ]


def skip_guard_on_globals_unsafe(guard_entries):
    """
    A common function to skip guards on all globals. This is unsafe to use by
    default. But if you don't expect any changes in the globals, you can just
    keep the tensor guards.

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_globals},
    >> )
    """

    return [not entry.is_global for entry in guard_entries]


def skip_all_guards_unsafe(guard_entries):
    """
    A function for skipping all guards on a compiled function.

    WARNING: This function will drop all the safety guarantees from Dynamo
             compiled function. Use this with caution.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
    >> )
    """
    return [False for entry in guard_entries]
