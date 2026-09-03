from collections.abc import Callable, Sequence
import functools
import sys
import types
from typing import Any

from .. import _compiler_state as _state

_torch = sys.modules[__name__.partition(".")[0]]
_MISSING_PARAMETER_TYPES = ()
_MISSING_BACKEND = object()
_BUILTIN_COMPILE_BACKENDS = frozenset(("eager", "inductor"))


__all__ = [
    "assume_constant_result",
    "reset",
    "list_backends",
    "register_backend",
    "disable",
    "set_default_backend",
    "get_default_backend",
    "set_enable_guard_collectives",
    "is_compiling",
    "is_dynamo_compiling",
    "is_exporting",
    "keep_portable_guards_unsafe",
    "skip_guard_on_inbuilt_nn_modules_unsafe",
    "skip_guard_on_all_nn_modules_unsafe",
    "keep_tensor_guards_unsafe",
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


def list_backends(exclude_tags=("debug", "experimental")) -> list[str]:
    """
    Return valid strings that can be passed to `torch.compile(..., backend="name")`.

    Args:
        exclude_tags(optional): A tuple of strings representing tags to exclude.
    """
    exclude_tags_set = set(exclude_tags or ())
    backends = [
        name
        for name in _state.registered_backends
        if name not in _state.registered_backend_fns
        or not exclude_tags_set.intersection(
            getattr(_state.registered_backend_fns[name], "_tags", ())
        )
    ]
    return sorted(backends)


def register_backend(
    compiler_fn: Callable[..., Any] | None = None,
    name: str | None = None,
    tags: Sequence[str] = (),
) -> Callable[..., Any]:
    """
    Decorator to add a given compiler to the registry to allow calling
    `torch.compile` with string shorthand.  Note: for projects not
    imported by default, it might be easier to pass a function directly
    as a backend and not use a string.

    Args:
        compiler_fn: Callable taking a FX graph and fake tensor inputs
        name: Optional name, defaults to `compiler_fn.__name__`
        tags: Optional set of string tags to categorize backend with
    """
    if name is not None and not isinstance(name, str):
        raise AssertionError(f"name must be str or None, got {type(name)}")
    if compiler_fn is None:
        return functools.partial(register_backend, name=name, tags=tags)
    if not callable(compiler_fn):
        raise AssertionError(f"compiler_fn must be callable, got {type(compiler_fn)}")

    backend_name = name or compiler_fn.__name__
    if not isinstance(backend_name, str):
        raise AssertionError(f"name must be str or None, got {type(backend_name)}")
    if backend_name in _state.registered_backend_fns:
        raise AssertionError(f"duplicate name: {backend_name}")

    backend_tags = tuple(tags)
    compiler_fn._tags = backend_tags
    _state.registered_backends.setdefault(backend_name, None)
    _state.registered_backend_fns[backend_name] = compiler_fn
    return compiler_fn


def _resolve_compile_backend(backend):
    if backend is None:
        backend = get_default_backend()

    if isinstance(backend, str):
        registered_backend = _state.registered_backend_fns.get(
            backend,
            _MISSING_BACKEND,
        )
        if registered_backend is not _MISSING_BACKEND:
            return registered_backend
        if (
            backend in _state.registered_backends
            or backend in _BUILTIN_COMPILE_BACKENDS
        ):
            return backend

        available_backends = sorted(
            {
                *_BUILTIN_COMPILE_BACKENDS,
                *(
                    name
                    for name in _state.registered_backends
                    if isinstance(name, str)
                ),
            }
        )
        if available_backends:
            available = ", ".join(repr(name) for name in available_backends)
            raise RuntimeError(
                "Invalid backend: "
                f"{backend!r}. Available backend names are: {available}"
            )
        raise RuntimeError(
            f"Invalid backend: {backend!r}. No backends are registered"
        )

    if callable(backend):
        return backend

    raise TypeError(f"backend must be a string or callable, got {type(backend)}")


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


def set_enable_guard_collectives(enabled: bool):
    """
    Enables use of collectives *during* guard evaluation to synchronize behavior
    across ranks.  This is expensive: we have to issue a collective every time
    we enter a compiled code region, even if no rank actually would need to
    compile.  This can help prevent NCCL hangs by ensuring that we never have a
    situation where one rank starts recompiling while other ranks don't compile;
    it is especially useful in conjunction with enable_compiler_collectives
    where such a situation would immediately cause a hang (as it is necessary
    for all ranks to compile at the same time to run compiler collectives).  Like
    compiler collectives, you can only run this on SPMD programs; you will hang
    otherwise.  Note that a guard collective is only issued if there is any
    compiled code to guard on; if this the first time we encounter a frame or
    the frame is skipped, we don't issue collectives.

    Returns the previous setting of enabled.
    """
    next_enabled = bool(enabled)
    return _state.exchange_enable_guard_collectives(next_enabled)


set_enable_guard_collectives._dynamo_forbidden = True


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


def skip_guard_on_inbuilt_nn_modules_unsafe(guard_entries):
    """
    A common function to skip guards on the inbuilt nn modules like
    torch.nn.Linear. This is unsafe to use by default. But for majority of
    torch.compile users, the model code does not modify the inbuilt nn module
    attributes. They can benefit from reduction in guard latency overhead using
    this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """
    return [
        not entry.orig_guard.source.is_unspecialized_builtin_nn_module()
        for entry in guard_entries
    ]


def skip_guard_on_all_nn_modules_unsafe(guard_entries):
    """
    A common function to skip guards on all nn modules, both user defined as
    well inbuilt nn modules (like torch.nn.Linear). This is unsafe to use by
    default. But for majority of torch.compile users, the model code does not
    modify the nn module attributes. They can benefit from reduction in guard
    latency overhead using this API.

    To use this API, use guard_filter_fn argument while calling torch.compile

    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
    >> )
    """

    return [
        not entry.orig_guard.source.is_unspecialized_nn_module()
        for entry in guard_entries
    ]


def keep_tensor_guards_unsafe(guard_entries, keep_parameters=False):
    """
    A common function to keep tensor guards on all tensors. This is unsafe to
    use by default. But if you don't expect any changes in the model code, you
    can just keep the tensor guards.


    >> opt_mod = torch.compile(
    >>     mod,
    >>     options={"guard_filter_fn": torch.compiler.keep_tensor_guards},
    >> )
    """

    keep_flags = []
    for entry in guard_entries:
        if entry.guard_type == "TENSOR_MATCH":
            value = entry.value
            try:
                parameter_type = _torch.nn.Parameter
            except AttributeError:
                parameter_type = _MISSING_PARAMETER_TYPES
            if not isinstance(value, parameter_type):
                keep_flags.append(True)
            elif keep_parameters:
                keep_flags.append(True)
            else:
                keep_flags.append(False)
        else:
            keep_flags.append(False)
    return keep_flags


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
