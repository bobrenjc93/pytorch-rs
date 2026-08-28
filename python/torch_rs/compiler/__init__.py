from collections.abc import Callable
import functools
import types
from typing import Any

from .. import _compiler_state as _state


__all__ = [
    "assume_constant_result",
    "reset",
    "allow_in_graph",
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


def allow_in_graph(fn):
    """
    Tells the compiler frontend (Dynamo) to skip symbolic introspection of the function
    and instead directly write it to the graph when encountered.

    If you are using :func:`torch.compile` (with backend="inductor" (the default)), or
    :func:`torch.export.export`, and trying to black-box a Python function throughout
    all tracing, do not use this API.
    Instead, please create a custom operator (see `PyTorch Custom Operators Landing Page
    <https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html>`_)

    .. warning::

        If you're a typical torch.compile user (e.g. you're applying torch.compile to
        a model to make it run faster), you probably don't want to use this function.
        :func:`allow_in_graph` is a footgun because it skips the compiler frontend
        (Dynamo) that is responsible for doing safety checks (graph breaks, handling
        closures, etc). Incorrect usage will lead to difficult-to-debug silent
        incorrectness issues.

    Given a Python function with no allow_in_graph decorator, regular execution
    of torch.compile traces through the function. :func:`allow_in_graph` changes
    it so that the frontend does not trace inside the function, but the compiler
    backend still traces through it. Compare this to custom operators, which
    treats a function as a black box throughout the torch.compile stack. The following
    table compares these mechanisms.

    +------------------------+-----------------------+--------------------------------+
    | Mechanism              | Frontend (Dynamo)     | Backend (AOTAutograd+Inductor) |
    +========================+=======================+================================+
    | no decorator           | trace inside          | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | allow_in_graph         | opaque callable       | trace inside                   |
    +------------------------+-----------------------+--------------------------------+
    | custom op              | opaque callable       | opaque callable                |
    +------------------------+-----------------------+--------------------------------+

    One common use case for :func:`allow_in_graph()` is as an escape hatch for the compiler
    frontend: if you know the function works w.r.t. to the downstream components of the
    compilation stack (AOTAutograd and Inductor) but there is a Dynamo bug that prevents it from
    symbolically introspecting the function properly (or if your code is in C/C++ and
    therefore cannot be introspected with Dynamo), then one can decorate said function
    with :func:`allow_in_graph` to bypass Dynamo.

    We require that ``fn`` adhere to the following restrictions. Failure to adhere
    results in undefined behavior:

    - The inputs to ``fn`` must be Proxy-able types in the FX graph. Valid types include:
      Tensor/int/bool/float/None/List[Tensor?]/List[int?]/List[float?]
      Tuple[Tensor?, ...]/Tuple[int?, ...]/Tuple[float?, ...]/torch.dtype/torch.device
    - The outputs to ``fn`` must be Proxy-able types in the FX graph (see previous bullet)
    - all Tensors used inside of ``fn`` must be passed directly as inputs to ``fn``
      (as opposed to being captured variables).

    Args:
        fn: A callable representing the function to be included in the graph.
            If ``fn`` is a list or tuple of callables it recursively applies
            :func:`allow_in_graph()` to each function and returns a new list or
            tuple containing the modified functions.

    Example::

        torch.compiler.allow_in_graph(my_custom_function)


        @torch.compile(...)
        def fn(x):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x


        fn(...)

    Will capture a single graph containing ``my_custom_function()``.

    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(item) for item in fn]
    if not callable(fn):
        raise AssertionError("allow_in_graph expects a callable")
    return fn


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
