"""PyTorch-compatible public package backed by the native extension."""

import builtins as _builtins
import copyreg as _copyreg
import functools as _functools
import multiprocessing.reduction as _multiprocessing_reduction
import sys as _sys
import types as _types
from math import e, inf, nan, pi

from . import torch_rs as _native
from .torch_rs import *

# PyTorch's built-in variable functions reduce through owners in ``torch._C``.
# Expose the native extension under the equivalent package-local name so those
# owners remain importable without creating or modifying a top-level ``torch``.
_C = _native
_sys.modules[f"{__name__}._C"] = _C
# TensorBase reports PyTorch's ``torch._C`` metadata, so retain its actual
# native class privately for package-local descriptor reconstruction.
_TensorBase = Tensor.__base__

# PyTorch's memory-format reducers use dotted public names such as
# ``torch.channels_last``. Mirror its module self-alias so those names resolve
# from this package without adding ``torch`` to wildcard imports.
torch = _sys.modules[__name__]

# Match PyTorch's NumPy-compatible spelling for inserting a singleton axis.
# This remains a package-level alias; ``torch._C`` does not expose it.
newaxis = None


# PyTorch exposes this private build probe through its immutable variable-
# function owner, but not through ``torch._C`` or wildcard imports.
_nnpack_available = _native._VariableFunctionsClass._nnpack_available


# Keep the public answer as the immutable build-time fact. Like PyTorch's
# generated package function, it must not observe reassignment or deletion of
# the writable compatibility metadata on ``torch._C``.
def compiled_with_cxx11_abi() -> _builtins.bool:
    r"""Returns whether PyTorch was built with _GLIBCXX_USE_CXX11_ABI=1"""
    return False


def are_deterministic_algorithms_enabled() -> _builtins.bool:
    r"""Returns True if the global deterministic flag is turned on. Refer to
    :func:`torch.use_deterministic_algorithms` documentation for more details.
    """
    return False


def _deterministic_algorithms_bool_type_name(value):
    value_type = _builtins.type(value)
    if value_type is Tensor:
        return "Tensor"
    if value_type is dtype:
        return "torch.dtype"
    if value_type is device:
        return "torch.device"
    if value_type is memory_format:
        return "torch.memory_format"
    if value_type is layout:
        return "torch.layout"
    if value_type is Size:
        return "torch.Size"
    if value_type is finfo:
        return "torch.finfo"

    name = _builtins.object.__getattribute__(value_type, "__name__")
    module = _builtins.object.__getattribute__(value_type, "__module__")
    if module == "numpy":
        return f"numpy.{name}"
    return name


def use_deterministic_algorithms(
    mode: _builtins.bool,
    *,
    warn_only: _builtins.bool = False,
) -> None:
    r"""Sets whether operations must use deterministic algorithms.

    This compatibility entrypoint currently accepts only ``mode=False`` with
    ``warn_only=False``. Deterministic algorithm enforcement and warning-only
    enforcement remain unsupported.

    Args:
        mode (:class:`bool`): If False, leave deterministic enforcement
            disabled. True is not supported.

    Keyword args:
        warn_only (:class:`bool`, optional): Must be False. Default: ``False``
    """
    if _builtins.type(mode) is not _builtins.bool:
        type_name = _deterministic_algorithms_bool_type_name(mode)
        raise TypeError(
            "_set_deterministic_algorithms(): argument 'mode' "
            f"(position 1) must be bool, not {type_name}"
        )
    if _builtins.type(warn_only) is not _builtins.bool:
        type_name = _deterministic_algorithms_bool_type_name(warn_only)
        raise TypeError(
            "_set_deterministic_algorithms(): argument 'warn_only' "
            f"must be bool, not {type_name}"
        )
    if mode:
        raise NotImplementedError(
            "use_deterministic_algorithms(): deterministic algorithm "
            "enforcement is not supported; only mode=False with "
            "warn_only=False is implemented"
        )
    if warn_only:
        raise NotImplementedError(
            "use_deterministic_algorithms(): warning-only deterministic "
            "enforcement is not supported; only mode=False with "
            "warn_only=False is implemented"
        )
    return None


def get_deterministic_debug_mode() -> _builtins.int:
    r"""Returns the current value of the debug mode for deterministic
    operations. Refer to :func:`torch.set_deterministic_debug_mode`
    documentation for more details.
    """
    return 0


def set_deterministic_debug_mode(debug_mode: _builtins.int | str) -> None:
    r"""Sets the debug mode for deterministic operations.

    .. note:: This is an alternative interface for
        :func:`torch.use_deterministic_algorithms`. Refer to that function's
        documentation for details about affected operations.

    Args:
        debug_mode(str or int): If "default" or 0, don't error or warn on
            nondeterministic operations. If "warn" or 1, warn on
            nondeterministic operations. If "error" or 2, error on
            nondeterministic operations.
    """
    if not _builtins.isinstance(
        debug_mode,
        (_builtins.int, _builtins.str),
    ):
        debug_mode_type = _builtins.type(debug_mode)
        if debug_mode_type is Tensor:
            type_name = "<class 'torch.Tensor'>"
        elif debug_mode_type is dtype:
            type_name = "<class 'torch.dtype'>"
        elif debug_mode_type is device:
            type_name = "<class 'torch.device'>"
        elif debug_mode_type is memory_format:
            type_name = "<class 'torch.memory_format'>"
        elif debug_mode_type is layout:
            type_name = "<class 'torch.layout'>"
        elif debug_mode_type is Size:
            type_name = "<class 'torch.Size'>"
        elif debug_mode_type is finfo:
            type_name = "<class 'torch.finfo'>"
        else:
            type_name = _builtins.str(debug_mode_type)
        raise TypeError(f"debug_mode must be str or int, but got {type_name}")

    requested_mode = debug_mode
    if _builtins.isinstance(debug_mode, _builtins.str):
        if debug_mode == "default":
            debug_mode = 0
        elif debug_mode == "warn":
            debug_mode = 1
        elif debug_mode == "error":
            debug_mode = 2
        else:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, "
                f"`warn`, `error`, but got {debug_mode}"
            )

    if debug_mode == 0:
        return None
    if debug_mode == 1 or debug_mode == 2:
        raise NotImplementedError(
            "set_deterministic_debug_mode(): debug_mode "
            f"{requested_mode!r} is not supported; only 0, False, and "
            "'default' are implemented"
        )
    raise RuntimeError(
        f"invalid value of debug_mode, expected 0, 1, or 2, but got {debug_mode}"
    )


def is_deterministic_algorithms_warn_only_enabled() -> _builtins.bool:
    r"""Returns True if the global deterministic flag is set to warn only.
    Refer to :func:`torch.use_deterministic_algorithms` documentation for more
    details.
    """
    return False


def get_default_device() -> "torch.device":
    r"""Gets the default ``torch.Tensor`` to be allocated on ``device``"""
    return torch.device("cpu")


def set_default_device(device: "Device") -> None:
    r"""Sets the default ``torch.Tensor`` to be allocated on ``device``.  This
    CPU-only compatibility entrypoint accepts only default-equivalent CPU
    requests: ``None``, ``"cpu"``, and unindexed ``torch.device("cpu")``
    values. It returns ``None`` and leaves factory output on the unindexed CPU
    device. Mutable default-device routing, CUDA/meta defaults, indexed CPU
    defaults, and ``with torch.device(...)`` device-context behavior remain
    unsupported.
    """
    if device is None:
        return None

    if _builtins.isinstance(device, torch.device):
        if device.type == "cpu" and device.index is None:
            return None
        if device.type == "cpu":
            raise NotImplementedError(
                "set_default_device(): indexed CPU default devices are not "
                "supported; only the unindexed CPU default is implemented"
            )
        raise RuntimeError(
            f"set_default_device(): device '{device}' is not supported; "
            "only the unindexed CPU default is implemented"
        )

    if _builtins.isinstance(device, str):
        if device == "cpu":
            return None
        if (
            device == "cuda"
            or device.startswith("cuda:")
            or device == "meta"
            or device.startswith("meta:")
        ):
            raise RuntimeError(
                f"set_default_device(): device '{device}' is not supported; "
                "only the unindexed CPU default is implemented"
            )
        parsed_device = torch.device(device)
        if parsed_device.type == "cpu" and parsed_device.index is not None:
            raise NotImplementedError(
                "set_default_device(): indexed CPU default devices are not "
                "supported; only the unindexed CPU default is implemented"
            )
        raise RuntimeError(
            f"set_default_device(): device '{parsed_device}' is not supported; "
            "only the unindexed CPU default is implemented"
        )

    device_type = _builtins.type(device)
    type_name = _builtins.object.__getattribute__(device_type, "__name__")
    raise TypeError(
        "set_default_device(): argument 'device' must be torch.device, str, "
        f"or None, not {type_name}"
    )


_COMPILE_UNSUPPORTED_MESSAGE = (
    "torch.compile(): only backend='eager', fullgraph=True straight-line "
    "Tensor neg/abs/relu/square/detach/float/add functions, plus one top-level "
    "if over an input Tensor.requires_grad selecting from that same subset, "
    "optionally inlining one exact same-module helper call, with one or two "
    "positional exact native CPU float32 Tensor inputs and Tensor or tuple/list "
    "Tensor-pytree outputs are supported; eager fallback, installed-PyTorch "
    "forwarding, callable backend invocation, CUDA compilation, and broader "
    "graph capture remain unsupported"
)
_COMPILE_DEFAULT_RECOMPILE_LIMIT = 8
_COMPILE_TENSOR_METHOD_GUARD_NAMES = (
    "__abs__",
    "__add__",
    "__getattribute__",
    "__neg__",
    "__radd__",
    "abs",
    "absolute",
    "add",
    "detach",
    "float",
    "neg",
    "negative",
    "relu",
    "square",
)
_COMPILE_TENSOR_METHOD_GUARD_MISSING = globals().get(
    "_COMPILE_TENSOR_METHOD_GUARD_MISSING",
    object(),
)
_COMPILE_TENSOR_METHOD_GUARDS = globals().get("_COMPILE_TENSOR_METHOD_GUARDS")


def _is_exact_python_function(model, _function_type=_types.FunctionType):
    return _builtins.type(model) is _function_type


def _set_compile_wrapper_metadata(
    compiled_model,
    *,
    fullgraph,
    dynamic,
    resolved_backend,
    mode,
    options,
    name,
    recompile_limit,
    isolate_recompiles,
    shapes_spec,
):
    compiled_model._torch_rs_compile_fullgraph = fullgraph
    compiled_model._torch_rs_compile_dynamic = dynamic
    compiled_model._torch_rs_compile_backend = resolved_backend
    compiled_model._torch_rs_compile_mode = mode
    compiled_model._torch_rs_compile_options = options
    compiled_model._torch_rs_compile_name = name
    compiled_model._torch_rs_compile_recompile_limit = recompile_limit
    compiled_model._torch_rs_compile_isolate_recompiles = isolate_recompiles
    compiled_model._torch_rs_compile_shapes_spec = shapes_spec
    return compiled_model


def _make_compile_wrapper(
    model,
    *,
    fullgraph,
    dynamic,
    resolved_backend,
    mode,
    options,
    name,
    recompile_limit,
    isolate_recompiles,
    shapes_spec,
    implementation,
):
    if implementation is None:

        def compiled_model(*args, **kwargs):
            raise NotImplementedError(_COMPILE_UNSUPPORTED_MESSAGE)

    else:

        def compiled_model(*args, **kwargs):
            return implementation(*args, **kwargs)

    import functools as _compile_functools

    if _is_exact_python_function(model):
        _compile_functools.update_wrapper(compiled_model, model)
    else:
        compiled_model.__wrapped__ = model
    return _set_compile_wrapper_metadata(
        compiled_model,
        fullgraph=fullgraph,
        dynamic=dynamic,
        resolved_backend=resolved_backend,
        mode=mode,
        options=options,
        name=name,
        recompile_limit=recompile_limit,
        isolate_recompiles=isolate_recompiles,
        shapes_spec=shapes_spec,
    )


def _supports_native_eager_compile(
    *,
    fullgraph,
    dynamic,
    resolved_backend,
    mode,
    options,
    isolate_recompiles,
    shapes_spec,
):
    return (
        _builtins.type(resolved_backend) is _builtins.str
        and resolved_backend == "eager"
        and fullgraph is True
        and dynamic is None
        and mode is None
        and options is None
        and isolate_recompiles is False
        and shapes_spec is None
    )


def _validated_compile_recompile_limit(recompile_limit):
    if recompile_limit is None:
        return _COMPILE_DEFAULT_RECOMPILE_LIMIT
    if _builtins.type(recompile_limit) is not _builtins.int:
        type_name = _builtins.object.__getattribute__(
            _builtins.type(recompile_limit),
            "__name__",
        )
        raise TypeError(
            "torch.compile(): argument 'recompile_limit' must be a "
            f"non-negative int or None, not {type_name}"
        )
    if recompile_limit < 0:
        raise ValueError(
            "torch.compile(): argument 'recompile_limit' must be "
            "non-negative"
        )
    return recompile_limit


def _snapshot_compile_tensor_method_guards():
    return tuple(
        (
            name,
            Tensor.__dict__.get(name, _COMPILE_TENSOR_METHOD_GUARD_MISSING),
            _TensorBase.__dict__.get(name, _COMPILE_TENSOR_METHOD_GUARD_MISSING),
        )
        for name in _COMPILE_TENSOR_METHOD_GUARD_NAMES
    )


def _initialize_compile_tensor_method_guards():
    global _COMPILE_TENSOR_METHOD_GUARDS
    if _COMPILE_TENSOR_METHOD_GUARDS is None:
        _COMPILE_TENSOR_METHOD_GUARDS = _snapshot_compile_tensor_method_guards()


def _ensure_compile_tensor_method_guards(
    compile_trace,
):
    _initialize_compile_tensor_method_guards()
    patched = []
    tensor_bindings = Tensor.__dict__
    tensor_base_bindings = _TensorBase.__dict__
    missing = _COMPILE_TENSOR_METHOD_GUARD_MISSING
    for name, expected_tensor, expected_tensor_base in _COMPILE_TENSOR_METHOD_GUARDS:
        if (
            tensor_bindings.get(name, missing) is not expected_tensor
            or tensor_base_bindings.get(name, missing) is not expected_tensor_base
        ):
            patched.append(f"Tensor.{name}")

    if patched:
        bindings = ", ".join(patched)
        raise compile_trace.CompileTraceUnsupportedError(
            "torch.compile trace bytecode lowering does not support patched "
            f"Tensor operation bindings: {bindings}"
        )


def _execute_native_eager_compile_graph(graph, inputs, compile_trace):
    return compile_trace.execute_compile_trace_graph(graph, *inputs)


def _native_eager_compile_implementation(model, name, recompile_limit):
    from . import _compiler_state as _compile_state

    cache = _compile_state.new_native_eager_compile_cache()
    program_descriptor = None

    def compiled_model(*args, **kwargs):
        nonlocal program_descriptor

        from . import _compile_bytecode as _compile_bytecode
        from . import _compile_trace as _compile_trace
        from . import overrides as _compile_overrides

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise _compile_trace.CompileTraceUnsupportedError(
                "torch.compile trace bytecode lowering does not support "
                f"keyword arguments: {names}"
            )
        if len(args) not in (1, 2):
            raise _compile_trace.CompileTraceUnsupportedError(
                "torch.compile trace bytecode lowering currently supports "
                "one or two positional Tensor arguments"
            )

        for input in args:
            if _builtins.type(input) is not Tensor:
                raise TypeError(
                    "torch.compile trace bytecode lowering expected exact "
                    f"native torch_rs Tensor input, got {_builtins.type(input)}"
                )

        if _compile_overrides._get_current_function_mode() is not None:
            raise _compile_trace.CompileTraceUnsupportedError(
                "torch.compile trace execution does not support active "
                "__torch_function__ modes"
            )

        _ensure_compile_tensor_method_guards(_compile_trace)
        input_metadatas = tuple(
            _compile_trace._metadata_from_native_tensor(input) for input in args
        )
        with cache.lock:
            compile_request = _compile_bytecode.prepare_compile_cache_request(
                model,
                input_metadatas,
                program_descriptor,
            )
            program_descriptor = compile_request.descriptor
            graph = cache.graphs.get(compile_request.key)
            if graph is None:
                if len(cache.graphs) >= recompile_limit:
                    raise _compile_trace.CompileTraceUnsupportedError(
                        "torch.compile trace bytecode lowering hit "
                        f"recompile_limit={recompile_limit}; no additional "
                        "input metadata variants will be compiled for this "
                        "wrapper"
                    )
                graph_name = name or getattr(model, "__name__", "compile_trace")
                if len(input_metadatas) == 1:
                    graph = _compile_bytecode.lower_one_input_compile_graph(
                        model,
                        input_metadatas[0],
                        name=graph_name,
                        compile_request=compile_request,
                    )
                else:
                    graph = _compile_bytecode.lower_compile_graph(
                        model,
                        input_metadatas,
                        name=graph_name,
                        compile_request=compile_request,
                    )
                cache.graphs[compile_request.key] = graph
        return _execute_native_eager_compile_graph(graph, args, _compile_trace)

    return compiled_model


def _validate_compile_configuration(mode, options):
    if mode is not None and options is not None:
        raise RuntimeError(
            "Either mode or options can be specified, but both can't be "
            "specified at the same time."
        )


def _validate_compile_model(model):
    if model is None:
        raise RuntimeError("Model can't be None")

    if not _builtins.callable(model):
        raise AssertionError(
            "A callable function is expected, but "
            f"{_builtins.type(model)} is provided."
        )


def _compile_bound_model(
    model,
    fullgraph,
    dynamic,
    backend,
    mode,
    options,
    name,
    disable,
    recompile_limit,
    isolate_recompiles,
    shapes_spec,
):
    _validate_compile_configuration(mode, options)
    resolved_backend = torch.compiler._resolve_compile_backend(backend)
    _validate_compile_model(model)

    if disable:
        return model

    implementation = None
    if _supports_native_eager_compile(
        fullgraph=fullgraph,
        dynamic=dynamic,
        resolved_backend=resolved_backend,
        mode=mode,
        options=options,
        isolate_recompiles=isolate_recompiles,
        shapes_spec=shapes_spec,
    ) and _is_exact_python_function(model):
        implementation = _native_eager_compile_implementation(
            model,
            name,
            _validated_compile_recompile_limit(recompile_limit),
        )

    return _make_compile_wrapper(
        model,
        fullgraph=fullgraph,
        dynamic=dynamic,
        resolved_backend=resolved_backend,
        mode=mode,
        options=options,
        name=name,
        recompile_limit=recompile_limit,
        isolate_recompiles=isolate_recompiles,
        shapes_spec=shapes_spec,
        implementation=implementation,
    )


def compile(
    model=None,
    *,
    fullgraph=False,
    dynamic=None,
    backend=None,
    mode=None,
    options=None,
    name=None,
    disable=False,
    recompile_limit=None,
    isolate_recompiles=False,
    shapes_spec=None,
):
    """Return a ``torch.compile`` compatibility shell.

    This entrypoint implements Python argument binding, ``disable=True``
    pass-through, and backend resolution through ``torch.compiler``. It also
    lowers exact Python functions with one or two positional exact native CPU
    ``float32`` Tensor inputs made only from Tensor ``neg``, ``abs``,
    ``relu``, ``square``, ``detach``, ``float``, binary ``add``, and one exact
    same-module helper call over Tensor arguments for ``backend="eager"``
    with ``fullgraph=True``. Those functions may contain one top-level
    ``if`` over an input Tensor's ``requires_grad`` metadata; the native path
    lowers the selected branch and returns either a Tensor or a tuple/list
    pytree with Tensor leaves.
    Eager fallback, installed-PyTorch forwarding, callable backend invocation,
    CUDA compilation, and broader graph capture remain unsupported.
    """
    if model is None:
        captured_backend = (
            torch.compiler.get_default_backend() if backend is None else backend
        )

        def compile_decorator(model):
            if model is None:
                raise RuntimeError("Model can't be None")

            return _compile_bound_model(
                model,
                fullgraph,
                dynamic,
                captured_backend,
                mode,
                options,
                name,
                disable,
                recompile_limit,
                isolate_recompiles,
                shapes_spec,
            )

        return compile_decorator

    return _compile_bound_model(
        model,
        fullgraph,
        dynamic,
        backend,
        mode,
        options,
        name,
        disable,
        recompile_limit,
        isolate_recompiles,
        shapes_spec,
    )


@_functools.cache
def get_device_module(device: torch.device | str | None = None):
    """
    Returns the module associated with a given device(e.g., torch.device('cuda'), "mtia:0", "xpu", ...).
    If no device is given, return the module for the current accelerator or CPU if none is present.
    """
    if isinstance(device, torch.device):
        device_module_name = device.type
    elif isinstance(device, str):
        device_module_name = torch.device(device).type
    elif device is None:
        # CPU is the only execution device implemented by the native backend.
        device_module_name = "cpu"
    else:
        raise RuntimeError(
            f"Invalid value of device '{device}', expect torch.device, str, or None"
        )
    device_module = getattr(torch, device_module_name, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_module_name}' does not have a corresponding module registered as 'torch.{device_module_name}'."
        )
    return device_module


def get_float32_matmul_precision() -> str:
    r"""Returns the current value of float32 matrix multiplication precision. Refer to
    :func:`torch.set_float32_matmul_precision` documentation for more details.
    """
    if _C._get_cublas_allow_tf32():
        return "high"
    return "highest"


def set_float32_matmul_precision(precision: str) -> None:
    r"""Sets the internal precision of float32 matrix multiplications.

    Running float32 matrix multiplications in lower precision may significantly increase
    performance, and in some programs the loss of precision has a negligible impact.

    Supports three settings:

        * "highest", float32 matrix multiplications use the float32 datatype (24 mantissa
          bits with 23 bits explicitly stored) for internal computations.
        * "high", float32 matrix multiplications either use the TensorFloat32 datatype (10
          mantissa bits explicitly stored) or treat each float32 number as the sum of two bfloat16 numbers
          (approximately 16 mantissa bits with 14 bits explicitly stored), if the appropriate fast matrix multiplication
          algorithms are available.  Otherwise float32 matrix multiplications are computed
          as if the precision is "highest".  See below for more information on the bfloat16
          approach.
        * "medium", float32 matrix multiplications use the bfloat16 datatype (8 mantissa
          bits with 7 bits explicitly stored) for internal computations, if a fast matrix multiplication algorithm
          using that datatype internally is available. Otherwise float32
          matrix multiplications are computed as if the precision is "high".

    When using "high" precision, float32 multiplications may use a bfloat16-based algorithm
    that is more complicated than simply truncating to some smaller number mantissa bits
    (e.g. 10 for TensorFloat32, 7 for bfloat16 explicitly stored).  Refer to [Henry2019]_ for a complete
    description of this algorithm.  To briefly explain here, the first step is to realize
    that we can perfectly encode a single float32 number as the sum of three bfloat16
    numbers (because float32 has 23 mantissa bits while bfloat16 has 7 explicitly stored, and both have the
    same number of exponent bits).  This means that the product of two float32 numbers can
    be exactly given by the sum of nine products of bfloat16 numbers.  We can then trade
    accuracy for speed by dropping some of these products.  The "high" precision algorithm
    specifically keeps only the three most significant products, which conveniently excludes
    all of the products involving the last 8 mantissa bits of either input.  This means that
    we can represent our inputs as the sum of two bfloat16 numbers rather than three.
    Because bfloat16 fused-multiply-add (FMA) instructions are typically >10x faster than
    float32 ones, it's faster to do three multiplications and 2 additions with bfloat16
    precision than it is to do a single multiplication with float32 precision.

    .. [Henry2019] http://arxiv.org/abs/1904.06376

    .. note::

        This does not change the output dtype of float32 matrix multiplications,
        it controls how the internal computation of the matrix multiplication is performed.

    .. note::

        This does not change the precision of convolution operations. Other flags,
        like `torch.backends.cudnn.allow_tf32`, may control the precision of convolution
        operations.

    .. note::

        This flag currently only affects one native device type: CUDA.
        If "high" or "medium" are set then the TensorFloat32 datatype will be used
        when computing float32 matrix multiplications, equivalent to setting
        `torch.backends.cuda.matmul.allow_tf32 = True`. When "highest" (the default)
        is set then the float32 datatype is used for internal computations, equivalent
        to setting `torch.backends.cuda.matmul.allow_tf32 = False`.

    Args:
        precision(str): can be set to "highest" (default), "high", or "medium" (see above).

    """
    if _builtins.isinstance(precision, _builtins.str):
        try:
            _builtins.str.encode(precision, "utf-8")
        except UnicodeEncodeError:
            raise RuntimeError("error unpacking string as utf-8") from None
        value = _builtins.str.__str__(precision)
    elif _builtins.isinstance(precision, _builtins.bytes):
        value = _builtins.bytes.decode(precision, "utf-8")
    else:
        precision_type = _builtins.type(precision)
        if precision_type is Tensor:
            type_name = "Tensor"
        elif precision_type is dtype:
            type_name = "torch.dtype"
        elif precision_type is device:
            type_name = "torch.device"
        elif precision_type is memory_format:
            type_name = "torch.memory_format"
        elif precision_type is layout:
            type_name = "torch.layout"
        elif precision_type is Size:
            type_name = "torch.Size"
        elif precision_type is finfo:
            type_name = "torch.finfo"
        else:
            type_name = _builtins.object.__getattribute__(precision_type, "__name__")
        raise RuntimeError(
            f"set_float32_matmul_precision expects a str, but got {type_name}"
        )

    if value == "highest":
        _C._set_cublas_allow_tf32(False)
        return None
    if value == "high":
        _C._set_cublas_allow_tf32(True)
        return None
    raise NotImplementedError(
        "set_float32_matmul_precision(): precision "
        f"{value!r} is not supported; only 'highest' and 'high' are implemented"
    )


def set_warn_always(b: _builtins.bool, /) -> None:
    r"""When this flag is False (default) then some PyTorch warnings may only
    appear once per process. This helps avoid excessive warning information.
    Setting it to True causes these warnings to always appear, which may be
    helpful when debugging.

    Args:
        b (:class:`bool`): If True, force warnings to always be emitted
                           If False, set to the default behaviour
    """
    _C._set_warnAlways(b)


def is_warn_always_enabled() -> _builtins.bool:
    r"""Returns True if the global warn_always flag is turned on. Refer to
    :func:`torch.set_warn_always` documentation for more details.
    """
    return _C._get_warnAlways()


def set_default_dtype(d: "torch.dtype", /) -> None:
    r"""

    Sets the default floating point dtype to :attr:`d`. Supports floating point dtype
    as inputs. Other dtypes will cause torch to raise an exception.

    When PyTorch is initialized its default floating point dtype is torch.float32,
    and the intent of set_default_dtype(torch.float64) is to facilitate NumPy-like
    type inference. The default floating point dtype is used to:

    1. Implicitly determine the default complex dtype. When the default floating type is float16,
       the default complex dtype is complex32. For float32, the default complex dtype is complex64.
       For float64, it is complex128. For bfloat16, an exception will be raised because
       there is no corresponding complex type for bfloat16.
    2. Infer the dtype for tensors constructed using Python floats or complex Python
       numbers. See examples below.
    3. Determine the result of type promotion between bool and integer tensors and
       Python floats and complex Python numbers.

    Args:
        d (:class:`torch.dtype`): the floating point dtype to make the default.

    Example:
        >>> # xdoctest: +SKIP("Other tests may have changed the default type. Can we reset it?")
        >>> # initial default for floating point is torch.float32
        >>> # Python floats are interpreted as float32
        >>> torch.tensor([1.2, 3]).dtype
        torch.float32
        >>> # initial default for floating point is torch.complex64
        >>> # Complex Python numbers are interpreted as complex64
        >>> torch.tensor([1.2, 3j]).dtype
        torch.complex64

        >>> torch.set_default_dtype(torch.float64)
        >>> # Python floats are now interpreted as float64
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float64
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex128

        >>> torch.set_default_dtype(torch.float16)
        >>> # Python floats are now interpreted as float16
        >>> torch.tensor([1.2, 3]).dtype  # a new floating point tensor
        torch.float16
        >>> # Complex Python numbers are now interpreted as complex128
        >>> torch.tensor([1.2, 3j]).dtype  # a new complex tensor
        torch.complex32

    """
    _C._set_default_dtype(d)


# PyTorch exposes ``strided`` as an attribute without including it in
# ``torch.__all__``. Bind it explicitly instead of widening wildcard imports.
strided = _native.strided


def _get_layout(name):
    """Return the canonical layout identified by its string representation."""
    if name == "torch.strided":
        return strided
    raise KeyError(name)


def _reduce_layout(value):
    return _get_layout, (str(value),)


_copyreg.pickle(layout, _reduce_layout)


# The default method-descriptor reduction tries to import ``torch._C`` and
# cannot resolve torch-rs's metadata-compatible TensorBase. Handle only that
# owner and delegate every other descriptor to the reducer already in place.
def _get_tensorbase_method_descriptor(name):
    """Return a native TensorBase method descriptor by name."""
    return getattr(_TensorBase, name)


def _make_tensorbase_method_descriptor_reducer(
    previous,
    modules=_sys.modules,
    module_name=__name__,
    fallback_resolver=_get_tensorbase_method_descriptor,
):
    if getattr(
        previous,
        "_torch_rs_tensorbase_method_descriptor_reducer",
        False,
    ):
        previous = getattr(
            previous,
            "_torch_rs_previous_method_descriptor_reducer",
            None,
        )

    def reducer(descriptor):
        if descriptor.__objclass__ is _TensorBase:
            module = modules.get(module_name)
            resolver = getattr(
                module,
                "_get_tensorbase_method_descriptor",
                fallback_resolver,
            )
            return resolver, (descriptor.__name__,)
        if previous is not None:
            return previous(descriptor)
        return descriptor.__reduce__()

    reducer._torch_rs_tensorbase_method_descriptor_reducer = True
    reducer._torch_rs_previous_method_descriptor_reducer = previous
    return reducer


_reduce_tensorbase_method_descriptor = _make_tensorbase_method_descriptor_reducer(
    _copyreg.dispatch_table.get(_types.MethodDescriptorType)
)
_copyreg.pickle(_types.MethodDescriptorType, _reduce_tensorbase_method_descriptor)

_reduce_tensorbase_method_descriptor_for_forking = (
    _make_tensorbase_method_descriptor_reducer(
        _multiprocessing_reduction.ForkingPickler._extra_reducers.get(
            _types.MethodDescriptorType
        )
    )
)
_multiprocessing_reduction.ForkingPickler.register(
    _types.MethodDescriptorType,
    _reduce_tensorbase_method_descriptor_for_forking,
)

__doc__ = _native.__doc__
# Keep package-only exports out of the native module's list, just as PyTorch's
# numeric constants live on ``torch`` rather than ``torch._C``.  A separate
# list also keeps reloading safe: the native wildcard import must only name
# attributes that the extension itself owns. PyTorch keeps ``__version__``
# directly importable but excludes it from package wildcard imports.
__all__ = [
    *(name for name in _native.__all__ if name != "__version__"),
    "are_deterministic_algorithms_enabled",
    "use_deterministic_algorithms",
    "get_deterministic_debug_mode",
    "set_deterministic_debug_mode",
    "is_deterministic_algorithms_warn_only_enabled",
    "get_default_device",
    "set_default_device",
    "compile",
    "get_device_module",
    "get_float32_matmul_precision",
    "set_float32_matmul_precision",
    "set_warn_always",
    "is_warn_always_enabled",
    "e",
    "pi",
    "nan",
    "inf",
    "newaxis",
]
# PyTorch lists ``matmul`` once among its hand-written package exports and once
# among generated variable functions. Preserve that observable duplicate while
# the native module continues to own the callable itself.
if "matmul" in _native.__all__:
    __all__.insert(0, "matmul")

from . import __future__ as __future__
from . import accelerator as accelerator
from . import autograd as autograd
from . import backends as backends
from . import compiler as compiler
from . import cpu as cpu
from . import cuda as cuda
from . import distributed as distributed
from . import functional as functional
from . import jit as jit
from . import nn as nn
from . import overrides as overrides
from . import _tensor as _tensor

_initialize_compile_tensor_method_guards()

from . import serialization as serialization
from . import utils as utils
from . import version as version
from .functional import atleast_1d as atleast_1d
from .functional import atleast_2d as atleast_2d
from .functional import atleast_3d as atleast_3d
from .functional import broadcast_shapes as broadcast_shapes
from .functional import broadcast_tensors as broadcast_tensors

del (
    _copyreg,
    _functools,
    _multiprocessing_reduction,
    _native,
    _sys,
    _types,
)
