"""
The weak_script annotation needs to be here instead of inside torch_rs/jit/ so it
can be used in other places in torch_rs/ (namely torch_rs.nn) without running into
circular dependency problems
"""

import collections
import types
import typing
import warnings
from typing import Callable, Final, TypeVar, Union, get_args, get_origin

from typing_extensions import ParamSpec


_P = ParamSpec("_P")
_R = TypeVar("_R")

BuiltinUnionType: type | tuple[type, ...] = types.UnionType


def is_scripting() -> bool:
    r"""
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
            if torch.jit.is_scripting():
                return torch.linear(x)
            else:
                return unsupported_linear_op(x)
    """
    return False


def raise_error_container_parameter_missing(target_type) -> None:
    if target_type.endswith("ict"):
        raise RuntimeError(
            f"Attempted to use {target_type} without "
            "contained types. Please add contained type, e.g. "
            f"{target_type}[int, int]"
        )
    raise RuntimeError(
        f"Attempted to use {target_type} without a "
        "contained type. Please add a contained type, e.g. "
        f"{target_type}[int]"
    )


_RAW_TYPE_NAME_MAPPING = {
    dict: "dict",
    list: "list",
    tuple: "tuple",
    typing.Dict: "Dict",
    typing.List: "List",
    typing.Optional: "Optional",
    typing.Tuple: "Tuple",
}


def check_args_exist(target_type) -> None:
    if name := _RAW_TYPE_NAME_MAPPING.get(target_type):
        raise_error_container_parameter_missing(name)


def check_empty_containers(obj) -> None:
    if obj == [] or obj == {} or obj == ():
        warnings.warn(
            "The inner type of a container is lost when "
            "calling torch.jit.isinstance in eager mode. For "
            "example, List[int] would become list and "
            "therefore falsely return True for List[float] or"
            " List[str].",
            stacklevel=2,
        )


def container_checker(obj, target_type) -> bool:
    origin_type = get_origin(target_type)
    check_args_exist(target_type)
    if origin_type is None:
        return False
    elif origin_type is list or origin_type is typing.List:
        check_empty_containers(obj)
        if not isinstance(obj, list):
            return False
        arg_type = get_args(target_type)[0]
        arg_origin = get_origin(arg_type)
        for el in obj:
            if arg_origin:
                if not container_checker(el, arg_type):
                    return False
            elif not isinstance(el, arg_type):
                return False
        return True
    elif origin_type is typing.Dict or origin_type is dict:
        check_empty_containers(obj)
        if not isinstance(obj, dict):
            return False
        key_type = get_args(target_type)[0]
        val_type = get_args(target_type)[1]
        for key, val in obj.items():
            if not isinstance(key, key_type):
                return False
            val_origin = get_origin(val_type)
            if val_origin:
                if not container_checker(val, val_type):
                    return False
            elif not isinstance(val, val_type):
                return False
        return True
    elif origin_type is typing.Tuple or origin_type is tuple:
        check_empty_containers(obj)
        if not isinstance(obj, tuple):
            return False
        arg_types = get_args(target_type)
        if len(obj) != len(arg_types):
            return False
        for el, el_type in zip(obj, arg_types):
            el_origin = get_origin(el_type)
            if el_origin:
                if not container_checker(el, el_type):
                    return False
            elif not isinstance(el, el_type):
                return False
        return True
    elif origin_type is Union or issubclass(origin_type, BuiltinUnionType):
        if obj is None:
            return True
        inner_types = get_args(target_type)
        for t in inner_types:
            t_origin = get_origin(t)
            if t_origin:
                return container_checker(obj, t)
            elif isinstance(obj, t):
                return True
    return False


def _isinstance(obj, target_type) -> bool:
    if isinstance(target_type, collections.abc.Container):
        if not isinstance(target_type, tuple):
            raise RuntimeError(
                "The second argument to "
                "`torch.jit.isinstance` must be a type "
                "or a tuple of types"
            )
        for t_type in target_type:
            if _isinstance(obj, t_type):
                return True
        return False

    origin_type = get_origin(target_type)
    if origin_type:
        return container_checker(obj, target_type)

    check_args_exist(target_type)

    return isinstance(obj, target_type)


class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """

    UNUSED = "unused (ignored and replaced with raising of an exception)"
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = "export (compile this function even if nothing calls it)"
    DEFAULT = "default (compile if called from an exported function / forward)"
    COPY_TO_SCRIPT_WRAPPER = (
        "if this method is not scripted, copy the python method onto the scripted model"
    )
    _DROP = "_drop (function is fully ignored, declaration can be unscriptable)"


def export(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """
    fn._torchscript_modifier = FunctionModifiers.EXPORT  # type: ignore[attr-defined]
    return fn


def unused(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn


            class MyModule(nn.Module):
                def __init__(self, use_memory_efficient):
                    super().__init__()
                    self.use_memory_efficient = use_memory_efficient

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb

                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10


            m = torch.jit.script(MyModule(use_memory_efficient=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    if isinstance(fn, property):
        prop = fn
        setattr(prop.fget, "_torchscript_modifier", FunctionModifiers.UNUSED)

        if prop.fset:
            setattr(prop.fset, "_torchscript_modifier", FunctionModifiers.UNUSED)

        return prop

    fn._torchscript_modifier = FunctionModifiers.UNUSED  # type: ignore[attr-defined]
    return fn


def ignore(drop=False, **kwargs):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    .. deprecated:: 2.5
        Please use :func:`torch.compile` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn


        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb

                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x


        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """

    if callable(drop):
        # used without any args, so drop is actually a function
        #   @torch.jit.ignore
        #   def fn(...):
        fn = drop
        fn._torchscript_modifier = FunctionModifiers.IGNORE  # type: ignore[attr-defined]
        return fn

    if not isinstance(drop, bool):
        raise RuntimeError(
            f"Argument to @torch.jit.ignore must be a bool or a function but got {drop}"
        )

    # for backwards compat
    drop_on_export = kwargs.pop("drop_on_export", None)
    if drop_on_export:
        warnings.warn(
            "ignore(drop_on_export=True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            stacklevel=2,
            category=FutureWarning,
        )

        drop = drop_on_export
    elif drop:
        warnings.warn(
            "ignore(True) has been deprecated. TorchScript will now drop the function "
            "call on compilation. Use torch.jit.unused now. {}",
            stacklevel=2,
            category=FutureWarning,
        )

    def decorator(fn):
        if drop:
            fn._torchscript_modifier = FunctionModifiers.UNUSED
        else:
            fn._torchscript_modifier = FunctionModifiers.IGNORE
        return fn

    return decorator
