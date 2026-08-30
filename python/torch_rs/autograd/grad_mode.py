"""Gradient-mode context managers."""

import copyreg as _copyreg

from ..torch_rs import enable_grad as enable_grad
from ..torch_rs import no_grad as no_grad


def _legacy_rebuild_no_grad(context_type):
    return no_grad.__new__(context_type)


def _legacy_rebuild_enable_grad(context_type):
    return enable_grad.__new__(context_type)


def _grad_mode_state(context):
    getstate = getattr(context, "__getstate__", None)
    if getstate is not None:
        return getstate()

    instance_state = getattr(context, "__dict__", None)
    slot_state = {}
    for name in _copyreg._slotnames(type(context)) or ():
        try:
            slot_state[name] = getattr(context, name)
        except AttributeError:
            pass
    if slot_state:
        return instance_state, slot_state
    return instance_state


def _grad_mode_newobj(context):
    context_type = type(context)
    getnewargs_ex = getattr(context, "__getnewargs_ex__", None)
    if getnewargs_ex is not None:
        newargs_ex = getnewargs_ex()
        if not isinstance(newargs_ex, tuple):
            raise TypeError(
                "__getnewargs_ex__ should return a tuple, "
                f"not '{type(newargs_ex).__name__}'"
            )
        if len(newargs_ex) != 2:
            raise ValueError(
                "__getnewargs_ex__ should return a tuple of length 2, "
                f"not {len(newargs_ex)}"
            )
        newargs, newkwargs = newargs_ex
        if not isinstance(newargs, tuple):
            raise TypeError(
                "first item of the tuple returned by __getnewargs_ex__ "
                f"must be a tuple, not '{type(newargs).__name__}'"
            )
        if not isinstance(newkwargs, dict):
            raise TypeError(
                "second item of the tuple returned by __getnewargs_ex__ "
                f"must be a dict, not '{type(newkwargs).__name__}'"
            )
        return _copyreg.__newobj_ex__, (context_type, newargs, newkwargs)

    getnewargs = getattr(context, "__getnewargs__", None)
    if getnewargs is not None:
        newargs = getnewargs()
        if not isinstance(newargs, tuple):
            raise TypeError(
                "__getnewargs__ should return a tuple, "
                f"not '{type(newargs).__name__}'"
            )
        return _copyreg.__newobj__, (context_type, *newargs)

    return _copyreg.__newobj__, (context_type,)


def _reduce_no_grad(context, protocol):
    if protocol < 2:
        return (
            _legacy_rebuild_no_grad,
            (type(context),),
            _grad_mode_state(context),
        )
    newobj, newargs = _grad_mode_newobj(context)
    return newobj, newargs, _grad_mode_state(context)


def _reduce_enable_grad(context, protocol):
    if protocol < 2:
        return (
            _legacy_rebuild_enable_grad,
            (type(context),),
            _grad_mode_state(context),
        )
    newobj, newargs = _grad_mode_newobj(context)
    return newobj, newargs, _grad_mode_state(context)


__all__ = ["no_grad", "enable_grad"]
