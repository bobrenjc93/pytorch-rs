"""Gradient-mode context managers."""

import copyreg as _copyreg

from ..torch_rs import no_grad as no_grad


def _rebuild_no_grad(context_type):
    return no_grad.__new__(context_type)


def _no_grad_state(context):
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


def _reduce_no_grad(context):
    return _rebuild_no_grad, (type(context),), _no_grad_state(context)


__all__ = ["no_grad"]
