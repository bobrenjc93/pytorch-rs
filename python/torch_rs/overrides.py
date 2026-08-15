"""Dynamic ``__torch_function__`` override modes."""

import warnings

from .torch_rs import (
    _get_function_stack_at,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


class TorchFunctionMode:
    """Override ``__torch_function__`` operations within a dynamic scope."""

    def __init__(self):
        pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode()

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just "
            "`with Mode()`",
            stacklevel=2,
        )
        return cls(*args, **kwargs)


def _push_mode(mode):
    _push_on_torch_function_stack(mode)


def _pop_mode():
    return _pop_torch_function_stack()


def _get_current_function_mode():
    stack_len = _len_torch_function_stack()
    return _get_function_stack_at(stack_len - 1) if stack_len > 0 else None


def _get_current_function_mode_stack():
    return [
        _get_function_stack_at(index)
        for index in range(_len_torch_function_stack())
    ]


__all__ = ["TorchFunctionMode"]
