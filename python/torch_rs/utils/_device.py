"""Default-device dispatch through the thread-local TorchFunctionMode stack."""

import functools

import torch_rs as torch
from torch_rs._C import _len_torch_function_stack
from torch_rs.overrides import TorchFunctionMode, _pop_mode, _push_mode


CURRENT_DEVICE = None


@functools.lru_cache(1)
def _device_constructors():
    return {
        torch.ones,
        torch.eye,
        torch.full,
        torch.zeros,
        torch.tensor,
        torch.scalar_tensor,
    }


class DeviceContext(TorchFunctionMode):
    def __init__(self, device) -> None:
        self.device = torch.device(device)
        self.prev_mode = None

    def __enter__(self):
        global CURRENT_DEVICE
        self.old_device = CURRENT_DEVICE
        CURRENT_DEVICE = self.device
        # Keep the default device at the bottom so temporary user modes can
        # forward through it without being displaced by set_default_device.
        current_stack = [
            _pop_mode() for _ in range(_len_torch_function_stack())
        ]

        _push_mode(self)

        for mode in reversed(current_stack):
            if isinstance(mode, DeviceContext):
                self.prev_mode = mode
            else:
                _push_mode(mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global CURRENT_DEVICE
        CURRENT_DEVICE = self.old_device
        current_stack = []
        # There is at most one DeviceContext on the live stack, at its bottom.
        for _ in range(_len_torch_function_stack() - 1):
            mode = _pop_mode()
            if isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Found nested DeviceContext on the mode stack where none expected"
                )
            current_stack.append(mode)

        if _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if not isinstance(mode, DeviceContext):
                raise AssertionError(
                    "Expected a DeviceContext at the bottom of the mode stack"
                )
        if self.prev_mode is not None:
            _push_mode(self.prev_mode)

        for mode in reversed(current_stack):
            _push_mode(mode)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get("device") is None:
            kwargs["device"] = self.device
        return func(*args, **kwargs)
