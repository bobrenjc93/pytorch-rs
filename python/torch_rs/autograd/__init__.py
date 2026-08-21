"""Automatic differentiation helpers."""

from .. import _C as _C
from . import grad_mode as grad_mode
from .grad_mode import no_grad as no_grad


is_multithreading_enabled = _C._is_multithreading_enabled
is_view_replay_enabled = _C._is_view_replay_enabled

__all__ = ["grad_mode", "no_grad"]

del _C
