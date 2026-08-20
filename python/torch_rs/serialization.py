__all__ = [
    "get_crc32_options",
    "set_crc32_options",
    "get_default_mmap_options",
    "set_default_mmap_options",
]

import mmap as _mmap
import sys as _sys
from typing import Any as _Any

from . import _serialization_state as _state


_IS_WINDOWS = _sys.platform == "win32"
if not _IS_WINDOWS:
    _MAP_PRIVATE = getattr(_mmap, "MAP_PRIVATE", None)
    _MAP_SHARED = getattr(_mmap, "MAP_SHARED", None)
else:
    _MAP_SHARED, _MAP_PRIVATE = None, None


def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
    return _state.compute_crc32


def set_crc32_options(compute_crc32: bool):
    """
    Set whether :func:`torch.save` computes and writes crc32 for each record.

    .. note::
        Setting this to ``False`` may make unzipping of the ``torch.save`` output
        fail or warn due to corrupted CRC32. However ``torch.load`` will be
        able to load the file.

    Args:
        compute_crc32 (bool): set crc32 computation flag
    """
    _state.compute_crc32 = compute_crc32


def get_default_mmap_options() -> int | None:
    """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """
    return _state.default_mmap_options


class set_default_mmap_options:
    """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """

    def __init__(self, flags: int) -> None:
        if _IS_WINDOWS:
            raise RuntimeError(
                "Changing the default mmap options is currently not supported for Windows"
            )
        if (
            _MAP_PRIVATE is None
            or _MAP_SHARED is None
            or (flags != _MAP_PRIVATE and flags != _MAP_SHARED)
        ):
            raise ValueError(
                "Invalid argument in function set_default_mmap_options, "
                f"expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got {flags}"
            )
        self.prev = _state.default_mmap_options
        _state.default_mmap_options = flags

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: _Any, exc_value: _Any, traceback: _Any) -> None:
        _state.default_mmap_options = self.prev
