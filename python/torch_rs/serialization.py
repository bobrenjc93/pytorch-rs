__all__ = [
    "check_module_version_greater_or_equal",
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
]

import mmap as _mmap
import sys as _sys
import warnings as _warnings
from enum import Enum as _Enum
from typing import Any as _Any

from . import _serialization_state as _state


_IS_WINDOWS = _sys.platform == "win32"
_MAP_PRIVATE = getattr(_mmap, "MAP_PRIVATE", None)
_MAP_SHARED = getattr(_mmap, "MAP_SHARED", None)


def check_module_version_greater_or_equal(
    module,
    req_version_tuple,
    error_if_malformed=True,
):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
    try:
        version_strs = module.__version__.split(".")
        # Cast module version fields to match the types of the required version
        module_version = tuple(
            type(req_field)(version_strs[idx])
            for idx, req_field in enumerate(req_version_tuple)
        )
        requirement_is_met = module_version >= req_version_tuple

    except Exception as e:
        message = (
            f"'{module.__name__}' module version string is malformed '{module.__version__}' and cannot be compared"
            f" with tuple {str(req_version_tuple)}"
        )
        if error_if_malformed:
            raise RuntimeError(message) from e
        else:
            _warnings.warn(
                message + ", but continuing assuming that requirement is met",
                stacklevel=2,
            )
            requirement_is_met = True

    return requirement_is_met


class LoadEndianness(_Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3


def get_default_load_endianness() -> LoadEndianness | None:
    """
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    """
    return _state.default_load_endianness


def set_default_load_endianness(endianness):
    """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """
    if not isinstance(endianness, LoadEndianness) and endianness is not None:
        raise TypeError("Invalid argument type in function set_default_load_endianness")
    _state.default_load_endianness = endianness


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
            (_MAP_PRIVATE is None or flags != _MAP_PRIVATE)
            and (_MAP_SHARED is None or flags != _MAP_SHARED)
        ):
            raise ValueError(
                "Invalid argument in function set_default_mmap_options, "
                f"expected mmap.MAP_PRIVATE or mmap.MAP_SHARED, but got {flags}"
            )
        self.prev = _state.default_mmap_options
        _state.default_mmap_options = flags

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: _Any,
        exc_value: _Any,
        traceback: _Any,
    ) -> None:
        _state.default_mmap_options = self.prev
