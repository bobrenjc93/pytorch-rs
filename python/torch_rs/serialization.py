__all__ = [
    "LoadEndianness",
    "get_crc32_options",
    "set_crc32_options",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "get_default_mmap_options",
    "set_default_mmap_options",
    "clear_safe_globals",
    "get_safe_globals",
    "add_safe_globals",
    "safe_globals",
]

import mmap as _mmap
import sys as _sys
from collections.abc import Callable as _Callable
from enum import Enum as _Enum
from typing import Any as _Any

from . import _serialization_state as _state


_IS_WINDOWS = _sys.platform == "win32"
_MAP_PRIVATE = getattr(_mmap, "MAP_PRIVATE", None)
_MAP_SHARED = getattr(_mmap, "MAP_SHARED", None)
_SafeGlobal = _Callable | tuple[_Callable, str]


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


def add_safe_globals(safe_globals: list[_SafeGlobal]) -> None:
    """
    Marks the given globals as safe for ``weights_only`` load. For example, functions
    added to this list can be called during unpickling, classes could be instantiated
    and have state set.

    Each item in the list can either be a function/class or a tuple of the form
    (function/class, string) where string is the full path of the function/class.

    Within the serialized format, each function is identified with its full
    path as ``{__module__}.{__qualname__}``. When calling this API, you can provide this
    full path that should match the one in the checkpoint otherwise the default
    ``{fn.__module__}.{fn.__qualname__}`` will be used.

    Args:
        safe_globals (List[Union[Callable, Tuple[Callable, str]]]): list of globals to mark as safe

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    """
    _state.safe_globals = _state.safe_globals.union(set(safe_globals))


def get_safe_globals() -> list[_SafeGlobal]:
    """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """
    return list(_state.safe_globals)


def clear_safe_globals() -> None:
    """
    Clears the list of globals that are safe for ``weights_only`` load.
    """
    _state.safe_globals = set()


def _remove_safe_globals(
    globals_to_remove: list[_SafeGlobal],
) -> None:
    _state.safe_globals = _state.safe_globals - set(globals_to_remove)


class _safe_globals:
    def __init__(self, safe_globals: list[_SafeGlobal]):
        self.safe_globals = safe_globals

    def __enter__(self):
        add_safe_globals(self.safe_globals)

    def __exit__(self, type, value, tb):
        _remove_safe_globals(self.safe_globals)


class safe_globals(_safe_globals):
    r"""Context-manager that adds certain globals as safe for ``weights_only`` load.

    Args:
        safe_globals: List of globals for weights_only load.

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     with torch.serialization.safe_globals([MyTensor]):
        ...         torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
        >>> assert torch.serialization.get_safe_globals() == []
    """


safe_globals.__annotations__ = {}
