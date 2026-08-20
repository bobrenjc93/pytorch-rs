__all__ = [
    "get_crc32_options",
    "set_crc32_options",
    "get_default_mmap_options",
]

from . import _serialization_state as _state


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
