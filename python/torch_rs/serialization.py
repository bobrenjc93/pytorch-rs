try:
    from mmap import MAP_PRIVATE as _DEFAULT_MMAP_OPTIONS
except ImportError:
    _DEFAULT_MMAP_OPTIONS = None


__all__ = ["get_crc32_options", "get_default_mmap_options"]


def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
    return True


def get_default_mmap_options() -> int | None:
    """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """
    return _DEFAULT_MMAP_OPTIONS
