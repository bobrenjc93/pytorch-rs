import mmap as _mmap


__all__ = ["get_crc32_options", "get_default_mmap_options"]

_DEFAULT_MMAP_OPTIONS = getattr(_mmap, "MAP_PRIVATE", None)


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
