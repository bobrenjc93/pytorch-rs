__all__ = ["get_crc32_options", "set_crc32_options"]


# Keep the option in the serialization module so every importer and thread in
# this process observes the same value.  Preserve it when the module is
# reloaded, matching the lifetime of PyTorch's backing serialization config.
if "_compute_crc32" not in globals():
    _compute_crc32 = True


def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
    return _compute_crc32


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
    global _compute_crc32
    _compute_crc32 = compute_crc32
