"""Internal accelerator discovery boundary for the native backend."""


def _discover_accelerator():
    """Return the compiled accelerator, runtime availability, and device count."""
    # The native Device enum currently contains only CPU, which is not an
    # accelerator. Keep these related facts together so a future accelerator
    # backend has one discovery boundary to replace.
    return None, False, 0
