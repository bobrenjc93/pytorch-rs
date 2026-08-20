# This private module outlives replacement imports of ``torch_rs.serialization``.
from contextvars import ContextVar as _ContextVar
import mmap as _mmap


compute_crc32 = True
default_mmap_options = _ContextVar(
    "torch_rs.serialization.default_mmap_options",
    default=getattr(_mmap, "MAP_PRIVATE", None),
)
