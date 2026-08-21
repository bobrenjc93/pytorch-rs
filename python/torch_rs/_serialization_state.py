# This private module outlives replacement imports of ``torch_rs.serialization``.
import mmap as _mmap
from contextvars import ContextVar as _ContextVar


compute_crc32 = True
default_load_endianness = _ContextVar(
    "torch_rs.serialization.default_load_endianness",
    default=None,
)
default_mmap_options = getattr(_mmap, "MAP_PRIVATE", None)
