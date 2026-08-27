# This private module outlives replacement imports of ``torch_rs.serialization``.
import mmap as _mmap


compute_crc32 = True
default_load_endianness = None
default_mmap_options = getattr(_mmap, "MAP_PRIVATE", None)
marked_safe_globals = set()
