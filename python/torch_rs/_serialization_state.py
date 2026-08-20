# This private module outlives replacement imports of ``torch_rs.serialization``.
compute_crc32 = True

try:
    from mmap import MAP_PRIVATE as default_mmap_options
except ImportError:
    default_mmap_options = None
