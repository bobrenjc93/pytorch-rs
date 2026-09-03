# This private module outlives replacement imports of ``torch_rs.compiler``.

import weakref

from .torch_rs import (
    _exchange_enable_guard_collectives as exchange_enable_guard_collectives,
)


default_backend = "inductor"
registered_backends = {}
registered_backend_fns = {}
native_eager_compile_caches = weakref.WeakSet()


class NativeEagerCompileCache:
    __slots__ = ("graphs", "__weakref__")

    def __init__(self):
        self.graphs = {}

    def clear(self):
        self.graphs.clear()


def new_native_eager_compile_cache():
    cache = NativeEagerCompileCache()
    native_eager_compile_caches.add(cache)
    return cache


def reset_compile_caches():
    for cache in tuple(native_eager_compile_caches):
        cache.clear()
