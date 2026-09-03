# This private module outlives replacement imports of ``torch_rs.compiler``.

import threading
import weakref

from .torch_rs import (
    _exchange_enable_guard_collectives as exchange_enable_guard_collectives,
)


default_backend = "inductor"
registered_backends = {}
registered_backend_fns = {}
native_eager_compile_caches = weakref.WeakSet()
native_eager_compile_caches_lock = threading.Lock()


class NativeEagerCompileCache:
    __slots__ = ("graphs", "lock", "__weakref__")

    def __init__(self):
        self.graphs = {}
        self.lock = threading.Lock()

    def clear(self):
        with self.lock:
            self.graphs.clear()


def new_native_eager_compile_cache():
    cache = NativeEagerCompileCache()
    with native_eager_compile_caches_lock:
        native_eager_compile_caches.add(cache)
    return cache


def reset_compile_caches():
    with native_eager_compile_caches_lock:
        caches = tuple(native_eager_compile_caches)
    for cache in caches:
        cache.clear()
