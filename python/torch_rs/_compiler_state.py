# This private module outlives replacement imports of ``torch_rs.compiler``.

import threading
import weakref

# Imported eagerly by package initialization, before callers can replace the
# native module's writable owner export. Lazy frontends must use this identity
# instead of reading that export at their first invocation.
from .torch_rs import (
    _VariableFunctionsClass as native_function_owner,
    _exchange_enable_guard_collectives as exchange_enable_guard_collectives,
)


default_backend = "inductor"
registered_backends = {}
registered_backend_fns = {}
native_eager_compile_caches = weakref.WeakSet()
native_eager_compile_caches_lock = threading.Lock()
native_cuda_compile_executors = weakref.WeakSet()
native_cuda_compile_executors_lock = threading.Lock()


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


def register_native_cuda_compile_executor(executor):
    with native_cuda_compile_executors_lock:
        native_cuda_compile_executors.add(executor)
    return executor


def reset_compile_caches():
    with native_eager_compile_caches_lock:
        caches = tuple(native_eager_compile_caches)
    for cache in caches:
        cache.clear()
    with native_cuda_compile_executors_lock:
        executors = tuple(native_cuda_compile_executors)
    for executor in executors:
        close = getattr(executor, "close", None)
        if close is not None:
            close()
