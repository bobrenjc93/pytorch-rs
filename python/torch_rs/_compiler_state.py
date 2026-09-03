# This private module outlives replacement imports of ``torch_rs.compiler``.

from .torch_rs import (
    _exchange_enable_guard_collectives as exchange_enable_guard_collectives,
)


default_backend = "inductor"
registered_backends = {}
registered_backend_fns = {}
