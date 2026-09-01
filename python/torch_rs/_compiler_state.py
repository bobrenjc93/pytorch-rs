# This private module outlives replacement imports of ``torch_rs.compiler``.

import weakref

from .torch_rs import (
    _exchange_enable_guard_collectives as exchange_enable_guard_collectives,
)


default_backend = "inductor"
_allowed_in_graph_callable_ids = set()
_allowed_in_graph_callable_finalizers = {}


def register_allow_in_graph_callable(fn):
    fn_id = id(fn)
    _allowed_in_graph_callable_ids.add(fn_id)

    def deregister():
        _allowed_in_graph_callable_ids.discard(fn_id)
        _allowed_in_graph_callable_finalizers.pop(fn_id, None)

    _allowed_in_graph_callable_finalizers[fn_id] = weakref.finalize(fn, deregister)
