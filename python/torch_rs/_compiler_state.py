# This private module outlives replacement imports of ``torch_rs.compiler``.

from .torch_rs import (
    _exchange_enable_guard_collectives as exchange_enable_guard_collectives,
)


default_backend = "inductor"

# These compatibility-only tables preserve the observable eager lifecycle of
# ``torch.compiler.allow_in_graph`` without providing a compiler or graph
# execution engine. They intentionally live outside ``torch_rs.compiler`` so
# module reloads do not discard registrations.
allow_in_graph_callable_ids: set[int] = set()
allow_in_graph_lazy_modules = {"einops": None}
