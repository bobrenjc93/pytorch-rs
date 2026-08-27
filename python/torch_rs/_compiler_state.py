# This private module outlives replacement imports of ``torch_rs.compiler``.

from _thread import allocate_lock


default_backend = "inductor"
enable_guard_collectives = False
_enable_guard_collectives_lock = allocate_lock()


def exchange_enable_guard_collectives(enabled: bool) -> bool:
    global enable_guard_collectives

    with _enable_guard_collectives_lock:
        previous_enabled = enable_guard_collectives
        enable_guard_collectives = enabled
        return previous_enabled
