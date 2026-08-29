from collections.abc import Callable as _Callable

from . import _serialization_state as _state


def _add_safe_globals(
    safe_globals: list[_Callable | tuple[_Callable, str]],
):
    _state.marked_safe_globals_set = _state.marked_safe_globals_set.union(
        set(safe_globals)
    )


def _get_safe_globals() -> list[_Callable | tuple[_Callable, str]]:
    return list(_state.marked_safe_globals_set)


def _clear_safe_globals():
    _state.marked_safe_globals_set = set()


def _remove_safe_globals(
    globals_to_remove: list[_Callable | tuple[_Callable, str]],
):
    _state.marked_safe_globals_set = _state.marked_safe_globals_set - set(
        globals_to_remove
    )


class _safe_globals:
    def __init__(self, safe_globals: list[_Callable | tuple[_Callable, str]]):
        self.safe_globals = safe_globals

    def __enter__(self):
        _add_safe_globals(self.safe_globals)

    def __exit__(self, type, value, tb):
        _remove_safe_globals(self.safe_globals)
