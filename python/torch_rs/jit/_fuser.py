# mypy: allow-untyped-defs
"""Fuser and graph-executor compatibility helpers."""

import contextlib as _contextlib


def _graph_executor_optimize_type_error(value):
    message = (
        "_set_graph_executor_optimize(): incompatible function arguments. "
        "The following argument types are supported:\n"
        "    1. (arg0: bool) -> None\n\n"
        f"Invoked with: {value!r}"
    )
    raise TypeError(message) from None


def _has_bool_slot(value):
    return any("__bool__" in cls.__dict__ for cls in type(value).__mro__)


def _validate_graph_executor_optimize_flag(value):
    if value is None or _has_bool_slot(value):
        try:
            bool(value)
        except BaseException:
            _graph_executor_optimize_type_error(value)
        return

    _graph_executor_optimize_type_error(value)


@_contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    _validate_graph_executor_optimize_flag(should_optimize)
    yield
