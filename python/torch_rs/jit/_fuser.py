"""JIT fusion helpers.

This module carries eager-only compatibility shims for the JIT fuser surface.
It does not enable TorchScript graph execution or fusion.
"""

import contextlib as _contextlib


def _repr_for_pybind_error(value):
    try:
        return repr(value)
    except BaseException:
        return "<repr raised Error>"


def _invalid_graph_executor_optimize_error(value):
    return TypeError(
        "_set_graph_executor_optimize(): incompatible function arguments. "
        "The following argument types are supported:\n"
        "    1. (arg0: bool) -> None\n\n"
        f"Invoked with: {_repr_for_pybind_error(value)}"
    )


def _validate_graph_executor_optimize(value):
    if value is None:
        return
    if not hasattr(type(value), "__bool__"):
        raise _invalid_graph_executor_optimize_error(value)
    try:
        bool(value)
    except BaseException:
        raise _invalid_graph_executor_optimize_error(value) from None


@_contextlib.contextmanager
def optimized_execution(should_optimize):
    """Context manager that controls whether the JIT's executor will run optimizations before executing a function."""
    _validate_graph_executor_optimize(should_optimize)
    yield
