"""Leaf-only collation helpers for :mod:`torch_rs.utils.data`."""

import sys as _sys

from torch_rs import Tensor as _Tensor


_UNSUPPORTED_ERROR = (
    "default_collate(): tensor, numeric, mapping, and nested sequence batches "
    "are not supported"
)


def _unsupported_collate_fn(batch, *, collate_fn_map=None):
    raise TypeError(_UNSUPPORTED_ERROR)


def _collate_str_fn(batch, *, collate_fn_map=None):
    return batch


def _get_default_collate_fn_map():
    collate_fn_map = {_Tensor: _unsupported_collate_fn}

    # NumPy is optional and importing torch_rs must not probe external
    # runtimes. Any real NumPy value necessarily arrives after its producer
    # has loaded NumPy, so reuse that module when present.
    numpy = _sys.modules.get("numpy")
    if numpy is not None:
        collate_fn_map[numpy.ndarray] = _unsupported_collate_fn
        collate_fn_map[
            (numpy.bool_, numpy.number, numpy.object_)
        ] = _unsupported_collate_fn

    collate_fn_map[float] = _unsupported_collate_fn
    collate_fn_map[int] = _unsupported_collate_fn
    collate_fn_map[str] = _collate_str_fn
    collate_fn_map[bytes] = _collate_str_fn
    return collate_fn_map


def default_collate(batch):
    r"""
    Return a supported string or bytes batch unchanged.

    A nonempty batch is classified from its first element using PyTorch's
    ordered Tensor, optional NumPy, numeric, string, and bytes dispatch. When
    the string or bytes handler is selected, the original batch object is
    returned and later elements are not inspected.

    Tensor, numeric, mapping, and nested sequence collation are outside this
    leaf-only implementation. Those inputs raise :class:`TypeError` without
    traversing their contents. Empty and non-subscriptable inputs retain the
    ordinary Python indexing errors raised while reading the first element.

    Args:
        batch: a single batch to be collated

    Returns:
        The original batch object.

    Raises:
        TypeError: If the first element is not a string or bytes value.

    Examples:
        >>> values = ["a", "b", "c"]
        >>> default_collate(values) is values
        True
    """
    elem = batch[0]
    elem_type = type(elem)
    collate_fn_map = _get_default_collate_fn_map()

    if elem_type in collate_fn_map:
        return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

    for collate_type, collate_fn in collate_fn_map.items():
        if isinstance(elem, collate_type):
            return collate_fn(batch, collate_fn_map=collate_fn_map)

    raise TypeError(_UNSUPPORTED_ERROR)
