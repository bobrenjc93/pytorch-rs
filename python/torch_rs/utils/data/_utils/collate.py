"""Leaf-only collation helpers for :mod:`torch_rs.utils.data`."""


_UNSUPPORTED_ERROR = (
    "default_collate(): tensor, numeric, mapping, and nested sequence batches "
    "are not supported"
)


def default_collate(batch):
    r"""
    Return a supported string or bytes batch unchanged.

    A nonempty batch is classified from its first element, matching PyTorch's
    dispatch rule. When that element is a :class:`str` or :class:`bytes`
    instance, the original batch object is returned and later elements are not
    inspected.

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
    if isinstance(elem, (str, bytes)):
        return batch
    raise TypeError(_UNSUPPORTED_ERROR)
