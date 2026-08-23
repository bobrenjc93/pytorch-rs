r"""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These methods are used to collate samples fetched from dataset into Tensor(s).
These **need** to be in global scope since Py2 doesn't support serializing
static methods.

`default_collate` and `default_convert` are exposed to users via 'dataloader.py'.
"""

import collections.abc

from torch_rs import Tensor


_NUMPY_ERROR = "default_convert(): NumPy arrays and scalars are not supported"
_CONTAINER_ERROR = (
    "default_convert(): recursive Mapping, sequence, and named-tuple inputs "
    "are not supported"
)


def _is_numpy_array_or_scalar(data):
    return any(
        base.__module__ == "numpy" and base.__name__ in {"generic", "ndarray"}
        for base in type(data).__mro__
    )


def default_convert(data):
    r"""
    Convert each NumPy array element into a :class:`torch.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
    If the input is not a NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~torch.utils.data.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~torch.utils.data.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with `int`
        >>> default_convert(0)
        0
        >>> # Example with NumPy array
        >>> default_convert(np.array([0, 1]))
        tensor([0, 1])
        >>> # Example with NamedTuple
        >>> Point = namedtuple("Point", ["x", "y"])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert(Point(np.array(0), np.array(0)))
        Point(x=tensor(0), y=tensor(0))
        >>> # Example with List
        >>> default_convert([np.array([0, 1]), np.array([2, 3])])
        [tensor([0, 1]), tensor([2, 3])]
    """
    if type(data) is Tensor:
        return data
    if _is_numpy_array_or_scalar(data):
        raise TypeError(_NUMPY_ERROR)
    if isinstance(data, collections.abc.Mapping):
        raise TypeError(_CONTAINER_ERROR)
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        raise TypeError(_CONTAINER_ERROR)
    if isinstance(data, collections.abc.Sequence) and not isinstance(
        data, (str, bytes)
    ):
        raise TypeError(_CONTAINER_ERROR)
    return data
