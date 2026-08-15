from typing import Generic, TypeVar

from torch_rs import Tensor


__all__ = ["Dataset", "TensorDataset"]

_T_co = TypeVar("_T_co", covariant=True)


class Dataset(Generic[_T_co]):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index) -> _T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")


def _leading_size(tensor):
    if isinstance(tensor, Tensor):
        if tensor.ndim == 0:
            raise IndexError("Dimension specified as 0 but tensor has no dimensions")
        return len(tensor)
    return tensor.size(0)


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    tensors: tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor) -> None:
        if any(
            _leading_size(tensors[0]) != _leading_size(tensor)
            for tensor in tensors
        ):
            raise AssertionError("Size mismatch between tensors")
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return _leading_size(self.tensors[0])
