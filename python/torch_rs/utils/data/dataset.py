import bisect
from collections.abc import Sequence
from typing import Generic, Iterable, TypeVar

from typing_extensions import deprecated

from torch_rs import Tensor


__all__ = ["Dataset", "TensorDataset", "ConcatDataset", "Subset"]

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

    def __add__(self, other: "Dataset[_T_co]") -> "ConcatDataset[_T_co]":
        return ConcatDataset([self, other])


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


class ConcatDataset(Dataset[_T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: list[Dataset[_T_co]]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        if len(self.datasets) == 0:
            raise AssertionError("datasets should not be an empty iterable")
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    @deprecated(
        "`cummulative_sizes` attribute is renamed to `cumulative_sizes`",
        category=FutureWarning,
    )
    def cummulative_sizes(self):
        return self.cumulative_sizes


class Subset(Dataset[_T_co]):
    r"""
    Subset of a dataset at specified indices.

    .. note::
        When subclassing `Subset` and overriding `__getitem__`, you **must** also
        override `__getitems__` to ensure `DataLoader` works correctly with your
        custom logic. If you override only `__getitem__`, a `NotImplementedError`
        will be raised when using `DataLoader`.

        A simple implementation of `__getitems__` can delegate to `__getitem__`:

        .. code-block:: python

            def __getitems__(self, indices):
                return [self.__getitem__(idx) for idx in indices]

        For better performance, consider implementing batch-aware logic in
        `__getitems__` instead of calling `__getitem__` multiple times.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    dataset: Dataset[_T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[_T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

        # Check if __getitem__ is overridden but __getitems__ is not
        if (
            type(self).__getitem__ is not Subset.__getitem__
            and type(self).__getitems__ is Subset.__getitems__
        ):
            raise NotImplementedError(
                f"{type(self).__name__} overrides __getitem__ but not __getitems__. "
                "When subclassing Subset and overriding __getitem__, you must also override "
                "__getitems__ to ensure DataLoader works correctly with your custom logic. "
                "A simple implementation:\n\n"
                "def __getitems__(self, indices):\n"
                "    return [self.__getitem__(idx) for idx in indices]"
            )

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list[_T_co]:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__(  # type: ignore[attr-defined]
                [self.indices[idx] for idx in indices]
            )
        else:
            return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)
