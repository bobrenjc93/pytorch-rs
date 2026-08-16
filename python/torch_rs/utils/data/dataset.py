import bisect
from collections.abc import Sequence
from typing import Generic, Iterable, TypeVar

from typing_extensions import deprecated

from torch_rs import Tensor


__all__ = ["Dataset", "TensorDataset", "StackDataset", "ConcatDataset", "Subset"]

_T_co = TypeVar("_T_co", covariant=True)
_T_dict = dict[str, _T_co]
_T_tuple = tuple[_T_co, ...]
_T_stack = TypeVar("_T_stack", _T_tuple, _T_dict)


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


class StackDataset(Dataset[_T_stack]):
    r"""Dataset as a stacking of multiple datasets.

    This class is useful to assemble different parts of complex input data, given as datasets.

    Example:
        >>> # xdoctest: +SKIP
        >>> images = ImageDataset()
        >>> texts = TextDataset()
        >>> tuple_stack = StackDataset(images, texts)
        >>> tuple_stack[0] == (images[0], texts[0])
        >>> dict_stack = StackDataset(image=images, text=texts)
        >>> dict_stack[0] == {"image": images[0], "text": texts[0]}

    Args:
        *args (Dataset): Datasets for stacking returned as tuple.
        **kwargs (Dataset): Datasets for stacking returned as dict.
    """

    datasets: tuple | dict

    def __init__(self, *args: Dataset[_T_co], **kwargs: Dataset[_T_co]) -> None:
        if args:
            if kwargs:
                raise ValueError(
                    "Supported either ``tuple``- (via ``args``) or"
                    "``dict``- (via ``kwargs``) like input/output, but both types are given."
                )
            self._length = len(args[0])
            if any(self._length != len(dataset) for dataset in args):
                raise ValueError("Size mismatch between datasets")
            self.datasets = args
        elif kwargs:
            tmp = list(kwargs.values())
            self._length = len(tmp[0])
            if any(self._length != len(dataset) for dataset in tmp):
                raise ValueError("Size mismatch between datasets")
            self.datasets = kwargs
        else:
            raise ValueError("At least one dataset should be passed")

    def __getitem__(self, index):
        if isinstance(self.datasets, dict):
            return {k: dataset[index] for k, dataset in self.datasets.items()}
        return tuple(dataset[index] for dataset in self.datasets)

    def __getitems__(self, indices: list):
        # add batched sampling support when parent datasets supports it.
        if isinstance(self.datasets, dict):
            dict_batch: list[_T_dict] = [{} for _ in indices]
            for k, dataset in self.datasets.items():
                if callable(getattr(dataset, "__getitems__", None)):
                    items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                    if len(items) != len(indices):
                        raise ValueError(
                            "Nested dataset's output size mismatch."
                            f" Expected {len(indices)}, got {len(items)}"
                        )
                    for data, d_sample in zip(items, dict_batch, strict=True):
                        d_sample[k] = data
                else:
                    for idx, d_sample in zip(indices, dict_batch, strict=True):
                        d_sample[k] = dataset[idx]
            return dict_batch

        # tuple data
        list_batch: list[list] = [[] for _ in indices]
        for dataset in self.datasets:
            if callable(getattr(dataset, "__getitems__", None)):
                items = dataset.__getitems__(indices)  # type: ignore[attr-defined]
                if len(items) != len(indices):
                    raise ValueError(
                        "Nested dataset's output size mismatch."
                        f" Expected {len(indices)}, got {len(items)}"
                    )
                for data, t_sample in zip(items, list_batch, strict=True):
                    t_sample.append(data)
            else:
                for idx, t_sample in zip(indices, list_batch, strict=True):
                    t_sample.append(dataset[idx])
        tuple_batch: list[_T_tuple] = [tuple(sample) for sample in list_batch]
        return tuple_batch

    def __len__(self) -> int:
        return self._length


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
