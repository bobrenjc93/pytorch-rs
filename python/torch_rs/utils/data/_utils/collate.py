from __future__ import annotations

import copy

import torch_rs as torch


_CPU_DEVICE = torch.device("cpu")
_UNSUPPORTED_TYPE_MESSAGE = (
    "default_collate(): only batches of exact native CPU float32 tensors and "
    "matching list, tuple, namedtuple, or dict containers are supported; found {}"
)
_UNSUPPORTED_TENSOR_MESSAGE = (
    "default_collate(): only exact native CPU float32 Tensor batches are supported"
)


def _is_namedtuple(value):
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _ensure_tensor_batch(batch):
    for index, value in enumerate(batch):
        if type(value) is not torch.Tensor:
            raise TypeError(
                "default_collate(): expected exact native Tensor as element "
                f"{index} in batch, but got {type(value).__name__}"
            )
        if (
            value.dtype is not torch.float32
            or value.device != _CPU_DEVICE
            or value.layout is not torch.strided
        ):
            raise NotImplementedError(_UNSUPPORTED_TENSOR_MESSAGE)


def _collate_dict(batch, elem):
    elem_type = type(elem)
    keys = tuple(elem.keys())
    key_set = set(keys)
    for value in batch:
        if type(value) is not elem_type:
            raise TypeError(
                "default_collate(): each dict batch element must have the same "
                "container type"
            )
        if set(value.keys()) != key_set:
            raise RuntimeError(
                "each element in dict batch should have the same keys"
            )

    collated = {key: _collate([value[key] for value in batch]) for key in keys}
    if elem_type is dict:
        return collated

    clone = copy.copy(elem)
    clone.clear()
    clone.update(collated)
    return clone


def _collate_namedtuple(batch, elem):
    elem_type = type(elem)
    for value in batch:
        if type(value) is not elem_type:
            raise TypeError(
                "default_collate(): each namedtuple batch element must have "
                "the same container type"
            )
    return elem_type(*(_collate(samples) for samples in zip(*batch)))


def _collate_sequence(batch, elem):
    elem_type = type(elem)
    elem_size = len(elem)
    for value in batch:
        if type(value) is not elem_type:
            raise TypeError(
                "default_collate(): each sequence batch element must have the "
                "same container type"
            )
        if len(value) != elem_size:
            raise RuntimeError("each element in list of batch should be of equal size")

    collated = [_collate(samples) for samples in zip(*batch)]
    if isinstance(elem, tuple):
        return collated
    if elem_type is list:
        return collated
    if isinstance(elem, list):
        clone = copy.copy(elem)
        for index, value in enumerate(collated):
            clone[index] = value
        return clone
    return elem_type(collated)


def _collate(batch):
    elem = batch[0]
    if type(elem) is torch.Tensor:
        _ensure_tensor_batch(batch)
        return torch.stack(batch, dim=0)
    if isinstance(elem, dict):
        return _collate_dict(batch, elem)
    if _is_namedtuple(elem):
        return _collate_namedtuple(batch, elem)
    if isinstance(elem, (list, tuple)):
        return _collate_sequence(batch, elem)
    raise TypeError(_UNSUPPORTED_TYPE_MESSAGE.format(type(elem)))


def default_collate(batch):
    r"""Collate a non-empty tensor batch or matching tensor container batch.

    The supported subset mirrors PyTorch's tensor default-collation behavior for
    exact native CPU float32 tensor leaves by returning ``torch.stack(batch,
    dim=0)``. Lists, plain tuples, namedtuples, and dicts are traversed
    recursively when every batch element has the same structure. Lists,
    namedtuples, and dicts preserve container type and dict key order; plain
    tuples return lists for PyTorch compatibility.

    NumPy arrays, strings, bytes, numeric scalars, arbitrary objects, worker
    shared-memory collation, and tensors outside the exact native CPU float32
    subset remain unsupported.
    """
    return _collate(batch)
