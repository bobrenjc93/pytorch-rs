from __future__ import annotations

import copy

import torch_rs as torch


_CPU_DEVICE = torch.device("cpu")
_UNSUPPORTED_TYPE_MESSAGE = (
    "default_collate(): only batches of exact native CPU float32 tensors, "
    "exact Python str or bytes leaves, and "
    "matching list, tuple, namedtuple, or dict containers are supported; found {}"
)
_UNSUPPORTED_TENSOR_MESSAGE = (
    "default_collate(): only exact native CPU float32 Tensor batches are supported"
)
_UNSUPPORTED_CONVERT_TYPE_MESSAGE = (
    "default_convert(): only exact native tensors, exact Python bool, int, "
    "float, complex, None, strings, bytes, and "
    "list, tuple, namedtuple, or dict containers are supported; found {}"
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


def _convert_dict(data):
    elem_type = type(data)
    converted = {key: _convert(data[key]) for key in data}
    if elem_type is dict:
        return converted

    clone = copy.copy(data)
    clone.clear()
    clone.update(converted)
    return clone


def _convert_namedtuple(data):
    elem_type = type(data)
    return elem_type(*(_convert(value) for value in data))


def _convert_sequence(data):
    converted = [_convert(value) for value in data]
    if isinstance(data, tuple):
        return converted
    if type(data) is list:
        return converted

    clone = copy.copy(data)
    for index, value in enumerate(converted):
        clone[index] = value
    return clone


def _convert(data):
    if type(data) is torch.Tensor:
        return data
    if data is None or type(data) in (bool, int, float, complex):
        return data
    if isinstance(data, (str, bytes)):
        return data
    if isinstance(data, dict):
        return _convert_dict(data)
    if _is_namedtuple(data):
        return _convert_namedtuple(data)
    if isinstance(data, (list, tuple)):
        return _convert_sequence(data)
    raise TypeError(_UNSUPPORTED_CONVERT_TYPE_MESSAGE.format(type(data)))


def _collate(batch):
    elem = batch[0]
    if type(elem) is torch.Tensor:
        _ensure_tensor_batch(batch)
        return torch.stack(batch, dim=0)
    if type(elem) in (str, bytes):
        for index, value in enumerate(batch):
            if type(value) is not type(elem):
                raise TypeError(
                    f"default_collate(): expected exact Python {type(elem).__name__} "
                    f"as element {index} in batch, but got {type(value).__name__}"
                )
        return batch
    if isinstance(elem, dict):
        return _collate_dict(batch, elem)
    if _is_namedtuple(elem):
        return _collate_namedtuple(batch, elem)
    if isinstance(elem, (list, tuple)):
        return _collate_sequence(batch, elem)
    raise TypeError(_UNSUPPORTED_TYPE_MESSAGE.format(type(elem)))


def default_collate(batch):
    r"""Collate a non-empty tensor or text batch, including matching containers.

    The supported subset mirrors PyTorch's tensor default-collation behavior for
    exact native CPU float32 tensor leaves by returning ``torch.stack(batch,
    dim=0)``. Homogeneous batches of exact Python ``str`` or ``bytes`` leaves
    are returned unchanged, preserving batch type and leaf identity. Lists,
    plain tuples, namedtuples, and dicts are traversed recursively when every
    batch element has the same structure. Lists, namedtuples, and dicts
    preserve container type and dict key order; plain
    tuples return lists for PyTorch compatibility.

    Text fields collected from dicts are lists; those transposed from sequences
    or namedtuples are tuples, matching PyTorch.

    NumPy inputs, numeric scalars, text subclasses, mixed leaf batches,
    arbitrary objects, worker shared-memory collation, and tensors outside the
    exact native CPU float32 subset remain unsupported.
    """
    return _collate(batch)


def default_convert(data):
    r"""Convert a supported single data point without batching.

    Exact native tensors, exact Python bool, int, float, and complex values,
    None, strings, and bytes are returned unchanged without constructing
    tensors. Lists, plain tuples, namedtuples, and dicts are traversed
    recursively; lists, namedtuples, and dicts preserve container type and dict
    key order, while plain tuples return lists for PyTorch compatibility.

    NumPy arrays and scalars, numeric scalar subclasses, foreign tensors,
    arbitrary objects, batch stacking, and the broader :func:`default_collate`
    conversion surface remain unsupported.
    """
    return _convert(data)
