"""Private helpers for PyTorch-compatible storage boundary metadata."""

from typing import Any, TypeGuard

__all__ = ["is_storage"]


class TypedStorage:
    """Private placeholder used only to resolve is_storage type hints."""


class UntypedStorage:
    """Private placeholder used only to resolve is_storage type hints."""


# Keep resolved type-hint reprs aligned with PyTorch without exposing storage
# classes or a torch.storage module from the public torch_rs package.
TypedStorage.__module__ = "torch.storage"
UntypedStorage.__module__ = "torch.storage"

# Storage objects remain unsupported, so the public PyTorch-compatible boundary
# predicate intentionally has no positive cases.
_storage_classes = ()


def is_storage(obj: Any, /) -> TypeGuard["TypedStorage | UntypedStorage"]:
    r"""Returns True if `obj` is a PyTorch storage object.

    Args:
        obj (Object): Object to test
    Example::

        >>> import torch
        >>> # UntypedStorage (recommended)
        >>> tensor = torch.tensor([1, 2, 3])
        >>> storage = tensor.untyped_storage()
        >>> torch.is_storage(storage)
        True
        >>>
        >>> # TypedStorage (legacy)
        >>> typed_storage = torch.TypedStorage(5, dtype=torch.float32)
        >>> torch.is_storage(typed_storage)
        True
        >>>
        >>> # regular tensor (should return False)
        >>> torch.is_storage(tensor)
        False
        >>>
        >>> # non-storage object
        >>> torch.is_storage([1, 2, 3])
        False
    """
    return type(obj) in _storage_classes


is_storage.__module__ = "torch_rs"
