"""Python-owned methods installed on :class:`torch_rs.Tensor`."""

from numbers import Integral as _Integral
from operator import index as _index

from .overrides import _dispatch_unary_torch_function
from .torch_rs import Tensor as _NativeTensor


_NATIVE_VIEW = _NativeTensor.view
_SIGNED_64_MIN = -(1 << 63)
_SIGNED_64_MAX = (1 << 63) - 1
_UNSIGNED_64_MODULUS = 1 << 64


def _is_shared_implementation(input):
    if type(input) is Tensor:
        # Native tensors use process-local owned or mutex-backed gradient
        # storage. Neither representation is operating-system shared memory.
        return False
    # Preserve PyTorch's unbound-call behavior without adding storage methods
    # to the native Tensor API.
    return input._typed_storage()._is_shared()


def _python_type_name(value):
    value_type = type(value)
    module = value_type.__module__
    if module == "numpy" or module.startswith("numpy."):
        return f"numpy.{value_type.__name__}"
    if module == "torch" or module.startswith("torch."):
        return f"torch.{value_type.__name__}"
    if module == "torch_rs" or module.startswith("torch_rs."):
        return f"torch.{value_type.__name__}"
    return value_type.__name__


def _parse_unflatten_dimension(dim):
    if type(dim) is bool or not isinstance(dim, _Integral):
        raise TypeError(
            "unflatten(): argument 'dim' (position 1) must be int, not "
            f"{_python_type_name(dim)}"
        )
    try:
        dimension = _index(dim)
    except Exception:
        raise TypeError(
            "unflatten(): argument 'dim' (position 1) must be int, not "
            f"{_python_type_name(dim)}"
        ) from None
    if dimension < _SIGNED_64_MIN or dimension > _SIGNED_64_MAX:
        raise ValueError("Overflow when unpacking long long")
    return dimension


def _unflatten_size_type_error(value):
    return TypeError(
        "unflatten(): argument 'sizes' (position 2) must be tuple of ints, "
        f"but found element of type {_python_type_name(value)} at pos 0"
    )


def _is_symbolic_integer(value):
    value_type = type(value)
    return value_type.__name__ == "SymInt" and (
        value_type.__module__ == "torch"
        or value_type.__module__.startswith("torch.")
    )


def _parse_unflatten_sizes(sizes):
    if not isinstance(sizes, (tuple, list)):
        raise TypeError(
            "unflatten(): argument 'sizes' (position 2) must be tuple of ints, "
            f"not {_python_type_name(sizes)}"
        )

    first = sizes[0]
    if type(first) is bool or _is_symbolic_integer(first):
        raise _unflatten_size_type_error(first)
    try:
        _index(first)
    except Exception:
        raise _unflatten_size_type_error(first) from None

    parsed = []
    for position, value in enumerate(sizes, 1):
        if _is_symbolic_integer(value):
            raise TypeError(
                "unflatten(): argument 'sizes' failed to unpack the object at "
                f'pos {position} with error "type must be tuple of ints,but got '
                f'{_python_type_name(value)}"'
            )
        try:
            dimension = _index(value)
        except Exception:
            raise TypeError(
                "unflatten(): argument 'sizes' failed to unpack the object at "
                f'pos {position} with error "type must be tuple of ints,but got '
                f'{_python_type_name(value)}"'
            ) from None
        if dimension < _SIGNED_64_MIN or dimension > _SIGNED_64_MAX:
            raise TypeError(
                "unflatten(): argument 'sizes' failed to unpack the object at "
                f'pos {position} with error "Overflow when unpacking long long"'
            )
        parsed.append(dimension)
    return parsed


def _wrapping_signed_64_product(values):
    product = 1
    for value in values:
        product = (product * value) % _UNSIGNED_64_MODULUS
        if product > _SIGNED_64_MAX:
            product -= _UNSIGNED_64_MODULUS
    return product


def _unflatten_unexpected_error(message):
    return RuntimeError(f"unflatten got an unexpected error:\n{message}")


def _resolve_unflatten_sizes(sizes, dimension_size):
    inferred_index = None
    for position, size in enumerate(sizes):
        if size == -1:
            if inferred_index is not None:
                raise _unflatten_unexpected_error(
                    "only one dimension can be inferred"
                )
            inferred_index = position
        elif size < 0:
            raise _unflatten_unexpected_error(
                f"invalid shape dimension {size} at index {position} of shape {sizes}"
            )

    specified_product = _wrapping_signed_64_product(
        size for size in sizes if size != -1
    )
    resolved = list(sizes)
    if inferred_index is not None:
        if specified_product == 0:
            if dimension_size == 0:
                raise _unflatten_unexpected_error(
                    "cannot reshape tensor of 0 elements into shape "
                    f"{sizes} because the unspecified dimension size -1 can be "
                    "any value and is ambiguous"
                )
            return None
        if specified_product < 0 or dimension_size % specified_product != 0:
            return None
        resolved[inferred_index] = dimension_size // specified_product

    if _wrapping_signed_64_product(resolved) != dimension_size:
        return None
    return resolved


def _unflatten_implementation(input, dim, sizes):
    if not sizes:
        raise RuntimeError("unflatten: sizes must be non-empty")

    dimension = _parse_unflatten_dimension(dim)
    parsed_sizes = _parse_unflatten_sizes(sizes)
    rank = len(input.shape)
    effective_rank = max(rank, 1)
    if dimension < -effective_rank or dimension >= effective_rank:
        raise IndexError(
            "Dimension out of range (expected to be in range of "
            f"[-{effective_rank}, {effective_rank - 1}], but got {dimension})"
        )
    if rank == 0:
        raise _unflatten_unexpected_error(
            "Dimension specified as 0 but tensor has no dimensions"
        )
    if dimension < 0:
        dimension += rank

    dimension_size = input.shape[dimension]
    resolved_sizes = _resolve_unflatten_sizes(parsed_sizes, dimension_size)
    if resolved_sizes is None:
        raise RuntimeError(
            f"unflatten: Provided sizes {parsed_sizes} don't multiply up to the "
            f"size of dim {dimension} ({dimension_size}) in the input tensor"
        )

    shape = (*input.shape[:dimension], *resolved_sizes, *input.shape[dimension + 1 :])
    try:
        return _NATIVE_VIEW(input, shape)
    except RuntimeError as error:
        if str(error) == "tensor element count overflowed usize":
            raise RuntimeError("numel: integer multiplication overflow") from None
        raise


# Defining the method in a Python class gives it PyTorch's function metadata.
# The module binding is replaced with the native class immediately afterward.
class Tensor:
    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        return _dispatch_unary_torch_function(
            Tensor.is_shared,
            _is_shared_implementation,
            self,
            {},
        )

    def unflatten(self, dim, sizes):
        r"""
        unflatten(dim, sizes) -> Tensor

        See :func:`torch.unflatten`.

        """
        return _dispatch_unary_torch_function(
            Tensor.unflatten,
            _unflatten_implementation,
            self,
            {},
            positional_arguments=(dim, sizes),
        )


_is_shared = Tensor.is_shared
_unflatten = Tensor.unflatten
Tensor = _NativeTensor
Tensor.is_shared = _is_shared
Tensor.unflatten = _unflatten

del _is_shared, _unflatten, _NativeTensor
