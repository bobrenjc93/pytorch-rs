"""PyTorch-compatible standalone size values."""

from numbers import Integral as _Integral
from operator import index as _index


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_UINT64_MODULUS = 1 << 64


def _type_name(value):
    value_type = type(value)
    if value_type.__module__ == "numpy":
        return f"numpy.{value_type.__name__}"
    return value_type.__name__


def _dimension_index(value, position):
    try:
        integer = _index(value)
    except Exception:
        raise TypeError(
            "torch.Size() takes an iterable of 'int' "
            f"(item {position} is '{_type_name(value)}')"
        ) from None

    return integer


def _unpack_long_long(value):
    integer = _index(value)
    if integer < _INT64_MIN or integer > _INT64_MAX:
        raise ValueError("Overflow when unpacking long long")
    return integer


def _preserves_integral_identity(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    return type(value).__module__ == "numpy" and isinstance(value, _Integral)


class Size(tuple):
    __slots__ = ()
    __module__ = "torch_rs"

    def __new__(cls, *args, **kwargs):
        # Let tuple bind the constructor so its positional-only and arity
        # diagnostics stay identical to PyTorch's tuple-backed native type.
        values = tuple(*args, **kwargs)
        normalized = []
        for position, value in enumerate(values):
            integer = _dimension_index(value, position)
            # PyTorch preserves integral scalar objects, including int and
            # NumPy integer subclasses, but canonicalizes bool and generic
            # __index__ providers to an ordinary Python int.
            if not _preserves_integral_identity(value):
                value = integer
            normalized.append(value)
        return tuple.__new__(cls, normalized)

    def __repr__(self):
        dimensions = ", ".join(
            str(_unpack_long_long(value)) for value in self
        )
        return f"torch.Size([{dimensions}])"

    def __getitem__(self, key):
        value = tuple.__getitem__(self, key)
        if isinstance(key, slice):
            return type(self)(value)
        return value

    def __add__(self, value):
        if not isinstance(value, tuple):
            raise TypeError(
                "can only concatenate tuple "
                f"(not {_type_name(value)}) to torch.Size"
            )
        return type(self)(tuple.__add__(self, value))

    def __radd__(self, value):
        if not isinstance(value, tuple):
            return NotImplemented
        return type(self)(tuple.__add__(value, self))

    def __mul__(self, value):
        return type(self)(tuple(self) * value)

    def __rmul__(self, value):
        return type(self)(tuple(self) * value)

    def numel(self, *args, **kwargs):
        r"""
        numel() -> int

        Returns the number of elements a :class:`torch.Tensor` with the given
        size would contain.
        """
        if kwargs:
            raise TypeError("Size.numel() takes no keyword arguments")
        if args:
            raise TypeError(
                f"Size.numel() takes no arguments ({len(args)} given)"
            )

        result = 1
        for value in self:
            result *= _unpack_long_long(value)
            result = (result - _INT64_MIN) % _UINT64_MODULUS + _INT64_MIN
        return result

    def __reduce__(self):
        return type(self), (tuple(self),)

    def __setattr__(self, name, value):
        raise AttributeError(
            f"'torch.Size' object has no attribute '{name}'"
        )

    def __setitem__(self, key, value):
        raise TypeError("'torch.Size' object does not support item assignment")

    def __init_subclass__(cls, **kwargs):
        raise TypeError("type 'Size' is not an acceptable base type")


# Keep the public constructor signature while retaining tuple's exact binding
# errors from the variadic implementation above.
Size.__new__.__text_signature__ = "($type, iterable=(), /)"
