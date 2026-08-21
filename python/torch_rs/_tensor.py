"""Python-owned methods installed on :class:`torch_rs.Tensor`."""

from .overrides import _dispatch_unary_torch_function
from .torch_rs import Tensor as _NativeTensor


def _is_shared_implementation(input):
    if isinstance(input, Tensor):
        # Native tensors use process-local owned or mutex-backed gradient
        # storage. Neither representation is operating-system shared memory.
        return False
    # Preserve PyTorch's unbound-call behavior without adding storage methods
    # to the native Tensor API.
    return input._typed_storage()._is_shared()


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


_is_shared = Tensor.is_shared
Tensor = _NativeTensor
Tensor.is_shared = _is_shared

del _is_shared, _NativeTensor
