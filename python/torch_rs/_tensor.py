"""Python-owned additions to :class:`torch_rs.Tensor`."""

from .overrides import _dispatch_unary_torch_function
from .torch_rs import Tensor


def _is_shared_impl(self):
    if isinstance(self, Tensor):
        # Both ordinary owned storage and mutex-backed accumulated-gradient
        # storage are process-local. Views retain the same local allocation.
        return False

    # Preserve PyTorch's duck-typed unbound-call behavior without exposing a
    # storage API on native Tensor objects.
    return self._typed_storage()._is_shared()


def is_shared(self):
    r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
    return _dispatch_unary_torch_function(
        Tensor.is_shared,
        _is_shared_impl,
        self,
        {},
    )


# PyTorch defines this method in ``torch._tensor.Tensor`` rather than on the
# native ``torch._C.TensorBase``. Install the Python function onto the native
# public class while retaining the same ownership metadata and pickle path.
is_shared.__qualname__ = "Tensor.is_shared"
Tensor.is_shared = is_shared
del is_shared
