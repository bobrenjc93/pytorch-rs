from typing import Optional

from . import _C as _C


__all__ = ["__version__", "cuda"]

__version__ = _C.__version__

# CUDA build metadata is supplied by the native extension, not discovered from
# host drivers. The current native build is CPU-only, so the false build flag
# short-circuits before requiring a CUDA toolkit version from a future backend.
cuda: Optional[str] = _C._cuda_version if _C._has_cuda else None
