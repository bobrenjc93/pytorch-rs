from typing import Optional

from . import torch_rs as _C


__all__ = ["__version__", "cuda"]
__version__ = _C.__version__
cuda: Optional[str] = None
if _C._has_cuda:
    cuda = _C._cuda_version
